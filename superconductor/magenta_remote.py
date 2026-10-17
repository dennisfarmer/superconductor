"""Magenta RealTime 2 (MRT2) client for `superconductor_server/mrt2_server.py`.

The server runs either locally (`make server`, mlx + mrt2_small) or on the
cluster (jax + mrt2_base, reached through an SSH tunnel); this client doesn't
care which. Same start / update_recipe / update_controls / poll_stats / stop /
connected interface as the old in-process LocalMagentaClient, plus set_paused.

The websocket runs in a separate *process* so that it never competes with the
tracking loop for the GIL. The parent process only sends small control messages
over a queue. Nothing plays here: the audio goes to whoever listens on WS /audio
(below), by default the camera page, or sc-audio (audio_receiver.py).

Flow control: `StartSession` grants the server `credits` blocks, and the player
acknowledges each block it finishes playing, which sends one `ReceivedChunk`, so
the server is always `credits` blocks ahead. Control latency is roughly
credits * frames_per_block * 40ms plus the one-way network delay. Buffered
audio is never dropped on a recipe change: the model state has already moved
past it, so dropping it would be an audible jump.

client_midi: the child process also serves a small HTTP API (default port 8470)
through which another program (e.g. SuperConductor Atlas) gives MRT2 its NOTES and
DRUMS input, while this client keeps providing the prompts:
    POST /pattern   Pattern body (see mrt2_server.py), forwarded to the server
    GET  /status    {connected, model, last_pattern_id}
    GET  /input     the prompts and controls this client sends to MRT2, as its camera
                    overlay shows them (collab.CollabFrontend.mrt2_input):
                    {connected, rows: [{prompt, source, value, share, color}], controls}

The same HTTP app serves the audio (see AUDIO_PROTOCOL.md):
    WS   /audio?role=clock|monitor&name=...
                    Every audio frame from the server is forwarded as is to every
                    connection. One of them is the clock: it answers {"type": "ack"}
                    each time it finishes playing a block, and only then is
                    ReceivedChunk sent. The clock is the newest connection that asked
                    for role=clock (the default); the others are monitors and never
                    ack. When the clock goes away, the next newest clock-asker takes
                    over and gets the blocks not acked yet again. Each connection is
                    told its role ({"type": "role", "role", "clock": name}). While no
                    clock is connected nothing is acked, so the server stops once its
                    credits are used up.
"""
import asyncio
import json
import logging
import multiprocessing as mp
import queue
from collections import deque

SAMPLE_RATE = 48000
CHANNELS = 2

logger = logging.getLogger(__name__)


# =========================
# CHILD PROCESS
# =========================

def _client_process(control_q, stats_q, uri, credits, midi_port):
    logging.basicConfig(level=logging.INFO)
    try:
        asyncio.run(_client_main(control_q, stats_q, uri, credits, midi_port))
    except Exception as e:  # pylint: disable=broad-exception-caught
        print(f"MRT2 connection to {uri} failed: {e}")
    stats_q.put({"disconnected": True})


class _Sink:
    def __init__(self, ws, name, wants_clock):
        self.ws, self.name, self.wants_clock = ws, name, wants_clock


class AudioSinks:
    """Everyone listening on WS /audio. `acks` gets one entry per block the clock
    finished playing (each becomes a ReceivedChunk)."""

    def __init__(self):
        self.acks = asyncio.Queue()
        self.pending = deque()  # frames sent and not acked by the clock yet, oldest first
        self.sinks = []  # in connection order
        self.clock = None  # the _Sink whose acks count
        self.underruns = 0

    async def _send(self, sink, data):
        ws = sink.ws
        if ws.closed:
            return
        try:
            if isinstance(data, bytes):
                await ws.send_bytes(data)
            else:
                await ws.send_str(json.dumps(data))
        except ConnectionError:
            pass

    async def play(self, frame):
        """a frame from the server ([uint32 seq][float32 (n, 2)]), to every listener"""
        self.pending.append(frame)
        for sink in list(self.sinks):
            await self._send(sink, frame)

    async def paused(self):
        """the silence while paused isn't an underrun"""
        for sink in list(self.sinks):
            await self._send(sink, {"type": "pause"})

    def buffered_s(self):
        return sum((len(f) - 4) // (4 * CHANNELS) for f in self.pending) / SAMPLE_RATE

    async def _update_roles(self):
        """the newest clock-asker is the clock; tell everyone their role, and give a new
        clock the blocks not acked yet"""
        clock = next((s for s in reversed(self.sinks) if s.wants_clock), None)
        changed = clock is not self.clock
        self.clock = clock
        name = clock.name if clock else None
        for sink in list(self.sinks):
            await self._send(sink, {"type": "role", "role": "clock" if sink is clock else "monitor",
                                    "clock": name})
        if changed and clock is not None:
            for frame in list(self.pending):  # what the server already sent, again
                await self._send(clock, frame)

    async def handle(self, request):
        from aiohttp import web

        ws = web.WebSocketResponse(heartbeat=20)
        await ws.prepare(request)
        role = request.query.get("role", "clock")
        sink = _Sink(ws, request.query.get("name") or "player", role != "monitor")
        self.sinks.append(sink)
        await self._update_roles()
        try:
            async for msg in ws:
                if msg.type != web.WSMsgType.TEXT or sink is not self.clock:
                    continue  # monitors don't ack
                try:
                    kind = json.loads(msg.data).get("type")
                except (ValueError, AttributeError):
                    continue
                if kind == "ack" and self.pending:
                    self.pending.popleft()
                    self.acks.put_nowait(None)
                elif kind == "underrun":
                    self.underruns += 1
        finally:
            self.sinks.remove(sink)
            await self._update_roles()
        return ws


async def _start_client_midi(port, link, sinks):
    """HTTP front door for note sources; `link` holds the server websocket once
    connected ({"ws", "model", "last_pattern_id"}). Also WS /audio for the players
    (`sinks`, an AudioSinks)."""
    from aiohttp import web

    cors = {"Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Content-Type",
            "Access-Control-Allow-Methods": "GET, POST, OPTIONS"}

    def reply(payload, status=200):
        return web.json_response(payload, status=status, headers=cors)

    async def status(_request):
        return reply({"connected": link.get("ws") is not None, "model": link.get("model"),
                      "last_pattern_id": link.get("last_pattern_id")})

    async def input_view(_request):
        view = link.get("input") or {"rows": [], "controls": {}}
        return reply({"connected": link.get("ws") is not None, **view})

    async def pattern(request):
        try:
            body = await request.json()
        except ValueError:
            return reply({"ok": False, "error": "body must be JSON"}, 400)
        if body is not None and not isinstance(body, dict):
            return reply({"ok": False, "error": "body must be an object or null"}, 400)
        ws = link.get("ws")
        if ws is None:
            return reply({"ok": False, "error": "not connected to the MRT2 server"}, 503)
        await ws.send(json.dumps({"type": "Pattern", "body": body}))
        link["last_pattern_id"] = (body or {}).get("id")
        return reply({"ok": True})

    async def options(_request):
        return web.Response(status=204, headers=cors)

    app = web.Application()
    app.router.add_get("/status", status)
    app.router.add_get("/input", input_view)
    app.router.add_post("/pattern", pattern)
    app.router.add_route("OPTIONS", "/{tail:.*}", options)
    app.router.add_get("/audio", sinks.handle)
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", port).start()
    print(f"client_midi on http://localhost:{port} (POST /pattern, WS /audio)")
    return runner


async def _client_main(control_q, stats_q, uri, credits, midi_port):
    import websockets

    link = {}  # client_midi's view of the connection
    sinks = AudioSinks()
    if not midi_port:
        raise RuntimeError("midi_port is needed: the players connect to client_midi's /audio")
    try:
        midi = await _start_client_midi(midi_port, link, sinks)
    except OSError as e:
        raise RuntimeError(f"client_midi: port {midi_port} unavailable ({e}); the players need it") from e
    try:
        await _stream(control_q, stats_q, uri, credits, link, websockets, sinks)
    finally:
        link.clear()
        await midi.cleanup()


async def _stream(control_q, stats_q, uri, credits, link, websockets, sinks):
    async with websockets.connect(uri, compression=None, max_size=None) as ws:
        ready = json.loads(await ws.recv())
        stats_q.put({"ready": True, **ready["body"]})
        link.update(ws=ws, model=ready["body"].get("model"))
        await ws.send(json.dumps({"type": "StartSession", "body": {"credits": credits}}))

        async def receive():
            async for message in ws:
                if isinstance(message, bytes):  # [uint32 LE seq][float32 LE samples (n, 2)]
                    await sinks.play(message)
                    continue
                msg = json.loads(message)
                if msg.get("type") == "Error":
                    print(f"MRT2 server: {msg['body'].get('error')}")
                elif msg.get("type") == "Stats":
                    stats_q.put({**msg["body"], "underruns": sinks.underruns,
                                 "buffered_s": sinks.buffered_s(),
                                 "player": sinks.clock.name if sinks.clock else None})

        async def send_acks():
            while True:
                await sinks.acks.get()
                await ws.send(json.dumps({"type": "ReceivedChunk", "body": None}))

        async def send_controls():
            while True:
                try:
                    msg = control_q.get_nowait()
                except queue.Empty:
                    await asyncio.sleep(0.01)
                    continue
                if msg["type"] == "stop":
                    await ws.send(json.dumps({"type": "EndSession", "body": None}))
                    return
                if msg["type"] == "recipe":
                    await ws.send(json.dumps({"type": "UpdateRecipe",
                                              "body": {"recipe": msg["recipe"]}}))
                elif msg["type"] == "controls":
                    await ws.send(json.dumps({"type": "UpdateControls",
                                              "body": msg["controls"]}))
                elif msg["type"] == "input_view":
                    link["input"] = msg["view"]
                elif msg["type"] == "pause":
                    if msg["paused"]:
                        await sinks.paused()
                    await ws.send(json.dumps({"type": "Pause" if msg["paused"] else "Resume",
                                              "body": None}))

        tasks = [asyncio.create_task(t()) for t in (receive, send_acks, send_controls)]
        try:
            # ends on "stop" (send_controls) or when the server goes away (receive)
            await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for t in tasks:
                t.cancel()


# =========================
# PARENT-SIDE CLIENT
# =========================

class MagentaClient:

    def __init__(self, server_url="ws://localhost:9100/stream", credits=3, midi_port=8470):
        """
        `server_url`: websocket of a running mrt2_server.py (local, or the
            local end of the SSH tunnel to the cluster)
        `credits`: blocks the server stays ahead of playback; control latency
            is roughly credits * frames_per_block * 40ms (+ network delay)
        `midi_port`: port of client_midi, where note sources POST patterns
            (NOTES / DRUMS input) and the players connect (WS /audio)
        """
        ctx = mp.get_context("spawn")
        self._control_q = ctx.Queue()
        self._stats_q = ctx.Queue()
        self._process = ctx.Process(
            target=_client_process,
            args=(self._control_q, self._stats_q, server_url, credits, midi_port),
            daemon=True,
        )
        self.connected = False
        self.info = {}  # the server's Ready message: model, backend, ...
        self.stats = {}
        self.paused = False

    def start(self):
        self._process.start()

    def poll_stats(self):
        """Drain stats from the client process; call from the UI loop."""
        try:
            while True:
                msg = self._stats_q.get_nowait()
                if msg.get("ready"):
                    self.connected = True
                    self.info = msg
                    print(f"MagentaRT2 ready: {msg.get('model')} ({msg.get('backend')})")
                elif msg.get("disconnected"):
                    self.connected = False
                else:
                    self.stats = msg
        except queue.Empty:
            pass
        return self.stats

    def update_recipe(self, recipe):
        """recipe example: {"Rock": 0.6, "Guitar": 0.8, "Jazz": 0.3}"""
        if recipe:
            self._control_q.put({"type": "recipe", "recipe": dict(recipe)})

    def update_controls(self, controls):
        """controls: any of {"temperature", "top_k", "cfg_musiccoca"}"""
        self._control_q.put({"type": "controls", "controls": dict(controls)})

    def update_input_view(self, view):
        """What client_midi's GET /input serves: {"rows": [...], "controls": {...}}
        (CollabFrontend.mrt2_input). Display only; nothing goes to the server."""
        self._control_q.put({"type": "input_view", "view": view})

    def set_paused(self, paused):
        """Pause: the server stops generating (after the ~credits blocks already
        sent have played, it goes silent). Resume continues the same music."""
        self.paused = bool(paused)
        self._control_q.put({"type": "pause", "paused": self.paused})

    def stop(self):
        if self._process.is_alive():
            self._control_q.put({"type": "stop"})
            self._process.join(timeout=3)
            if self._process.is_alive():
                self._process.terminate()
        self.connected = False
