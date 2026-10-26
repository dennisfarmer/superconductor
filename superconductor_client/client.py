"""Magenta RealTime 2 (MRT2) client for `superconductor_server/mrt2_server.py`.

The server runs either locally (`make server`, mlx + mrt2_small) or on the
cluster (jax + mrt2_base, reached through an SSH tunnel); this client doesn't
care which. Same start / update_recipe / update_controls / poll_stats / stop /
connected interface as the old in-process LocalMagentaClient, plus set_paused.

The websocket runs in a separate *process* so that it never competes with the
tracking loop for the GIL. The parent process only sends small control messages
over a queue. This process opens the server's `GET /stream/input` and sends
recipe, controls, pattern, and pause. It does not play audio and it does not
grant credits. A player (the camera page, the playback plugin, or
audio_receiver.py) opens `GET /stream/playback`, sends `Credits` once, and
`ReceivedChunk` for each block it finishes.

Until the server answers (it may still be loading the model, or not started yet), and
again whenever it goes away, the child process retries every RETRY_S seconds;
client_midi stays up meanwhile, so the order things are started in doesn't matter.

client_midi: the child process also serves a small HTTP API (default port 8470)
through which another program (e.g. SuperConductor Atlas) gives MRT2 its NOTES and
DRUMS input, while this client keeps providing the prompts:
    POST /pattern   Pattern body (see mrt2_server.py), forwarded to the server
    POST /controls  UpdateControls body ({temperature?, top_k?, cfg_scales?}), forwarded
                    to the server
    POST /recipe    {prompt: weight, ...}: the style recipe, forwarded as UpdateRecipe
    POST /pause     {"paused": true|false}
    POST /input     the camera overlay's view of prompts and controls (display only)
    POST /prompts   {prompt: weight, ...}: extra style prompts, added to the recipe this
                    client sends (tracking's recipe, source "atlas"); weight 0
                    removes a prompt. Each POST replaces the previous set.
    GET  /status    {connected, model, last_pattern_id, paused, info, stats, prompts}
    GET  /input     the prompts and controls this client sends to MRT2, as its camera
                    overlay shows them (collab.CollabFrontend.mrt2_input):
                    {connected, rows: [{prompt, source, value, share, color}], controls}
"""
import argparse
import asyncio
import json
import logging
import multiprocessing as mp
import queue
import urllib.error
import urllib.request

logger = logging.getLogger(__name__)

RETRY_S = 2.0  # between attempts to reach the MRT2 server


def _publish(stats_q, link, msg):
    """Hand a status message to the parent queue, and keep a copy on `link` so
    GET /status can answer when this process is the client itself."""
    stats_q.put(msg)
    if link is None:
        return
    if msg.get("ready"):
        link["info"] = msg
    elif msg.get("disconnected"):
        link.pop("info", None)
    elif "prompts" in msg:
        link["prompts"] = msg["prompts"]
    else:
        link["stats"] = msg


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


async def _start_client_midi(port, link, stats_q, control_q):
    """HTTP front door for note sources; `link` holds the server websocket once
    connected ({"ws", "model", "last_pattern_id"}). POST /prompts goes to the
    parent over `stats_q`."""
    from aiohttp import web

    cors = {"Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Content-Type",
            "Access-Control-Allow-Methods": "GET, POST, OPTIONS"}

    def reply(payload, status=200):
        return web.json_response(payload, status=status, headers=cors)

    async def status(_request):
        return reply({"connected": link.get("ws") is not None, "model": link.get("model"),
                      "last_pattern_id": link.get("last_pattern_id"),
                      "paused": bool(link.get("paused")),
                      "info": link.get("info") or {},
                      "stats": link.get("stats") or {},
                      "prompts": link.get("prompts") or {}})

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

    async def controls(request):
        try:
            body = await request.json()
        except ValueError:
            return reply({"ok": False, "error": "body must be JSON"}, 400)
        if not isinstance(body, dict):
            return reply({"ok": False, "error": "body must be an object"}, 400)
        ws = link.get("ws")
        if ws is None:
            return reply({"ok": False, "error": "not connected to the MRT2 server"}, 503)
        await ws.send(json.dumps({"type": "UpdateControls", "body": body}))
        return reply({"ok": True})

    async def prompts(request):
        try:
            body = await request.json()
            weights = {str(p): float(w) for p, w in body.items()}
        except (ValueError, TypeError, AttributeError):
            return reply({"ok": False, "error": "body must be an object of prompt: weight"}, 400)
        if any(w < 0 for w in weights.values()):
            return reply({"ok": False, "error": "weights must be >= 0"}, 400)
        _publish(stats_q, link, {"prompts": {p: w for p, w in weights.items() if w > 0}})
        return reply({"ok": True})

    async def recipe(request):
        try:
            body = await request.json()
        except ValueError:
            return reply({"ok": False, "error": "body must be JSON"}, 400)
        if not isinstance(body, dict):
            return reply({"ok": False, "error": "body must be an object"}, 400)
        control_q.put({"type": "recipe", "recipe": body})
        return reply({"ok": True})

    async def post_input(request):
        try:
            view = await request.json()
        except ValueError:
            return reply({"ok": False, "error": "body must be JSON"}, 400)
        if not isinstance(view, dict):
            return reply({"ok": False, "error": "body must be an object"}, 400)
        control_q.put({"type": "input_view", "view": view})
        return reply({"ok": True})

    async def pause(request):
        try:
            body = await request.json()
        except ValueError:
            return reply({"ok": False, "error": "body must be JSON"}, 400)
        if not isinstance(body, dict):
            return reply({"ok": False, "error": "body must be an object"}, 400)
        paused = bool(body.get("paused"))
        link["paused"] = paused
        control_q.put({"type": "pause", "paused": paused})
        return reply({"ok": True})

    async def options(_request):
        return web.Response(status=204, headers=cors)

    app = web.Application()
    app.router.add_get("/status", status)
    app.router.add_get("/input", input_view)
    app.router.add_post("/pattern", pattern)
    app.router.add_post("/controls", controls)
    app.router.add_post("/prompts", prompts)
    app.router.add_post("/recipe", recipe)
    app.router.add_post("/input", post_input)
    app.router.add_post("/pause", pause)
    app.router.add_route("OPTIONS", "/{tail:.*}", options)
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", port).start()
    print(f"client_midi on http://localhost:{port} "
          f"(POST /pattern /controls /recipe /pause /prompts)")
    return runner


async def _client_main(control_q, stats_q, uri, credits, midi_port):
    import websockets

    link = {}  # client_midi's view of the connection
    if not midi_port:
        raise RuntimeError("midi_port is needed: note sources POST patterns to client_midi")
    try:
        midi = await _start_client_midi(midi_port, link, stats_q, control_q)
    except OSError as e:
        raise RuntimeError(f"client_midi: port {midi_port} unavailable ({e}); note sources need it") from e
    try:
        reported = False  # the current outage was printed
        while True:
            try:
                if await _stream(control_q, stats_q, uri, credits, link, websockets):
                    return  # stop
                error = "it went away"
            except (OSError, asyncio.TimeoutError, websockets.exceptions.WebSocketException) as e:
                error = f"not reachable ({e})"
            if link.pop("ws", None) is not None:  # was connected: a new outage
                _publish(stats_q, link, {"disconnected": True})
                reported = False
            if not reported:
                print(f"MRT2 server at {uri}: {error}; retrying every {RETRY_S:.0f} s")
                reported = True
            if await _wait_for_retry(control_q, link):
                return  # stop
    finally:
        link.clear()
        await midi.cleanup()


async def _wait_for_retry(control_q, link):
    """Sleep RETRY_S while not connected, still taking control messages: True on "stop".
    The page's input view is kept; recipes, controls and pauses are dropped (the
    parent sends the recipe again every update_interval)."""
    deadline = asyncio.get_running_loop().time() + RETRY_S
    while asyncio.get_running_loop().time() < deadline:
        try:
            msg = control_q.get_nowait()
        except queue.Empty:
            await asyncio.sleep(0.05)
            continue
        if msg["type"] == "stop":
            return True
        if msg["type"] == "input_view":
            link["input"] = msg["view"]
    return False


async def _stream(control_q, stats_q, uri, credits, link, websockets):
    """One input connection to the server. True when it ended on "stop", False when
    the server went away (connection errors while connecting are raised).
    `credits` is unused: the player grants the lead on /stream/playback."""
    async with websockets.connect(uri, compression=None, max_size=None) as ws:
        ready = json.loads(await ws.recv())
        _publish(stats_q, link, {"ready": True, **ready["body"]})
        link.update(ws=ws, model=ready["body"].get("model"))

        async def receive():
            async for message in ws:
                if isinstance(message, bytes):
                    continue
                msg = json.loads(message)
                if msg.get("type") == "Error":
                    print(f"MRT2 server: {msg['body'].get('error')}")

        async def send_controls():
            while True:
                try:
                    msg = control_q.get_nowait()
                except queue.Empty:
                    await asyncio.sleep(0.01)
                    continue
                if msg["type"] == "stop":
                    await ws.send(json.dumps({"type": "EndSession", "body": None}))
                    return True
                if msg["type"] == "recipe":
                    await ws.send(json.dumps({"type": "UpdateRecipe",
                                              "body": {"recipe": msg["recipe"]}}))
                elif msg["type"] == "controls":
                    await ws.send(json.dumps({"type": "UpdateControls",
                                              "body": msg["controls"]}))
                elif msg["type"] == "input_view":
                    link["input"] = msg["view"]
                elif msg["type"] == "pause":
                    link["paused"] = bool(msg["paused"])
                    await ws.send(json.dumps({"type": "Pause" if msg["paused"] else "Resume",
                                              "body": None}))

        tasks = [asyncio.create_task(t()) for t in (receive, send_controls)]
        try:
            # ends on "stop" (send_controls) or when the server goes away (receive)
            await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for t in tasks:
                t.cancel()
        control = tasks[1]
        return control.done() and not control.cancelled() and control.exception() is None and bool(control.result())


# =========================
# PARENT-SIDE CLIENT
# =========================

class MagentaClient:

    def __init__(self, server_url="ws://localhost:9100/stream/input", credits=3, midi_port=8470):
        """
        `server_url`: input websocket of a running mrt2_server.py (local
            `ws://localhost:9100/stream/input`, or the local end of the SSH tunnel
            to the cluster, `ws://localhost:9100/stream/input`)
        `credits`: kept so callers don't break; the player grants the lead on
            `/stream/playback`, not this process
        `midi_port`: port of client_midi, where note sources POST patterns
            (NOTES / DRUMS input)
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
        self.prompts = {}  # extra style prompts from client_midi's POST /prompts (Atlas)

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
                elif "prompts" in msg:
                    self.prompts = msg["prompts"]
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
        """Pause: the server stops generating. Resume continues the same music."""
        self.paused = bool(paused)
        self._control_q.put({"type": "pause", "paused": self.paused})

    def stop(self):
        if self._process.is_alive():
            self._control_q.put({"type": "stop"})
            self._process.join(timeout=3)
            if self._process.is_alive():
                self._process.terminate()
        self.connected = False


class HttpMagentaClient:
    """The client when it is already running (make client). Same calls as
    MagentaClient; they go to client_midi over HTTP. start/stop do not own
    that process."""

    def __init__(self, base_url):
        self.base = base_url.rstrip("/")
        self.connected = False
        self.info = {}
        self.stats = {}
        self.paused = False
        self.prompts = {}

    def start(self):
        self.poll_stats()

    def stop(self):
        pass

    def poll_stats(self):
        data = self._get("/status")
        if data is None:
            self.connected = False
            return self.stats
        self.connected = bool(data.get("connected"))
        self.info = data.get("info") or {}
        self.stats = data.get("stats") or {}
        self.prompts = data.get("prompts") or {}
        self.paused = bool(data.get("paused"))
        return self.stats

    def update_recipe(self, recipe):
        if recipe:
            self._post("/recipe", dict(recipe))

    def update_controls(self, controls):
        self._post("/controls", dict(controls))

    def update_input_view(self, view):
        self._post("/input", view)

    def set_paused(self, paused):
        self.paused = bool(paused)
        self._post("/pause", {"paused": self.paused})

    def _get(self, path):
        try:
            with urllib.request.urlopen(self.base + path, timeout=2) as resp:
                return json.loads(resp.read().decode())
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            return None

    def _post(self, path, payload):
        req = urllib.request.Request(
            self.base + path, data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"})
        try:
            urllib.request.urlopen(req, timeout=2).read()
        except (urllib.error.URLError, TimeoutError):
            self.connected = False


def main():
    """The client process: the MRT2 websocket and client_midi. Tracking, midi
    and Atlas talk to it; this process does not run the camera."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--server", default=None,
                        help="MRT2 input websocket. Default: the local entry in addresses.toml.")
    parser.add_argument("--credits", type=int, default=3)
    parser.add_argument("--midi-port", type=int, default=8470)
    args = parser.parse_args()
    server = args.server
    if server is None:
        from superconductor_client.addresses import load_place
        server = load_place("local").direct.input
    _client_process(queue.Queue(), queue.Queue(), server, args.credits, args.midi_port)


if __name__ == "__main__":
    main()
