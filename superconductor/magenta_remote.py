"""Magenta RealTime 2 (MRT2) client for `superconductor_server/mrt2_server.py`.

The server runs either locally (`make server`, mlx + mrt2_small) or on the
cluster (jax + mrt2_base, reached through an SSH tunnel); this client doesn't
care which. Same start / update_recipe / update_controls / poll_stats / stop /
connected interface as the old in-process LocalMagentaClient, plus set_paused.

The websocket and the sounddevice callback run in a separate *process* so that
they never compete with the webcam / YOLO loop for the GIL. The parent process
only sends small control messages over a queue.

Flow control: `StartSession` grants the server `credits` blocks, and the audio
callback sends one `ReceivedChunk` each time it finishes playing a block, so the
server is always `credits` blocks ahead. Control latency is roughly
credits * frames_per_block * 40ms plus the one-way network delay. Buffered
audio is never dropped on a recipe change: the model state has already moved
past it, so dropping it would be an audible jump.

client_midi: the child process also serves a small HTTP API (default port 8470)
through which another program (e.g. Harmonic Atlas) gives MRT2 its NOTES and
DRUMS input, while this client keeps providing the prompts:
    POST /pattern   Pattern body (see mrt2_server.py), forwarded to the server
    GET  /status    {connected, model, last_pattern_id}
"""
import asyncio
import json
import logging
import multiprocessing as mp
import queue
import struct
from collections import deque

import numpy as np

SAMPLE_RATE = 48000
CHANNELS = 2

logger = logging.getLogger(__name__)


# =========================
# CHILD PROCESS
# =========================

def _client_process(control_q, stats_q, uri, credits, output_device, midi_port):
    logging.basicConfig(level=logging.INFO)
    try:
        asyncio.run(_client_main(control_q, stats_q, uri, credits, output_device, midi_port))
    except Exception as e:  # pylint: disable=broad-exception-caught
        print(f"MRT2 connection to {uri} failed: {e}")
    stats_q.put({"disconnected": True})


async def _start_client_midi(port, link):
    """HTTP front door for note sources; `link` holds the server websocket once
    connected ({"ws", "model", "last_pattern_id"})."""
    from aiohttp import web

    cors = {"Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Content-Type",
            "Access-Control-Allow-Methods": "GET, POST, OPTIONS"}

    def reply(payload, status=200):
        return web.json_response(payload, status=status, headers=cors)

    async def status(_request):
        return reply({"connected": link.get("ws") is not None, "model": link.get("model"),
                      "last_pattern_id": link.get("last_pattern_id")})

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
    app.router.add_post("/pattern", pattern)
    app.router.add_route("OPTIONS", "/{tail:.*}", options)
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", port).start()
    print(f"client_midi on http://localhost:{port} (POST /pattern)")
    return runner


async def _client_main(control_q, stats_q, uri, credits, output_device, midi_port):
    import sounddevice as sd
    import websockets

    link = {}  # client_midi's view of the connection
    midi = None
    if midi_port:
        try:
            midi = await _start_client_midi(midi_port, link)
        except OSError as e:
            print(f"client_midi: port {midi_port} unavailable ({e}); notes/drums input off")
    try:
        await _stream(control_q, stats_q, uri, credits, output_device, link, sd, websockets)
    finally:
        link.clear()
        if midi is not None:
            await midi.cleanup()


async def _stream(control_q, stats_q, uri, credits, output_device, link, sd, websockets):

    loop = asyncio.get_running_loop()
    acks = asyncio.Queue()  # one entry per block the callback finished playing
    buffer = deque()  # np.ndarray blocks of shape (n, 2)
    underruns = [0]
    started = [False]

    def callback(outdata, frames, time_info, status):
        filled = 0
        while filled < frames and buffer:
            block = buffer[0]
            n = min(len(block), frames - filled)
            outdata[filled:filled + n] = block[:n]
            filled += n
            if n == len(block):
                buffer.popleft()
                loop.call_soon_threadsafe(acks.put_nowait, None)
            else:
                buffer[0] = block[n:]
        if filled < frames:
            outdata[filled:].fill(0)
            if started[0]:  # don't count the silence before the first block
                underruns[0] += 1

    async with websockets.connect(uri, compression=None, max_size=None) as ws:
        ready = json.loads(await ws.recv())
        stats_q.put({"ready": True, **ready["body"]})
        link.update(ws=ws, model=ready["body"].get("model"))
        await ws.send(json.dumps({"type": "StartSession", "body": {"credits": credits}}))

        stream = sd.OutputStream(samplerate=SAMPLE_RATE, channels=CHANNELS,
                                 callback=callback, blocksize=1024,
                                 device=output_device)
        stream.start()

        async def receive():
            async for message in ws:
                if isinstance(message, bytes):
                    # frame layout: [uint32 LE seq][float32 LE samples (n, 2)]
                    (_seq,) = struct.unpack_from("<I", message)
                    buffer.append(np.frombuffer(message[4:], dtype="<f4").reshape(-1, CHANNELS))
                    started[0] = True
                    continue
                msg = json.loads(message)
                if msg.get("type") == "Error":
                    print(f"MRT2 server: {msg['body'].get('error')}")
                elif msg.get("type") == "Stats":
                    stats_q.put({**msg["body"], "underruns": underruns[0],
                                 "buffered_s": sum(len(b) for b in buffer) / SAMPLE_RATE})

        async def send_acks():
            while True:
                await acks.get()
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
                elif msg["type"] == "pause":
                    if msg["paused"]:
                        started[0] = False  # the silence while paused isn't an underrun
                    await ws.send(json.dumps({"type": "Pause" if msg["paused"] else "Resume",
                                              "body": None}))

        tasks = [asyncio.create_task(t()) for t in (receive, send_acks, send_controls)]
        try:
            # ends on "stop" (send_controls) or when the server goes away (receive)
            await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for t in tasks:
                t.cancel()
            stream.stop()
            stream.close()


# =========================
# PARENT-SIDE CLIENT
# =========================

class MagentaClient:

    def __init__(self, server_url="ws://localhost:9100/stream", credits=3,
                 output_device=None, midi_port=8470):
        """
        `server_url`: websocket of a running mrt2_server.py (local, or the
            local end of the SSH tunnel to the cluster)
        `credits`: blocks the server stays ahead of playback; control latency
            is roughly credits * frames_per_block * 40ms (+ network delay)
        `midi_port`: port of client_midi, where note sources POST patterns
            (NOTES / DRUMS input); 0 = off
        """
        ctx = mp.get_context("spawn")
        self._control_q = ctx.Queue()
        self._stats_q = ctx.Queue()
        self._process = ctx.Process(
            target=_client_process,
            args=(self._control_q, self._stats_q, server_url, credits, output_device,
                  midi_port),
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
