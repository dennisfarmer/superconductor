"""Magenta RealTime 2 streaming server.

Replaces the MRT1 `superconductor_server.py` + `scheduler.py` pair for MRT2.
The same script runs locally (`--backend mlx`, e.g. mrt2_small on a MacBook)
and on the cluster (`--backend jax`, e.g. mrt2_base on a GPU); only the flags
differ, and only `load_model` knows which backend is in use.

Why there is no scheduler / fork logic any more:
  * MRT1 generated 2s chunks, so the queue held ~10s of audio and a recipe
    change had to rewind to a saved state to be heard in time. MRT2 generates
    40ms frames (sent in small blocks), the lead over playback is a few
    hundred ms, and a new recipe simply applies to the next block.
  * The generation state never leaves this process: passing it per block over
    HTTP would cost more than generating, and the JAX backend donates the state
    buffer to each step, so old states can't be replayed without copying.
  * Buffered client audio is never dropped: the model state has already moved
    past it, so dropping it would be an audible jump.

Flow control is credit-based: the client's `StartSession` grants N blocks, and
every `ReceivedChunk` (sent when the client finishes playing a block) grants one
more. The client's sound card paces generation, and the lead (~control latency)
is N blocks.

Websocket protocol (`GET /stream`, one client at a time):
  client -> server, JSON text frames {"type": ..., "body": ...}:
    StartSession    {recipe?, controls?, credits}
    UpdateRecipe    {recipe}                        e.g. {"jazz": 0.6, "flute": 0.3}
    UpdateControls  {temperature?, top_k?, cfg_musiccoca?}
    ReceivedChunk   null                            one more block of credit
    EndSession      null
  server -> client:
    JSON text  {"type": "Ready", "body": {model, backend, sample_rate, frames_per_block, host}}
    JSON text  {"type": "Stats", "body": {rtf, gen_ms_per_frame}}   (about once a second)
    binary     [uint32 LE seq][float32 LE samples, row-major (num_samples, 2)]

Tempo (see tempo.py): every block is time-stretched toward a target tempo
before it is sent, so blocks are 200ms / speed long. Plain HTTP, any client:
  GET  /tempo           test page: set an exact bpm, watch the estimated bpm
  GET  /conduct         conductor page: space bar = one beat
  POST /tempo/beat      {t?}     one conductor beat (t: seconds, sender's clock)
  POST /tempo/target    {bpm}    exact target tempo
  POST /tempo/free      null     no target (speed 1)
  GET  /tempo/status    mode, speed, model_bpm, heard_bpm, conductor_bpm, ...
"""

import asyncio
import json
import logging
import queue
import socket
import struct
import threading
import time
from pathlib import Path

import numpy as np
from absl import app as absl_app
from absl import flags
from aiohttp import web

from tempo import TempoController

SAMPLE_RATE = 48000
CHANNELS = 2
FRAME_SAMPLES = 1920  # one MRT2 frame = 40ms @ 48kHz
DEFAULT_RECIPE = {"jazz": 1.0}

MODELS = ["mrt2_small", "mrt2_base"]
PAGES = Path(__file__).parent / "pages"

_BACKEND = flags.DEFINE_enum(
    "backend", "mlx", ["mlx", "jax"], "mlx: Apple Silicon; jax: NVIDIA GPU (cluster)."
)
_MODEL = flags.DEFINE_enum("model", "mrt2_small", MODELS, "MRT2 model size.")
_PORT = flags.DEFINE_integer("port", 9100, "Port to listen on.")
_FRAMES_PER_BLOCK = flags.DEFINE_integer(
    "frames_per_block", 5, "MRT2 frames (40ms each) generated and sent per block."
)

logger = logging.getLogger(__name__)


def load_model(backend, model):
    """The only backend-specific code."""
    if backend == "jax":
        from magenta_rt import MagentaRT2Jax
        return MagentaRT2Jax(size=model)

    from magenta_rt import MagentaRT2StdMlxfn
    return MagentaRT2StdMlxfn(size=model)


def _blend_styles(embed, cache, recipe):
    """Weighted average of MusicCoCa embeddings, e.g. {"jazz": 0.6, "flute": 0.3}."""
    total = sum(w for w in recipe.values() if w > 0)
    if total <= 0:
        return None
    mix = None
    for prompt, weight in recipe.items():
        if weight <= 0:
            continue
        if prompt not in cache:
            cache[prompt] = np.asarray(embed(prompt), dtype=np.float32)
        term = cache[prompt] * (weight / total)
        mix = term if mix is None else mix + term
    return mix


class Generator:
    """Owns the model and the generation state; runs on one dedicated thread.

    Loading, embedding and generation all happen on this thread (as in the old
    in-process client), so the model is never used from two threads; MLX
    streams are thread-local, so the model must also be *loaded* here. The
    websocket handler only posts control messages to `control_q`.
    """

    def __init__(self, load, frames_per_block):
        from magenta_rt.config import MUSICCOCA
        self._load = load
        self._mrt = None
        self._musiccoca_key = MUSICCOCA.key
        self._frames_per_block = frames_per_block
        self._cache = {}
        self.tempo = TempoController()
        self.control_q = queue.Queue()
        self.load_error = None
        self.loaded = threading.Event()
        threading.Thread(target=self._run, daemon=True).start()

    def _embed(self, prompt):
        return self._mrt.embed_style(prompt, use_mapper=True)

    def _run(self):
        try:
            self._mrt = self._load()
        except BaseException as e:
            self.load_error = e
            return
        finally:
            self.loaded.set()
        emit = None  # callback of the active session; None = no session
        style, controls, state = None, {}, None
        credits, seq = 0, 0
        gen_time = gen_audio = 0.0
        last_stats = time.time()
        while True:
            # apply all pending control messages (latest wins); block while idle
            idle = emit is None or credits <= 0 or style is None
            try:
                while True:
                    msg = self.control_q.get(timeout=0.5) if idle else self.control_q.get_nowait()
                    idle = False  # drain the rest without blocking
                    kind = msg["type"]
                    if kind == "start":
                        emit, state, seq = msg["emit"], None, 0
                        self.tempo.reset()
                        credits = msg["credits"]
                        controls = msg["controls"]
                        style = _blend_styles(self._embed, self._cache, msg["recipe"])
                    elif kind == "end":
                        emit, state, credits = None, None, 0
                    elif kind == "recipe":
                        new_style = _blend_styles(self._embed, self._cache, msg["recipe"])
                        if new_style is not None:
                            style = new_style
                    elif kind == "controls":
                        controls = msg["controls"]
                    elif kind == "credit":
                        credits += 1
            except queue.Empty:
                pass
            if emit is None or credits <= 0 or style is None:
                continue

            t0 = time.time()
            cfg_scales = None
            if "cfg_musiccoca" in controls:
                cfg_scales = {"musiccoca": controls["cfg_musiccoca"]}
            wav, state = self._mrt.generate(
                conditioning={self._musiccoca_key: style},
                cfg_scales=cfg_scales,
                temperature=controls.get("temperature"),
                top_k=int(controls["top_k"]) if "top_k" in controls else None,
                frames=self._frames_per_block,
                state=state,
            )
            samples = np.asarray(wav.samples, dtype="<f4").reshape(-1, CHANNELS)
            samples = self.tempo.process(samples).astype("<f4")
            emit(struct.pack("<I", seq) + samples.tobytes())
            seq += 1
            credits -= 1
            gen_time += time.time() - t0
            gen_audio += self._frames_per_block * FRAME_SAMPLES / SAMPLE_RATE

            now = time.time()
            if now - last_stats > 1.0:
                # realtime factor > 1.0 means generating faster than playback
                frames = gen_audio * SAMPLE_RATE / FRAME_SAMPLES
                emit(json.dumps({"type": "Stats", "body": {
                    "rtf": gen_audio / max(gen_time, 1e-6),
                    "gen_ms_per_frame": 1000 * gen_time / max(frames, 1),
                }}))
                gen_time = gen_audio = 0.0
                last_stats = now


class MRT2Server:
    """Single-session websocket front-end for a `Generator`."""

    def __init__(self, generator, ready_info, port):
        self._generator = generator
        self._ready_info = ready_info
        self._port = port
        self._session_ws = None
        self._session_lock = asyncio.Lock()
        self._app = web.Application()
        self._app.router.add_get("/stream", self._handle_ws)
        self._app.router.add_get("/tempo", self._page("tempo.html"))
        self._app.router.add_get("/conduct", self._page("conduct.html"))
        self._app.router.add_post("/tempo/beat", self._beat)
        self._app.router.add_post("/tempo/target", self._target)
        self._app.router.add_post("/tempo/free", self._free)
        self._app.router.add_get("/tempo/status", self._status)

    def run(self):
        web.run_app(self._app, port=self._port)

    def _post(self, msg):
        self._generator.control_q.put(msg)

    @staticmethod
    def _page(name):
        async def handler(request):
            return web.FileResponse(PAGES / name)
        return handler

    async def _beat(self, request):
        body = await request.json() if request.can_read_body else {}
        self._generator.tempo.beat((body or {}).get("t"))
        return web.json_response(self._generator.tempo.status())

    async def _target(self, request):
        self._generator.tempo.set_target((await request.json())["bpm"])
        return web.json_response(self._generator.tempo.status())

    async def _free(self, request):
        self._generator.tempo.set_free()
        return web.json_response(self._generator.tempo.status())

    async def _status(self, request):
        return web.json_response(self._generator.tempo.status())

    async def _handle_ws(self, request):
        ws = web.WebSocketResponse(compress=False)  # float audio doesn't compress
        await ws.prepare(request)

        async with self._session_lock:
            if self._session_ws is not None and not self._session_ws.closed:
                await ws.close(message=b"Session already active")
                return ws
            self._session_ws = ws

        client_id = f"{request.remote}"
        print(f"WS connected: {client_id}")

        # The generator thread hands frames to this queue; one sender task keeps
        # them in order.
        loop = asyncio.get_running_loop()
        outbox = asyncio.Queue()

        def emit(frame):
            loop.call_soon_threadsafe(outbox.put_nowait, frame)

        async def sender():
            while True:
                frame = await outbox.get()
                if ws.closed:
                    return
                if isinstance(frame, bytes):
                    await ws.send_bytes(frame)
                else:
                    await ws.send_str(frame)

        send_task = asyncio.create_task(sender())
        await ws.send_str(json.dumps({"type": "Ready", "body": self._ready_info}))

        try:
            async for msg in ws:
                if msg.type != web.WSMsgType.TEXT:
                    if msg.type == web.WSMsgType.ERROR:
                        print(f"WS error from {client_id}: {ws.exception()}")
                        break
                    continue
                try:
                    data = json.loads(msg.data)
                except json.JSONDecodeError:
                    print(f"WS bad json: {msg.data!r}")
                    continue

                msg_type = data.get("type")
                body = data.get("body") or {}

                if msg_type == "StartSession":
                    self._post({"type": "start", "emit": emit,
                                "recipe": body.get("recipe") or DEFAULT_RECIPE,
                                "controls": dict(body.get("controls") or {}),
                                "credits": int(body.get("credits", 3))})
                elif msg_type == "UpdateRecipe":
                    if body.get("recipe"):
                        self._post({"type": "recipe", "recipe": dict(body["recipe"])})
                elif msg_type == "UpdateControls":
                    self._post({"type": "controls", "controls": dict(body)})
                elif msg_type == "ReceivedChunk":
                    self._post({"type": "credit"})
                elif msg_type == "EndSession":
                    break
                else:
                    print(f"WS unknown message type: {msg_type}")
        finally:
            self._post({"type": "end"})
            send_task.cancel()
            async with self._session_lock:
                if self._session_ws is ws:
                    self._session_ws = None
            if not ws.closed:
                await ws.close()
            print(f"WS disconnected: {client_id}")

        return ws


def main(_):
    logging.basicConfig(level=logging.INFO)
    generator = Generator(
        lambda: load_model(_BACKEND.value, _MODEL.value),
        _FRAMES_PER_BLOCK.value,
    )
    generator.loaded.wait()
    if generator.load_error is not None:
        raise generator.load_error
    ready_info = {
        "model": _MODEL.value,
        "backend": _BACKEND.value,
        "sample_rate": SAMPLE_RATE,
        "frames_per_block": _FRAMES_PER_BLOCK.value,
        "host": socket.gethostname(),  # lets the client tell a local server from a remote one
    }
    print(f"Serving {_MODEL.value} ({_BACKEND.value}) on port {_PORT.value}", flush=True)
    MRT2Server(generator, ready_info, _PORT.value).run()


if __name__ == "__main__":
    absl_app.run(main)
