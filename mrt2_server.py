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
    Pause           null                            stop generating (session and state kept)
    Resume          null                            continue from where it paused
    Pattern         {notes?, drums?, cfg_notes?, cfg_drums?, id?}   NOTES / DRUMS input, below
    EndSession      null
  server -> client:
    JSON text  {"type": "Ready", "body": {model, backend, sample_rate, frames_per_block, host}}
    JSON text  {"type": "Stats", "body": {rtf, gen_ms_per_frame}}   (about once a second)
    JSON text  {"type": "Error", "body": {error}}   a message was rejected (e.g. a bad Pattern)
    binary     [uint32 LE seq][float32 LE samples, row-major (num_samples, 2)]

NOTES / DRUMS (Pattern): besides the style, MRT2 takes two inputs per 40ms frame.
  notes: 128 slots (MIDI pitch), each -1 model chooses, 0 off, 1 held,
         2 onset (attack now), 3 on (model picks onset or held)
  drums: 1 slot, -1 model chooses, 0 no hit, 1 hit now
A Pattern sets either lane or both; each lane loops on its own:
  {"notes": {"loop": true, "steps": [{"frames": 36, "pitches": {"36": 2, "60": -1}, "default": 0}]},
   "drums": {"loop": true, "steps": [{"frames": 9, "first": 1, "rest": 0}]},
   "cfg_notes": 3.0, "cfg_drums": 1.0}
  notes step: `frames` x 40ms; `pitches` sets some slots, `default` (-1) the rest;
      after the first frame an onset (2) becomes held (1)
  drums step: `first` on its first frame, `rest` on the others (0 after a hit, else `first`)
  loop false: play once, then back to the lane's last looping steps
  a lane left out is unchanged; a lane set to null is cleared (all -1, as with no Pattern)
A change applies from the next generated frame (heard after the client's lead).

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


NOTE_STATES = (-1, 0, 1, 2, 3)
DRUM_STATES = (-1, 0, 1)


def _state(value, allowed, where):
    value = int(value)
    if value not in allowed:
        raise ValueError(f"{where}: {value} is not one of {allowed}")
    return value


def expand_lane(kind, lane):
    """Validate one Pattern lane and expand it to one value per frame.

    Returns (frames, loop): for "notes" each frame is a 128-tuple, for "drums"
    an int. Raises ValueError on bad input."""
    if not isinstance(lane, dict) or not lane.get("steps"):
        raise ValueError(f"{kind}: needs a non-empty 'steps' list")
    frames = []
    for i, step in enumerate(lane["steps"]):
        where = f"{kind} step {i}"
        n = int(step.get("frames", 1))
        if n < 1:
            raise ValueError(f"{where}: frames must be >= 1")
        if kind == "notes":
            first = [_state(step.get("default", -1), NOTE_STATES, where)] * 128
            for pitch, value in (step.get("pitches") or {}).items():
                pitch = int(pitch)
                if not 0 <= pitch < 128:
                    raise ValueError(f"{where}: pitch {pitch} outside 0-127")
                first[pitch] = _state(value, NOTE_STATES, where)
            later = tuple(1 if v == 2 else v for v in first)  # an onset happens once
            frames += [tuple(first)] + [later] * (n - 1)
        else:
            first = _state(step.get("first", -1), DRUM_STATES, where)
            rest = _state(step.get("rest", 0 if first == 1 else first), DRUM_STATES, where)
            frames += [first] + [rest] * (n - 1)
    return frames, bool(lane.get("loop", True))


class Lane:
    """One Pattern lane (notes or drums), consumed one frame at a time."""

    def __init__(self):
        self.set(None)

    def set(self, frames, loop=True):
        """frames=None clears the lane (model chooses, as with no Pattern)."""
        if frames is None or loop:
            self.fallback = frames  # where a play-once lane returns to
        self.frames, self.loop, self.pos = frames, loop, 0

    def next(self):
        """Value for the next generated frame; None = no instruction (masked)."""
        if self.frames is not None and self.pos >= len(self.frames):
            if not self.loop:
                self.frames, self.loop = self.fallback, True
            self.pos = 0
        if self.frames is None:
            return None
        value = self.frames[self.pos]
        self.pos += 1
        return value


class Generator:
    """Owns the model and the generation state; runs on one dedicated thread.

    Loading, embedding and generation all happen on this thread (as in the old
    in-process client), so the model is never used from two threads; MLX
    streams are thread-local, so the model must also be *loaded* here. The
    websocket handler only posts control messages to `control_q`.
    """

    def __init__(self, load, frames_per_block):
        from magenta_rt.config import DRUM_PIANOROLL, MUSICCOCA, PIANOROLL_WITH_ONSETS
        self._load = load
        self._mrt = None
        self._musiccoca_key = MUSICCOCA.key
        self._notes_key = PIANOROLL_WITH_ONSETS.key
        self._drums_key = DRUM_PIANOROLL.key
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
        notes, drums, pattern_cfg = Lane(), Lane(), {}  # NOTES / DRUMS input (Pattern)
        paused = False
        credits, seq = 0, 0
        gen_time = gen_audio = 0.0
        last_stats = time.time()
        while True:
            # apply all pending control messages (latest wins); block while idle
            idle = emit is None or credits <= 0 or style is None or paused
            try:
                while True:
                    msg = self.control_q.get(timeout=0.5) if idle else self.control_q.get_nowait()
                    idle = False  # drain the rest without blocking
                    kind = msg["type"]
                    if kind == "start":
                        emit, state, seq, paused = msg["emit"], None, 0, False
                        notes, drums, pattern_cfg = Lane(), Lane(), {}
                        self.tempo.reset()
                        credits = msg["credits"]
                        controls = msg["controls"]
                        style = _blend_styles(self._embed, self._cache, msg["recipe"])
                    elif kind == "end":
                        emit, state, credits, paused = None, None, 0, False
                        notes, drums, pattern_cfg = Lane(), Lane(), {}
                    elif kind == "pattern":
                        # lanes were validated and expanded by the websocket handler
                        for name, lane in (("notes", notes), ("drums", drums)):
                            if name in msg["lanes"]:
                                lane.set(*msg["lanes"][name])
                        pattern_cfg.update(msg["cfg"])
                    elif kind == "recipe":
                        new_style = _blend_styles(self._embed, self._cache, msg["recipe"])
                        if new_style is not None:
                            style = new_style
                    elif kind == "controls":
                        controls = msg["controls"]
                    elif kind == "credit":
                        credits += 1
                    elif kind == "pause":
                        # the state is kept, so Resume continues the music; credits keep
                        # coming in while the client plays out its buffer, so Resume
                        # starts with the full lead again
                        paused = msg["paused"]
            except queue.Empty:
                pass
            if emit is None or credits <= 0 or style is None or paused:
                continue

            t0 = time.time()
            cfg_scales = dict(pattern_cfg)
            if "cfg_musiccoca" in controls:
                cfg_scales["musiccoca"] = controls["cfg_musiccoca"]
            # generate() uses one conditioning for all the frames it makes, so the
            # block is split wherever the notes / drums input changes. Without a
            # Pattern this is a single call, as before.
            frame_inputs = [(notes.next(), drums.next()) for _ in range(self._frames_per_block)]
            pieces = []
            i = 0
            while i < len(frame_inputs):
                j = i + 1
                while j < len(frame_inputs) and frame_inputs[j] == frame_inputs[i]:
                    j += 1
                note_tokens, drum_token = frame_inputs[i]
                conditioning = {self._musiccoca_key: style}
                if note_tokens is not None:
                    conditioning[self._notes_key] = list(note_tokens)
                if drum_token is not None:
                    conditioning[self._drums_key] = [drum_token]
                wav, state = self._mrt.generate(
                    conditioning=conditioning,
                    cfg_scales=cfg_scales or None,
                    temperature=controls.get("temperature"),
                    top_k=int(controls["top_k"]) if "top_k" in controls else None,
                    frames=j - i,
                    state=state,
                )
                pieces.append(np.asarray(wav.samples, dtype="<f4").reshape(-1, CHANNELS))
                i = j
            samples = np.concatenate(pieces) if len(pieces) > 1 else pieces[0]
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

    @staticmethod
    def _parse_pattern(body):
        """Pattern body -> generator message; a null body clears both lanes."""
        if body is None:
            body = {"notes": None, "drums": None}
        if not isinstance(body, dict):
            raise ValueError("body must be an object or null")
        lanes = {}
        for kind in ("notes", "drums"):
            if kind in body:
                lanes[kind] = (None, True) if body[kind] is None else expand_lane(kind, body[kind])
        cfg = {}
        for key, name in (("cfg_notes", "notes"), ("cfg_drums", "drums")):
            if body.get(key) is not None:
                cfg[name] = min(7.0, max(-1.0, float(body[key])))
        if body.get("id"):
            print(f"Pattern {body['id']}: {', '.join(lanes) or 'cfg only'}")
        return {"type": "pattern", "lanes": lanes, "cfg": cfg}

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
                elif msg_type in ("Pause", "Resume"):
                    self._post({"type": "pause", "paused": msg_type == "Pause"})
                elif msg_type == "Pattern":
                    try:
                        self._post(self._parse_pattern(data.get("body")))
                    except (ValueError, TypeError, AttributeError) as e:
                        print(f"WS bad Pattern: {e}")
                        await ws.send_str(json.dumps({"type": "Error", "body": {"error": f"Pattern: {e}"}}))
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
