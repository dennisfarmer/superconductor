"""Local Magenta RealTime 2 (MRT2) client running on Apple Silicon via MLX.

Drop-in alternative to `magenta_client_stream_dennis.MagentaClient` (same
start / update_recipe / stop / connected interface) that generates audio on
this machine instead of talking to the Lighthouse scheduler.

Generation runs in a separate *process* so that the MLX generation loop and the
sounddevice callback never compete with the webcam / YOLO loop for the GIL.
The parent process only sends small control messages over a queue.
"""
import logging
import multiprocessing as mp
import queue
import time
from collections import deque

import numpy as np

SAMPLE_RATE = 48000
CHANNELS = 2
FRAME_SAMPLES = 1920  # one MRT2 frame = 40ms @ 48kHz

logger = logging.getLogger(__name__)


# =========================
# CHILD PROCESS
# =========================

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


def _generation_process(control_q, stats_q, model, frames_per_block,
                        max_buffered_blocks, output_device, model_dir):
    from pathlib import Path

    import sounddevice as sd
    from magenta_rt import MagentaRT2StdMlxfn, paths
    from magenta_rt.config import MUSICCOCA

    if model_dir:
        # load <model_dir>/<model>/<model>.mlxfn (e.g. a 4-bit re-export)
        # instead of ~/Documents/Magenta/magenta-rt-v2/models/<model>
        paths.models_dir = lambda: Path(model_dir).expanduser().resolve()

    logging.basicConfig(level=logging.INFO)
    mrt = MagentaRT2StdMlxfn(size=model)

    def embed(prompt):
        return mrt.embed_style(prompt, use_mapper=True)

    cache = {}
    recipe = {"jazz": 1.0}
    controls = {}  # temperature / top_k / cfg_musiccoca overrides
    style = _blend_styles(embed, cache, recipe)
    state = None

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
            else:
                buffer[0] = block[n:]
        if filled < frames:
            outdata[filled:].fill(0)
            if started[0]:  # don't count the silence before the first block
                underruns[0] += 1

    stream = sd.OutputStream(samplerate=SAMPLE_RATE, channels=CHANNELS,
                             callback=callback, blocksize=1024,
                             device=output_device)
    stream.start()
    stats_q.put({"ready": True})

    gen_time = 0.0
    gen_audio = 0.0
    last_stats = time.time()
    running = True
    while running:
        # apply all pending control messages (latest wins)
        try:
            while True:
                msg = control_q.get_nowait()
                if msg["type"] == "stop":
                    running = False
                elif msg["type"] == "recipe":
                    recipe = msg["recipe"]
                    new_style = _blend_styles(embed, cache, recipe)
                    if new_style is not None:
                        style = new_style
                elif msg["type"] == "controls":
                    controls = msg["controls"]
        except queue.Empty:
            pass
        if not running:
            break

        # keep only a small lead over playback so control changes are heard fast
        if len(buffer) >= max_buffered_blocks:
            time.sleep(0.005)
            continue

        t0 = time.time()
        cfg_scales = None
        if "cfg_musiccoca" in controls:
            cfg_scales = {"musiccoca": controls["cfg_musiccoca"]}
        wav, state = mrt.generate(
            conditioning={MUSICCOCA.key: style},
            cfg_scales=cfg_scales,
            temperature=controls.get("temperature"),
            top_k=int(controls["top_k"]) if "top_k" in controls else None,
            frames=frames_per_block,
            state=state,
        )
        buffer.append(np.asarray(wav.samples, dtype=np.float32).reshape(-1, CHANNELS))
        started[0] = True
        gen_time += time.time() - t0
        gen_audio += frames_per_block * FRAME_SAMPLES / SAMPLE_RATE

        now = time.time()
        if now - last_stats > 1.0:
            # realtime factor > 1.0 means generating faster than playback
            stats_q.put({"rtf": gen_audio / max(gen_time, 1e-6),
                         "underruns": underruns[0],
                         "buffered_s": sum(len(b) for b in buffer) / SAMPLE_RATE})
            gen_time = gen_audio = 0.0
            last_stats = now

    stream.stop()
    stream.close()


# =========================
# PARENT-SIDE CLIENT
# =========================

class LocalMagentaClient:

    def __init__(self, model="mrt2_small", frames_per_block=5,
                 max_buffered_blocks=3, output_device=None, model_dir=None):
        """
        `model_dir`: folder containing <model>/<model>.mlxfn, to load a custom
            export (e.g. quantized/ with a 4-bit mrt2_base); None = stock models
        `frames_per_block`: MRT2 frames (40ms each) generated per call
        `max_buffered_blocks`: generation lead over playback; control latency
            is roughly frames_per_block * 40ms * max_buffered_blocks
        """
        ctx = mp.get_context("spawn")
        self._control_q = ctx.Queue()
        self._stats_q = ctx.Queue()
        self._process = ctx.Process(
            target=_generation_process,
            args=(self._control_q, self._stats_q, model, frames_per_block,
                  max_buffered_blocks, output_device, model_dir),
            daemon=True,
        )
        self.connected = False
        self.stats = {}

    def start(self):
        self._process.start()

    def poll_stats(self):
        """Drain stats from the generation process; call from the UI loop."""
        try:
            while True:
                msg = self._stats_q.get_nowait()
                if msg.get("ready"):
                    self.connected = True
                    print("MagentaRT2 ready")
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

    def stop(self):
        if self._process.is_alive():
            self._control_q.put({"type": "stop"})
            self._process.join(timeout=3)
            if self._process.is_alive():
                self._process.terminate()
        self.connected = False
