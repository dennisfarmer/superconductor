"""Render the same short progression under different sampling settings, for A/B listening.

Usage: python audio_engine/sweep.py [--seconds-per-chord 2.4] [--out audio_samples/sweep]
Writes one wav per setting and prints generation speed (ms/frame) and hiss-band energy.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from engine import Chord, Engine, Timeline  # noqa: E402

PROGRESSION = [  # I - IV6 - V7 - I in C, the app's home voicings
    Chord([48, 52, 60, 67], "I", "C"),
    Chord([45, 53, 60, 65], "IV6", "F/A"),
    Chord([43, 53, 59, 62], "V7", "G7"),
    Chord([48, 52, 60, 64], "I", "C"),
]
GRID = [(t, k) for t in (1.2, 0.9, 0.7, 0.5) for k in (40, 10)]


def hiss_share(y: np.ndarray) -> float:
    spec = np.abs(np.fft.rfft(y * np.hanning(len(y)))) ** 2
    f = np.fft.rfftfreq(len(y), 1 / 48000)
    return float(spec[(f > 8000) & (f < 20000)].sum() / spec[(f > 50) & (f < 20000)].sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default=str(Path(__file__).parent / "presets" / "warm_piano.json"))
    ap.add_argument("--seconds-per-chord", type=float, default=2.4)
    ap.add_argument("--out", default="audio_samples/sweep")
    args = ap.parse_args()
    logging.disable(logging.INFO)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    preset = json.loads(Path(args.preset).read_text())
    eng = Engine("mrt2_small", 8, None, "x", preset)
    from magenta_rt.audio import Waveform

    beats = 4
    eng.settings.beats_per_chord = beats
    eng.settings.bpm = 60.0 * beats / args.seconds_per_chord
    frames = int(round(25 * args.seconds_per_chord)) * len(PROGRESSION) * 2  # progression twice
    for temp, k in GRID:
        eng.settings.temperature = temp
        eng.mrt.top_k = k
        eng.state = None
        eng.conductor = type(eng.conductor)(eng.settings)
        eng.conductor.set_timeline(Timeline("preview", PROGRESSION, "sweep"))
        chunks, times = [], []
        for _ in range(frames):
            t0 = time.perf_counter()
            samples, _ = eng.generate_one()
            times.append((time.perf_counter() - t0) * 1000)
            chunks.append(samples)
        y = np.concatenate(chunks)
        path = out / f"temp{temp}_topk{k}.wav"
        Waveform(y, 48000).write(str(path))
        print(f"temp {temp:<4} top_k {k:<3} ms/frame {np.mean(times):5.1f}  hiss(8-20k) {100 * hiss_share(y.mean(axis=1)):.3f}%  -> {path}", flush=True)


if __name__ == "__main__":
    main()
