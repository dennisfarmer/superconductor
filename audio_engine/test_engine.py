"""Headless check of the engine: drive the conductor through a scripted session
(idle vamp -> hover preview -> release -> style change) without a sound card,
then verify with chroma analysis that each audible chord matches its voicing.

Usage: python audio_engine/test_engine.py [--model mrt2_small] [--out session.wav]
"""

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from engine import FRAME_RATE, SAMPLE_RATE, Chord, Engine, Timeline  # noqa: E402

NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

I = Chord([48, 52, 60, 67], "I", "C")
IV6 = Chord([45, 53, 60, 65], "IV⁶", "F/A")
V7 = Chord([43, 53, 59, 62], "V⁷", "G7")
VI = Chord([45, 52, 60, 69], "vi", "Am")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mrt2_small")
    ap.add_argument("--out", default="engine_session.wav")
    args = ap.parse_args()

    eng = Engine(args.model, buffer_frames=6, record=None, prompt="warm grand piano, intimate, slow")
    eng.settings.bpm = 100
    eng.settings.beats_per_chord = 4
    eng.conductor.set_timeline(Timeline("idle", [I], "idle"))

    script = [
        (0.0, lambda: None, "idle vamp on I"),
        (3.0, lambda: eng.conductor.set_timeline(Timeline("preview", [I, IV6, V7, I], "p1")), "hover: I-IV6-V7-I"),
        (14.0, lambda: eng.conductor.release(), "release -> idle"),
        (17.0, lambda: eng.set_style("lofi rhodes electric piano"), "style change"),
    ]
    total_s = 21.0
    frames = int(total_s * FRAME_RATE)
    audio, metas, events = [], [], []
    step = 0
    t_start = time.perf_counter()
    for f in range(frames):
        t = f / FRAME_RATE
        while step < len(script) and t >= script[step][0]:
            script[step][1]()
            events.append((t, script[step][2]))
            step += 1
        t0 = time.perf_counter()
        samples, meta = eng.generate_one()
        dt = (time.perf_counter() - t0) * 1000
        audio.append(samples)
        metas.append((meta, dt))
    wall = time.perf_counter() - t_start
    y = np.concatenate(audio, axis=0)

    from magenta_rt.audio import Waveform

    Waveform(y, SAMPLE_RATE).write(args.out)
    step_ms = np.array([d for _, d in metas])
    print(f"generated {total_s:.0f}s in {wall:.1f}s  (ms/frame mean {step_ms.mean():.1f}, "
          f"p99 {np.percentile(step_ms, 99):.1f}, max {step_ms.max():.1f}; budget 40)")
    print("events:", "; ".join(f"{t:.2f}s {e}" for t, e in events))

    # Segment by (role, index) runs and check chroma against the intended voicing.
    import librosa

    mono = y.mean(axis=1)
    runs = []
    for f, (m, _) in enumerate(metas):
        key = (m.role, m.timeline_id, m.index, m.label)
        if runs and runs[-1][0] == key:
            runs[-1][2] = f
        else:
            runs.append([key, f, f])
    by_label = {c.label: c for c in (I, IV6, V7, VI)}
    ok = 0
    checked = 0
    for (role, tid, idx, label), f0, f1 in runs:
        n = f1 - f0 + 1
        t0 = f0 / FRAME_RATE
        # ignore the attack (first 0.3 s) and very short fragments
        a = (f0 + 8) * (SAMPLE_RATE // FRAME_RATE)
        b = (f1 + 1) * (SAMPLE_RATE // FRAME_RATE)
        if n < 20:
            print(f"  {t0:5.2f}s {role:7s} {label:4s} ({n} frames, too short to check)")
            continue
        chroma = librosa.feature.chroma_cqt(y=mono[a:b], sr=SAMPLE_RATE).mean(axis=1)
        want = sorted({p % 12 for p in by_label[label].notes})
        top = list(np.argsort(chroma)[::-1][: len(want)])
        hit = len(set(top) & set(want))
        checked += 1
        ok += hit == len(want)
        print(f"  {t0:5.2f}s {role:7s} {label:4s} want {[NAMES[p] for p in want]} "
              f"top {[NAMES[p] for p in top]}  {hit}/{len(want)}")
    print(f"chord segments fully matched: {ok}/{checked}; wrote {args.out}")


if __name__ == "__main__":
    main()
