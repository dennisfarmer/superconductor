"""Feasibility probe: can mrt2_small run frame-by-frame in real time with
notes conditioning, and does the notes input actually steer pitch?

Renders C major -> A minor -> F major -> G major (2 s each) one frame at a
time, reports ms/step, writes a wav, and prints the dominant pitch classes
detected in each chord segment (chroma) so we can check the conditioning.

Usage: python audio_engine/probe_magenta.py [--auto-strum] [--prompt "..."]
"""

import argparse
import time

import numpy as np

from magenta_rt import MagentaRT2StdMlxfn
from magenta_rt.config import MUSICCOCA, PIANOROLL_WITH_ONSETS, DRUM_PIANOROLL

FRAME_RATE = 25
NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

PROGRESSION = [
    ("C", [48, 55, 64, 72]),
    ("Am", [45, 57, 64, 72]),
    ("F", [41, 57, 65, 72]),
    ("G", [43, 55, 62, 71]),
]


def notes_frame(pitches, prev, onset_frame, auto_strum):
    tokens = [0] * 128
    for p in pitches:
        if onset_frame:
            tokens[p] = 1 if p in prev else 2
        else:
            tokens[p] = 3 if auto_strum else 1
    return tokens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="warm grand piano, slow ballad")
    ap.add_argument("--seconds-per-chord", type=float, default=2.0)
    ap.add_argument("--auto-strum", action="store_true")
    ap.add_argument("--cfg-notes", type=float, default=3.0)
    ap.add_argument("--out", default="probe.wav")
    args = ap.parse_args()

    t0 = time.time()
    mrt = MagentaRT2StdMlxfn(
        size="mrt2_small",
        cfg_scales={"musiccoca": 3.0, "notes": args.cfg_notes, "drums": 1.0},
    )
    print(f"load: {time.time() - t0:.1f}s")
    style = mrt.embed_style(args.prompt, use_mapper=True)
    style_tokens = mrt.tokenize_style(style).tolist()[:MUSICCOCA.rvq_truncation_level]
    print(f"style tokens: {len(style_tokens)}")

    frames_per_chord = int(args.seconds_per_chord * FRAME_RATE)
    state = None
    chunks, step_times = [], []
    prev = []
    for name, pitches in PROGRESSION:
        for f in range(frames_per_chord):
            cond = {
                MUSICCOCA.key: style_tokens,
                PIANOROLL_WITH_ONSETS.key: notes_frame(pitches, prev, f == 0, args.auto_strum),
                DRUM_PIANOROLL.key: [0],
            }
            ts = time.perf_counter()
            wav, state = mrt.generate(conditioning=cond, frames=1, state=state)
            step_times.append(time.perf_counter() - ts)
            chunks.append(wav.samples)
        prev = pitches

    audio = np.concatenate(chunks, axis=0)
    st = np.array(step_times[5:]) * 1000
    print(f"ms/step: mean {st.mean():.1f}  p95 {np.percentile(st, 95):.1f}  "
          f"max {st.max():.1f}  (real-time budget 40ms)")

    from magenta_rt.audio import Waveform
    Waveform(audio, 48_000).write(args.out)
    print(f"wrote {args.out}  shape {audio.shape}")

    import librosa
    mono = audio.mean(axis=1)
    seg = int(args.seconds_per_chord * 48_000)
    for i, (name, pitches) in enumerate(PROGRESSION):
        # skip the first half-second of each segment (transition)
        y = mono[i * seg + 12_000:(i + 1) * seg]
        chroma = librosa.feature.chroma_cqt(y=y, sr=48_000).mean(axis=1)
        top = [NAMES[j] for j in np.argsort(chroma)[::-1][:4]]
        want = sorted({NAMES[p % 12] for p in pitches})
        print(f"{name:3s} wanted {want}  top chroma {top}")


if __name__ == "__main__":
    main()
