"""Benchmark hand landmarks for 4/4 beat and tempo tracking.

The video is decoded once to cache numeric MediaPipe hand landmarks. No frame
images are displayed or saved. Subsequent candidates use the same timestamps
and landmarks, so comparisons isolate the tracked point or point average.

Usage:
    python bench/benchmark_landmarks.py 44test_new.mp4 --bpm 168

The command resumes completed candidates in the output directory. It writes
results.csv after every candidate, then ranked.csv and summary.md at the end.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys
import time
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import mediapipe as mp
import numpy as np

from annotate.beat_tracker import BeatTracker
from annotate.filterconst import MEAS_VARIANCE, POS_VARIANCE, TEMPO_VARIANCE, VEL_VARIANCE
from annotate.kalman_filters import Kalman1D, Kalman2D
from annotate.vibe import VIBE_FOURFOUR
from live.hand_tracking import create_hand_landmarker, ensure_task_file

RESULT_FIELDS = [
    "candidate", "landmarks", "status", "error", "detected_beats",
    "grid_matches", "expected_beats", "beat_precision", "beat_recall",
    "beat_f1", "grid_phase_s", "bpm_coverage", "bpm_median",
    "bpm_median_abs_error", "bpm_p90_abs_error", "bpm_within_10_fraction",
    "first_bpm_s", "elapsed_s",
]


def candidate_specs():
    """Try every landmark, every distinct pair, and useful hand regions."""
    singles = [12, 0] + [i for i in range(21) if i not in (12, 0)]
    for index in singles:
        yield f"landmark_{index:02d}", (index,)
    for left, right in combinations(range(21), 2):
        yield f"pair_{left:02d}_{right:02d}", (left, right)
    groups = {
        "all_landmarks": tuple(range(21)),
        "four_fingertips": (8, 12, 16, 20),
        "five_fingertips": (4, 8, 12, 16, 20),
        "palm_center": (0, 5, 9, 13, 17),
        "four_knuckles": (5, 9, 13, 17),
        "index_finger": (5, 6, 7, 8),
        "middle_finger": (9, 10, 11, 12),
        "finger_landmarks": tuple(range(5, 21)),
    }
    for name, indices in groups.items():
        yield name, indices


def video_signature(video: Path):
    info = video.stat()
    return {
        "video": str(video.resolve()),
        "video_size": info.st_size,
        "video_mtime_ns": info.st_mtime_ns,
    }


def extract_landmarks(video: Path, cache: Path):
    """Decode once; save timestamps and hand coordinates, never images."""
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video}")

    task_path = ROOT / "hand_landmarker.task"
    ensure_task_file(task_path)
    landmarker = create_hand_landmarker(task_path)
    timestamps = []
    positions = []
    previous_timestamp_ms = -1
    started = time.monotonic()

    try:
        while True:
            ok, image = cap.read()
            if not ok:
                break
            position_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            if not math.isfinite(position_ms) or position_ms < 0:
                raise RuntimeError("Video has no usable frame timestamps")
            timestamp_ns = int(position_ms * 1e6)
            timestamp_ms = int(timestamp_ns / 1e6)
            if timestamp_ms <= previous_timestamp_ms:
                raise RuntimeError(
                    f"Non-increasing timestamp at frame {len(timestamps)}: "
                    f"{timestamp_ms} <= {previous_timestamp_ms}"
                )
            previous_timestamp_ms = timestamp_ms
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if result.hand_landmarks:
                hand = result.hand_landmarks[0]
                xy = np.array([(point.x, point.y) for point in hand], dtype=np.float64)
            else:
                xy = np.full((21, 2), np.nan, dtype=np.float64)
            positions.append(xy)
            timestamps.append(timestamp_ns)

            if len(timestamps) % 100 == 0:
                print(
                    f"Extracted {len(timestamps)} frames in "
                    f"{time.monotonic() - started:.0f}s",
                    flush=True,
                )
    finally:
        landmarker.close()
        cap.release()

    if not timestamps:
        raise RuntimeError("Video contained no readable frames")
    temp = cache.with_name(cache.name + ".tmp")
    with temp.open("wb") as handle:
        np.savez_compressed(
            handle,
            timestamps_ns=np.asarray(timestamps, dtype=np.int64),
            positions=np.asarray(positions, dtype=np.float64),
            video_size=np.int64(video.stat().st_size),
            video_mtime_ns=np.int64(video.stat().st_mtime_ns),
        )
    os.replace(temp, cache)
    print(f"Cached {len(timestamps)} frames at {cache}", flush=True)


def load_landmarks(video: Path, cache: Path):
    if not cache.exists():
        extract_landmarks(video, cache)
    with np.load(cache, allow_pickle=False) as data:
        signature = video_signature(video)
        if int(data["video_size"]) != signature["video_size"]:
            raise RuntimeError("Cached landmarks belong to a different video size")
        if int(data["video_mtime_ns"]) != signature["video_mtime_ns"]:
            raise RuntimeError("Video changed since landmarks were cached")
        timestamps = data["timestamps_ns"].copy()
        positions = data["positions"].copy()
    if positions.shape != (len(timestamps), 21, 2):
        raise RuntimeError(f"Unexpected landmark cache shape: {positions.shape}")
    return timestamps, positions


def score_beats(beats_ns, duration_s, target_bpm):
    """Find the best fixed phase of the known beat period, with one match per pulse."""
    period = 60.0 / target_bpm
    tolerance = min(0.08, period * 0.25)
    beat_times = np.asarray(beats_ns, dtype=np.float64) * 1e-9
    best = None
    for phase in np.linspace(0.0, period, 181, endpoint=False):
        expected = max(0, math.floor((duration_s - phase) / period) + 1)
        indices = np.rint((beat_times - phase) / period).astype(np.int64)
        error = np.abs(beat_times - (phase + indices * period))
        valid = (indices >= 0) & (indices < expected) & (error <= tolerance)
        matches = len(np.unique(indices[valid]))
        denominator = len(beat_times) + expected
        f1 = 2 * matches / denominator if denominator else 0.0
        score = (f1, matches, -float(np.sum(error[valid])))
        if best is None or score > best[0]:
            best = (score, float(phase), expected, matches)

    _, phase, expected, matches = best
    count = len(beat_times)
    return {
        "detected_beats": count,
        "grid_matches": matches,
        "expected_beats": expected,
        "beat_precision": matches / count if count else 0.0,
        "beat_recall": matches / expected if expected else 0.0,
        "beat_f1": 2 * matches / (count + expected) if count + expected else 0.0,
        "grid_phase_s": phase,
    }


def evaluate(timestamps, positions, indices, target_bpm):
    position_kf = Kalman2D(
        pos_var=POS_VARIANCE, vel_var=VEL_VARIANCE, meas_var=MEAS_VARIANCE
    )
    tempo_kf = Kalman1D(meas_var=TEMPO_VARIANCE)
    tracker = BeatTracker(
        100, vibe_pattern=VIBE_FOURFOUR, max_beats=3000, mirror_pattern=True
    )
    beat_timestamps = []
    frame_bpms = []
    last_bpm = None
    first_bpm_s = None

    for frame_index, timestamp in enumerate(timestamps):
        timestamp_ns = int(timestamp)
        tracker.apply_missed_beat_backup(timestamp_ns)
        selected = positions[frame_index, indices, :]
        if np.isfinite(selected).all():
            raw_x = 1.0 - float(np.mean(selected[:, 0]))
            raw_y = float(np.mean(selected[:, 1]))
            x, y = position_kf.update(raw_x, raw_y, timestamp_ns)
            tracker.add_sample(x, y, timestamp_ns)
            beat_ts, _, counted_for_tempo = tracker.analyze_vibe_beat()
            if beat_ts is not None:
                beat_timestamps.append(int(beat_ts))
            recent = tracker.get_recent_beats()
            if counted_for_tempo and len(recent) >= 2:
                measured_bpm = 60 * 1e9 / (recent[-1] - recent[-2])
                last_bpm = tempo_kf.update(float(measured_bpm), int(recent[-1]))
                if first_bpm_s is None:
                    first_bpm_s = (timestamp_ns - int(timestamps[0])) * 1e-9
        if last_bpm is not None:
            frame_bpms.append(float(last_bpm))

    duration_s = (int(timestamps[-1]) - int(timestamps[0])) * 1e-9
    if len(timestamps) > 1:
        duration_s += float(np.median(np.diff(timestamps))) * 1e-9
    metrics = score_beats(beat_timestamps, duration_s, target_bpm)
    errors = np.abs(np.asarray(frame_bpms) - target_bpm)
    metrics.update({
        "bpm_coverage": len(frame_bpms) / len(timestamps),
        "bpm_median": statistics.median(frame_bpms) if frame_bpms else "",
        "bpm_median_abs_error": float(np.median(errors)) if len(errors) else "",
        "bpm_p90_abs_error": float(np.percentile(errors, 90)) if len(errors) else "",
        "bpm_within_10_fraction": float(np.mean(errors <= 10)) if len(errors) else 0.0,
        "first_bpm_s": first_bpm_s if first_bpm_s is not None else "",
    })
    return metrics


def write_ranked(output_dir, rows, config):
    ranked = sorted(
        (row for row in rows if row["status"] == "ok"),
        key=lambda row: (
            -float(row["beat_f1"]),
            float(row["bpm_median_abs_error"])
            if row["bpm_median_abs_error"] != "" else math.inf,
        ),
    )
    ranked_path = output_dir / "ranked.csv"
    with ranked_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(ranked)

    lines = [
        "# Landmark benchmark",
        "",
        f"Video: {config['video']}",
        f"Reference tempo: {config['bpm']} BPM (4/4)",
        f"Candidates completed: {len(ranked)} / {config['candidate_count']}",
        "",
        "Each candidate uses the same detected hand and frame timestamps. "
        "Pairs and groups use an equal-weight average of their landmark positions.",
        "A beat matches the reference grid when it is within 80 ms, with at most "
        "one detected beat credited per reference pulse. Phase is fitted separately "
        "for each candidate; this is a relative comparison, not manual beat labeling.",
        "Frame BPM metrics reproduce the current pipeline's tempo filter.",
        "",
        "| Rank | Candidate | Beat F1 | Grid matches | BPM median | BPM median error | Within 10 BPM |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for rank, row in enumerate(ranked[:20], 1):
        lines.append(
            f"| {rank} | {row['candidate']} | {float(row['beat_f1']):.3f} | "
            f"{row['grid_matches']}/{row['expected_beats']} | "
            f"{float(row['bpm_median']):.1f} | "
            f"{float(row['bpm_median_abs_error']):.1f} | "
            f"{float(row['bpm_within_10_fraction']):.1%} |"
        )
    lines += ["", "Baseline candidates:", ""]
    by_name = {row["candidate"]: row for row in ranked}
    for name in ("landmark_12", "landmark_00"):
        row = by_name.get(name)
        if row:
            lines.append(
                f"- {name}: {row['grid_matches']} grid matches, "
                f"F1 {float(row['beat_f1']):.3f}, "
                f"median BPM {float(row['bpm_median']):.1f}"
            )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "video", nargs="?", type=Path,
        default=ROOT / "44test_new.mp4",
    )
    parser.add_argument("--bpm", type=float, default=168.0)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--limit", type=int, help="Evaluate only the first N candidates (for a smoke test)"
    )
    args = parser.parse_args()
    if args.bpm <= 0:
        parser.error("--bpm must be positive")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    video = args.video.resolve()
    if not video.exists():
        parser.error(f"Video does not exist: {video}")

    output_dir = (
        args.output_dir.resolve() if args.output_dir
        else video.parent / "outputs" / "benchmarks" / f"{video.stem}_landmark_benchmark"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    specs = list(candidate_specs())
    config = {
        **video_signature(video),
        "bpm": args.bpm,
        "candidate_count": len(specs),
        "pattern": "VIBE_FOURFOUR",
        "mirror_pattern": True,
        "version": 1,
    }
    config_path = output_dir / "config.json"
    if config_path.exists():
        saved = json.loads(config_path.read_text(encoding="utf-8"))
        if saved != config:
            raise RuntimeError(
                "Benchmark inputs changed; choose a new --output-dir to keep results separate"
            )
    else:
        config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    timestamps, positions = load_landmarks(video, output_dir / "landmarks.npz")
    results_path = output_dir / "results.csv"
    rows = []
    if results_path.exists():
        with results_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != RESULT_FIELDS:
                raise RuntimeError("Existing results.csv has a different format")
            rows = list(reader)
    completed = {row["candidate"] for row in rows if row["status"] == "ok"}
    requested = specs[:args.limit] if args.limit else specs
    remaining = [(name, indices) for name, indices in requested if name not in completed]
    print(
        f"Loaded {len(timestamps)} frames; {len(specs)} candidates total, "
        f"{len(completed)} complete, {len(remaining)} to run.",
        flush=True,
    )

    with results_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        if handle.tell() == 0:
            writer.writeheader()
            handle.flush()
        for position, (name, indices) in enumerate(remaining, 1):
            started = time.monotonic()
            row = {
                "candidate": name,
                "landmarks": ",".join(str(index) for index in indices),
                "status": "ok",
                "error": "",
            }
            try:
                row.update(evaluate(timestamps, positions, indices, args.bpm))
            except Exception as exc:
                row["status"] = "error"
                row["error"] = f"{type(exc).__name__}: {exc}"
            row["elapsed_s"] = round(time.monotonic() - started, 3)
            writer.writerow(row)
            handle.flush()
            rows.append(row)
            print(
                f"[{position}/{len(remaining)}] {name}: {row['status']}, "
                f"beats={row.get('detected_beats', '')}, "
                f"F1={row.get('beat_f1', '')}, "
                f"{row['elapsed_s']}s",
                flush=True,
            )

    write_ranked(output_dir, rows, config)
    print(f"Results: {output_dir / 'results.csv'}", flush=True)
    print(f"Ranked: {output_dir / 'ranked.csv'}", flush=True)
    print(f"Summary: {output_dir / 'summary.md'}", flush=True)


if __name__ == "__main__":
    main()
