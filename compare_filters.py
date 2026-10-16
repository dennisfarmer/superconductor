"""Compare beat-hit margins and position/tempo filters using cached landmarks.

This script never opens video frames when a landmark cache already exists.
It runs the 4/4 BeatTracker once per candidate and position-filter setting,
then replays tempo measurements through several display filters.

Usage:
    python superconductor/compare_filters.py superconductor/44test_new.mp4 --bpm 168
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from collections import deque
from pathlib import Path

import numpy as np

from benchmark_landmarks import candidate_specs, load_landmarks, video_signature
from beat_tracker import BeatTracker
from filterconst import MEAS_VARIANCE, POS_VARIANCE, VEL_VARIANCE
from kalman_filters import Kalman1D, Kalman2D
from vibe import VIBE_FOURFOUR

POSITION_FILTERS = ("kalman", "none")
TEMPO_FILTERS = ("kalman_r0.3", "kalman_r3", "kalman_r30", "median5", "none")
FIELDS = [
    "candidate", "landmarks", "position_filter", "tempo_filter", "status", "error",
    "detected_beats", "expected_beats", "hits_40ms", "f1_40ms", "hits_80ms",
    "f1_80ms", "bpm_coverage", "bpm_median", "bpm_median_abs_error",
    "bpm_p90_abs_error", "bpm_within_10_fraction", "first_bpm_s",
]


def collect_motion(timestamps, positions, indices, position_filter):
    position_kf = (
        Kalman2D(
            pos_var=POS_VARIANCE, vel_var=VEL_VARIANCE, meas_var=MEAS_VARIANCE
        )
        if position_filter == "kalman" else None
    )
    tracker = BeatTracker(
        100, vibe_pattern=VIBE_FOURFOUR, max_beats=3000, mirror_pattern=True
    )
    beats = []
    measurements = []

    for frame_index, timestamp in enumerate(timestamps):
        timestamp_ns = int(timestamp)
        tracker.apply_missed_beat_backup(timestamp_ns)
        selected = positions[frame_index, indices, :]
        if not np.isfinite(selected).all():
            continue
        raw_x = 1.0 - float(np.mean(selected[:, 0]))
        raw_y = float(np.mean(selected[:, 1]))
        if position_kf is None:
            x, y = raw_x, raw_y
        else:
            x, y = position_kf.update(raw_x, raw_y, timestamp_ns)
        tracker.add_sample(x, y, timestamp_ns)
        beat_ts, _, counted_for_tempo = tracker.analyze_vibe_beat()
        if beat_ts is not None:
            beats.append(int(beat_ts))
        recent = tracker.get_recent_beats()
        if counted_for_tempo and len(recent) >= 2:
            measured_bpm = 60 * 1e9 / (recent[-1] - recent[-2])
            measurements.append((frame_index, int(recent[-1]), float(measured_bpm)))
    return beats, measurements


def score_grid(beats_ns, duration_s, target_bpm):
    period = 60.0 / target_bpm
    phases = np.linspace(0.0, period, 1000, endpoint=False)
    expected = np.maximum(0, np.floor((duration_s - phases) / period).astype(int) + 1)
    beats_s = np.asarray(beats_ns, dtype=np.float64) * 1e-9
    if len(beats_s):
        grid_indices = np.rint((beats_s[None, :] - phases[:, None]) / period).astype(int)
        errors = np.abs(beats_s[None, :] - (phases[:, None] + grid_indices * period))
        in_range = (grid_indices >= 0) & (grid_indices < expected[:, None])
    else:
        errors = np.empty((len(phases), 0))
        in_range = np.empty((len(phases), 0), dtype=bool)

    scores = {"detected_beats": len(beats_s)}
    for margin_ms in (40, 80):
        hits = np.sum(in_range & (errors <= margin_ms / 1000.0), axis=1)
        # BeatTracker requires >200 ms between detections, so two detections
        # cannot occupy the same 160 ms acceptance window.
        denominator = len(beats_s) + expected
        f1 = np.divide(
            2 * hits, denominator, out=np.zeros_like(hits, dtype=float),
            where=denominator > 0,
        )
        best = int(np.argmax(f1))
        scores[f"hits_{margin_ms}ms"] = int(hits[best])
        scores[f"f1_{margin_ms}ms"] = float(f1[best])
        if margin_ms == 40:
            scores["expected_beats"] = int(expected[best])
    return scores


def tempo_metrics(timestamps, measurements, target_bpm, policy):
    values = np.full(len(timestamps), np.nan, dtype=float)
    filter_1d = None
    if policy.startswith("kalman_r"):
        measurement_variance = float(policy.removeprefix("kalman_r"))
        filter_1d = Kalman1D(meas_var=measurement_variance)
    history = deque(maxlen=int(policy[6:]) if policy.startswith("median") else 5)
    next_event = 0
    current = None
    first_bpm_s = None

    for frame_index in range(len(timestamps)):
        while next_event < len(measurements) and measurements[next_event][0] == frame_index:
            _, beat_ts, measured_bpm = measurements[next_event]
            if filter_1d is not None:
                current = filter_1d.update(measured_bpm, beat_ts)
            elif policy.startswith("median"):
                history.append(measured_bpm)
                current = statistics.median(history)
            else:
                current = measured_bpm
            if first_bpm_s is None:
                first_bpm_s = (int(timestamps[frame_index]) - int(timestamps[0])) * 1e-9
            next_event += 1
        if current is not None:
            values[frame_index] = current

    valid = values[np.isfinite(values)]
    errors = np.abs(valid - target_bpm)
    return {
        "bpm_coverage": len(valid) / len(timestamps),
        "bpm_median": float(np.median(valid)) if len(valid) else "",
        "bpm_median_abs_error": float(np.median(errors)) if len(errors) else "",
        "bpm_p90_abs_error": float(np.percentile(errors, 90)) if len(errors) else "",
        "bpm_within_10_fraction": float(np.mean(errors <= 10)) if len(errors) else 0.0,
        "first_bpm_s": first_bpm_s if first_bpm_s is not None else "",
    }


def write_summary(output_dir, rows, config):
    position_filters = config["position_filters"]
    tempo_filters = config["tempo_filters"]
    successful = [row for row in rows if row["status"] == "ok"]
    by_key = {
        (row["candidate"], row["position_filter"], row["tempo_filter"]): row
        for row in successful
    }
    lines = [
        "# Filter comparison",
        "",
        f"Video: {config['video']}",
        f"Reference: {config['bpm']} BPM, 4/4",
        f"Completed rows: {len(by_key)} / {config['candidate_count'] * len(position_filters) * len(tempo_filters)}",
        "",
        "A hit means a detected beat is within the stated margin of a fitted "
        f"{config['bpm']} BPM grid pulse. The phase is fitted separately for each candidate. "
        "This is a proxy for beat accuracy because individual beats were not "
        "manually labeled.",
        "",
        "## Landmark 12 baseline",
        "",
        "| Position filter | Tempo filter | Hits ±40 ms | Hits ±80 ms | Median BPM | Median BPM error | Within 10 BPM |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for pos in position_filters:
        for tempo in tempo_filters:
            row = by_key.get(("landmark_12", pos, tempo))
            if row:
                lines.append(
                    f"| {pos} | {tempo} | {row['hits_40ms']}/{row['expected_beats']} | "
                    f"{row['hits_80ms']} | "
                    f"{float(row['bpm_median']):.1f} | "
                    f"{float(row['bpm_median_abs_error']):.1f} | "
                    f"{float(row['bpm_within_10_fraction']):.1%} |"
                )
    lines += [
        "",
        "## Best beat timing by position filter",
        "",
        "| Position filter | Candidate | Hits ±40 ms | F1 ±40 ms | Hits ±80 ms | F1 ±80 ms |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for pos in position_filters:
        candidates = [
            row for row in successful
            if row["position_filter"] == pos and row["tempo_filter"] == tempo_filters[0]
        ]
        if candidates:
            row = max(candidates, key=lambda item: float(item["f1_40ms"]))
            lines.append(
                f"| {pos} | {row['candidate']} | {row['hits_40ms']} | "
                f"{float(row['f1_40ms']):.3f} | {row['hits_80ms']} | "
                f"{float(row['f1_80ms']):.3f} |"
            )
    lines += [
        "",
        "The BPM measurements use the current BeatTracker events. Removing the "
        "position Kalman filter can change those events; changing only the tempo "
        "filter cannot. A small BPM error on one constant-tempo video does not "
        "establish good response to real tempo changes.",
        "",
    ]
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "video", nargs="?", type=Path,
        default=Path(__file__).with_name("44test_new.mp4"),
    )
    parser.add_argument("--bpm", type=float, default=168.0)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--only", action="append", metavar="CANDIDATE",
        help="Evaluate this candidate; repeat for more (default: all)",
    )
    parser.add_argument(
        "--position-filter", action="append", choices=POSITION_FILTERS,
        help="Repeat to select position filters (default: both)",
    )
    parser.add_argument(
        "--tempo-filter", action="append",
        help="Repeat: none, kalman_r<number>, or median<N> (default: standard sweep)",
    )
    args = parser.parse_args()
    position_filters = tuple(dict.fromkeys(args.position_filter or POSITION_FILTERS))
    tempo_filters = tuple(dict.fromkeys(args.tempo_filter or TEMPO_FILTERS))
    for policy in tempo_filters:
        try:
            valid = (
                policy == "none"
                or (policy.startswith("kalman_r")
                    and np.isfinite(float(policy[8:])) and float(policy[8:]) > 0)
                or (policy.startswith("median") and int(policy[6:]) >= 1)
            )
        except ValueError:
            valid = False
        if not valid:
            parser.error(f"Invalid tempo filter: {policy}")
    if args.bpm <= 0:
        parser.error("--bpm must be positive")
    video = args.video.resolve()
    if not video.exists():
        parser.error(f"Video does not exist: {video}")
    output_dir = (
        args.output_dir.resolve() if args.output_dir
        else video.parent / "outputs" / "benchmarks" / f"{video.stem}_filter_benchmark"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    cache = video.parent / "outputs" / "benchmarks" / f"{video.stem}_landmark_benchmark" / "landmarks.npz"
    timestamps, positions = load_landmarks(video, cache)
    specs = list(candidate_specs())
    valid_names = {name for name, _ in specs}
    if args.only:
        unknown = set(args.only) - valid_names
        if unknown:
            parser.error(f"Unknown candidate(s): {', '.join(sorted(unknown))}")
        specs = [(name, indices) for name, indices in specs if name in args.only]
    config = {
        **video_signature(video), "bpm": args.bpm, "version": 1,
        "candidate_count": len(list(candidate_specs())),
        "position_filters": list(position_filters),
        "tempo_filters": list(tempo_filters),
        "margins_ms": [40, 80],
    }
    config_path = output_dir / "config.json"
    if config_path.exists():
        if json.loads(config_path.read_text(encoding="utf-8")) != config:
            raise RuntimeError("Benchmark configuration changed; use a new output directory")
    else:
        config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    results_path = output_dir / "results.csv"
    rows = []
    if results_path.exists():
        with results_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != FIELDS:
                raise RuntimeError("Existing results.csv has a different format")
            rows = list(reader)
    done = {
        (r["candidate"], r["position_filter"], r["tempo_filter"])
        for r in rows if r["status"] == "ok" and r["f1_40ms"] not in ("", None)
    }
    duration_s = (int(timestamps[-1]) - int(timestamps[0])) * 1e-9
    duration_s += float(np.median(np.diff(timestamps))) * 1e-9
    total = len(specs) * len(position_filters) * len(tempo_filters)
    remaining = total - sum(
        (name, pos, tempo) in done
        for name, _ in specs for pos in position_filters for tempo in tempo_filters
    )
    print(f"Loaded {len(timestamps)} cached frames; {remaining} of {total} rows remain.", flush=True)

    with results_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        if handle.tell() == 0:
            writer.writeheader()
            handle.flush()
        for candidate, indices in specs:
            for pos in position_filters:
                pending = [
                    tempo for tempo in tempo_filters
                    if (candidate, pos, tempo) not in done
                ]
                if not pending:
                    continue
                started = time.monotonic()
                try:
                    beats, measurements = collect_motion(timestamps, positions, indices, pos)
                    beat_metrics = score_grid(beats, duration_s, args.bpm)
                    for tempo in pending:
                        row = {
                            "candidate": candidate,
                            "landmarks": ",".join(str(index) for index in indices),
                            "position_filter": pos,
                            "tempo_filter": tempo,
                            "status": "ok",
                            "error": "",
                            **beat_metrics,
                            **tempo_metrics(timestamps, measurements, args.bpm, tempo),
                        }
                        writer.writerow(row)
                        rows.append(row)
                    handle.flush()
                    print(
                        f"{candidate}/{pos}: beats={len(beats)}, "
                        f"F1±40ms={beat_metrics['f1_40ms']:.3f}, "
                        f"{time.monotonic() - started:.1f}s",
                        flush=True,
                    )
                except Exception as exc:
                    for tempo in pending:
                        row = {
                            "candidate": candidate,
                            "landmarks": ",".join(str(index) for index in indices),
                            "position_filter": pos,
                            "tempo_filter": tempo,
                            "status": "error",
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                        writer.writerow(row)
                        rows.append(row)
                    handle.flush()
                    print(f"{candidate}/{pos}: ERROR {exc}", flush=True)

    write_summary(output_dir, rows, config)
    print(f"Results: {results_path}", flush=True)
    print(f"Summary: {output_dir / 'summary.md'}", flush=True)


if __name__ == "__main__":
    main()
