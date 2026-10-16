"""Compare whole-gesture tempo monitoring across recordings with known BPM."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from benchmark_landmarks import candidate_specs, load_landmarks, video_signature
from cycle_tempo import CycleTempoEstimator

FIELDS = [
    "recording", "reference_bpm", "candidate", "landmarks", "coverage",
    "median_bpm", "min_bpm", "max_bpm", "median_abs_error", "p90_abs_error",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording", nargs=2, action="append", metavar=("VIDEO", "BPM"))
    parser.add_argument("--window", type=float, default=3.0)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "outputs" / "benchmarks" / "cycle_benchmark")
    args = parser.parse_args()
    recordings = args.recording or [
        [str(Path(__file__).with_name("44test_new.mp4")), "168"],
        [str(Path(__file__).with_name("44_bpm132.mp4")), "132"],
    ]
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = {
        "window_seconds": args.window, "version": 1,
        "recordings": [{**video_signature(Path(path).resolve()), "bpm": float(bpm)}
                       for path, bpm in recordings],
    }
    config_path = output / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise RuntimeError("Configuration changed: choose a new output directory")
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    results = output / "results.csv"
    rows = []
    if results.exists():
        with results.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
    done = {(r["recording"], r["candidate"]) for r in rows}
    with results.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        if handle.tell() == 0:
            writer.writeheader()
        for path, reference in recordings:
            video = Path(path).resolve()
            target = float(reference)
            cache = video.parent / "outputs" / "benchmarks" / (video.stem + "_landmark_benchmark") / "landmarks.npz"
            timestamps, positions = load_landmarks(video, cache)
            for index, (name, indices) in enumerate(candidate_specs(), 1):
                if (video.name, name) in done:
                    continue
                estimator = CycleTempoEstimator(window_seconds=args.window)
                values, trace, eligible = [], [], 0
                for frame, ts in enumerate(timestamps):
                    selected = positions[frame, indices, :]
                    x, y = np.mean(selected, axis=0)
                    value = estimator.update(x, y, int(ts))
                    elapsed = (int(ts) - int(timestamps[0])) * 1e-9
                    if name == "landmark_12":
                        trace.append((frame, elapsed, value, estimator.confidence))
                    if elapsed >= args.window:
                        eligible += 1
                        if value is not None:
                            values.append(value)
                errors = np.abs(np.array(values) - target)
                row = {
                    "recording": video.name, "reference_bpm": target,
                    "candidate": name, "landmarks": ",".join(map(str, indices)),
                    "coverage": len(values) / eligible if eligible else 0,
                    "median_bpm": float(np.median(values)) if values else "",
                    "min_bpm": min(values) if values else "",
                    "max_bpm": max(values) if values else "",
                    "median_abs_error": float(np.median(errors)) if values else "",
                    "p90_abs_error": float(np.percentile(errors, 90)) if values else "",
                }
                writer.writerow(row)
                handle.flush()
                rows.append(row)
                if name == "landmark_12":
                    with (output / (video.stem + "_tempo.csv")).open("w", newline="") as trace_file:
                        trace_writer = csv.writer(trace_file)
                        trace_writer.writerow(["frame_index", "time_s", "tempo_bpm", "correlation"])
                        trace_writer.writerows(trace)
                if index % 40 == 0 or index == 1:
                    print(f"{video.name} {index}/239: {name}, median error={row['median_abs_error']}", flush=True)

    grouped = {}
    for row in rows:
        grouped.setdefault(row["candidate"], []).append(row)
    ranked = []
    for name, group in grouped.items():
        if len(group) != len(recordings) or any(float(r["coverage"]) < 0.8 for r in group):
            continue
        if any(r["median_abs_error"] == "" for r in group):
            continue
        ranked.append({
            "candidate": name,
            "landmarks": group[0]["landmarks"],
            "worst_recording_median_error": max(float(r["median_abs_error"]) for r in group),
            "mean_recording_median_error": float(np.mean([float(r["median_abs_error"]) for r in group])),
            "worst_recording_p90_error": max(float(r["p90_abs_error"]) for r in group),
            "minimum_coverage": min(float(r["coverage"]) for r in group),
        })
    ranked.sort(key=lambda r: (r["worst_recording_median_error"], r["mean_recording_median_error"]))
    if ranked:
        with (output / "ranked.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(ranked[0]))
            writer.writeheader()
            writer.writerows(ranked)
    lines = [
        "# Whole-gesture tempo benchmark", "",
        f"Window: {args.window} seconds. All input is causal; target BPM is used only for scoring.",
        "Coverage and errors are measured after the initial full-window warmup.",
        "Rank uses the worst recording's median absolute BPM error; candidates must report on at least 80% of eligible frames in every clip.",
        "This measures tempo monitoring, not individual beat detection or cue latency.", "",
        "| Candidate | Landmarks | Worst median BPM error | Worst 90th-percentile error | Minimum coverage |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in ranked[:15]:
        lines.append(f"| {row['candidate']} | {row['landmarks']} | {row['worst_recording_median_error']:.3f} | {row['worst_recording_p90_error']:.3f} | {row['minimum_coverage']:.1%} |")
    lines += ["", "Middle fingertip baseline:", ""]
    for row in grouped.get("landmark_12", []):
        lines.append(
            f"- {row['recording']}: median {float(row['median_bpm']):.2f} BPM, "
            f"range {float(row['min_bpm']):.2f}–{float(row['max_bpm']):.2f}, "
            f"median error {float(row['median_abs_error']):.3f} BPM, "
            f"coverage {float(row['coverage']):.1%}."
        )
    lines += ["", "These are exploratory rankings on two recordings; small differences need another recording to confirm."]
    (output / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"Results: {output}", flush=True)


if __name__ == "__main__":
    main()
