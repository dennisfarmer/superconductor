import argparse
import csv

import matplotlib.pyplot as plt
import numpy as np


VIBE_COLOR_MAP = {
    "UP": "red",
    "DOWN": "deepskyblue",
    "LEFT": "gold",
    "RIGHT": "limegreen",
}


def robustfloat(value, default=np.nan):
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def parse_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    text = str(value).strip().lower()
    return text in {"1", "true", "t", "yes", "y"}


def add_beat_lines(axis, beat_events):
    shown_labels = set()
    for beat_time_s, vibe_direction in beat_events:
        vibe_key = (vibe_direction or "UNKNOWN").upper()
        color = VIBE_COLOR_MAP.get(vibe_key, "gray")
        label = f"Beat-{vibe_key}"
        if label in shown_labels:
            label = None
        else:
            shown_labels.add(label)

        axis.axvline(beat_time_s, color=color, linestyle="-", linewidth=1.2, alpha=0.65, label=label)


def main():
    parser = argparse.ArgumentParser(description="Plot camera frame kinematics with beat timestamp markers.")
    parser.add_argument("--csv", default="camera_frame_data.csv", help="Input frame metadata CSV file")
    parser.add_argument("--output", default="hand_movement_plots.png", help="Output image path")
    parser.add_argument("--no-show", action="store_true", help="Save figure without opening an interactive window")
    args = parser.parse_args()

    timestamps_ns = []
    x_positions = []
    y_positions = []
    x_velocities = []
    x_accelerations = []
    beat_events_ns = []

    with open(args.csv, "r", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            ts_ns = robustfloat(row.get("timestamp_ns"), np.nan)
            timestamps_ns.append(ts_ns)
            x_positions.append(robustfloat(row.get("x"), np.nan))
            y_positions.append(robustfloat(row.get("y"), np.nan))
            x_velocities.append(robustfloat(row.get("vx"), np.nan))
            x_accelerations.append(robustfloat(row.get("ax"), np.nan))

            if parse_bool(row.get("is_beat_frame")) and not np.isnan(ts_ns):
                beat_events_ns.append((ts_ns, row.get("vibe_direction", "UNKNOWN")))

    timestamps_ns = np.array(timestamps_ns, dtype=np.float64)
    x_positions = np.array(x_positions, dtype=np.float64)
    y_positions = np.array(y_positions, dtype=np.float64)
    x_velocities = np.array(x_velocities, dtype=np.float64)
    x_accelerations = np.array(x_accelerations, dtype=np.float64)

    if timestamps_ns.size == 0:
        raise RuntimeError(f"No rows loaded from CSV: {args.csv}")

    t0_ns = np.nanmin(timestamps_ns)
    time_s = (timestamps_ns - t0_ns) / 1e9
    beat_events_s = [((ts - t0_ns) / 1e9, vibe_dir) for ts, vibe_dir in beat_events_ns]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    ax1.plot(time_s, x_positions, label="X Position", color="blue")
    ax1.plot(time_s, y_positions, label="Y Position", color="orange")
    add_beat_lines(ax1, beat_events_s)
    ax1.set_title("Hand Position Over Time")
    ax1.set_ylabel("Normalized Position (0-1)")
    ax1.legend(loc="upper right")
    ax1.grid(True)

    ax2.plot(time_s, x_velocities, label="X Velocity", color="green")
    ax2.plot(time_s, x_accelerations, label="X Acceleration", color="purple", alpha=0.8)
    add_beat_lines(ax2, beat_events_s)
    ax2.set_title("X Velocity / Acceleration Over Time")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Value")
    ax2.legend(loc="upper right")
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()