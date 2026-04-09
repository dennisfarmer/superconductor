"""
Video output and metadata file generation.

This module handles writing annotated video files with overlay information
and saving frame-by-frame metadata to JSON and CSV formats.

Functions:
    - write_annotated_video: Creates video with beat markers and overlay info
    - save_frame_metadata: Exports metadata to JSON and CSV files
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2


def _fmt_num(value, digits=4):
    if value is None:
        return "NA"
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def draw_direction_arrow(
    frame_img,
    direction_str: str,
    center_x: int = None,
    center_y: int = None,
    arrow_length: int = 50,
    color=(0, 255, 0),
    thickness=3,
):
    if center_x is None:
        center_x = frame_img.shape[1] - 80
    if center_y is None:
        center_y = 80

    direction_map = {
        "UP": (0, -1),
        "DOWN": (0, 1),
        "LEFT": (-1, 0),
        "RIGHT": (1, 0),
    }

    if direction_str not in direction_map:
        return

    dx, dy = direction_map[direction_str]
    end_x = center_x + dx * arrow_length
    end_y = center_y + dy * arrow_length

    cv2.arrowedLine(frame_img, (center_x, center_y), (end_x, end_y), color, thickness, tipLength=0.3)
    cv2.putText(frame_img, direction_str, (center_x - 20, center_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)


def render_annotated_frame(frame_img, row, width: int, height: int):
    """Render full overlay for a single frame and return the annotated copy."""
    rendered = frame_img.copy()

    cv2.putText(rendered, f"frame={row['frame_index']} t_ns={row['timestamp_ns']}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(rendered, f"x={_fmt_num(row['x'])} y={_fmt_num(row['y'])}", (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(rendered, f"vx={_fmt_num(row['vx'])} vy={_fmt_num(row['vy'])}", (10, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(rendered, f"ax={_fmt_num(row['ax'])} ay={_fmt_num(row['ay'])}", (10, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(rendered, f"tempo_bpm={_fmt_num(row['tempo_bpm'], 2)}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 210, 0), 1, cv2.LINE_AA)
    cv2.putText(rendered, f"closest_beat_frame={row['closest_beat_frame_index']} dt_ms={_fmt_num(row['closest_beat_dt_ms'], 2)}", (10, 144), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 200, 100), 1, cv2.LINE_AA)

    if row.get("is_beat_frame", False):
        cv2.circle(rendered, (24, 24), 12, (0, 0, 255), thickness=-1)
        cv2.putText(rendered, "BEAT", (42, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)

    vibe_dir = row.get("vibe_direction", "UNKNOWN")
    draw_direction_arrow(rendered, vibe_dir, center_x=width - 80, center_y=80, arrow_length=50, color=(100, 200, 255), thickness=3)

    if row.get("x") is not None and row.get("y") is not None:
        finger_x = int(row["x"] * width)
        finger_y = int(row["y"] * height)
        cv2.circle(rendered, (finger_x, finger_y), 15, (0, 255, 0), thickness=2)
        cv2.circle(rendered, (finger_x, finger_y), 5, (0, 255, 0), thickness=-1)

    return rendered


def write_annotated_video(
    captured_frames,
    output_video_path: str,
    frame_records,
    fps: float,
    width: int,
    height: int,
    show_output_preview: bool = False,
    frames_are_annotated: bool = False,
):
    """Write video with beat markers, overlay text, and visual indicators.

    Adds to each frame:
    - Frame index and timestamp
    - Position (x, y), velocity, acceleration
    - Tempo (BPM)
    - Beat indicators (red circle + "BEAT" text on beat frames)
    - Direction arrow showing current vibe goal
    - Green circle at detected hand position

    Args:
        captured_frames: List of frame images
        output_video_path: Output file path (.mp4)
        frame_records: Frame metadata with kinematic data
        fps: Frame rate
        width: Frame width
        height: Frame height
        show_output_preview: If True, show exactly the rendered frame being written
        frames_are_annotated: If True, frames already contain overlays and are written as-is
    """
    writer = cv2.VideoWriter(
        output_video_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (width, height),
    )

    for idx, row in enumerate(frame_records):
        if idx >= len(captured_frames):
            break
        frame_img = captured_frames[idx]
        if frame_img is None:
            continue

        if frames_are_annotated:
            rendered = frame_img
        else:
            rendered = render_annotated_frame(frame_img, row, width, height)
        writer.write(rendered)

        if show_output_preview:
            cv2.imshow("Annotated Output", rendered)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                show_output_preview = False
                cv2.destroyWindow("Annotated Output")

    writer.release()


def save_frame_metadata(frame_records, beat_frame_mappings, json_path: Path, csv_path: Path):
    """Save frame metadata to JSON and CSV files.

    JSON structure:
        - frame_count: Total number of frames
        - beat_count: Number of beats detected
        - beat_frame_mappings: List of beat-to-frame associations
        - frames: Array of frame records with kinematic data

    CSV columns:
        frame_index, timestamp_ns, x, y, vx, vy, ax, ay,
        tempo_bpm, is_beat_frame, closest_beat_timestamp_ns,
        closest_beat_frame_index, closest_beat_dt_ms, vibe_direction
    """
    payload = {
        "frame_count": len(frame_records),
        "beat_count": len(beat_frame_mappings),
        "beat_frame_mappings": beat_frame_mappings,
        "frames": frame_records,
    }

    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "frame_index",
        "timestamp_ns",
        "x",
        "y",
        "vx",
        "vy",
        "ax",
        "ay",
        "tempo_bpm",
        "is_beat_frame",
        "closest_beat_timestamp_ns",
        "closest_beat_frame_index",
        "closest_beat_dt_ms",
        "vibe_direction",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in frame_records:
            writer.writerow({k: row.get(k) for k in fieldnames})
