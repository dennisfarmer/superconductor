from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from beat_tracker import BeatTracker
from hand_tracking import create_hand_landmarker, ensure_task_file, frame
from kalman_filters import Kalman1D, Kalman2D
from tempo_constants import MIN_DT_SECONDS, NS_TO_SECONDS
from video_output import save_frame_metadata, write_annotated_video
from filterconst import TEMPO_VARIANCE, POS_VARIANCE, VEL_VARIANCE, MEAS_VARIANCE

def process_video_collect_kinematics(cap, hand_landmarker_input, show_preview: bool = False, release_cap: bool = False):
    if cap is None or not cap.isOpened():
        raise RuntimeError("Could not use provided video capture object.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    position_kf = Kalman2D(pos_var=POS_VARIANCE, vel_var=VEL_VARIANCE, meas_var=MEAS_VARIANCE)
    tempo_kf = Kalman1D(meas_var=TEMPO_VARIANCE)
    tracker = BeatTracker(100, max_beats=3000,mirror_pattern=True)

    frame_records = []
    detected_beat_timestamps = []
    captured_frames = []

    prev_ts = None
    prev_x = None
    prev_y = None
    prev_vx = 0.0
    prev_vy = 0.0
    last_tempo_bpm = None

    frame_index = 0
    while True:
        ok, landmarks, timestamp_ns, frame_img = frame(cap, hand_landmarker_input, show_preview=show_preview)
        if not ok:
            break

        captured_frames.append(frame_img.copy() if frame_img is not None else None)

        did_backup_switch = tracker.apply_missed_beat_backup(timestamp_ns)
        if did_backup_switch:
            print("Backup: interval too long, assuming a missed beat and switching direction.")

        record = {
            "frame_index": frame_index,
            "timestamp_ns": int(timestamp_ns),
            "x": None,
            "y": None,
            "vx": None,
            "vy": None,
            "ax": None,
            "ay": None,
            "tempo_bpm": last_tempo_bpm,
            "is_beat_frame": False,
            "closest_beat_timestamp_ns": None,
            "closest_beat_frame_index": None,
            "closest_beat_dt_ms": None,
            "vibe_direction": tracker.get_current_goal_direction(),
        }

        if landmarks:
            raw_x, raw_y = 1 - landmarks[0][12].x, landmarks[0][12].y
            filt_x, filt_y = position_kf.update(raw_x, raw_y, timestamp_ns)
            tracker.add_sample(filt_x, filt_y, timestamp_ns)

            vx = 0.0
            vy = 0.0
            ax = 0.0
            ay = 0.0
            if prev_ts is not None:
                dt = max((timestamp_ns - prev_ts) * NS_TO_SECONDS, MIN_DT_SECONDS)
                vx = (filt_x - prev_x) / dt
                vy = (filt_y - prev_y) / dt
                ax = (vx - prev_vx) / dt
                ay = (vy - prev_vy) / dt

            prev_ts = int(timestamp_ns)
            prev_x = float(filt_x)
            prev_y = float(filt_y)
            prev_vx = float(vx)
            prev_vy = float(vy)

            record["x"] = float(filt_x)
            record["y"] = float(filt_y)
            record["vx"] = float(vx)
            record["vy"] = float(vy)
            record["ax"] = float(ax)
            record["ay"] = float(ay)

            beat_ts, _, counted_for_tempo = tracker.analyze_vibe_beat()
            if beat_ts is not None:
                detected_beat_timestamps.append(int(beat_ts))

            if counted_for_tempo and len(tracker.get_recent_beats()) >= 2:
                beat_times = tracker.get_recent_beats()
                measured_bpm = 60 * 1e9 / (beat_times[-1] - beat_times[-2])
                filtered_bpm = tempo_kf.update(float(measured_bpm), int(beat_times[-1]))
                last_tempo_bpm = float(filtered_bpm)
                record["tempo_bpm"] = last_tempo_bpm

        frame_records.append(record)
        frame_index += 1

    if release_cap:
        cap.release()
    if show_preview:
        cv2.destroyAllWindows()
    return frame_records, detected_beat_timestamps, captured_frames, float(fps), width, height


def map_beats_to_closest_frames(frame_records, beat_timestamps):
    if not frame_records or not beat_timestamps:
        return [], set()

    frame_times = np.array([row["timestamp_ns"] for row in frame_records], dtype=np.int64)
    beat_frame_mappings = []
    beat_frame_indices = set()

    for beat_ts in beat_timestamps:
        nearest_idx = int(np.argmin(np.abs(frame_times - int(beat_ts))))
        nearest_frame = frame_records[nearest_idx]["frame_index"]
        dt_ms = abs(frame_records[nearest_idx]["timestamp_ns"] - int(beat_ts)) / 1e6
        beat_frame_mappings.append(
            {
                "beat_timestamp_ns": int(beat_ts),
                "closest_frame_index": int(nearest_frame),
                "closest_frame_timestamp_ns": int(frame_records[nearest_idx]["timestamp_ns"]),
                "dt_ms": float(dt_ms),
            }
        )
        beat_frame_indices.add(int(nearest_frame))

    return beat_frame_mappings, beat_frame_indices


def attach_closest_beat_info_to_frames(frame_records, beat_frame_mappings, beat_frame_indices):
    if not frame_records:
        return

    if not beat_frame_mappings:
        for row in frame_records:
            row["is_beat_frame"] = False
            row["closest_beat_timestamp_ns"] = None
            row["closest_beat_frame_index"] = None
            row["closest_beat_dt_ms"] = None
        return

    beat_timestamps = np.array([m["beat_timestamp_ns"] for m in beat_frame_mappings], dtype=np.int64)
    beat_frames = np.array([m["closest_frame_index"] for m in beat_frame_mappings], dtype=np.int64)

    for row in frame_records:
        row["is_beat_frame"] = row["frame_index"] in beat_frame_indices
        idx = int(np.argmin(np.abs(beat_timestamps - int(row["timestamp_ns"]))))
        row["closest_beat_timestamp_ns"] = int(beat_timestamps[idx])
        row["closest_beat_frame_index"] = int(beat_frames[idx])
        row["closest_beat_dt_ms"] = float(abs(int(row["timestamp_ns"]) - int(beat_timestamps[idx])) / 1e6)


def run_tempo_pipeline(
    cap,
    source_name: str = "capture",
    output_video_path: str | None = None,
    output_json_path: str | None = None,
    output_csv_path: str | None = None,
    show_preview: bool = False,
    release_cap: bool = False,
):
    input_path = Path(source_name)
    if output_video_path is None:
        output_video_path = str(input_path.with_name(f"{input_path.stem}_annotated.mp4"))
    if output_json_path is None:
        output_json_path = str(input_path.with_name(f"{input_path.stem}_frame_data.json"))
    if output_csv_path is None:
        output_csv_path = str(input_path.with_name(f"{input_path.stem}_frame_data.csv"))

    task_file = Path("hand_landmarker.task")
    ensure_task_file(task_file)
    hand_landmarker_input = create_hand_landmarker(task_file)

    frame_records, beat_timestamps, captured_frames, fps, width, height = process_video_collect_kinematics(
        cap=cap,
        hand_landmarker_input=hand_landmarker_input,
        show_preview=show_preview,
        release_cap=release_cap,
    )

    beat_frame_mappings, beat_frame_indices = map_beats_to_closest_frames(frame_records, beat_timestamps)
    attach_closest_beat_info_to_frames(frame_records, beat_frame_mappings, beat_frame_indices)

    write_annotated_video(captured_frames, output_video_path, frame_records, fps, width, height)
    save_frame_metadata(frame_records, beat_frame_mappings, Path(output_json_path), Path(output_csv_path))

    print(f"Annotated video: {output_video_path}")
    print(f"Metadata JSON: {output_json_path}")
    print(f"Metadata CSV: {output_csv_path}")
    print(f"Frames processed: {len(frame_records)} | Beats mapped: {len(beat_frame_mappings)}")
