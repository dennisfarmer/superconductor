"""
Core tempo detection pipeline for SuperConductor system.

This module processes video frames to detect hand movements, track beats,
and calculate tempo (BPM) using MediaPipe hand tracking and Kalman filtering.

Key Functions:
    - process_video_collect_kinematics(): Main processing loop that extracts
      kinematic data from video frames
    - map_beats_to_closest_frames(): Associates beat timestamps with video frames
    - attach_closest_beat_info_to_frames(): Attaches beat metadata to frame records
    - run_tempo_pipeline(): Complete end-to-end pipeline entry point

Data Flow:
    Video Input → Hand Tracking → Beat Detection → Tempo Calculation → Output

Kalman Filter Architecture:
    - Kalman2D: Filters 2D hand position (x, y) with velocity state variables
    - Kalman1D: Filters 1D tempo (BPM) measurements to smooth tempo output
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

from beat_tracker import BeatTracker
from hand_tracking import create_hand_landmarker, ensure_task_file, frame
from kalman_filters import Kalman1D, Kalman2D
from tempo_constants import MIN_DT_SECONDS, NS_TO_SECONDS
from video_output import render_annotated_frame, save_frame_metadata, write_annotated_video
from filterconst import TEMPO_VARIANCE, POS_VARIANCE, VEL_VARIANCE, MEAS_VARIANCE


def estimate_effective_fps(frame_records, fallback_fps: float) -> float:
    """Estimate real FPS from frame timestamps to avoid playback speed mismatch."""
    if not frame_records or len(frame_records) < 2:
        return float(fallback_fps)

    timestamps = np.array([row["timestamp_ns"] for row in frame_records], dtype=np.int64)
    dt_ns = np.diff(timestamps)
    dt_ns = dt_ns[dt_ns > 0]
    if dt_ns.size == 0:
        return float(fallback_fps)

    median_dt_ns = float(np.median(dt_ns))
    if median_dt_ns <= 0:
        return float(fallback_fps)

    effective_fps = 1e9 / median_dt_ns
    if effective_fps < 1.0 or effective_fps > 240.0:
        return float(fallback_fps)
    return float(effective_fps)


def process_video_collect_kinematics(cap, hand_landmarker_input, show_preview: bool = False, release_cap: bool = False):
    """
    Process video frames to collect kinematic hand movement data.

    This is the main processing loop that:
    1. Reads frames from the video capture
    2. Detects hand landmarks using MediaPipe
    3. Applies Kalman filtering for smooth position tracking
    4. Detects beats based on hand movement patterns
    5. Calculates velocity, acceleration, and tempo
    6. Records all metadata for each frame

    Args:
        cap: OpenCV VideoCapture object (video file or camera)
        hand_landmarker_input: MediaPipe HandLandmarker instance
        show_preview: If True, display live preview window during processing
        release_cap: If True, release VideoCapture when done

    Returns:
        tuple containing:
            - frame_records: List of dicts with kinematic data for each frame
            - detected_beat_timestamps: List of timestamps when beats were detected
            - captured_frames: List of raw frame images
            - fps: Frame rate of the video
            - width: Video width in pixels
            - height: Video height in pixels

    Kinematic Data Recorded:
        - x, y: Normalized hand position (0-1)
        - vx, vy: Velocity (units per second)
        - ax, ay: Acceleration (units per second^2)
        - tempo_bpm: Estimated tempo in beats per minute
        - vibe_direction: Current directional goal for beat detection
        - is_beat_frame: Whether this frame contains a detected beat
    """
    # Validate video capture object
    if cap is None or not cap.isOpened():
        raise RuntimeError("Could not use provided video capture object.")

    # Extract video metadata
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0  # Default FPS if not available
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Initialize Kalman filters for position (2D) and tempo (1D)
    # These filters smooth noisy measurements over time
    position_kf = Kalman2D(pos_var=POS_VARIANCE, vel_var=VEL_VARIANCE, meas_var=MEAS_VARIANCE)
    tempo_kf = Kalman1D(meas_var=TEMPO_VARIANCE)

    # Initialize beat tracker with mirror pattern for alternating directional beats
    tracker = BeatTracker(100, max_beats=3000, mirror_pattern=False)

    # Storage for processing results
    frame_records = []  # Metadata for each frame
    detected_beat_timestamps = []  # Raw beat detection timestamps
    captured_frames = []  # Raw frame images for video output

    # Real-time FPS tracking: use perf_counter for accurate elapsed time measurement
    # This gives process_video_collect_kinematics access to live FPS during processing
    frame_count = 0
    loop_start_ns = time.perf_counter_ns()
    current_fps = fps  # Start with reported FPS, update in real-time

    # Previous frame state for kinematic calculations
    prev_ts = None  # Previous frame timestamp
    prev_x = None  # Previous X position
    prev_y = None  # Previous Y position
    prev_vx = 0.0  # Previous X velocity
    prev_vy = 0.0  # Previous Y velocity
    last_tempo_bpm = None  # Smoothes tempo output across frames

    # Main processing loop - process each frame
    frame_index = 0
    try:
        while True:
            # Get next frame with hand landmarks
            ok, landmarks, timestamp_ns, frame_img = frame(cap, hand_landmarker_input, show_preview=False)
            if not ok:
                break  # End of video or user requested stop (q)

            # Store raw frame for later video writing
            captured_frames.append(frame_img.copy() if frame_img is not None else None)

            # Check for missed beats (backup switching logic)
            # This handles cases where the beat tracker might miss detection
            did_backup_switch = tracker.apply_missed_beat_backup(timestamp_ns)
            if did_backup_switch:
                print("Backup: interval too long, assuming a missed beat and switching direction.")

            # Real-time FPS calculation: track elapsed time and frames processed
            frame_count += 1
            elapsed_ns = time.perf_counter_ns() - loop_start_ns
            if elapsed_ns > 500_000_000:  # Update FPS every 0.5 seconds minimum
                current_fps = frame_count / (elapsed_ns / 1e9)
                loop_start_ns = time.perf_counter_ns()
                frame_count = 0

            # Initialize frame record with default values
            record = {
                "frame_index": frame_index,
                "timestamp_ns": int(timestamp_ns),
                "fps": current_fps,  # Real-time FPS accessible here
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

            # Process hand landmarks if detected
            if landmarks:
                # Extract wrist position (landmark 12), apply horizontal flip
                raw_x, raw_y = 1 - landmarks[0][12].x, landmarks[0][12].y

                # Apply Kalman filter to smooth position and track velocity
                filt_x, filt_y = position_kf.update(raw_x, raw_y, timestamp_ns)

                # Add filtered position to beat tracker
                tracker.add_sample(filt_x, filt_y, timestamp_ns)

                # Calculate kinematics (velocity and acceleration)
                vx = 0.0
                vy = 0.0
                ax = 0.0
                ay = 0.0
                if prev_ts is not None:
                    # Calculate time difference in seconds
                    dt = max((timestamp_ns - prev_ts) * NS_TO_SECONDS, MIN_DT_SECONDS)

                    # Velocity = (current_pos - prev_pos) / dt
                    vx = (filt_x - prev_x) / dt
                    vy = (filt_y - prev_y) / dt

                    # Acceleration = (current_vel - prev_vel) / dt
                    ax = (vx - prev_vx) / dt
                    ay = (vy - prev_vy) / dt

                # Update previous state for next iteration
                prev_ts = int(timestamp_ns)
                prev_x = float(filt_x)
                prev_y = float(filt_y)
                prev_vx = float(vx)
                prev_vy = float(vy)

                # Store kinematic data in record
                record["x"] = float(filt_x)
                record["y"] = float(filt_y)
                record["vx"] = float(vx)
                record["vy"] = float(vy)
                record["ax"] = float(ax)
                record["ay"] = float(ay)

                # Analyze hand movement for beat detection
                beat_ts, _, counted_for_tempo = tracker.analyze_vibe_beat(curr)
                if beat_ts is not None:
                    detected_beat_timestamps.append(int(beat_ts))
                    record["is_beat_frame"] = True

                # Calculate tempo from beat intervals
                if counted_for_tempo and len(tracker.get_recent_beats()) >= 2:
                    beat_times = tracker.get_recent_beats()
                    # BPM = 60 seconds / beat_interval in nanoseconds * 1e9
                    measured_bpm = 60 * 1e9 / (beat_times[-1] - beat_times[-2])

                    # Apply Kalman filter to smooth tempo measurements
                    filtered_bpm = tempo_kf.update(float(measured_bpm), int(beat_times[-1]))
                    last_tempo_bpm = float(filtered_bpm)
                    record["tempo_bpm"] = last_tempo_bpm

            # Store frame record
            frame_records.append(record)

            # Live annotated preview (camera mode): show exactly what will be saved
            if show_preview and frame_img is not None:
                rendered_preview = render_annotated_frame(frame_img, record, width, height)
                captured_frames[-1] = rendered_preview.copy()
                cv2.imshow("Annotated Preview", rendered_preview)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

            frame_index += 1
    finally:
        if release_cap:
            cap.release()
        if show_preview:
            cv2.destroyAllWindows()

    return frame_records, detected_beat_timestamps, captured_frames, float(current_fps), width, height


def map_beats_to_closest_frames(frame_records, beat_timestamps):
    """
    Map beat timestamps to the nearest video frames.

    Since beat detection can occur at arbitrary times between frames,
    each beat is associated with the frame that is temporally closest.

    Args:
        frame_records: List of frame metadata dicts
        beat_timestamps: List of beat timestamps in nanoseconds

    Returns:
        tuple containing:
            - beat_frame_mappings: List of dicts mapping each beat to its nearest frame
            - beat_frame_indices: Set of frame indices that contain beats

    Example mapping:
        Beat at 835ms → Frame at 833ms (nearest available frame)
    """
    if not frame_records or not beat_timestamps:
        return [], set()

    # Extract frame timestamps for efficient search
    frame_times = np.array([row["timestamp_ns"] for row in frame_records], dtype=np.int64)
    beat_frame_mappings = []
    beat_frame_indices = set()

    # For each beat, find the closest frame
    for beat_ts in beat_timestamps:
        # Find index of frame with minimum absolute time difference
        nearest_idx = int(np.argmin(np.abs(frame_times - int(beat_ts))))
        nearest_frame = frame_records[nearest_idx]["frame_index"]

        # Calculate timing difference in milliseconds
        dt_ms = abs(frame_records[nearest_idx]["timestamp_ns"] - int(beat_ts)) / 1e6

        # Store the mapping
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
    """
    Attach beat metadata to each frame record.

    Each frame gets information about the nearest beat, including:
    - is_beat_frame: Whether this frame is marked as containing a beat
    - closest_beat_timestamp_ns: Timestamp of the nearest beat
    - closest_beat_frame_index: Index of the frame containing the nearest beat
    - closest_beat_dt_ms: Time difference from this frame to nearest beat

    Args:
        frame_records: List of frame metadata dicts (modified in place)
        beat_frame_mappings: List of beat-to-frame mappings
        beat_frame_indices: Set of frame indices that contain beats
    """
    if not frame_records:
        return

    # Handle case with no detected beats
    if not beat_frame_mappings:
        for row in frame_records:
            row["is_beat_frame"] = False
            row["closest_beat_timestamp_ns"] = None
            row["closest_beat_frame_index"] = None
            row["closest_beat_dt_ms"] = None
        return

    # Extract beat data for efficient lookup
    beat_timestamps = np.array([m["beat_timestamp_ns"] for m in beat_frame_mappings], dtype=np.int64)
    beat_frames = np.array([m["closest_frame_index"] for m in beat_frame_mappings], dtype=np.int64)

    # Attach beat info to each frame
    for row in frame_records:
        # Mark if this frame contains a beat
        row["is_beat_frame"] = row["frame_index"] in beat_frame_indices

        # Find the nearest beat to this frame
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
    """
    Run the complete tempo detection pipeline.

    This function orchestrates the entire process:
    1. Ensures MediaPipe model is downloaded
    2. Creates hand landmarker
    3. Processes video to detect beats and calculate tempo
    4. Maps beats to video frames
    5. Writes annotated video and metadata files

    Args:
        cap: OpenCV VideoCapture object (video file or camera)
        source_name: Identifies the source (used for output file naming)
        output_video_path: Path for annotated video output (auto-generated if None)
        output_json_path: Path for JSON metadata output (auto-generated if None)
        output_csv_path: Path for CSV metadata output (auto-generated if None)
        show_preview: Display live preview during processing
        release_cap: Release VideoCapture when done

    Output Files (auto-named if paths not provided):
        - {source_name}_annotated.mp4: Video with beat annotations
        - {source_name}_frame_data.json: JSON metadata
        - {source_name}_frame_data.csv: CSV metadata
    """
    # Generate output file paths from source name if not provided
    input_path = Path(source_name)
    if output_video_path is None:
        output_video_path = str(input_path.with_name(f"{input_path.stem}_annotated.mp4"))
    if output_json_path is None:
        output_json_path = str(input_path.with_name(f"{input_path.stem}_frame_data.json"))
    if output_csv_path is None:
        output_csv_path = str(input_path.with_name(f"{input_path.stem}_frame_data.csv"))

    # Ensure MediaPipe hand model is downloaded
    task_file = Path("hand_landmarker.task")
    ensure_task_file(task_file)

    # Create hand landmarker configured for VIDEO mode
    hand_landmarker_input = create_hand_landmarker(task_file)

    # Process video and collect kinematic data
    frame_records, beat_timestamps, captured_frames, fps, width, height = process_video_collect_kinematics(
        cap=cap,
        hand_landmarker_input=hand_landmarker_input,
        show_preview=show_preview,
        release_cap=release_cap,
    )

    # Map detected beats to nearest video frames
    beat_frame_mappings, beat_frame_indices = map_beats_to_closest_frames(frame_records, beat_timestamps)

    # Attach beat metadata to each frame record
    attach_closest_beat_info_to_frames(frame_records, beat_frame_mappings, beat_frame_indices)

    # Estimate effective FPS from timestamps to keep output speed aligned with capture cadence
    output_fps = estimate_effective_fps(frame_records, fps)

    # Write annotated video output
    write_annotated_video(
        captured_frames,
        output_video_path,
        frame_records,
        output_fps,
        width,
        height,
        show_output_preview=False,
        frames_are_annotated=show_preview,
    )

    # Save metadata files (JSON and CSV)
    save_frame_metadata(frame_records, beat_frame_mappings, Path(output_json_path), Path(output_csv_path))

    # Print summary
    print(f"Annotated video: {output_video_path}")
    print(f"Metadata JSON: {output_json_path}")
    print(f"Metadata CSV: {output_csv_path}")
    print(f"Video FPS used: {output_fps:.2f} (capture reported: {fps:.2f})")
    print(f"Frames processed: {len(frame_records)} | Beats mapped: {len(beat_frame_mappings)}")
