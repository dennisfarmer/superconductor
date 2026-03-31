from __future__ import annotations

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import json
import csv
from pathlib import Path
from urllib.request import urlretrieve
import time
import matplotlib.pyplot as plt
from collections import deque
from filterpy.kalman import KalmanFilter as FilterPyKalmanFilter
from filterpy.common import Q_discrete_white_noise
from typing import overload
from scipy.interpolate import Akima1DInterpolator
from vibe import VIBE_FOURFOUR,VIBE_DOWN,VIBE_UP,VIBE_LEFT,VIBE_RIGHT,Vibe
# pip install mediapipe opencv-python filterpy
NS_TO_SECONDS = 1e-9
MIN_DT_SECONDS = 1e-3


def frame(cap, hand_landmarker, show_preview=False):
    for _ in range(1):
        cap.grab()

    success, img = cap.read() # This will now be a fresh frame
    _ = cv2.waitKey(1) & 0xFF
    if not success:
        return False, [], 0, None

    pos_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
    if pos_msec is not None and pos_msec >= 0:
        frame_timestamp_ns = int(pos_msec * 1e6)
    else:
        frame_timestamp_ns = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height, width, n_channels = img.shape

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=img_rgb
    )

    # retrieve landmark coordinates and handedness from camera input
    results = hand_landmarker.detect_for_video(mp_image, int(frame_timestamp_ns / 1e6))

    hand_landmarks_list = results.hand_landmarks
   
    handedness_list = results.handedness

    # cv2.putText doesn't seem to allow horizonally flipping text,
    # so this draws text onto an unflipped overlay which is then
    # added to the flipped image prior to calling cv2.imshow()
    text_mask = np.zeros((height, width, n_channels), dtype="uint8")

    for idx in range(len(results.hand_landmarks)):
        # Retrieve the wrist coordinate:
        wrist_coordinates = hand_landmarks_list[idx][0]
        wrist_x = wrist_coordinates.x
        wrist_y = wrist_coordinates.y
        #wrist_z = wrist_coordinates.z
        wrist_handedness = handedness_list[idx][0].category_name


        hand_landmarks = hand_landmarks_list[idx]
        # =====================================================================
        # Draw handedness (left or right hand) on the image.
        # =====================================================================
        # Draw text at the top left corner of the detected hand's bounding box.
        #x_coordinates = [landmark.x for landmark in hand_landmarks]
        #y_coordinates = [landmark.y for landmark in hand_landmarks]
        #text_x = width - int(max(x_coordinates) * width)
        #text_y = int(min(y_coordinates) * height)
        #hand_label = f"{handedness[0].category_name}"

        # Draw text at the wrist coordinate (index 0)
        hand_label = wrist_handedness
        text_x = width - int(wrist_x * width)
        text_y = int(wrist_y * height)

        # =====================================================================
        font_scale = 1  # negative value to flip text vertically
        font_thickness = 1
        font_color = (88, 205, 54) # vibrant green

        cv2.putText(text_mask, hand_label,
                    (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                    font_scale, font_color, font_thickness, cv2.LINE_AA)

        # =====================================================================
        # =====================================================================


        # Draw the hand landmarks.
        vision.drawing_utils.draw_landmarks(
            img,
            hand_landmarks,
            vision.HandLandmarksConnections.HAND_CONNECTIONS,
            vision.drawing_styles.get_default_hand_landmarks_style(),
            vision.drawing_styles.get_default_hand_connections_style())


    img = cv2.flip(img, 1)
    img = cv2.add(img, text_mask)

    if show_preview:
        cv2.imshow("Image", img)

    # if frame_counter % 25 == 0 and len(hand_landmarks_list) != 0:
    #     output_str = ""
    #     # note: output is horizontally flipped, so x=1-x, y=y
    #     #       is to keep top left as the origin (0,0)
    #     # 
    #     #       (1-x)*width, (y)*height is pixel coordinate
    #     for hand, coordinate in zip(handedness_list, hand_landmarks_list):
    #         output_str += f"{hand[0].category_name}: x={1-coordinate[0].x}, y={coordinate[0].y}, "

    #     print(f"wrist coordinates: {output_str}")
    #     frame_counter = 0
    return True, hand_landmarks_list, frame_timestamp_ns, img


def xytopolar(x, y):
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return r, theta


class ConstantVelocityKalman:
    """N-dimensional constant-velocity Kalman filter backed by FilterPy.

    State: [p0, v0, p1, v1, ...]
    Measurement: [p0, p1, ...]
    """

    def __init__(self, dimensions: int, pos_var=1e-4, vel_var=1e-3, meas_var=1e-3):
        if dimensions < 1:
            raise ValueError("dimensions must be >= 1")

        self.dimensions = dimensions
        self.pos_var = float(pos_var)
        self.vel_var = float(vel_var)
        self.meas_var = float(meas_var)
        self.last_t_ns = None
        self.kf = FilterPyKalmanFilter(dim_x=2 * dimensions, dim_z=dimensions)

        # Initial covariance and measurement model.
        self.kf.P *= 1e-1
        self.kf.R = np.eye(dimensions, dtype=float) * self.meas_var
        self.kf.H = np.zeros((dimensions, 2 * dimensions), dtype=float)
        for axis in range(dimensions):
            self.kf.H[axis, 2 * axis] = 1.0

    def _set_dynamic_model(self, dt_seconds: float):
        dt = max(float(dt_seconds), MIN_DT_SECONDS)

        # F is block diagonal with [ [1, dt], [0, 1] ] per axis.
        self.kf.F = np.eye(2 * self.dimensions, dtype=float)
        for axis in range(self.dimensions):
            self.kf.F[2 * axis, 2 * axis + 1] = dt

        # Q is block diagonal with one 2x2 CV process noise block per axis.
        q_pos = max(self.pos_var, 1e-12)
        q_vel = max(self.vel_var, 1e-12)
        q_block = Q_discrete_white_noise(dim=2, dt=dt, var=q_pos)
        q_block[1, 1] += q_vel * dt

        self.kf.Q = np.zeros((2 * self.dimensions, 2 * self.dimensions), dtype=float)
        for axis in range(self.dimensions):
            start = 2 * axis
            self.kf.Q[start:start + 2, start:start + 2] = q_block

    def initialize(self, measurement, timestamp_ns: int):
        z = np.atleast_1d(np.asarray(measurement, dtype=float))
        if z.shape[0] != self.dimensions:
            raise ValueError(f"expected {self.dimensions}D measurement, got shape {z.shape}")

        x0 = np.zeros((2 * self.dimensions, 1), dtype=float)
        for axis in range(self.dimensions):
            x0[2 * axis, 0] = z[axis]
        self.kf.x = x0
        self.last_t_ns = int(timestamp_ns)

    def update(self, measurement, timestamp_ns: int):
        z = np.atleast_1d(np.asarray(measurement, dtype=float))
        if z.shape[0] != self.dimensions:
            raise ValueError(f"expected {self.dimensions}D measurement, got shape {z.shape}")

        if self.last_t_ns is None:
            self.initialize(z, timestamp_ns)
            return self.get_position()

        dt_seconds = (int(timestamp_ns) - self.last_t_ns) * NS_TO_SECONDS
        self._set_dynamic_model(dt_seconds)
        self.kf.predict()
        self.kf.update(z)
        self.last_t_ns = int(timestamp_ns)
        return self.get_position()

    def get_position(self):
        position = np.array([self.kf.x[2 * axis, 0] for axis in range(self.dimensions)], dtype=float)
        if self.dimensions == 1:
            return float(position[0])
        return tuple(float(v) for v in position)


class Kalman1D(ConstantVelocityKalman):
    def __init__(self, pos_var=1e-4, vel_var=1e-3, meas_var=1e-3):
        super().__init__(dimensions=1, pos_var=pos_var, vel_var=vel_var, meas_var=meas_var)

    def update(self, measurement: float, timestamp_ns: int):
        return float(super().update([measurement], timestamp_ns))


class Kalman2D(ConstantVelocityKalman):
    def __init__(self, pos_var=1e-4, vel_var=1e-3, meas_var=1e-3):
        super().__init__(dimensions=2, pos_var=pos_var, vel_var=vel_var, meas_var=meas_var)
    
    def update(self, measurement_x: float, measurement_y: float, timestamp_ns: int):
        filtered_x, filtered_y = super().update([measurement_x, measurement_y], timestamp_ns)
        return float(filtered_x), float(filtered_y)


def create_hand_landmarker(task_path: Path):
    base_options = python.BaseOptions(model_asset_path=str(task_path))
    options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2, running_mode=mp.tasks.vision.RunningMode.VIDEO)
    return vision.HandLandmarker.create_from_options(options)
def get_physics_series(pos_buffer, speed_buffer):
    """Computes arrays for X, Y, V, and A using numpy vectorization."""
    pos_arr = np.array(pos_buffer)  # [x, y, ts]
    speed_arr = np.array(speed_buffer) # [vx, vy]
    
    # Calculate dt for acceleration
    dts = np.diff(pos_arr[:, 2]) * NS_TO_SECONDS
    dts = np.where(dts <= 0, MIN_DT_SECONDS, dts)
    
    # Acceleration (diff of speed / dt)
    # We match the length by taking the dts corresponding to the speed samples
    acc_arr = np.diff(speed_arr, axis=0) / dts[-len(speed_arr)+1:][:, None] if len(speed_arr) > 1 else np.empty((0,2))
    
    return {
        "x": pos_arr[:, 0], "y": pos_arr[:, 1],
        "vx": speed_arr[:, 0], "vy": speed_arr[:, 1],
        "ax": acc_arr[:, 0], "ay": acc_arr[:, 1]
    }

class BeatTracker:
    def __init__(self, size=20, vibe_pattern=VIBE_FOURFOUR, max_beats=5, decay_factor=0.8, miss_threshold=1.2):
        self.size = size
        self.data = np.zeros((size, 3))  # [x, y, timestamp]
        self.ptr = 0
        self.is_full = False
        
        self.vibe = Vibe(vibe_pattern)
        self.current_goal = self.vibe.next()
        self.last_beat_ts = 0
        self.beat_timestamps = deque(maxlen=max_beats)
        self.decay_factor = float(decay_factor)
        self.miss_threshold = float(miss_threshold)
        self.pending_missed_confirmation = False
        self.pending_expected_interval_ns = None

    def add_sample(self, x, y, ts):
        self.data[self.ptr] = [x, y, ts]
        self.ptr = (self.ptr + 1) % self.size
        if self.ptr == 0: self.is_full = True

    def get_linear_view(self):
        if not self.is_full: return self.data[:self.ptr]
        return np.roll(self.data, -self.ptr, axis=0)

    def get_recent_beats(self):
        return list(self.beat_timestamps)

    def get_current_goal_direction(self):
        direction_map = {
            VIBE_DOWN: "DOWN",
            VIBE_UP: "UP",
            VIBE_LEFT: "LEFT",
            VIBE_RIGHT: "RIGHT",
        }
        return direction_map.get(self.current_goal, "UNKNOWN")

    def get_anticipation_message(self):
        direction = self.get_current_goal_direction()
        return f"Anticipation: what direction is the beat looking for? {direction}"

    def get_smoothed_bpm(self):
        if len(self.beat_timestamps) < 2:
            return None

        beat_times = np.array(self.beat_timestamps, dtype=float)
        intervals_ns = np.diff(beat_times)
        valid_intervals_ns = intervals_ns[intervals_ns > 0]
        if len(valid_intervals_ns) == 0:
            return None

        bpm_series = 60 * 1e9 / valid_intervals_ns

        # Exponential decay weights: newest intervals receive higher weight.
        n = len(bpm_series)
        exponents = np.arange(n - 1, -1, -1, dtype=float)
        weights = np.power(self.decay_factor, exponents)
        weights_sum = np.sum(weights)
        if weights_sum <= 0:
            return float(np.mean(bpm_series))

        smoothed_bpm = float(np.sum(bpm_series * weights) / weights_sum)
        return smoothed_bpm

    def get_expected_interval_ns(self):
        if len(self.beat_timestamps) < 2:
            return None

        beat_times = np.array(self.beat_timestamps, dtype=float)
        intervals_ns = np.diff(beat_times)
        valid_intervals_ns = intervals_ns[intervals_ns > 0]
        if len(valid_intervals_ns) == 0:
            return None

        n = len(valid_intervals_ns)
        exponents = np.arange(n - 1, -1, -1, dtype=float)
        weights = np.power(self.decay_factor, exponents)
        weights_sum = np.sum(weights)
        if weights_sum <= 0:
            return float(np.mean(valid_intervals_ns))

        return float(np.sum(valid_intervals_ns * weights) / weights_sum)

    def apply_missed_beat_backup(self, current_ts):
        if self.pending_missed_confirmation:
            return False
        if self.last_beat_ts <= 0:
            return False

        expected_interval_ns = self.get_expected_interval_ns()
        if expected_interval_ns is None:
            return False

        elapsed_ns = float(current_ts - self.last_beat_ts)
        if elapsed_ns > self.miss_threshold * expected_interval_ns:
            self.current_goal = self.vibe.next()
            self.pending_missed_confirmation = True
            self.pending_expected_interval_ns = expected_interval_ns
            self.last_beat_ts = float(current_ts)
            return True

        return False

    def analyze_vibe_beat(self):
        view = self.get_linear_view()
        if len(view) < 10: return None, view,False

        ts = view[:, 2]
        # We always create both splines, but only use one for root finding
        ix = Akima1DInterpolator(ts, view[:, 0])
        iy = Akima1DInterpolator(ts, view[:, 1])

        # Determine which axis and direction to look for based on Vibe
        # UP/DOWN = Y axis, LEFT/RIGHT = X axis
        target_spline = iy if abs(self.current_goal) == 2 else ix
        
        # Find roots of the first derivative
        dy = target_spline.derivative()
        roots = dy.roots()
        
        # Filter for recent valid roots
        valid_roots = roots[(roots > ts[0]) & (roots <= ts[-1])]
        
        if len(valid_roots) > 0:
            potential_ts = valid_roots[-1]
            
            # Use 2nd derivative to check if it's the CORRECT extreme
            # VIBE_DOWN (-2) wants a minimum (accel > 0)
            # VIBE_UP   (+2) wants a maximum (accel < 0)
            # VIBE_LEFT (-1) wants a minimum
            # VIBE_RIGHT(+1) wants a maximum
            accel = target_spline.derivative(2)(potential_ts)
            
            is_correct_extreme = (accel > 0) if self.current_goal < 0 else (accel < 0)

            if is_correct_extreme and (potential_ts > self.last_beat_ts + 0.2*1e9):
                self.last_beat_ts = potential_ts
                actual_beat_ts = potential_ts

                should_count_for_tempo = True
                if self.pending_missed_confirmation:
                    should_count_for_tempo = False
                    if len(self.beat_timestamps) > 0 and self.pending_expected_interval_ns is not None:
                        interval_ns = actual_beat_ts - self.beat_timestamps[-1]
                        if interval_ns >= self.miss_threshold * self.pending_expected_interval_ns:
                            should_count_for_tempo = True

                    self.pending_missed_confirmation = False
                    self.pending_expected_interval_ns = None

                if should_count_for_tempo:
                    self.beat_timestamps.append(actual_beat_ts)
                # Advance the vibe to the next expected movement!
                self.current_goal = self.vibe.next()
                return actual_beat_ts, view, should_count_for_tempo

        return None, view, False
def ensure_task_file(task_file: Path):
    if not task_file.exists():
        urlretrieve(
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
            task_file,
        )


def process_video_collect_kinematics(cap, hand_landmarker_input, show_preview: bool = False, release_cap: bool = False):
    if cap is None or not cap.isOpened():
        raise RuntimeError("Could not use provided video capture object.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    position_kf = Kalman2D(pos_var=1e-4, vel_var=1e-3, meas_var=1e-4)
    tempo_kf = Kalman1D(meas_var=3.0)
    tracker = BeatTracker(100, max_beats=3000)

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
            raw_x, raw_y = 1 - landmarks[0][8].x, landmarks[0][8].y
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


def _fmt_num(value, digits=4):
    if value is None:
        return "NA"
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def draw_direction_arrow(frame_img, direction_str: str, center_x: int = None, center_y: int = None, arrow_length: int = 50, color=(0, 255, 0), thickness=3):
    """Draw an arrow showing the anticipated direction."""
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


def write_annotated_video(captured_frames, output_video_path: str, frame_records, fps: float, width: int, height: int):
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

        cv2.putText(frame_img, f"frame={row['frame_index']} t_ns={row['timestamp_ns']}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame_img, f"x={_fmt_num(row['x'])} y={_fmt_num(row['y'])}", (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame_img, f"vx={_fmt_num(row['vx'])} vy={_fmt_num(row['vy'])}", (10, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame_img, f"ax={_fmt_num(row['ax'])} ay={_fmt_num(row['ay'])}", (10, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame_img, f"tempo_bpm={_fmt_num(row['tempo_bpm'], 2)}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 210, 0), 1, cv2.LINE_AA)
        cv2.putText(frame_img, f"closest_beat_frame={row['closest_beat_frame_index']} dt_ms={_fmt_num(row['closest_beat_dt_ms'], 2)}", (10, 144), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 200, 100), 1, cv2.LINE_AA)

        if row.get("is_beat_frame", False):
            cv2.circle(frame_img, (24, 24), 12, (0, 0, 255), thickness=-1)
            cv2.putText(frame_img, "BEAT", (42, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)

        vibe_dir = row.get("vibe_direction", "UNKNOWN")
        draw_direction_arrow(frame_img, vibe_dir, center_x=width - 80, center_y=80, arrow_length=50, color=(100, 200, 255), thickness=3)

        # Highlight fingertip position
        if row.get("x") is not None and row.get("y") is not None:
            finger_x = int(row["x"] * width)
            finger_y = int(row["y"] * height)
            cv2.circle(frame_img, (finger_x, finger_y), 15, (0, 255, 0), thickness=2)
            cv2.circle(frame_img, (finger_x, finger_y), 5, (0, 255, 0), thickness=-1)

        writer.write(frame_img)

    writer.release()



def save_frame_metadata(frame_records, beat_frame_mappings, json_path: Path, csv_path: Path):
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


def main():
    cap = cv2.VideoCapture("/home/danny-wenjue-zhang/Videos/Webcam/video_fixed.mp4")
    if not cap.isOpened():
        raise RuntimeError("Could not open video source in main().")

    run_tempo_pipeline(
        cap=cap,
        source_name="/home/danny-wenjue-zhang/Videos/Webcam/video_fixed.mp4",
        output_video_path=None,
        output_json_path=None,
        output_csv_path=None,
        show_preview=False,
        release_cap=True,
    )

if __name__ == "__main__":
    main()