from __future__ import annotations

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
from pathlib import Path
from urllib.request import urlretrieve
import time
import matplotlib.pyplot as plt
from collections import deque
from filterpy.kalman import KalmanFilter as FilterPyKalmanFilter
from filterpy.common import Q_discrete_white_noise
from typing import overload
from scipy.interpolate import Akima1DInterpolator
from vibe import VIBE_FOURFOUR,VIBE_DOWN,VIBE_UP,VIBE_LEFT,VIBE_RIGHT
# pip install mediapipe opencv-python filterpy
NS_TO_SECONDS = 1e-9
MIN_DT_SECONDS = 1e-3


def frame(cap, hand_landmarker):
    for _ in range(1):
        cap.grab()

    success, img = cap.read() # This will now be a fresh frame
    _ = cv2.waitKey(1) & 0xFF
    if not success:
        return [], time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height, width, n_channels = img.shape

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=img_rgb
    )

    # retrieve landmark coordinates and handedness from camera input
    results = hand_landmarker.detect_for_video(mp_image, int(time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW) / 1e6))

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
    return hand_landmarks_list, time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)


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
import numpy as np
from scipy.interpolate import Akima1DInterpolator

class MotionTracker:
    def __init__(self, size=100):
        self.size = size
        # Pre-allocate one contiguous block of memory
        self.data = np.zeros((size, 3))  # [x, y, ts]
        self.ptr = 0
        self.is_full = False

    def add_sample(self, x, y, ts):
        self.data[self.ptr] = [x, y, ts]
        self.ptr = (self.ptr + 1) % self.size
        if self.ptr == 0:
            self.is_full = True

    def get_view(self):
        """Returns a sorted view of the data without copying the whole array."""
        if not self.is_full:
            return self.data[:self.ptr]
        # np.roll creates a view/copy to linearize the circular buffer
        return np.roll(self.data, -self.ptr, axis=0)

    def interpolate(self, num_between=10):
        view = self.get_view()
        if len(view) < 5:
            return None
        
        ts = view[:, 2]
        # new_ts is a reference to a new array, but calculations are vectorized
        new_ts = np.linspace(ts[0], ts[-1], (len(ts) - 1) * num_between + 1)
        
        # Akima fitting on the existing memory views
        ix = Akima1DInterpolator(ts, view[:, 0])
        iy = Akima1DInterpolator(ts, view[:, 1])
        
        return ix(new_ts), iy(new_ts), new_ts
def main():
   
    pos_buffer = deque(maxlen=20)
    beat_buffer = deque(maxlen=20)
    pos_counter=0
    task_file = Path("hand_landmarker.task")

    # download hand_landmarker.task for MediaPipe
    if not task_file.exists():
        urlretrieve(
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
            task_file
            )

    speed_buffer = deque(maxlen=20)
    hand_landmarker_input = create_hand_landmarker(task_file)
#     try:
#         Cap = cv2.VideoCapture(1)
#     except:
#         print("Error: Could not open video stream.")
#         return
#     Cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
#     Cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#     Cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480) 
#     Cap.set(cv2.CAP_PROP_FPS, 120)
#     Cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
#     Cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25) 
# # Set a fast exposure value (higher = darker but faster)
#     Cap.set(cv2.CAP_PROP_EXPOSURE, -5)
    Cap = cv2.VideoCapture(0)
    plt.ion()
    fig, axes = plt.subplots(3, 2, figsize=(12, 8))
    axes = axes.flatten()
    plot_specs = [
        ("x", "X Position"),
        ("y", "Y Position"),
        ("vx", "Velocity X"),
        ("vy", "Velocity Y"),
        ("ax", "Acceleration X"),
        ("ay", "Acceleration Y"),
    ]
    lines = {}
    for plot_axis, (key, title) in zip(axes, plot_specs):
        line, = plot_axis.plot([], [], linewidth=1.5)
        lines[key] = line
        plot_axis.set_title(title)
        plot_axis.set_xlabel("Sample index")
        plot_axis.set_ylabel(key)
        plot_axis.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.show(block=False)
    position_kf = Kalman2D()
    initial_bpm = None
    tempo_kf = Kalman1D(meas_var=5.0)
    while True:
        # --- Main Loop Logic ---
        pos_counter += 1
        landmarks, timestamp = frame(Cap, hand_landmarker_input)

        if landmarks:
            # 1. Update Position (Kalman)
            raw_x, raw_y = 1 - landmarks[0][8].x, landmarks[0][8].y
            filt_x, filt_y = position_kf.update(raw_x, raw_y, timestamp)
            pos_buffer.append((filt_x, filt_y, timestamp))

            # 2. Velocity & Beat Detection
            if len(pos_buffer) >= 2:
                dt = max((pos_buffer[-1][2] - pos_buffer[-2][2]) * NS_TO_SECONDS, MIN_DT_SECONDS)
                vx = (pos_buffer[-1][0] - pos_buffer[-2][0]) / dt
                vy = (pos_buffer[-1][1] - pos_buffer[-2][1]) / dt
                speed_buffer.append((vx, vy))

        # 3. BPM Processing
        if len(beat_buffer) >= 2:
            delta_t = (beat_buffer[-1] - beat_buffer[-2])
            detected_bpm = 60 * 1e9 / delta_t
            
            if len(beat_buffer) == 2:
                initial_bpm = detected_bpm
            else:
                smoothed_bpm = tempo_kf.update(detected_bpm, beat_buffer[-1])
                initial_bpm = (initial_bpm * 0.95) + (smoothed_bpm * 0.05)
                print(f"BPM: {detected_bpm:.1f} | Smooth: {initial_bpm:.1f}")

        # 4. Visualization (Every 5 frames)
        if pos_counter % 5 == 0 and len(pos_buffer) > 5:
            series = get_physics_series(pos_buffer, speed_buffer)
            
            for ax, (key, _) in zip(axes, plot_specs):
                data = series[key]
                if len(data) == 0: continue
                    
                # Update line data
                x_idx = np.arange(len(data))
                lines[key].set_data(x_idx, data)
                
                # Auto-scale Y axis with padding
                d_min, d_max = np.min(data), np.max(data)
                pad = (d_max - d_min) * 0.15 if d_min != d_max else 0.1
                ax.set_xlim(0, max(1, len(data)-1))
                ax.set_ylim(d_min - pad, d_max + pad)

            fig.canvas.draw_idle()
            fig.canvas.flush_events()
                    
        
if __name__ == "__main__":
    main()