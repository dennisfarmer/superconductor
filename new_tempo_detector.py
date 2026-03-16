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

# pip install mediapipe opencv-python filterpy

NS_TO_SECONDS = 1e-9
MIN_DT_SECONDS = 1e-3

class Vibe:
    def __init__(self, pattern):
        self.pattern = pattern
        self.index = 0

    def next(self):
        current_vibe = self.pattern[self.index]
        self.index = (self.index + 1) % len(self.pattern)
        return current_vibe
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
    results = hand_landmarker.detect(mp_image)

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
    options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
    return vision.HandLandmarker.create_from_options(options)

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
    fig, ax = plt.subplots()
    # create a dummy initial point so the PathCollection and colorbar initialize
    # make the initial marker larger and visible (edge + alpha)
    scatter = ax.scatter([0.0], [0.0], c=[0.0], cmap='hsv', vmin=0, vmax=2*np.pi, s=120, edgecolor='k', alpha=0.9)
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Direction (degrees)")
    # show degree ticks on colorbar (convert radians ticks to degree labels)
    cbar.set_ticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
    cbar.set_ticklabels(["0°", "90°", "180°", "270°", "360°"])
    ax.set_title("Real-time Speed")
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Speed magnitude (units/s)")
    plt.show(block=False)
    position_kf = Kalman2D()
    initial_bpm = None
    tempo_kf = Kalman1D(meas_var=5.0)
    while True:
        pos_counter += 1
        hand_landmarker, timestamp = frame(Cap, hand_landmarker_input)
       
        # print(f"size: {len(hand_landmarker)}")
        if len(hand_landmarker) != 0:
            filt_x, filt_y = position_kf.update(1 - hand_landmarker[0][8].x, hand_landmarker[0][8].y, timestamp)
            pos_buffer.append((filt_x, filt_y, timestamp)) # 8 is the tip of index finger
        if len(pos_buffer) > 2:
            dt = (pos_buffer[-1][2] - pos_buffer[-2][2]) * NS_TO_SECONDS
            if dt <= 0:
                dt = MIN_DT_SECONDS
            speed = (
                (pos_buffer[-1][0] - pos_buffer[-2][0]) / dt,
                (pos_buffer[-1][1] - pos_buffer[-2][1]) / dt,
            )
            speed_buffer.append(speed)
            if len(speed_buffer) > 2:
                speed_old, speed_dir_old = xytopolar(speed_buffer[-2][0], speed_buffer[-2][1])
                speed_new, speed_dir_new = xytopolar(speed_buffer[-1][0], speed_buffer[-1][1])
                angle_delta = abs(speed_dir_new - speed_dir_old)
                if np.pi / 4 < angle_delta < 3 * np.pi / 4 and speed_new * speed_old > 0.2:
                    beat_buffer.append(pos_buffer[-1][2])

        if len(beat_buffer) == 2:
            initial_bpm = 60 * 1e9 / (beat_buffer[-1] - beat_buffer[-2])
        if len(beat_buffer) > 2:
            detected_bpm = 60 * 1e9 / (beat_buffer[-1] - beat_buffer[-2])
            smoothed_bpm = tempo_kf.update(detected_bpm, beat_buffer[-1])
            initial_bpm = initial_bpm * 0.95 + smoothed_bpm * 0.05 if initial_bpm is not None else smoothed_bpm

            print(f"original bpm: {60*1e9/(beat_buffer[-1]-beat_buffer[-2])}, smoothed bpm: {initial_bpm}")
        if pos_counter % 5 == 0 and len(speed_buffer) > 1:
            # 1. Prepare data
            x_data = np.arange(len(speed_buffer))
            # Calculate magnitudes (Speed)
            y_data = np.array([np.sqrt(v[0]**2 + v[1]**2) for v in speed_buffer])
            # Calculate directions (0 to 2pi)
            angles = np.array([np.arctan2(v[1], v[0]) for v in speed_buffer])
            angles = (angles + 2 * np.pi) % (2 * np.pi)

            # 2. Update Scatter Object
            offsets = np.column_stack((x_data, y_data))
            scatter.set_offsets(offsets)
            
            # Explicitly set the color array and the limits
            scatter.set_array(angles)
            
            # 3. Dynamic Marker Sizing
            # Use a slightly more stable scaling for visibility
            max_y = np.max(y_data) if np.max(y_data) > 0 else 1
            sizes = (y_data / max_y) * 200 + 50
            scatter.set_sizes(sizes)

            # 4. Critical: Update Axis Limits
            ax.set_xlim(0, len(speed_buffer))
            # Give the Y axis 20% headroom
            ax.set_ylim(0, max_y * 1.2)
            
            # 5. Redraw
            fig.canvas.draw_idle() # Better than draw() for real-time
            fig.canvas.flush_events()
            
            
        
if __name__ == "__main__":
    main()