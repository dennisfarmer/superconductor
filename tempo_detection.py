from enum import auto
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
from pathlib import Path
from urllib.request import urlretrieve
import time
import timeit
import matplotlib.pyplot as plt
from collections import deque
from scipy.interpolate import make_interp_spline
import sys
import filterpy as fpy
# pip install mediapipe opencv-python
from vibe import VIBE_FOURFOUR,VIBE_DOWN,VIBE_UP,VIBE_LEFT,VIBE_RIGHT
def frame(cap,hand_landmarker) -> tuple[list[list[auto]], float]:
    
   
    
    for _ in range(1): 
        cap.grab()
    
    success, img = cap.read() # This will now be a fresh frame
    key = cv2.waitKey(1) & 0xFF

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height, width, n_channels = img.shape

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=img_rgb
    )

    # retrieve landmark coordinates and handedness from camera input
    results = hand_landmarker.detect_for_video(mp_image,int(time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)/1e6))

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
        handedness = handedness_list[idx]

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
    return hand_landmarks_list,time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
def xytopolar(x,y):
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return r, theta 
class Vibe:
    def __init__(self, pattern):
        self.pattern = pattern
        self.index = 0

    def next(self):
        current_vibe = self.pattern[self.index]
        self.index = (self.index + 1) % len(self.pattern)
        return current_vibe
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
    base_options = python.BaseOptions(model_asset_path=str(task_file))
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2,
        running_mode=mp.tasks.vision.RunningMode.VIDEO
    )
    speed_buffer=deque(maxlen=20)
    hand_landmarker_input = vision.HandLandmarker.create_from_options(options)
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
    initial_beat=0
    last_beat_time = 0
    COOLDOWN_NS = 200_000_000 # 200ms debounce
    ment, 1 means detect x axis movement. This is a simple binary pattern that can be expanded to more complex rhythms.
    fourfour = Vibe(vibe)
    while True:
        pos_counter+=1
        hand_landmarker, timestamp = frame(Cap,hand_landmarker_input)
       
        # print(f"size: {len(hand_landmarker)}")
        if len(hand_landmarker) != 0:
            pos_buffer.append((hand_landmarker[0][8].x, hand_landmarker[0][8].y, timestamp)) # 8 is the tip of index finger
        if len(pos_buffer) > 2:
            speed = ((pos_buffer[-1][0] - pos_buffer[-2][0])/((pos_buffer[-1][2] - pos_buffer[-2][2]) / 1e9), (pos_buffer[-1][1] - pos_buffer[-2][1])/((pos_buffer[-1][2] - pos_buffer[-2][2]) / 1e9))
            speed_buffer.append(speed)
        if len(speed_buffer) >= 5:
            vx_old = np.mean([speed_buffer[-5][0], speed_buffer[-4][0]])
            vy_old = np.mean([speed_buffer[-5][1], speed_buffer[-4][1]])
            
            vx_ictus = speed_buffer[-3][0]
            vy_ictus = speed_buffer[-3][1]
            
            vx_new = np.mean([speed_buffer[-2][0], speed_buffer[-1][0]])
            vy_new = np.mean([speed_buffer[-2][1], speed_buffer[-1][1]])

            # 2. Calculate speeds (magnitudes)
            mag_old = np.sqrt(vx_old**2 + vy_old**2)
            mag_ictus = np.sqrt(vx_ictus**2 + vy_ictus**2)
            mag_new = np.sqrt(vx_new**2 + vy_new**2)

            # 3. Detect the Speed Dip (Local Minimum)
            # The ictus speed should be noticeably lower than the speed before and after it.
            # Multiplying by 0.8 ensures it's a true dip, not just a tiny fluctuation.
            is_speed_dip = (mag_ictus < mag_old * 0.8) and (mag_ictus < mag_new * 0.8)

            # Prevent division by zero for the angle math
            if mag_old > 0.001 and mag_new > 0.001:
                # 4. Detect the Direction Change (Dot Product)
                dot_product = (vx_old * vx_new) + (vy_old * vy_new)
                cos_theta = np.clip(dot_product / (mag_old * mag_new), -1.0, 1.0)
                angle_change = np.arccos(cos_theta)

                current_time = pos_buffer[-3][2] # Get the timestamp of the ictus itself!

                # 5. The Trigger: Sharp angle change + Speed dip + Cooldown
                # We use np.pi/4 (45 degrees) as it's more forgiving for the fluid motions of conducting
                if angle_change > np.pi/4 and is_speed_dip:
                    if current_time - last_beat_time > COOLDOWN_NS:
                        beat_buffer.append(current_time)
                        last_beat_time = current_time
                        sys.stdout.write('\a')
                        sys.stdout.flush()
        # detecting the last but two frame of speed. Getting coresponding timeframe
        if(len(beat_buffer) == 2):
            initial_bpm=60*1e9/(beat_buffer[-1]-beat_buffer[-2]+1)
        if(len(beat_buffer) > 2):
            initial_bpm=initial_bpm*0.5+(60*1e9/(beat_buffer[-1]-beat_buffer[-2]+1))*0.5
            print(f"original bpm: {60*1e9/(beat_buffer[-1]-beat_buffer[-2]+1)}, smoothed bpm: {initial_bpm},fps: {1e9/(pos_buffer[-1][2]-pos_buffer[-2][2]+1)}")
        if pos_counter % 5 == 0 and len(pos_buffer) > 2 and len(speed_buffer) > 1:
            pos_array = np.array(pos_buffer, dtype=float)
            speed_array = np.array(speed_buffer, dtype=float)

            x_series = pos_array[:, 0]
            y_series = pos_array[:, 1]
            vx_series = speed_array[:, 0]
            vy_series = speed_array[:, 1]

            timestamp_deltas = np.diff(pos_array[:, 2]) / 1e9
            needed = max(0, len(vx_series) - 1)
            if len(timestamp_deltas) >= needed and needed > 0:
                dt_for_acc = timestamp_deltas[-needed:]
            elif needed > 0:
                dt_for_acc = np.ones(needed)
            else:
                dt_for_acc = np.array([])

            if len(vx_series) > 1:
                safe_dt = np.where(dt_for_acc == 0, 1e-9, dt_for_acc)
                ax_series = np.diff(vx_series) / safe_dt
                ay_series = np.diff(vy_series) / safe_dt
            else:
                ax_series = np.array([])
                ay_series = np.array([])

            series_map = {
                "x": x_series,
                "y": y_series,
                "vx": vx_series,
                "vy": vy_series,
                "ax": ax_series,
                "ay": ay_series,
            }

            for plot_axis, (key, _) in zip(axes, plot_specs):
                data = np.array(series_map[key], dtype=float)
                if len(data) == 0:
                    lines[key].set_data([], [])
                    plot_axis.set_xlim(0, 1)
                    plot_axis.set_ylim(-1, 1)
                    continue

                x_index = np.arange(len(data))
                lines[key].set_data(x_index, data)

                plot_axis.set_xlim(0, max(1, len(data) - 1))

                data_min = float(np.min(data))
                data_max = float(np.max(data))
                if np.isclose(data_min, data_max):
                    pad = max(0.01, abs(data_max) * 0.1 + 0.01)
                else:
                    pad = (data_max - data_min) * 0.15
                plot_axis.set_ylim(data_min - pad, data_max + pad)

            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            
            
        
if __name__ == "__main__":
    main()