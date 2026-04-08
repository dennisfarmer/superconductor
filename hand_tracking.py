from __future__ import annotations

import time
from pathlib import Path
from urllib.request import urlretrieve

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


def frame(cap, hand_landmarker, show_preview=False):
   

    success, img = cap.read()
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

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

    results = hand_landmarker.detect_for_video(mp_image, int(frame_timestamp_ns / 1e6))

    hand_landmarks_list = results.hand_landmarks
    handedness_list = results.handedness

    text_mask = np.zeros((height, width, n_channels), dtype="uint8")

    for idx in range(len(results.hand_landmarks)):
        wrist_coordinates = hand_landmarks_list[idx][0]
        wrist_x = wrist_coordinates.x
        wrist_y = wrist_coordinates.y
        wrist_handedness = handedness_list[idx][0].category_name

        hand_landmarks = hand_landmarks_list[idx]

        hand_label = wrist_handedness
        text_x = width - int(wrist_x * width)
        text_y = int(wrist_y * height)

        font_scale = 1
        font_thickness = 1
        font_color = (88, 205, 54)

        cv2.putText(
            text_mask,
            hand_label,
            (text_x, text_y),
            cv2.FONT_HERSHEY_DUPLEX,
            font_scale,
            font_color,
            font_thickness,
            cv2.LINE_AA,
        )

        vision.drawing_utils.draw_landmarks(
            img,
            hand_landmarks,
            vision.HandLandmarksConnections.HAND_CONNECTIONS,
            vision.drawing_styles.get_default_hand_landmarks_style(),
            vision.drawing_styles.get_default_hand_connections_style(),
        )

    img = cv2.flip(img, 1)
    img = cv2.add(img, text_mask)

    if show_preview:
        cv2.imshow("Image", img)

    return True, hand_landmarks_list, frame_timestamp_ns, img


def create_hand_landmarker(task_path: Path):
    base_options = python.BaseOptions(model_asset_path=str(task_path))
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2,
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
    )
    return vision.HandLandmarker.create_from_options(options)


def ensure_task_file(task_file: Path):
    if not task_file.exists():
        urlretrieve(
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
            task_file,
        )
