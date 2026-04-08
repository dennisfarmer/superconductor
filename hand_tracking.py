"""
Hand tracking module for MediaPipe hand landmark detection.

This module provides functions to capture video frames and detect hand landmarks
using Google's MediaPipe library. It is used by the tempo detection pipeline to
track hand movements for beat detection.

Key Functions:
    - frame(): Capture and process a single frame with hand detection
    - create_hand_landmarker(): Create and configure MediaPipe HandLandmarker
    - ensure_task_file(): Download MediaPipe model if not present

Timestamp Handling:
    - Video files: Uses CAP_PROP_POS_MSEC from OpenCV (reliable, based on file metadata)
    - Camera/Unknown: Falls back to CLOCK_MONOTONIC_RAW (session-relative time)

Note: There was a frame-skipping bug where cap.grab() was called before cap.read(),
causing exactly half the frames to be skipped from video files, resulting in 2x playback
speed in the output video. This has been removed for proper frame retrieval.

MediaPipe Hand Landmarks:
    - 21 landmarks per hand
    - Landmark 0: Wrist (used in tempo detection)
    - Normalized coordinates: x, y (0-1)

Horizontal Flip:
    - Frames are horizontally flipped (mirror effect) for more natural interaction
    - Coordinates are adjusted: raw_x is transformed to (1 - raw_x) for flipped view
"""

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
    """
    Capture and process a single video frame with hand landmark detection.

    This function:
    1. Reads a frame from the video capture
    2. Converts BGR to RGB (MediaPipe expects RGB)
    3. Detects hand landmarks using the provided HandLandmarker
    4. Draws landmarks and handedness labels on the frame
    5. Horizontally flips the image for mirror effect
    6. Optionally displays a preview window

    Args:
        cap: OpenCV VideoCapture object (video file or camera)
        hand_landmarker: MediaPipe HandLandmarker instance configured for VIDEO mode
        show_preview: If True, display frame in preview window

    Returns:
        tuple containing:
            - success (bool): True if frame was successfully processed
            - hand_landmarks_list: List of hand landmarks, each hand has 21 landmarks
            - frame_timestamp_ns (int): Frame timestamp in nanoseconds
            - frame_img: The processed frame image (or None if failed)

    Timestamp Handling:
        - If cap.get(cv2.CAP_PROP_POS_MSEC) returns valid value (video files):
          Uses that value (milliseconds * 1e6 → nanoseconds)
        - Otherwise (cameras or unsupported sources):
          Uses time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)

    Coordinates:
        - Original: (0,0) is top-left, (1,1) is bottom-right
        - After flip: Image is mirrored horizontally for natural interaction
        - Wrist detection: Uses landmark 12 from detected hands
    """
    # Read the next frame from the video capture
    success, img = cap.read()

    # Short wait for OpenCV's event loop (does nothing if no window)
    _ = cv2.waitKey(1) & 0xFF

    if not success:
        return False, [], 0, None

    # Get frame timestamp
    # Try to use video's embedded timestamp first (for video files)
    pos_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
    if pos_msec is not None and pos_msec >= 0:
        # Convert milliseconds to nanoseconds
        frame_timestamp_ns = int(pos_msec * 1e6)
    else:
        # Fall back to monotonic clock for cameras or unsupported sources
        frame_timestamp_ns = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)

    # Convert BGR (OpenCV default) to RGB (MediaPipe expects RGB)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height, width, n_channels = img.shape

    # Create MediaPipe Image object
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

    # Detect hand landmarks (VIDEO mode expects timestamp in milliseconds)
    # This uses internal tracking across frames for better accuracy
    results = hand_landmarker.detect_for_video(mp_image, int(frame_timestamp_ns / 1e6))

    hand_landmarks_list = results.hand_landmarks
    handedness_list = results.handedness

    # Create a blank mask for text rendering
    # This allows text to be rendered before the image is flipped, preventing mirrored text
    text_mask = np.zeros((height, width, n_channels), dtype="uint8")

    # Draw landmarks and handedness labels for each detected hand
    for idx in range(len(results.hand_landmarks)):
        # Get wrist coordinates (landmark 0 is wrist in MediaPipe)
        wrist_coordinates = hand_landmarks_list[idx][0]
        wrist_x = wrist_coordinates.x  # Normalized X coordinate (0-1)
        wrist_y = wrist_coordinates.y  # Normalized Y coordinate (0-1)
        wrist_handedness = handedness_list[idx][0].category_name  # "Left" or "Right"

        hand_landmarks = hand_landmarks_list[idx]

        # Calculate text position (mirrored horizontally for pre-flip rendering)
        # Since image will be flipped, we mirror the X coordinate here
        hand_label = wrist_handedness
        text_x = width - int(wrist_x * width)
        text_y = int(wrist_y * height)

        # Text styling
        font_scale = 1
        font_thickness = 1
        font_color = (88, 205, 54)  # Vibrant green

        # Render handedness label onto the text mask
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

        # Draw hand landmarks and connections on the image
        vision.drawing_utils.draw_landmarks(
            img,
            hand_landmarks,
            vision.HandLandmarksConnections.HAND_CONNECTIONS,
            vision.drawing_styles.get_default_hand_landmarks_style(),
            vision.drawing_styles.get_default_hand_connections_style(),
        )

    # Horizontally flip the image for mirror effect (more natural interaction)
    img = cv2.flip(img, 1)

    # Add the text mask (which contains pre-mirrored text) to the flipped image
    img = cv2.add(img, text_mask)

    # Display preview if requested
    if show_preview:
        cv2.imshow("Image", img)

    return True, hand_landmarks_list, frame_timestamp_ns, img


def create_hand_landmarker(task_path: Path):
    """
    Create and configure a MediaPipe HandLandmarker instance for VIDEO mode.

    The HandLandmarker is configured to:
    - Run in VIDEO mode (for temporal consistency across frames)
    - Detect up to 2 hands simultaneously
    - Use the specified .task model file

    Args:
        task_path: Path to the hand_landmarker.task model file

    Returns:
        vision.HandLandmarker: Configured hand landmarker instance

    Note:
        VIDEO mode is important for temporal consistency - the landmarker maintains
        state across frames, which improves tracking accuracy and reduces jitter.
    """
    base_options = python.BaseOptions(model_asset_path=str(task_path))
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2,  # Detect up to 2 hands simultaneously
        running_mode=mp.tasks.vision.RunningMode.VIDEO,  # Use VIDEO mode for temporal tracking
    )
    return vision.HandLandmarker.create_from_options(options)


def ensure_task_file(task_file: Path):
    """
    Download the MediaPipe hand_landmarker.task model if not present.

    The model is downloaded from Google's storage if it doesn't exist locally.
    This enables the code to run without requiring manual model download.

    Args:
        task_file: Path where the model file should be stored

    Model Info:
        - URL: https://storage.googleapis.com/mediapipe-models/...
        - Size: ~20MB (float16 quantized version)
        - Purpose: Detects hand landmarks and handedness
    """
    if not task_file.exists():
        urlretrieve(
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
            task_file,
        )
