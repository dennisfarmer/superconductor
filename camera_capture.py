"""Shared webcam setup for the live conductor and standalone camera preview."""
import sys
import warnings

import cv2


def open_camera(index):
    # Request the Linux camera backend explicitly for USB cameras exposed to WSL.
    backend = cv2.CAP_V4L2 if sys.platform.startswith("linux") else cv2.CAP_ANY
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open camera {index}")
    try:
        # Compressed frames reduce USB transport bandwidth. Incomplete YUYV
        # frames can leave a green region even when the device opens correctly.
        if not cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG")):
            warnings.warn("Camera rejected MJPEG; using its negotiated format.", RuntimeWarning)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_CONVERT_RGB, 1)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap
    except BaseException:
        cap.release()
        raise
