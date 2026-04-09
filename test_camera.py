#!/usr/bin/env python3
"""
Camera test utility for verifying webcam connectivity.

This script opens the default camera and displays a live preview with overlay
information including resolution, frame rate, and metrics.

Usage:
    python test_camera.py

Controls:
    - Press 'q' to exit

Features:
    - Auto-detects working camera (tries indices 0, 1, 2)
    - Displays camera properties (resolution, FPS)
    - Shows live preview with real-time metrics
"""

import cv2


def test_camera(camera_index=0):
    """Test camera at specified index and display preview.

    Args:
        camera_index: Camera device index to try (default: 0)

    Returns:
        bool: True if camera was opened successfully
    """
    print(f"Opening camera at index {camera_index}...")
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"ERROR: Could not open camera at index {camera_index}")
        return False

    # Get camera properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    backend = cap.getBackendName()

    print(f"Camera opened successfully!")
    print(f"  Backend: {backend}")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps}")
    print(f"  Press 'q' to exit...")

    frame_count = 0
    while True:
        ret, frame = cap.read()

        if not ret:
            if frame_count == 0:
                print("ERROR: Could not read frame from camera")
            else:
                print("Camera stopped providing frames")
            break

        frame_count += 1
        cv2.putText(
            frame,
            f"Frame: {frame_count} | FPS: {fps:.2f} | Resolution: {width}x{height}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )
        cv2.putText(
            frame,
            "Press 'q' to exit",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
            cv2.LINE_AA
        )

        cv2.imshow("Camera Test", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Test ended. Total frames captured: {frame_count}")
    return True


def main():
    # Try camera index 0 first, then try others if needed
    for idx in range(3):
        print(f"\n--- Trying camera index {idx} ---")
        if test_camera(idx):
            return

    print("\nERROR: Could not find a working camera. Tried indices 0, 1, 2")


if __name__ == "__main__":
    main()
