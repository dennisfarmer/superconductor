"""
Main entry point for the SuperConductor tempo detection system.


This module provides a simple command-line interface to run the tempo detection
pipeline on a video file. It captures hand movements, tracks beats, calculates
tempo (BPM), and outputs annotated videos with beat markers and metadata files.

Usage:
    new_tempo_detector.py

The default processes "./42test.mp4" and outputs:
    - 42test_annotated.mp4 (video with beat annotations)
    - 42test_frame_data.json (frame-by-frame metadata)
    - 42test_frame_data.csv (tabular metadata)
"""

from __future__ import annotations

import cv2

from tempo_pipeline import run_tempo_pipeline


def main():
    """Run the tempo pipeline on the default test video file.

    This function:
    1. Opens the video file using OpenCV VideoCapture
    2. Runs the complete tempo detection pipeline
    3. Outputs annotated video and metadata files

    Raises:
        RuntimeError: If the video file cannot be opened
    """
    # Open the source video file
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open video source in main().")

    # Run the tempo detection pipeline
    # This will:
    # - Process each frame through hand tracking
    # - Detect beats based on hand movement patterns
    # - Calculate tempo (BPM) from beat intervals
    # - Annotate video with beat markers and overlay info
    # - Save metadata to JSON and CSV files
    try:
        run_tempo_pipeline(
            cap=cap,
            source_name="./camera",
            output_video_path=None,  # Default: camera_annotated.mp4
            output_json_path=None,   # Default: camera_frame_data.json
            output_csv_path=None,    # Default: camera_frame_data.csv
            show_preview=True,       # Required for q-key stop (focus the preview window)
            release_cap=True,        # Release VideoCapture when done
        )
    except KeyboardInterrupt:
        print("Interrupted by user. Exiting cleanly.")


# Entry point: Run the main function when this script is executed directly
if __name__ == "__main__":
    main()