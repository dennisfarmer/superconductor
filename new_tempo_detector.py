from __future__ import annotations

import cv2

from tempo_pipeline import run_tempo_pipeline


def main():
    cap = cv2.VideoCapture("./42test.mp4")
    if not cap.isOpened():
        raise RuntimeError("Could not open video source in main().")

    run_tempo_pipeline(
        cap=cap,
        source_name="./42test.mp4",
        output_video_path=None,
        output_json_path=None,
        output_csv_path=None,
        show_preview=False,
        release_cap=True,
    )


if __name__ == "__main__":
    main()