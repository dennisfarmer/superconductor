# SuperConductor Frontend

Real-time hand-motion tempo detection using MediaPipe + Kalman filters.

The pipeline captures camera frames, detects hand landmarks, tracks motion, detects beat events, estimates tempo (BPM), and exports:

- an annotated video,
- per-frame metadata (`.json` + `.csv`),
- optional plots with beat markers.

![diagram](media/diagram.jpeg)

## Features

- Real-time camera capture and beat detection.
- Live annotated preview with `q`-to-stop.
- Saved annotated video aligned with preview overlays.
- Per-frame kinematics (`x/y`, velocity, acceleration, tempo, beat flags).
- Beat-to-frame mapping and nearest-beat metadata.
- Plot utility with beat timestamp vertical bars colored by `vibe_direction`.

## Project Layout

- `new_tempo_detector.py`: Main camera entrypoint.
- `tempo_pipeline.py`: End-to-end processing pipeline.
- `hand_tracking.py`: MediaPipe hand landmarking and frame acquisition.
- `beat_tracker.py`: Beat logic and direction/vibe handling.
- `video_output.py`: Annotated video + metadata writers.
- `plot.py`: Plot frame data and beat markers from CSV.
- `palm_opened_closed_detection/`: Optional palm open/closed classifier utilities.

## Requirements

- Python 3.10+ (3.12 also used in this workspace).
- Webcam/camera device.
- OS packages needed by OpenCV GUI (for preview windows).

Python dependencies are listed in:

- `palm_opened_closed_detection/requirements.txt`

If you only use the tempo pipeline and plotting, you still need core packages like `opencv-python`, `mediapipe`, `numpy`, and `matplotlib`.

## Setup

From project root (`superconductor_frontend`):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r palm_opened_closed_detection/requirements.txt
```

The hand model file (`hand_landmarker.task`) is expected in the repo root. The pipeline will also ensure it exists.

## Run Camera Tempo Detection

```bash
python new_tempo_detector.py
```

Behavior:

- Opens a live annotated preview window.
- Press `q` (with preview window focused) to stop capture cleanly.
- `Ctrl+C` is also handled gracefully.

## Default Output Files

With current camera defaults (`source_name="./camera"`), outputs are written in project root:

- `camera_annotated.mp4`
- `camera_frame_data.json`
- `camera_frame_data.csv`

Legacy/example files may also exist (for example `42test_*`).

## Plotting Beat Timestamps

Generate plots from CSV metadata:

```bash
python plot.py
```

Options:

```bash
python plot.py --csv camera_frame_data.csv --output hand_movement_plots.png --no-show
```

What `plot.py` shows:

- Position curves (`x`, `y`) over time.
- Velocity/acceleration curves.
- Vertical beat bars at `is_beat_frame` timestamps.
- Beat bar color by `vibe_direction`:
    - `UP` → red
    - `DOWN` → deepskyblue
    - `LEFT` → gold
    - `RIGHT` → limegreen
    - unknown → gray

## Notes

- If preview window does not respond to keys, click/focus the OpenCV window first.
- If camera-reported FPS is unreliable, the pipeline estimates effective FPS from timestamps for output video timing.

## Palm Open/Closed Dataset

- Dataset: https://www.kaggle.com/datasets/dennisfj/open-closed-palm-dataset
