"""Live camera conducting with tempo-following music.

Run without a video path for live mode, or pass a video for offline annotation.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2

from annotate.tempo_pipeline import run_tempo_pipeline


def main(argv=None):
    """Run live conducting by default, or annotate an explicit video file."""
    parser = argparse.ArgumentParser(description="Conduct music with a live camera, or annotate a video.")
    parser.add_argument(
        "video",
        nargs="?",
        type=Path,
        default=None,
        help="Optional input video for offline annotation; omit for live camera",
    )
    parser.add_argument("--tempo-method", choices=("cycle", "beats"), default="cycle",
                        help="Stable 4/4 cycle monitoring (default), or legacy beat intervals")
    parser.add_argument("--tempo-window", type=float, default=5.0,
                        help="Cycle history in seconds, at least 2 (default: 5)")
    parser.add_argument("--camera", type=int, default=0, help="Live camera device index (default: 0)")
    parser.add_argument("--playback", choices=("server", "local"), default="server",
                        help="Server HTTP tempo control (default), or local Bungee music")
    parser.add_argument("--server-url", default="http://localhost:9100",
                        help="Tempo server URL; use http://localhost:9100 for the Lighthouse tunnel")
    parser.add_argument("--music", type=Path, default=Path(__file__).with_name("testmusic-80.mp3"),
                        help="Local playback only: music file (default: testmusic-80.mp3)")
    parser.add_argument("--music-bpm", type=float,
                        help="Local playback only: original track BPM (80 for testmusic-80.mp3, 190 for testmusic.mp3; otherwise auto-estimate)")
    args = parser.parse_args(argv)
    if not math.isfinite(args.tempo_window) or args.tempo_window < 2:
        parser.error("--tempo-window must be finite and at least 2 seconds")
    if args.music_bpm is not None and (not math.isfinite(args.music_bpm) or args.music_bpm <= 0):
        parser.error("--music-bpm must be finite and positive")
    if args.video is None:
        if args.tempo_method != "cycle":
            parser.error("Live camera mode uses --tempo-method cycle")
        from live.live_conductor import run_live_conductor
        try:
            run_live_conductor(args.camera, args.music, args.music_bpm, args.tempo_window,
                               playback=args.playback, server_url=args.server_url)
        except KeyboardInterrupt:
            pass
        except (RuntimeError, OSError, ValueError) as exc:
            parser.exit(1, f"Live conducting failed: {exc}\n")
        return
    input_video = args.video
    cap = cv2.VideoCapture(str(input_video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {input_video}")

    # Run the tempo detection pipeline
    # This will:
    # - Process each frame through hand tracking
    # - Detect beats based on hand movement patterns
    # - Estimate tempo from repeating 4/4 motion (or legacy beat intervals)
    # - Annotate video with beat markers and overlay info
    # - Save metadata to JSON and CSV files
    run_tempo_pipeline(
        cap=cap,
        tempo_method=args.tempo_method,
        tempo_window_seconds=args.tempo_window,
        source_name=str(input_video),
        output_video_path=None,  # Default: outputs/annotations/42test_annotated.mp4
        output_json_path=None,   # Default: outputs/annotations/42test_frame_data.json
        output_csv_path=None,    # Default: outputs/annotations/42test_frame_data.csv
        show_preview=False,      # Set True to see live preview during processing
        release_cap=True,        # Release VideoCapture when done
    )


# Entry point: Run the main function when this script is executed directly
if __name__ == "__main__":
    main()
