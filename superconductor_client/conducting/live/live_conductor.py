"""Live conducting: latest camera frame -> motion BPM -> music speed."""
from collections import deque
from threading import Condition, Thread
import time
from pathlib import Path

import cv2

from .camera_capture import open_camera
from .cycle_tempo import CycleTempoEstimator
from .hand_tracking import create_hand_landmarker, ensure_task_file, frame
from .music_playback import MusicPlayer, estimate_music_bpm
from .stop_gesture import StopGestureDetector
from .server_playback import ServerTempoPlayer

PROJECT = Path(__file__).resolve().parents[1]


class LatestCamera:
    """Drain capture on a worker, keeping one frame instead of accumulating lag."""
    def __init__(self, index):
        self.cap = open_camera(index)
        self.condition = Condition()
        self.latest = None
        self.stopped = False
        self.error = None
        self.read_timestamp_ns = None
        self.worker = Thread(target=self._capture, daemon=True)
        self.worker.start()

    def _capture(self):
        try:
            while not self.stopped:
                ok, image = self.cap.read()
                timestamp = time.monotonic_ns()
                with self.condition:
                    if not ok:
                        self.stopped = True
                    else:
                        self.latest = (image, timestamp)
                    self.condition.notify_all()
        except Exception as exc:
            with self.condition:
                self.error = exc
                self.stopped = True
                self.condition.notify_all()
        finally:
            self.cap.release()

    def read(self):
        with self.condition:
            if not self.condition.wait_for(lambda: self.latest is not None or self.stopped, timeout=2):
                raise RuntimeError("Camera stopped delivering frames")
            if self.error:
                raise RuntimeError("Camera capture failed") from self.error
            if self.stopped:
                return False, None
            image, self.read_timestamp_ns = self.latest
            self.latest = None
            return True, image

    def close(self):
        with self.condition:
            self.stopped = True
            self.condition.notify_all()
        self.worker.join(timeout=3)


def run_live_conductor(camera_index, music_path=None, music_bpm=None, tempo_window=5.0,
                       *, playback="server", server_url="http://localhost:9100"):
    if playback == "server":
        player_context = ServerTempoPlayer(server_url)
        print(f"Tempo server: {server_url}. Connect an audio client to hear music.")
        print("Stop gestures / F release tempo control; server music continues. R resumes conducting.")
    elif playback == "local":
        music_path, music_bpm = resolve_music(music_path, music_bpm)
        player_context = MusicPlayer(music_path, music_bpm)
    else:
        raise ValueError("playback must be server or local")
    return _run_live(camera_index, tempo_window, player_context, playback, music_bpm)


def resolve_music(music_path, music_bpm):
    if music_path is None:
        music_path = PROJECT / "testmusic-80.mp3"
    music_path = Path(music_path)
    if not music_path.is_file():
        raise FileNotFoundError(f"Music file not found: {music_path}")
    if music_bpm is None:
        known_tempos = {"testmusic-80.mp3": 80.0, "testmusic.mp3": 190.0}
        for name, source_bpm in known_tempos.items():
            if music_path.resolve() == (PROJECT / name).resolve():
                music_bpm = source_bpm
                break
    if music_bpm is None:
        print("Analyzing music tempo (first 90 seconds)...", flush=True)
        music_bpm = estimate_music_bpm(music_path)
        print(f"Estimated music tempo: {music_bpm:.2f} BPM; override with --music-bpm if needed.")
    return music_path, music_bpm


def _run_live(camera_index, tempo_window, player_context, playback, music_bpm):
    estimator = CycleTempoEstimator(window_seconds=tempo_window)
    task = PROJECT / "hand_landmarker.task"
    ensure_task_file(task)
    title = "SuperConductor - live camera"
    trail = deque()
    stop_detector = StopGestureDetector()
    camera = None
    try:
        with player_context as player, create_hand_landmarker(task) as landmarker:
            camera = LatestCamera(camera_index)
            print("Conduct a repeating 4/4 pattern. Q or Esc exits. Music keeps playing through tracking gaps.")
            while True:
                ok, hands, timestamp, image = frame(camera, landmarker, live=True)
                if not ok:
                    raise RuntimeError("Camera disconnected")
                x, y = (1 - hands[0][12].x, hands[0][12].y) if hands else (None, None)
                height, width = image.shape[:2]
                reason = stop_detector.update(hands, timestamp, estimator.confidence, width / height)
                if reason:
                    if not player.paused:
                        player.set_intentional_stop()
                    bpm = None
                else:
                    bpm = estimator.update(x, y, timestamp)
                    player.set_tempo(bpm)
                if hands:
                    trail.append((timestamp, (int(x * width), int(y * height))))
                else:
                    trail.clear()
                while trail and timestamp - trail[0][0] > 1_000_000_000:
                    trail.popleft()
                points = list(trail)
                for previous, current in zip(points, points[1:]):
                    cv2.line(image, previous[1], current[1], (0, 220, 255), 2)
                status = f"{bpm:.1f} BPM | {player.speed:.2f}x" if bpm is not None else f"Holding {player.speed:.2f}x - waiting for motion"
                if reason:
                    label = "Released" if playback == "server" else "Stopped"
                    status = f"{label}: {reason} - R to resume"
                cv2.putText(image, status, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
                detail = player.display_status() if playback == "server" else f"Music: {music_bpm:.1f} BPM"
                cv2.putText(image, detail, (12, 58),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
                cv2.putText(image, "Q / Esc: quit | F: release / stop | R: resume", (12, 82),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.imshow(title, image)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("f"):
                    stop_detector.reason = "Manual release"
                    player.set_intentional_stop()
                if key == ord("r") and stop_detector.reason:
                    stop_detector = StopGestureDetector()
                    estimator = CycleTempoEstimator(window_seconds=tempo_window)
                    player.set_intentional_stop(False)
                    trail.clear()
                if key in (ord("q"), 27):
                    break
                if cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                    break
    finally:
        if camera is not None:
            camera.close()
        cv2.destroyAllWindows()
