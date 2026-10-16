"""Regression tests for wall-clock capture and continuous tempo-following audio."""
from pathlib import Path
import tempfile
from threading import Event
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from hand_tracking import frame
from live_conductor import LatestCamera, run_live_conductor
from music_playback import MusicPlayer


class CameraTimingTests(unittest.TestCase):
    def test_live_camera_ignores_zero_file_timestamp(self):
        cap = Mock()
        cap.read.return_value = (True, np.zeros((8, 8, 3), dtype=np.uint8))
        cap.get.return_value = 0
        cap.read_timestamp_ns = None
        landmarker = Mock()
        landmarker.detect_for_video.return_value = SimpleNamespace(hand_landmarks=[], handedness=[])
        with patch('hand_tracking.time.monotonic_ns', side_effect=[1_000_000_000, 1_040_000_000]):
            first = frame(cap, landmarker, live=True)
            second = frame(cap, landmarker, live=True)
        self.assertEqual(second[2] - first[2], 40_000_000)
        self.assertEqual([c.args[1] for c in landmarker.detect_for_video.call_args_list], [1000, 1040])
        cap.get.assert_not_called()

    def test_capture_timestamp_survives_processing_delay(self):
        cap = Mock(read_timestamp_ns=2_123_000_000)
        cap.read.return_value = (True, np.zeros((8, 8, 3), dtype=np.uint8))
        landmarker = Mock()
        landmarker.detect_for_video.return_value = SimpleNamespace(hand_landmarks=[], handedness=[])
        self.assertEqual(frame(cap, landmarker, live=True)[2], 2_123_000_000)
        cap.get.return_value = 42
        self.assertEqual(frame(cap, landmarker)[2], 42_000_000)

    def test_latest_frame_replaces_backlog(self):
        ready, finish = Event(), Event()
        count = 0
        def read():
            nonlocal count
            count += 1
            if count <= 5:
                return True, count
            ready.set()
            finish.wait(2)
            return False, None
        cap = Mock()
        cap.read.side_effect = read
        with patch('live_conductor.cv2.VideoCapture', return_value=cap):
            camera = LatestCamera(0)
            try:
                self.assertTrue(ready.wait(2))
                self.assertEqual(camera.read(), (True, 5))
                self.assertIsNotNone(camera.read_timestamp_ns)
            finally:
                finish.set()
                camera.close()
        cap.release.assert_called_once()


class MusicTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / 'music.mp3'
        self.path.touch()

    def test_hold_without_pausing_and_reject_invalid_rates(self):
        player = MusicPlayer(self.path, 120)
        player.set_tempo(None)
        self.assertFalse(player.paused)
        self.assertTrue(player.started.is_set())
        player.set_tempo(180)
        self.assertEqual(player.target_speed, 1.5)
        self.assertEqual(player.speed, 1.0)  # Only audio rendering advances speed.
        player.speed = 1.2
        player.set_tempo(None)
        self.assertEqual(player.target_speed, 1.2)
        for bpm in (0, float('nan'), 100000):
            with self.assertRaises(ValueError):
                player.set_tempo(bpm)

    def test_wslg_selects_pulse_and_failed_start_cleans_up(self):
        for server, expected in [('unix:/mnt/wslg/PulseServer', 'pulseaudio'), ('', 'dummy')]:
            player = MusicPlayer(self.path, 120)
            decoded = SimpleNamespace(returncode=0, stdout=np.zeros((256, 2), dtype='<f4').tobytes())
            with patch.dict('music_playback.os.environ', {'PULSE_SERVER': server, 'SDL_AUDIODRIVER': 'dummy'}), \
                 patch('music_playback.shutil.which', return_value='/bin/tool'), \
                 patch('music_playback.subprocess.run', return_value=decoded), \
                 patch('music_playback.BungeeStream') as stream, \
                 patch('music_playback.subprocess.Popen', side_effect=OSError('test stop')) as launch:
                with self.assertRaises(OSError):
                    player.__enter__()
                self.assertEqual(launch.call_args.kwargs['env']['SDL_AUDIODRIVER'], expected)
                stream.return_value.close.assert_called_once()
                self.assertTrue(player.log.closed)

    def test_small_jitter_is_ignored_but_sustained_change_is_followed(self):
        player = MusicPlayer(self.path, 80)
        for bpm in (80.5, 79, 81.5, 80):
            player.set_tempo(bpm)
            player._advance_speed(0.25)
        self.assertEqual(player.speed, 1.0)
        player.set_tempo(100)
        for _ in range(100):
            before = player.speed
            player._advance_speed(0.01)
            self.assertLessEqual((player.speed - before) * 80, 0.040001)
        self.assertAlmostEqual(player.speed * 80, 84, delta=0.1)
        for _ in range(1900):
            player._advance_speed(0.01)
        self.assertAlmostEqual(player.speed * 80, 100, delta=0.1)
        player.set_tempo(60)
        before = player.speed
        player._advance_speed(0.1)
        self.assertAlmostEqual((before - player.speed) * 80, 0.4)
        player.set_tempo(None)
        held = player.speed
        player._advance_speed(10)
        self.assertEqual(player.speed, held)
        self.assertFalse(player.paused)

    def test_worker_error_reaches_live_loop(self):
        player = MusicPlayer(self.path, 120)
        player.error = OSError('output disconnected')
        with self.assertRaisesRegex(RuntimeError, 'Bungee playback failed: output disconnected'):
            player.set_tempo(None)

    def test_invalid_source_bpm(self):
        for bpm in (0, -1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                MusicPlayer(self.path, bpm)

    def test_live_loop_forwards_bpm_and_cleans_up_on_camera_failure(self):
        with patch('live_conductor.ensure_task_file'), \
             patch('live_conductor.create_hand_landmarker') as model, \
             patch('live_conductor.MusicPlayer') as player_class, \
             patch('live_conductor.LatestCamera') as camera_class, \
             patch('live_conductor.CycleTempoEstimator') as estimator_class, \
             patch('live_conductor.frame') as next_frame, \
             patch('live_conductor.cv2.imshow'), \
             patch('live_conductor.cv2.waitKey', return_value=-1), \
             patch('live_conductor.cv2.getWindowProperty', return_value=1), \
             patch('live_conductor.cv2.destroyAllWindows') as destroy:
            player = player_class.return_value.__enter__.return_value
            player.speed = 1.25
            estimator_class.return_value.update.side_effect = [None, 150, None]
            img = np.zeros((80, 80, 3), dtype=np.uint8)
            next_frame.side_effect = [(True, [], i * 40_000_000, img) for i in range(3)] + [(False, [], 0, None)]
            with self.assertRaisesRegex(RuntimeError, 'disconnected'):
                run_live_conductor(0, self.path, 120, playback='local')
            self.assertEqual([c.args for c in player.set_tempo.call_args_list], [(None,), (150,), (None,)])
            camera_class.return_value.close.assert_called_once()
            player_class.return_value.__exit__.assert_called_once()
            model.return_value.__exit__.assert_called_once()
            destroy.assert_called_once()


if __name__ == '__main__':
    unittest.main()
