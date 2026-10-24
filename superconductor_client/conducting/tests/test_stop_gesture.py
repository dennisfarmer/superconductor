"""Synthetic stop-cue regressions; these do not establish real-world accuracy."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock
import unittest

import numpy as np

from live.music_playback import MusicPlayer
from live.stop_gesture import StopGestureDetector


def hand(x=0.0, fist=False):
    points = [SimpleNamespace(x=0.5+x, y=0.5, z=0.0) for _ in range(21)]
    for (mcp, pip, dip, tip), offset in zip(((5,6,7,8), (9,10,11,12), (13,14,15,16), (17,18,19,20)), (-0.04,0,0.03,0.06)):
        for index, y in ((mcp,0.10), (pip,0.16), (dip,0.13 if fist else 0.20), (tip,0.09 if fist else 0.24)):
            points[index] = SimpleNamespace(x=0.5+x+offset, y=0.5+y, z=0.0)
    return points


class StopCueTests(unittest.TestCase):
    def test_fist_requires_confirmation_and_latches(self):
        detector = StopGestureDetector()
        for i in range(4):
            self.assertIsNone(detector.update([hand(fist=True)], i*40_000_000))
        detector.update([hand(fist=True)], 160_000_000)
        self.assertEqual(detector.update([hand(fist=True)], 200_000_000), 'Closed fist')
        self.assertEqual(detector.update([], 240_000_000), 'Closed fist')

    def test_inward_fist_and_rotated_views(self):
        inward = hand(fist=True)
        for pip, dip, tip in ((6,7,8), (10,11,12), (14,15,16), (18,19,20)):
            inward[dip].y, inward[dip].z = 0.67, 0.045
            inward[tip].y, inward[tip].z = 0.64, 0.09
        for angle in (0, np.pi/3, np.pi, -np.pi/2):
            c, s = np.cos(angle), np.sin(angle)
            rotation = np.array([[1,0,0], [0,c,-s], [0,s,c]])
            points = (np.array([(p.x,p.y,p.z) for p in inward]) - [0.5,0.5,0]) @ rotation.T
            points += [0.5,0.5,0]
            pose = [SimpleNamespace(x=x,y=y,z=z) for x,y,z in points]
            detector = StopGestureDetector()
            for i in range(13):
                result = detector.update([pose], i*40_000_000)
                if i < 12:
                    self.assertIsNone(result)
            self.assertEqual(result, 'Closed fist')

    def test_one_uncertain_finger_allowed_but_pointing_is_not(self):
        pose = hand(fist=True)
        pose[19].y, pose[19].z = 0.67, 0.045
        pose[20].y, pose[20].z = 0.68, 0.10
        detector = StopGestureDetector()
        for i in range(13):
            result = detector.update([pose], i*40_000_000)
        self.assertEqual(result, 'Closed fist')
        opened = hand()
        for index in (17,18,19,20):
            pose[index] = opened[index]
        detector = StopGestureDetector()
        for i in range(20):
            self.assertIsNone(detector.update([pose], i*40_000_000))

    def test_rotated_open_hand_and_collapsed_finger_do_not_stop(self):
        opened = hand()
        for p in opened:
            p.y, p.z = 0.5, p.y-0.5  # Open fingers pointing along camera depth.
        detector = StopGestureDetector()
        for i in range(20):
            self.assertIsNone(detector.update([opened], i*40_000_000))
        pose = hand(fist=True)
        pose[19] = pose[20] = pose[18]
        detector = StopGestureDetector()
        for i in range(20):
            self.assertIsNone(detector.update([pose], i*40_000_000))

    def test_brief_fist_missing_hand_and_ambiguous_hands_do_not_stop(self):
        detector = StopGestureDetector()
        for i in range(40):
            hands = [hand(fist=True)] if i % 5 < 3 else []
            self.assertIsNone(detector.update(hands, i*40_000_000))
        for i in range(40,60):
            self.assertIsNone(detector.update([hand(fist=True), hand()], i*40_000_000))

    def test_motion_cutoff_requires_prior_motion_and_confidence(self):
        for confidence in (None, 0.5, 0.9):
            detector = StopGestureDetector()
            reason = None
            for i in range(35):
                t = i*0.04
                reason = detector.update([hand(x=0.2*min(t,0.6))], round(t*1e9), confidence)
            self.assertEqual(reason, 'Rapid motion cutoff' if confidence == 0.9 else None)
        detector = StopGestureDetector()
        for i in range(40):
            self.assertIsNone(detector.update([hand()], i*40_000_000, 0.99))

    def test_regular_turnarounds_and_slow_deceleration_do_not_stop(self):
        detector = StopGestureDetector()
        for i in range(300):
            t = i*0.04
            self.assertIsNone(detector.update([hand(x=0.12*np.sin(2*np.pi*t/1.5))], round(t*1e9), 0.9))
        detector = StopGestureDetector()
        x = 0
        for i in range(100):
            t = i*0.04
            x += 0.04*max(0, 0.25*(1-t/2))
            self.assertIsNone(detector.update([hand(x=x)], round(t*1e9), 0.9))

    def test_gap_invalidates_motion_confirmation(self):
        detector = StopGestureDetector()
        for i in range(15):
            detector.update([hand(x=i*0.008)], i*40_000_000, 0.9)
        detector.update([], 600_000_000, 0.9)
        for i in range(16,45):
            self.assertIsNone(detector.update([hand(x=0.12)], i*40_000_000, 0.9))

    def test_pause_fades_freezes_playhead_and_only_explicitly_resumes(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'track.mp3'
            path.touch()
            player = MusicPlayer(path, 80)
            player.stream = Mock()
            player.stream.render.side_effect = lambda speed: np.ones((240,2), dtype=np.float32)
            player.set_tempo(None)
            for _ in range(4):
                player._render_output()
            self.assertEqual(player.output_gain, 1.0)
            player.set_intentional_stop()
            fade = np.concatenate([player._render_output() for _ in range(4)])
            self.assertTrue(np.all(np.diff(fade[:,0]) <= 0))
            self.assertEqual(fade[-1,0], 0)
            calls = player.stream.render.call_count
            player.set_tempo(100)
            self.assertTrue(player.paused)
            self.assertFalse(player._render_output().any())
            self.assertEqual(player.stream.render.call_count, calls)
            player.set_intentional_stop(False)
            self.assertFalse(player.paused)
            self.assertGreater(player._render_output().max(), 0)


if __name__ == '__main__':
    unittest.main()
