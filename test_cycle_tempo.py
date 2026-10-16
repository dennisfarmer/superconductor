import contextlib
import io
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import unittest
import numpy as np
from cycle_tempo import CycleTempoEstimator


def motion(bpm, seconds=24, switch_to=None):
    phase = 0.0
    for i in range(int(seconds * 30)):
        t = i / 30
        tempo = switch_to if switch_to is not None and t >= 12 else bpm
        phase += 2 * np.pi * tempo / (240 * 30)
        x = 0.5 + 0.2 * np.sin(phase) + 0.06 * np.sin(2 * phase)
        y = 0.5 + 0.2 * np.cos(phase) + 0.04 * np.cos(3 * phase)
        yield int(round(t * 1e9)), x, y


class CycleTempoTests(unittest.TestCase):
    def test_tempo_without_target_and_warmup(self):
        for target in (100, 132, 168, 210):
            estimator = CycleTempoEstimator()
            outputs = []
            for ts, x, y in motion(target):
                value = estimator.update(x, y, ts)
                if ts < 3e9:
                    self.assertIsNone(value)
                elif value is not None:
                    outputs.append(value)
            self.assertGreater(len(outputs), 300)
            self.assertLess(abs(np.median(outputs) - target), 1.5)

    def test_noise_and_brief_dropout(self):
        rng = np.random.default_rng(7)
        estimator = CycleTempoEstimator()
        outputs = []
        for ts, x, y in motion(168):
            if 10 <= ts / 1e9 < 10.3:
                x = y = None
            else:
                x += rng.normal(0, 0.002)
                y += rng.normal(0, 0.002)
            value = estimator.update(x, y, ts)
            if ts > 12e9 and value is not None:
                outputs.append(value)
        self.assertGreater(len(outputs), 200)
        self.assertLess(np.percentile(np.abs(np.array(outputs) - 168), 90), 2)

    def test_loss_and_stopped_motion_clear_estimate(self):
        for missing in (False, True):
            estimator = CycleTempoEstimator()
            for ts, x, y in motion(132, seconds=12):
                estimator.update(x, y, ts)
            self.assertIsNotNone(estimator.bpm)
            for i in range(60):
                estimator.update(None if missing else 0.5,
                                 None if missing else 0.5,
                                 int(round((12 + i / 30) * 1e9)))
            self.assertIsNone(estimator.bpm)

    def test_accepts_real_tempo_change_after_window(self):
        estimator = CycleTempoEstimator()
        last = None
        for ts, x, y in motion(132, seconds=26, switch_to=168):
            last = estimator.update(x, y, ts)
        self.assertIsNotNone(last)
        self.assertLess(abs(last - 168), 2)

    def test_duplicate_timestamp_is_rejected(self):
        estimator = CycleTempoEstimator()
        estimator.update(0.5, 0.5, 0)
        with self.assertRaises(ValueError):
            estimator.update(0.5, 0.5, 0)


class PipelineTempoTests(unittest.TestCase):
    def test_default_pipeline_uses_cycle_and_exports_metadata(self):
        import cv2
        from tempo_pipeline import process_video_collect_kinematics
        from video_output import save_frame_metadata

        class Capture:
            def isOpened(self):
                return True

            def get(self, prop):
                return 30 if prop == cv2.CAP_PROP_FPS else 32

        stream = []
        for ts, x, y in motion(132, seconds=14):
            hand = [SimpleNamespace(x=x, y=y)] * 21
            stream.append((True, [hand], ts, None))
        # A sustained loss must clear the displayed tempo, even without beats.
        stream.extend((True, [], int(round((14 + i / 30) * 1e9)), None)
                      for i in range(30))
        stream.append((False, [], 0, None))
        with patch("tempo_pipeline.frame", side_effect=stream), contextlib.redirect_stdout(io.StringIO()):
            records, beats, *_ = process_video_collect_kinematics(Capture(), None)
        self.assertTrue(all(row["tempo_bpm"] is None for row in records[:90]))
        self.assertAlmostEqual(records[410]["tempo_bpm"], 132, delta=2)
        self.assertEqual(records[410]["tempo_method"], "cycle")
        self.assertGreater(records[410]["tempo_correlation"], 0.5)
        self.assertIsNone(records[-1]["tempo_bpm"])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_frame_metadata(records, [], root / "frames.json", root / "frames.csv")
            payload = json.loads((root / "frames.json").read_text())
            self.assertEqual(payload["frame_count"], len(records))
            self.assertIn("tempo_correlation", (root / "frames.csv").read_text().splitlines()[0])


if __name__ == "__main__":
    unittest.main()
