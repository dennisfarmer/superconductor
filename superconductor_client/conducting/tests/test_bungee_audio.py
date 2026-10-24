"""Native engine checks; no camera or audible output is used."""
import os
from pathlib import Path
import shutil
import tempfile
import time
import unittest
from unittest.mock import patch
import wave

import numpy as np

from live.bungee_audio import BungeeStream, SAMPLE_RATE
from live.music_playback import MusicPlayer

BUILT = any((Path(__file__).resolve().parents[1] / '.build/native' / name).exists()
            for name in ('libconductor_bungee.so', 'libconductor_bungee.dylib'))


@unittest.skipUnless(BUILT, 'Run bash build_bungee.sh first')
class NativeAudioTests(unittest.TestCase):
    def setUp(self):
        t = np.arange(SAMPLE_RATE) / SAMPLE_RATE
        self.samples = np.column_stack([0.1*np.sin(2*np.pi*440*t),
                                        0.1*np.sin(2*np.pi*660*t)])

    def test_pitch_and_duration_at_three_speeds(self):
        for speed in (0.75, 1.0, 1.25):
            stream = BungeeStream(self.samples)
            try:
                rendered = np.concatenate([stream.render(speed).copy() for _ in range(400)])
                self.assertLessEqual(abs(len(rendered) - 400*256/speed), 1)
                self.assertTrue(np.isfinite(rendered).all())
                for channel, expected in enumerate((440, 660)):
                    window = rendered[24000:72000, channel] * np.hanning(48000)
                    frequency = np.argmax(abs(np.fft.rfft(window)))
                    self.assertAlmostEqual(frequency, expected, delta=2)
            finally:
                stream.close()

    def test_changing_rate_retains_stream_and_loops(self):
        stream = BungeeStream(self.samples[:4800])
        handle = stream.handle
        try:
            rates = np.r_[np.linspace(1, 0.7, 150), np.linspace(0.7, 1.3, 150)]
            output = np.concatenate([stream.render(float(speed)).copy() for speed in rates])
            self.assertEqual(stream.handle, handle)
            self.assertEqual(stream.position, 300*256 % 4800)
            self.assertTrue(np.isfinite(output).all())
            self.assertGreater(np.std(output), 0.01)
            self.assertLess(np.max(np.abs(output)), 0.5)
        finally:
            stream.close()
        stream.close()  # Safe repeated cleanup.

    def test_invalid_speed_and_closed_engine(self):
        stream = BungeeStream(self.samples)
        try:
            for speed in (0, -1, float('nan'), 21):
                with self.assertRaises(ValueError):
                    stream.render(speed)
        finally:
            stream.close()
        with self.assertRaises(RuntimeError):
            stream.render(1)

    @unittest.skipUnless(shutil.which('ffplay') and shutil.which('ffmpeg'), 'FFmpeg required')
    def test_real_decode_render_output_and_shutdown_without_sound(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'silence.wav'
            with wave.open(str(path), 'wb') as audio:
                audio.setparams((2, 2, SAMPLE_RATE, 0, 'NONE', 'not compressed'))
                audio.writeframes(bytes(SAMPLE_RATE*4))
            # SDL dummy output consumes audio without touching speakers.
            with patch.dict(os.environ, {'PULSE_SERVER': '', 'SDL_AUDIODRIVER': 'dummy'}):
                with MusicPlayer(path, 80) as player:
                    player.set_tempo(100)
                    time.sleep(1.2)
                    player.check_health()
                    self.assertGreater(player.speed, 1.01)
                    self.assertLess(player.speed, 1.1)
                    player.set_tempo(None)
                    held = player.speed
                    time.sleep(0.15)
                    self.assertAlmostEqual(player.speed, held, places=6)
                    self.assertFalse(player.paused)
                    player.set_intentional_stop()
                    time.sleep(0.3)
                    position = player.stream.position
                    player.set_tempo(110)
                    time.sleep(0.15)
                    self.assertEqual(player.stream.position, position)
                    self.assertTrue(player.paused)
                    player.set_intentional_stop(False)
                    time.sleep(0.15)
                    player.check_health()
                    self.assertNotEqual(player.stream.position, position)
                self.assertFalse(player.worker.is_alive())
                self.assertIsNotNone(player.process.poll())
                self.assertIsNone(player.error)
