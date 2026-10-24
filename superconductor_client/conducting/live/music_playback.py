"""Continuous Bungee Basic playback; FFmpeg decodes and ffplay outputs PCM."""
from __future__ import annotations

import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
import time

import numpy as np
from .bungee_audio import BungeeStream, INPUT_FRAMES, SAMPLE_RATE

TEMPO_SMOOTHING_SECONDS = 2.5
TEMPO_DEADBAND_BPM = 2.0
MAX_TEMPO_CHANGE_BPM_PER_SECOND = 4.0

def positive_bpm(value):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("Music BPM must be finite and positive")
    return value


def estimate_music_bpm(path):
    try:
        import librosa
    except ImportError as exc:
        raise RuntimeError("Install librosa, or supply the track's known --music-bpm") from exc
    # Bound startup work and memory; a fixed source tempo assumes steady music.
    audio, sample_rate = librosa.load(str(path), sr=22050, mono=True, duration=90)
    tempo, beats = librosa.beat.beat_track(y=audio, sr=sample_rate)
    if len(beats) < 4:
        raise RuntimeError("Cannot estimate music tempo reliably; supply --music-bpm")
    return positive_bpm(tempo.item())


class MusicPlayer:
    def __init__(self, path, source_bpm):
        self.path = Path(path).resolve()
        if not self.path.is_file():
            raise FileNotFoundError(f'Music file not found: {self.path}')
        self.source_bpm = positive_bpm(source_bpm)
        self.speed = self.target_speed = 1.0
        self.paused = True
        self.intentional_stop = False
        self.output_gain = 0.0
        self.lock = threading.Lock()
        self.stopped = threading.Event()
        self.started = threading.Event()
        self.worker = self.process = self.stream = self.log = None
        self.error = None

    def __enter__(self):
        if not shutil.which('ffmpeg') or not shutil.which('ffplay'):
            raise RuntimeError('Install FFmpeg (including ffplay) for decoding and audio output')
        try:
            decoded = subprocess.run([
                'ffmpeg', '-nostdin', '-v', 'error', '-i', str(self.path),
                '-vn', '-f', 'f32le', '-acodec', 'pcm_f32le', '-ar', str(SAMPLE_RATE),
                '-ac', '2', 'pipe:1',
            ], capture_output=True, timeout=120)
            if decoded.returncode:
                raise RuntimeError('Music decode failed: ' + decoded.stderr.decode(errors='replace')[-1000:])
            samples = np.frombuffer(decoded.stdout, dtype='<f4').reshape(-1, 2)
            self.stream = BungeeStream(samples)
            environment = os.environ.copy()
            if '/mnt/wslg/' in environment.get('PULSE_SERVER', ''):
                environment['SDL_AUDIODRIVER'] = 'pulseaudio'
            self.log = tempfile.TemporaryFile()
            self.process = subprocess.Popen([
                'ffplay', '-nodisp', '-autoexit', '-loglevel', 'error',
                '-f', 'f32le', '-ar', str(SAMPLE_RATE), '-ac', '2',
                '-probesize', '32', '-analyzeduration', '0', '-i', 'pipe:0',
            ], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                stderr=self.log, env=environment, bufsize=0)
            self.worker = threading.Thread(target=self._play, name='bungee-audio', daemon=True)
            self.worker.start()
            return self
        except BaseException:
            self.close()
            raise

    def set_tempo(self, bpm):
        self.check_health()
        target = None if bpm is None else positive_bpm(bpm) / self.source_bpm
        if target is not None and not 0.05 <= target <= 20:
            raise ValueError('Tempo ratio must be between 0.05 and 20')
        with self.lock:
            if self.intentional_stop:
                return
            # Missing tracking freezes the current speed, never pauses playback.
            if target is None:
                self.target_speed = self.speed
            elif abs(target - self.target_speed) * self.source_bpm >= TEMPO_DEADBAND_BPM:
                self.target_speed = target
            self.paused = False
        self.started.set()

    def set_intentional_stop(self, stopped=True):
        self.check_health()
        with self.lock:
            self.intentional_stop = stopped
            self.paused = stopped
            self.target_speed = self.speed
        self.started.set()

    def _render_output(self):
        with self.lock:
            stopped = self.intentional_stop
        if stopped and self.output_gain == 0.0:
            return np.zeros((INPUT_FRAMES, 2), dtype=np.float32)
        if stopped:
            speed = self.speed
        else:
            speed = self._advance_speed(INPUT_FRAMES / (SAMPLE_RATE * self.speed))
        output = self.stream.render(speed)
        # Twenty milliseconds to/from silence avoids a discontinuity at the cutoff.
        direction = -1.0 if stopped else 1.0
        gains = np.clip(self.output_gain + direction * np.arange(1, len(output)+1) / (0.02*SAMPLE_RATE), 0, 1)
        self.output_gain = float(gains[-1])
        return np.asarray(output * gains[:, None], dtype=np.float32)

    def _advance_speed(self, seconds):
        """Advance by rendered audio time, independently of camera frame rate."""
        with self.lock:
            alpha = -math.expm1(-seconds / TEMPO_SMOOTHING_SECONDS)
            step = alpha * (self.target_speed - self.speed)
            limit = MAX_TEMPO_CHANGE_BPM_PER_SECOND * seconds / self.source_bpm
            self.speed += max(-limit, min(limit, step))
            return self.speed

    def check_health(self):
        if self.error is not None:
            raise RuntimeError(f'Bungee playback failed: {self.error}') from self.error
        if self.process is not None and self.process.poll() is not None and not self.stopped.is_set():
            self.log.seek(0)
            message = self.log.read().decode(errors='replace')[-1000:]
            raise RuntimeError('Audio output exited: ' + message)

    def _play(self):
        try:
            self.started.wait()
            # Pace rendering by the audio clock as well as pipe backpressure.
            # This bounds queued audio, so camera tempo changes stay responsive.
            deadline = time.monotonic()
            while not self.stopped.is_set():
                if self.stopped.wait(max(0.0, deadline - time.monotonic() - 0.10)):
                    break
                output = self._render_output()
                pending = memoryview(output).cast('B')
                while pending and not self.stopped.is_set():
                    count = self.process.stdin.write(pending)
                    if not count:
                        raise RuntimeError('Audio output pipe closed')
                    pending = pending[count:]
                deadline = max(deadline, time.monotonic() - 0.10) + len(output) / SAMPLE_RATE
        except Exception as exc:
            if not self.stopped.is_set():
                self.error = exc
        finally:
            self.stream.close()

    def close(self):
        self.stopped.set()
        self.started.set()
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        if self.worker is not None:
            self.worker.join(timeout=3)
            if self.worker.is_alive():
                raise RuntimeError('Bungee rendering thread did not stop')
        elif self.stream is not None:
            self.stream.close()
        if self.process is not None and self.process.stdin is not None:
            self.process.stdin.close()
        if self.log is not None:
            self.log.close()

    def __exit__(self, *exc):
        self.close()
