"""Adaptive playback speed for the MRT2 server.

MRT2 picks its own tempo, so the server measures it and time-stretches the
generated audio (changing the speed, not the pitch) toward a target tempo:

    model audio --+--> TempoEstimator ("model bpm", the tempo MRT2 is playing)
                  +--> Stretcher(speed) --> client      heard bpm = model bpm * speed

The target tempo comes either from the test page (an exact bpm) or from the
conductor's beats, smoothed by a small Kalman filter (`BeatTracker`). The speed
is never set directly: every block it moves a fraction of the way toward
`target / model bpm`, so tempo changes are gradual and audio is never skipped.

Only numpy, so it also runs in the Lighthouse venv.
"""

import threading
import time

import numpy as np

SAMPLE_RATE = 48000

MIN_SPEED, MAX_SPEED = 0.75, 1.3  # speeding up needs rtf > speed
SPEED_TIME_CONSTANT = 1.5  # seconds for the speed to move ~63% of the way to its goal
BEAT_TIMEOUT = 3.0  # seconds without beats before the conductor tempo is dropped


class TempoEstimator:
    """Tempo of the last few seconds of audio: onset strength + autocorrelation.

    1. mono, downsampled 4x to 12kHz
    2. onset envelope = spectral flux (how much louder each frequency got
       since the previous 5ms frame), summed over frequencies
    3. autocorrelation of the envelope: the strongest repeat period between
       60 and 200 bpm, mildly preferring tempos near 110 bpm
    The result is the median of the last few estimates, so one bad window
    doesn't move it.
    """

    DEC = 4
    RATE = SAMPLE_RATE // DEC
    FRAME, HOP = 512, 64
    WINDOW_S, MIN_S = 8.0, 4.0  # analysed audio; audio needed for a first estimate
    EVERY_S = 1.0  # analyse once per second of audio
    PRIOR_BPM, PRIOR_OCTAVES = 110.0, 1.0

    def __init__(self):
        self._ring = np.zeros(int(self.WINDOW_S * self.RATE), dtype=np.float32)
        self._filled = 0
        self._since = 0
        self._recent = []
        self.bpm = None  # median of recent estimates

    def push(self, samples):
        """`samples`: (n, 2) float32 at 48kHz."""
        n = len(samples) // self.DEC * self.DEC
        if n == 0:
            return
        mono = samples[:n].mean(axis=1).reshape(-1, self.DEC).mean(axis=1)
        mono = mono[-len(self._ring):]
        self._ring = np.roll(self._ring, -len(mono))
        self._ring[-len(mono):] = mono
        self._filled = min(len(self._ring), self._filled + len(mono))
        self._since += len(mono)
        if self._since >= self.EVERY_S * self.RATE and self._filled >= self.MIN_S * self.RATE:
            self._since = 0
            bpm = self.estimate(self._ring[-self._filled:])
            if bpm is not None:
                self._recent = (self._recent + [bpm])[-5:]
                self.bpm = float(np.median(self._recent))

    @classmethod
    def estimate(cls, audio):
        frames = np.lib.stride_tricks.sliding_window_view(audio, cls.FRAME)[::cls.HOP]
        spec = np.log1p(100 * np.abs(np.fft.rfft(frames * np.hanning(cls.FRAME), axis=1)))
        flux = np.maximum(np.diff(spec, axis=0), 0).sum(axis=1)
        env_rate = cls.RATE / cls.HOP
        # remove the slowly changing loudness, keep the onsets
        k = int(env_rate / 2)
        flux = np.maximum(flux - np.convolve(flux, np.ones(k) / k, mode="same"), 0)
        if not flux.any():
            return None
        spec_ac = np.fft.rfft(flux, 2 * len(flux))
        ac = np.fft.irfft(spec_ac * np.conj(spec_ac))[:len(flux)]
        lags = np.arange(int(env_rate * 60 / 200), int(env_rate * 60 / 60) + 1)
        bpms = 60 * env_rate / lags
        prior = np.exp(-0.5 * (np.log2(bpms / cls.PRIOR_BPM) / cls.PRIOR_OCTAVES) ** 2)
        score = ac[lags] * prior
        i = int(np.argmax(score))
        lag = float(lags[i])
        if 0 < i < len(score) - 1:  # parabolic interpolation between lags
            a, b, c = score[i - 1], score[i], score[i + 1]
            if a - 2 * b + c < 0:
                lag += 0.5 * (a - c) / (a - 2 * b + c)
        return 60 * env_rate / lag


class BeatTracker:
    """Conductor tempo from beat times: a 1-D Kalman filter on the beat period.

    Each interval between two beats is a noisy measurement of the period. The
    filter trusts a new interval more when it is unsure of the period
    (start-up) and less once it has settled, which smooths hand jitter but
    still follows a gradual tempo change. An interval far from the current
    period (a missed or doubled beat) is ignored, unless two in a row agree,
    which means the conductor really changed tempo.
    """

    MEASUREMENT_VAR = 0.04 ** 2  # human timing jitter, ~40ms
    PROCESS_VAR = 0.015 ** 2  # how much the real period may drift per beat
    OUTLIER = 0.3  # an interval more than 30% off the period is an outlier

    def __init__(self):
        self.reset()

    def reset(self):
        self.period = None
        self.var = None
        self.last_beat = None
        self._outliers = []

    def beat(self, t):
        last, self.last_beat = self.last_beat, t
        if last is None:
            return
        x = t - last
        if not 0.25 <= x <= 2.0:  # outside 30-240 bpm: a pause, start over
            self.period = None
            return
        if self.period is None:
            self.period, self.var = x, self.MEASUREMENT_VAR
            return
        if abs(x - self.period) > self.OUTLIER * self.period:
            self._outliers.append(x)
            if len(self._outliers) >= 2 and abs(self._outliers[-1] - self._outliers[-2]) < self.OUTLIER * x:
                self.period, self.var = x, self.MEASUREMENT_VAR
                self._outliers = []
            return
        self._outliers = []
        self.var += self.PROCESS_VAR  # predict
        gain = self.var / (self.var + self.MEASUREMENT_VAR)  # update
        self.period += gain * (x - self.period)
        self.var *= 1 - gain

    @property
    def bpm(self):
        return None if self.period is None else 60.0 / self.period


class Stretcher:
    """Streaming WSOLA time-stretch: changes the speed, keeps the pitch.

    Cuts the input into overlapping 43ms pieces and lays them out again with a
    fixed spacing in the output, while walking through the input faster or
    slower (`speed`). Each piece is shifted by up to +-11ms so that it lines up
    with the previous one (the most similar waveform), which avoids clicks.
    """

    N, HS, DELTA, DEC = 2048, 1024, 512, 4  # piece, output hop, max shift, search decimation

    def __init__(self):
        self._window = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(self.N) / self.N)).astype(np.float32)[:, None]
        self._in = np.zeros((0, 2), dtype=np.float32)
        self._base = 0  # input sample index of self._in[0]
        self._pos = 0.0  # nominal input position of the next piece
        self._prev = None  # input position of the previous piece
        self._acc = np.zeros((self.N, 2), dtype=np.float32)

    def process(self, samples, speed):
        self._in = np.concatenate([self._in, samples])
        end = self._base + len(self._in)
        out = []
        while True:
            if self._prev is None:
                if self.N > end:
                    break
                start = 0
            else:
                a = int(round(self._pos))
                nat = self._prev + self.HS  # where the previous piece naturally continues
                if max(a + self.DELTA, nat) + self.N > end:
                    break
                start = self._best_start(a - self.DELTA, a + self.DELTA, nat)
            piece = self._in[start - self._base:start - self._base + self.N]
            self._acc += piece * self._window
            out.append(self._acc[:self.HS].copy())
            self._acc = np.concatenate([self._acc[self.HS:], np.zeros((self.HS, 2), np.float32)])
            self._prev = start
            self._pos += self.HS * speed
            keep = min(int(round(self._pos)) - self.DELTA, self._prev + self.HS)
            if keep > self._base:
                self._in = self._in[keep - self._base:]
                self._base = keep
        return np.concatenate(out) if out else np.zeros((0, 2), np.float32)

    def _best_start(self, lo, hi, nat):
        mono = lambda s, n: self._in[s - self._base:s - self._base + n].mean(axis=1)
        template = mono(nat, self.N)

        def similarity(region, tmpl):
            corr = np.correlate(region, tmpl, mode="valid")
            energy = np.convolve(region ** 2, np.ones(len(tmpl)), mode="valid")
            return corr / np.sqrt(energy + 1e-9)

        # coarse search on every DEC-th sample, then refine around the best
        region = mono(lo, hi - lo + self.N)
        coarse = similarity(region[::self.DEC], template[::self.DEC])
        best = lo + int(np.argmax(coarse)) * self.DEC
        flo, fhi = max(lo, best - self.DEC), min(hi, best + self.DEC)
        fine = similarity(mono(flo, fhi - flo + self.N), template)
        return flo + int(np.argmax(fine))


def _fold(ratio):
    """Pick ratio, 2x or 1/2x, whichever is closest to 1 (half/double-time)."""
    return min((ratio * f for f in (0.5, 1.0, 2.0)), key=lambda r: abs(np.log(r)))


class TempoController:
    """Thread-safe: the generator thread calls `process`; the HTTP handlers
    call `beat`, `set_target`, `set_free` and `status`."""

    def __init__(self):
        self._lock = threading.Lock()
        self._beats = BeatTracker()
        self.mode = "free"  # free (speed 1) | target (exact bpm) | conduct (beats)
        self.target_bpm = None
        self.speed = 1.0
        self._last_beat_wall = 0.0
        self.reset()

    def reset(self):
        """New session: new audio, so new tempo estimates."""
        with self._lock:
            self._model = TempoEstimator()  # what MRT2 generates
            self._output = TempoEstimator()  # what the listener hears (a check)
            self._stretcher = Stretcher()
            self.speed = 1.0

    def beat(self, t=None):
        with self._lock:
            self._beats.beat(time.time() if t is None else t)
            self._last_beat_wall = time.time()
            if self._beats.bpm is not None:
                self.mode = "conduct"

    def set_target(self, bpm):
        with self._lock:
            self.mode, self.target_bpm = "target", float(bpm)

    def set_free(self):
        with self._lock:
            self.mode, self.target_bpm = "free", None
            self._beats.reset()

    def _goal_speed(self):
        if self.mode == "conduct":
            if time.time() - self._last_beat_wall > BEAT_TIMEOUT:
                return self.speed  # conductor stopped: hold the current tempo
            target = self._beats.bpm
        elif self.mode == "target":
            target = self.target_bpm
        else:
            return 1.0
        if target is None or self._model.bpm is None:  # still calibrating
            return self.speed
        return float(np.clip(_fold(target / self._model.bpm), MIN_SPEED, MAX_SPEED))

    def process(self, samples):
        """Model audio in, stretched audio out. Called once per generated block."""
        with self._lock:
            self._model.push(samples)
            dt = len(samples) / SAMPLE_RATE
            self.speed += (self._goal_speed() - self.speed) * min(1.0, dt / SPEED_TIME_CONSTANT)
            out = self._stretcher.process(samples, self.speed)
            self._output.push(out)
            return out

    def status(self):
        with self._lock:
            model = self._model.bpm
            return {
                "mode": self.mode,
                "speed": self.speed,
                "model_bpm": model,  # tempo MRT2 is generating
                "heard_bpm": None if model is None else model * self.speed,
                "measured_output_bpm": self._output.bpm,  # independent check of heard_bpm
                "target_bpm": self.target_bpm,
                "conductor_bpm": self._beats.bpm,
            }
