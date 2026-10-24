"""Causal tempo estimate from repeated four-beat conducting motion.

Uses only samples received so far. A short rolling window estimates the complete gesture's repetition period. No reference
BPM is supplied to the estimator. Beat markers remain a separate task.
"""
from collections import deque
import math

import numpy as np
from scipy.signal import detrend, find_peaks


class CycleTempoEstimator:
    def __init__(self, window_seconds=3.0, update_seconds=0.25,
                 sample_rate=30.0, min_bpm=60.0, max_bpm=240.0):
        if not all(math.isfinite(v) for v in
                   (window_seconds, update_seconds, sample_rate, min_bpm, max_bpm)):
            raise ValueError("Settings must be finite")
        if window_seconds < 2 or update_seconds <= 0 or sample_rate < 10:
            raise ValueError("Use a window >=2 s, positive update interval, and >=10 Hz")
        if not 0 < min_bpm < max_bpm:
            raise ValueError("Expected 0 < min_bpm < max_bpm")
        self.window_seconds = float(window_seconds)
        self.update_seconds = float(update_seconds)
        self.sample_rate = float(sample_rate)
        self.min_bpm = float(min_bpm)
        self.max_bpm = float(max_bpm)
        self.samples = deque()
        self.started = None
        self.last_timestamp = None
        self.last_update = None
        self.last_seen = None
        self.bpm = None
        self.confidence = None

    def update(self, x, y, timestamp_ns):
        t = int(timestamp_ns) * 1e-9
        if self.last_timestamp is not None and t <= self.last_timestamp:
            raise ValueError("Frame timestamps must increase")
        self.last_timestamp = t
        if self.started is None:
            self.started = t
        valid = x is not None and y is not None and np.isfinite([x, y]).all()
        self.samples.append((t, float(x) if valid else np.nan,
                             float(y) if valid else np.nan))
        if valid:
            self.last_seen = t
        cutoff = t - self.window_seconds
        # Keep a sample preceding the window for interpolation at its boundary.
        while len(self.samples) > 1 and self.samples[1][0] < cutoff:
            self.samples.popleft()
        if self.last_seen is None or t - self.last_seen > 0.5:
            self.bpm = self.confidence = None
            return None
        if t - self.started < self.window_seconds:
            return None
        if self.last_update is not None and t - self.last_update < self.update_seconds:
            return self.bpm
        self.last_update = t
        self.bpm, self.confidence = self._estimate(t)
        return self.bpm

    def _estimate(self, now):
        samples = np.asarray(self.samples)
        visible = np.isfinite(samples[:, 1:]).all(axis=1)
        inside = samples[:, 0] >= now - self.window_seconds
        if not inside.any() or np.mean(visible[inside]) < 0.7:
            return None, None
        good = samples[visible]
        gaps = np.diff(np.r_[now - self.window_seconds,
                            good[:, 0].clip(now - self.window_seconds, now), now])
        if len(good) < 10 or np.max(gaps) > 0.5:
            return None, None
        recent = good[good[:, 0] >= now - 1.0, 1:]
        if len(recent) < 3 or np.max(np.std(recent, axis=0)) < 0.005:
            return None, None

        count = int(round(self.window_seconds * self.sample_rate))
        grid = now - self.window_seconds + np.arange(count) / self.sample_rate
        signal = np.column_stack([
            np.interp(grid, good[:, 0], good[:, axis]) for axis in (1, 2)
        ])
        signal = detrend(signal, axis=0)
        spread = signal.std(axis=0)
        if np.max(spread) < 0.005:
            return None, None
        signal /= np.maximum(spread, 0.005)
        minimum_lag = max(2, math.ceil(240 * self.sample_rate / self.max_bpm))
        maximum_lag = min(count - max(6, math.ceil(0.4 * self.sample_rate)),
                          math.floor(240 * self.sample_rate / self.min_bpm))
        if maximum_lag <= minimum_lag:
            return None, None
        lags = np.arange(minimum_lag - 1, maximum_lag + 2)
        correlation = []
        for lag in lags:
            left, right = signal[:-lag], signal[lag:]
            denominator = np.sqrt(np.sum(left * left) * np.sum(right * right))
            correlation.append(np.sum(left * right) / denominator if denominator else 0)
        correlation = np.asarray(correlation)
        peaks, _ = find_peaks(correlation, prominence=0.1)
        peaks = peaks[correlation[peaks] > 0.5]
        if not len(peaks):
            return None, None
        near_best = peaks[correlation[peaks] >= np.max(correlation[peaks]) - 0.1]
        index = int(near_best[0])
        lag = float(lags[index])
        curvature = correlation[index - 1] - 2 * correlation[index] + correlation[index + 1]
        if curvature:
            lag += 0.5 * (correlation[index - 1] - correlation[index + 1]) / curvature
        bpm = 240 * self.sample_rate / lag
        if not self.min_bpm <= bpm <= self.max_bpm:
            return None, None
        return float(bpm), float(correlation[index])
