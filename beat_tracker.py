"""
Beat detection module using spline interpolation and directional patterns.

This module provides the BeatTracker class which detects beats from hand movement
by analyzing velocity zero-crossings along specific directional axes defined by a
"vibe" pattern. The system follows a directional goal and detects when hand motion
reaches peaks/troughs in that direction.

Directional Goals (vibe):
    - VIBE_UP (2): Look for upward movement peaks
    - VIBE_DOWN (-2): Look for downward movement troughs
    - VIBE_LEFT (-1): Look for leftward movement peaks
    - VIBE_RIGHT (1): Look for rightward movement troughs
"""

from __future__ import annotations

from collections import deque

import numpy as np
from scipy.interpolate import Akima1DInterpolator

from vibe import VIBE_DOWN, VIBE_LEFT, VIBE_RIGHT, VIBE_TWOFOUR,VIBE_FOURFOUR, VIBE_UP, Vibe, mirror_vibe_pattern


class BeatTracker:
    """
    Detects beats from hand movement using spline interpolation and directional goals.

    The tracker uses a circular buffer to collect recent hand position samples and
    applies Akima spline interpolation to find smooth trajectories. Beats are detected
    at velocity zero-crossings that match the current directional goal.
    """
    def __init__(self, size=20, vibe_pattern=VIBE_TWOFOUR, max_beats=5, decay_factor=0.8, miss_threshold=1.2,mirror_pattern=False):
        """Initialize BeatTracker with circular buffer and directional pattern.

        Args:
            size: Number of samples in circular buffer (needs >=10 for analysis)
            vibe_pattern: Directional pattern (VIBE_TWOFOUR, VIBE_THREEFOUR, VIBE_FOURFOUR)
            max_beats: Maximum beats to store for tempo calculation
            decay_factor: Weight for tempo smoothing (0-1, lower = more smoothing)
            miss_threshold: Factor for missed beat detection (1.2 = 120% of expected)
            mirror_pattern: Mirror pattern for alternating directions
        """
        self.size = size
        self.data = np.zeros((size, 3))
        self.ptr = 0
        self.is_full = False

        self.vibe = Vibe(vibe_pattern) if not mirror_pattern else Vibe(mirror_vibe_pattern(vibe_pattern))
        self.current_goal = self.vibe.next()
        self.last_beat_ts = 0
        self.beat_timestamps = deque(maxlen=max_beats)
        self.decay_factor = float(decay_factor)
        self.miss_threshold = float(miss_threshold)
        self.pending_missed_confirmation = False
        self.pending_expected_interval_ns = None

    def add_sample(self, x, y, ts):
        """Add hand position sample to circular buffer.

        Args:
            x: Normalized X position (0-1)
            y: Normalized Y position (0-1)
            ts: Timestamp in nanoseconds
        """
        self.data[self.ptr] = [x, y, ts]
        self.ptr = (self.ptr + 1) % self.size
        if self.ptr == 0:
            self.is_full = True

    def get_linear_view(self):
        """Get circular buffer data as linear array sorted by time."""
        if not self.is_full:
            return self.data[:self.ptr]
        return np.roll(self.data, -self.ptr, axis=0)

    def get_recent_beats(self):
        return list(self.beat_timestamps)

    def get_current_goal_direction(self):
        direction_map = {
            VIBE_DOWN: "DOWN",
            VIBE_UP: "UP",
            VIBE_LEFT: "LEFT",
            VIBE_RIGHT: "RIGHT",
        }
        return direction_map.get(self.current_goal, "UNKNOWN")

    def get_anticipation_message(self):
        direction = self.get_current_goal_direction()
        return f"Anticipation: what direction is the beat looking for? {direction}"

    def get_smoothed_bpm(self):
        """Calculate smoothed tempo (BPM) using exponential moving average.

        Returns:
            float: Smoothed BPM or None if insufficient data
        """
        if len(self.beat_timestamps) < 2:
            return None

        beat_times = np.array(self.beat_timestamps, dtype=float)
        intervals_ns = np.diff(beat_times)
        valid_intervals_ns = intervals_ns[intervals_ns > 0]
        if len(valid_intervals_ns) == 0:
            return None

        bpm_series = 60 * 1e9 / valid_intervals_ns

        n = len(bpm_series)
        exponents = np.arange(n - 1, -1, -1, dtype=float)
        weights = np.power(self.decay_factor, exponents)
        weights_sum = np.sum(weights)
        if weights_sum <= 0:
            return float(np.mean(bpm_series))

        return float(np.sum(bpm_series * weights) / weights_sum)

    def get_expected_interval_ns(self):
        """Calculate expected time between beats using exponential smoothing.

        Returns:
            float: Expected interval in nanoseconds or None
        """
        if len(self.beat_timestamps) < 2:
            return None

        beat_times = np.array(self.beat_timestamps, dtype=float)
        intervals_ns = np.diff(beat_times)
        valid_intervals_ns = intervals_ns[intervals_ns > 0]
        if len(valid_intervals_ns) == 0:
            return None

        n = len(valid_intervals_ns)
        exponents = np.arange(n - 1, -1, -1, dtype=float)
        weights = np.power(self.decay_factor, exponents)
        weights_sum = np.sum(weights)
        if weights_sum <= 0:
            return float(np.mean(valid_intervals_ns))

        return float(np.sum(valid_intervals_ns * weights) / weights_sum)

    def apply_missed_beat_backup(self, current_ts):
        """Check for missed beats and advance direction if needed.

        If elapsed time exceeds miss_threshold * expected_interval, assumes missed beat.

        Returns:
            bool: True if missed beat detected and direction advanced
        """
        if self.pending_missed_confirmation:
            return False
        if self.last_beat_ts <= 0:
            return False

        expected_interval_ns = self.get_expected_interval_ns()
        if expected_interval_ns is None:
            return False

        elapsed_ns = float(current_ts - self.last_beat_ts)
        if elapsed_ns > self.miss_threshold * expected_interval_ns:
            self.current_goal = self.vibe.next()
            self.pending_missed_confirmation = True
            self.pending_expected_interval_ns = expected_interval_ns
            self.last_beat_ts = float(current_ts)
            return True

        return False

    def analyze_vibe_beat(self):
        """Analyze hand movement to detect a beat based on current directional goal.

        Fits Akima spline to position data, finds velocity zero-crossings,
        and verifies acceleration matches expected direction.

        Returns:
            tuple: (beat_timestamp_ns, linear_view, counted_for_tempo)
        """
        view = self.get_linear_view()
        if len(view) < 10:
            return None, view, False

        ts = view[:, 2]
        ix = Akima1DInterpolator(ts, view[:, 0])
        iy = Akima1DInterpolator(ts, view[:, 1])

        target_spline = iy if abs(self.current_goal) == 2 else ix

        dy = target_spline.derivative()
        roots = dy.roots()

        valid_roots = roots[(roots > ts[0]) & (roots <= ts[-1])]

        if len(valid_roots) > 0:
            potential_ts = valid_roots[-1]
            accel = target_spline.derivative(2)(potential_ts)
            is_correct_extreme = (accel > 0) if self.current_goal < 0 else (accel < 0)

            if is_correct_extreme and (potential_ts > self.last_beat_ts + 0.2 * 1e9):
                self.last_beat_ts = potential_ts
                actual_beat_ts = potential_ts

                should_count_for_tempo = True
                if self.pending_missed_confirmation:
                    should_count_for_tempo = False
                    if len(self.beat_timestamps) > 0 and self.pending_expected_interval_ns is not None:
                        interval_ns = actual_beat_ts - self.beat_timestamps[-1]
                        if interval_ns >= self.miss_threshold * self.pending_expected_interval_ns:
                            should_count_for_tempo = True

                    self.pending_missed_confirmation = False
                    self.pending_expected_interval_ns = None

                if should_count_for_tempo:
                    self.beat_timestamps.append(actual_beat_ts)

                self.current_goal = self.vibe.next()
                return actual_beat_ts, view, should_count_for_tempo

        return None, view, False
