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
from lzma import FILTER_DELTA

import numpy as np
from scipy.interpolate import Akima1DInterpolator
from scipy.signal import butter, filtfilt, find_peaks
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

    def analyze_vibe_beat(self, sample_rate):
        """Analyze hand movement to detect a beat based on current directional goal.

        Pipeline:
            1. Get raw position (x, y) from circular buffer
            2. Compute raw velocity by differentiating position
            3. Apply Butterworth low-pass filter to VELOCITY (removes noise)
            4. Find zero-crossings using LINEAR INTERPOLATION

        Direction handling:
            - VIBE_UP (2):    Y velocity negative→positive (upward peak)
            - VIBE_DOWN (-2): Y velocity positive→negative (downward trough)
            - VIBE_RIGHT (1): X velocity negative→positive (rightward peak)
            - VIBE_LEFT (-1): X velocity positive→negative (leftward trough)

        Linear interpolation formula:
            t_zero = t[i] + (0 - v[i]) * (t[i+1] - t[i]) / (v[i+1] - v[i])

        Returns:
            tuple: (beat_timestamp_ns, linear_view, counted_for_tempo)
        """
        # Step 1: Get raw position data from circular buffer
        view = self.get_linear_view()
        if len(view) < 10:
            return None, view, False

        pos_x = view[:, 0]
        pos_y = view[:, 1]
        timestamps = view[:, 2]

        # Step 2: Compute VELOCITY (derivative of position)
        dt_ns = np.diff(timestamps)
        dt_ns = dt_ns[dt_ns > 0]
        if len(dt_ns) == 0:
            return None, view, False

        velocity_x = np.diff(pos_x) / (dt_ns * 1e-9)
        velocity_y = np.diff(pos_y) / (dt_ns * 1e-9)
        timestamps_vel = timestamps[1:len(velocity_x)+1]

        # Step 3: Apply LOW-PASS FILTER to VELOCITY (not position)
        butter_filter = butter(3, 5, btype='low', fs=sample_rate, output='sos')
        filtered_vel_x = filtfilt(butter_filter, velocity_x)
        filtered_vel_y = filtfilt(butter_filter, velocity_y)

        # Step 4: Find zero-crossings with LINEAR INTERPOLATION
        def find_zero_crossings(velocity, time_arr,direction='x'):
            """Find zero-crossings with linear interpolation for X axis."""
            crossings = []
            for i in range(len(velocity) - 1):
                v_curr, v_next = velocity[i], velocity[i + 1]
                if v_curr * v_next < 0:
                    t_zero = time_arr[i] + (0 - v_curr) * (time_arr[i + 1] - time_arr[i]) / (v_next - v_curr)
                    crossings.append(t_zero)
            return crossings

        

        # Select axis and crossing direction based on current_goal
        if self.current_goal in (VIBE_UP, VIBE_DOWN):
            all_crossings = find_zero_crossings(filtered_vel_y, timestamps_vel)
            valid_roots = []
            for ts in all_crossings:
                idx = np.searchsorted(timestamps_vel, ts)
                if 0 < idx < len(filtered_vel_y):
                    v_before, v_after = filtered_vel_y[idx - 1], filtered_vel_y[idx]
                    if self.current_goal == VIBE_UP and v_before < 0 and v_after > 0:
                        valid_roots.append(ts)
                    elif self.current_goal == VIBE_DOWN and v_before > 0 and v_after < 0:
                        valid_roots.append(ts)
        else:
            all_crossings = find_zero_crossings(filtered_vel_x, timestamps_vel)
            valid_roots = []
            for ts in all_crossings:
                idx = np.searchsorted(timestamps_vel, ts)
                if 0 < idx < len(filtered_vel_x):
                    v_before, v_after = filtered_vel_x[idx - 1], filtered_vel_x[idx]
                    if self.current_goal == VIBE_RIGHT and v_before < 0 and v_after > 0:
                        valid_roots.append(ts)
                    elif self.current_goal == VIBE_LEFT and v_before > 0 and v_after < 0:
                        valid_roots.append(ts)

        # Process valid zero-crossings
        if len(valid_roots) > 0:
            potential_ts = valid_roots[-1]
            if potential_ts > self.last_beat_ts + 0.2 * 1e9:
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
