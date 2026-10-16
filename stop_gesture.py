"""Conservative, causal stop cues. Geometric thresholds are not calibrated probabilities."""
from collections import deque
import numpy as np


class StopGestureDetector:
    def __init__(self):
        self.reason = None
        self._clear()

    def _clear(self):
        self.previous = None
        self.velocities = deque()
        self.fist_since = self.still_since = None
        self.fist_frames = self.still_frames = 0
        self.fist_required_seconds = 0.18
        self.armed_until = None

    def update(self, hands, timestamp_ns, confidence=None, aspect_ratio=1.0):
        if self.reason:
            return self.reason
        t = timestamp_ns * 1e-9
        # Multiple hands, gaps, invalid geometry, or identity jumps cannot confirm a stop.
        if len(hands) != 1 or len(hands[0]) != 21:
            self._clear()
            return None
        points = np.array([(p.x * aspect_ratio, p.y, getattr(p, 'z', 0.0) * aspect_ratio)
                           for p in hands[0]])
        scale = np.linalg.norm(points[9] - points[0])
        if not np.isfinite(points).all() or scale < 0.025:
            self._clear()
            return None
        center = points[[0, 5, 9, 13, 17]].mean(axis=0)
        previous = self.previous
        if previous is not None:
            dt = t - previous[0]
            jump = np.linalg.norm(center - previous[1]) / scale
            if dt <= 0 or dt > 0.12 or jump > 1.0 or not 0.7 < scale / previous[2] < 1.4:
                self._clear()
                previous = None
        self.previous = (t, center, scale)
        curled = extended = uncertain = strong = 0
        palm_axis = (points[9] - points[0]) / scale
        for mcp, pip, dip, tip in ((5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20)):
            a, b = points[mcp] - points[pip], points[tip] - points[pip]
            denominator = np.linalg.norm(a) * np.linalg.norm(b)
            lengths = [np.linalg.norm(points[j] - points[i])
                       for i, j in ((mcp, pip), (pip, dip), (dip, tip))]
            if denominator < scale * scale * 0.01 or min(lengths) < scale * 0.03:
                continue  # Collapsed landmarks cannot vote for a fist.
            cosine = np.dot(a, b) / denominator
            # The endpoint-to-chain ratio measures folding in 3D rather than
            # relying on a hidden tip's radial distance from the wrist.
            reach = np.linalg.norm(points[tip] - points[mcp]) / sum(lengths)
            tip_forward = np.dot(points[tip] - points[mcp], palm_axis)
            pip_forward = np.dot(points[pip] - points[mcp], palm_axis)
            folded_back = tip_forward <= pip_forward + 0.05 * scale
            is_curled = folded_back and (cosine > 0.35 or reach < 0.65)
            is_extended = cosine < -0.5 and reach > 0.85
            curled += is_curled
            strong += is_curled and cosine > 0.35
            extended += is_extended
            uncertain += not is_curled and not is_extended and reach < 0.85
        # Three clear curls can tolerate one uncertain hidden finger, but never
        # a clearly extended finger or a collapsed/invalid landmark chain.
        if extended == 0 and (curled == 4 or (curled == 3 and uncertain == 1)):
            if self.fist_since is None:
                self.fist_since = t
                self.fist_required_seconds = 0.18
            # If any frame depends on inferred hidden fingers, require a longer
            # continuous hold. A single clearer frame must not bypass that delay.
            if strong < 4:
                self.fist_required_seconds = 0.45
            self.fist_frames += 1
            if t - self.fist_since >= self.fist_required_seconds and self.fist_frames >= 4:
                self.reason = 'Closed fist'
                return self.reason
        else:
            self.fist_since = None
            self.fist_frames = 0
        if previous is None or extended < 3:
            self.velocities.clear()
            self.still_since = self.armed_until = None
            self.still_frames = 0
            return None
        velocity = np.linalg.norm(center - previous[1]) / ((t - previous[0]) * scale)
        self.velocities.append((t, velocity))
        while self.velocities and t - self.velocities[0][0] > 0.65:
            self.velocities.popleft()
        # Arming needs coherent conducting plus sustained recent palm movement.
        fast = [sample_t for sample_t, v in self.velocities if v >= 1.5]
        if confidence is not None and confidence >= 0.75 and len(fast) >= 4 and fast[-1] - fast[0] >= 0.15:
            self.armed_until = fast[-1] + 0.65
        if self.armed_until is not None and t <= self.armed_until and velocity < 0.20:
            if self.still_since is None:
                self.still_since = t
            self.still_frames += 1
            if t - self.still_since >= 0.32 and self.still_frames >= 6:
                self.reason = 'Rapid motion cutoff'
        else:
            self.still_since = None
            self.still_frames = 0
        return self.reason
