from __future__ import annotations

import numpy as np
from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import KalmanFilter as FilterPyKalmanFilter

from tempo_constants import MIN_DT_SECONDS, NS_TO_SECONDS


class ConstantVelocityKalman:
    """N-dimensional constant-velocity Kalman filter backed by FilterPy.

    State: [p0, v0, p1, v1, ...]
    Measurement: [p0, p1, ...]
    """

    def __init__(self, dimensions: int, pos_var=1e-4, vel_var=1e-3, meas_var=1e-3):
        if dimensions < 1:
            raise ValueError("dimensions must be >= 1")

        self.dimensions = dimensions
        self.pos_var = float(pos_var)
        self.vel_var = float(vel_var)
        self.meas_var = float(meas_var)
        self.last_t_ns = None
        self.kf = FilterPyKalmanFilter(dim_x=2 * dimensions, dim_z=dimensions)

        self.kf.P *= 1e-1
        self.kf.R = np.eye(dimensions, dtype=float) * self.meas_var
        self.kf.H = np.zeros((dimensions, 2 * dimensions), dtype=float)
        for axis in range(dimensions):
            self.kf.H[axis, 2 * axis] = 1.0

    def _set_dynamic_model(self, dt_seconds: float):
        dt = max(float(dt_seconds), MIN_DT_SECONDS)

        self.kf.F = np.eye(2 * self.dimensions, dtype=float)
        for axis in range(self.dimensions):
            self.kf.F[2 * axis, 2 * axis + 1] = dt

        q_pos = max(self.pos_var, 1e-12)
        q_vel = max(self.vel_var, 1e-12)
        q_block = Q_discrete_white_noise(dim=2, dt=dt, var=q_pos)
        q_block[1, 1] += q_vel * dt

        self.kf.Q = np.zeros((2 * self.dimensions, 2 * self.dimensions), dtype=float)
        for axis in range(self.dimensions):
            start = 2 * axis
            self.kf.Q[start:start + 2, start:start + 2] = q_block

    def initialize(self, measurement, timestamp_ns: int):
        z = np.atleast_1d(np.asarray(measurement, dtype=float))
        if z.shape[0] != self.dimensions:
            raise ValueError(f"expected {self.dimensions}D measurement, got shape {z.shape}")

        x0 = np.zeros((2 * self.dimensions, 1), dtype=float)
        for axis in range(self.dimensions):
            x0[2 * axis, 0] = z[axis]
        self.kf.x = x0
        self.last_t_ns = int(timestamp_ns)

    def update(self, measurement, timestamp_ns: int):
        z = np.atleast_1d(np.asarray(measurement, dtype=float))
        if z.shape[0] != self.dimensions:
            raise ValueError(f"expected {self.dimensions}D measurement, got shape {z.shape}")

        if self.last_t_ns is None:
            self.initialize(z, timestamp_ns)
            return self.get_position()

        dt_seconds = (int(timestamp_ns) - self.last_t_ns) * NS_TO_SECONDS
        self._set_dynamic_model(dt_seconds)
        self.kf.predict()
        self.kf.update(z)
        self.last_t_ns = int(timestamp_ns)
        return self.get_position()

    def get_position(self):
        position = np.array([self.kf.x[2 * axis, 0] for axis in range(self.dimensions)], dtype=float)
        if self.dimensions == 1:
            return float(position[0])
        return tuple(float(v) for v in position)


class Kalman1D(ConstantVelocityKalman):
    def __init__(self, pos_var=1e-4, vel_var=1e-3, meas_var=1e-3):
        super().__init__(dimensions=1, pos_var=pos_var, vel_var=vel_var, meas_var=meas_var)

    def update(self, measurement: float, timestamp_ns: int):
        return float(super().update([measurement], timestamp_ns))


class Kalman2D(ConstantVelocityKalman):
    def __init__(self, pos_var=1e-4, vel_var=1e-3, meas_var=1e-3):
        super().__init__(dimensions=2, pos_var=pos_var, vel_var=vel_var, meas_var=meas_var)

    def update(self, measurement_x: float, measurement_y: float, timestamp_ns: int):
        filtered_x, filtered_y = super().update([measurement_x, measurement_y], timestamp_ns)
        return float(filtered_x), float(filtered_y)
