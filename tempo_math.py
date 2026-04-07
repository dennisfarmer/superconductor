from __future__ import annotations

import numpy as np

from tempo_constants import MIN_DT_SECONDS, NS_TO_SECONDS


def xytopolar(x, y):
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return r, theta


def get_physics_series(pos_buffer, speed_buffer):
    """Computes arrays for X, Y, V, and A using numpy vectorization."""
    pos_arr = np.array(pos_buffer)
    speed_arr = np.array(speed_buffer)

    dts = np.diff(pos_arr[:, 2]) * NS_TO_SECONDS
    dts = np.where(dts <= 0, MIN_DT_SECONDS, dts)

    if len(speed_arr) > 1:
        acc_arr = np.diff(speed_arr, axis=0) / dts[-len(speed_arr) + 1 :][:, None]
    else:
        acc_arr = np.empty((0, 2))

    return {
        "x": pos_arr[:, 0],
        "y": pos_arr[:, 1],
        "vx": speed_arr[:, 0],
        "vy": speed_arr[:, 1],
        "ax": acc_arr[:, 0],
        "ay": acc_arr[:, 1],
    }
