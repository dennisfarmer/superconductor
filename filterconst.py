"""
Kalman filter variance parameters.

These constants control the behavior of the Kalman filters:

    POS_VARIANCE: Expected variance in position measurements (lower = trust measurements more)
    VEL_VARIANCE: Expected variance in velocity changes (lower = more stable tracking)
    MEAS_VARIANCE: Measurement noise (lower = trust measurements more)
    TEMPO_VARIANCE: Expected variance in tempo measurements (higher = more smoothing)
"""

POS_VARIANCE=1e-4
VEL_VARIANCE=1e-2
MEAS_VARIANCE=1e-5
TEMPO_VARIANCE=3.0
