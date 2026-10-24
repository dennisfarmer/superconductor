// Kalman filter of Ultralytics' KalmanFilterXYWH (ultralytics/trackers/utils/kalman_filter.py),
// the one BOTSORT uses. State: [x, y, w, h, vx, vy, vw, vh] (box center, size, their velocities).
//
// Ultralytics does this with 8x8 matrices, but every matrix involved is diagonal or only links a
// coordinate with its own velocity: the motion matrix adds v_i to p_i, the observation matrix
// picks p_i, and the initial, process and measurement covariances are diagonal. So the covariance
// stays block-diagonal and the filter is exactly four independent 2x2 filters, one per coordinate.
// Per coordinate i the covariance is [[a_i, b_i], [b_i, c_i]] (position variance, position-velocity
// covariance, velocity variance). This holds only without camera-motion compensation (GMC), which
// rotates coordinates into each other; superconductor's tracker config has it off.
//
// `mean` is a Float64Array(8). `cov` is a Float64Array(12): a[0..3], b[4..7], c[8..11].

const STD_POS = 1 / 20;
const STD_VEL = 1 / 160;

/** New track from a measurement (x, y, w, h): zero velocity, uncertainty relative to its size. */
export function initiate(mean, cov, x, y, w, h) {
  mean[0] = x; mean[1] = y; mean[2] = w; mean[3] = h;
  mean[4] = 0; mean[5] = 0; mean[6] = 0; mean[7] = 0;
  for (let i = 0; i < 4; i++) {
    const size = i % 2 === 0 ? w : h;  // x and w scale with the width, y and h with the height
    const sp = 2 * STD_POS * size;
    const sv = 10 * STD_VEL * size;
    cov[i] = sp * sp;
    cov[4 + i] = 0;
    cov[8 + i] = sv * sv;
  }
}

/** One frame ahead: p += v, covariance F P F^T + Q (noise from the size before the step). */
export function predict(mean, cov) {
  const w = mean[2], h = mean[3];
  for (let i = 0; i < 4; i++) {
    const size = i % 2 === 0 ? w : h;
    const qp = STD_POS * size, qv = STD_VEL * size;
    const a = cov[i], b = cov[4 + i], c = cov[8 + i];
    mean[i] += mean[4 + i];
    cov[i] = a + 2 * b + c + qp * qp;
    cov[4 + i] = b + c;
    cov[8 + i] = c + qv * qv;
  }
}

/** Correct with a measurement (x, y, w, h); measurement noise from the predicted size. */
export function update(mean, cov, x, y, w, h) {
  const pw = mean[2], ph = mean[3];
  for (let i = 0; i < 4; i++) {
    const size = i % 2 === 0 ? pw : ph;
    const r = STD_POS * size;
    const a = cov[i], b = cov[4 + i], c = cov[8 + i];
    const s = a + r * r;               // innovation variance
    const kp = a / s, kv = b / s;      // Kalman gain for position, velocity
    const z = i === 0 ? x : i === 1 ? y : i === 2 ? w : h;
    const innovation = z - mean[i];
    mean[i] += kp * innovation;
    mean[4 + i] += kv * innovation;
    cov[i] = a - kp * a;               // P - K S K^T, per block
    cov[4 + i] = b - kp * b;
    cov[8 + i] = c - kv * b;
  }
}
