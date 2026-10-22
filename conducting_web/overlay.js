// Draws over the live video what live_conductor.py draws into its OpenCV preview: the
// hands (MediaPipe's default hand style), the 1-second trail of the middle fingertip, the
// handedness labels, and the three status lines. Landmarks are 0..1 in the camera's frame,
// unmirrored; the video is shown mirrored, so x is flipped here. Text is never mirrored.
// The label and frameMapping helpers are the camera page's (client-web superconductor/overlay.js).

/** Where the frame (fw x fh) sits in a w x h element with object-fit: contain, mirrored. */
export function frameMapping(fw, fh, w, h) {
  const scale = Math.min(w / fw, h / fh);
  const ox = (w - fw * scale) / 2, oy = (h - fh * scale) / 2;
  return {
    scale, ox, oy, fw, fh,
    x: (x) => ox + (fw - x) * scale,
    y: (y) => oy + y * scale,
  };
}

// mediapipe drawing_styles.get_default_hand_landmarks_style / _connections_style
const PALM = "#ff3030", THUMB = "#ffe5b4", INDEX = "#804080", MIDDLE = "#ffcc00", RING = "#30ff30", PINKY = "#1565c0";
const POINT_COLORS = [PALM, PALM, THUMB, THUMB, THUMB, PALM, INDEX, INDEX, INDEX, PALM, MIDDLE, MIDDLE, MIDDLE,
                      PALM, RING, RING, RING, PALM, PINKY, PINKY, PINKY];
const CONNECTIONS = [
  [0, 1, "#808080", 3], [1, 5, "#808080", 3], [5, 9, "#808080", 3], [9, 13, "#808080", 3],
  [13, 17, "#808080", 3], [0, 17, "#808080", 3],
  [1, 2, THUMB, 2], [2, 3, THUMB, 2], [3, 4, THUMB, 2], [5, 6, INDEX, 2], [6, 7, INDEX, 2], [7, 8, INDEX, 2],
  [9, 10, MIDDLE, 2], [10, 11, MIDDLE, 2], [11, 12, MIDDLE, 2], [13, 14, RING, 2], [14, 15, RING, 2],
  [15, 16, RING, 2], [17, 18, PINKY, 2], [18, 19, PINKY, 2], [19, 20, PINKY, 2],
];
const TRAIL = "#ffdc00";       // (0, 220, 255) BGR
const STATUS = "#ffff00";      // (0, 255, 255) BGR
const HANDEDNESS = "#36cd58";  // (88, 205, 54) BGR

/**
 * Draw on the 2D context `g` of a canvas laid over the video element (`w` x `h` CSS px).
 * `frame`: the newest tracked frame {width, height, hands, handedness}; `trail`: middle
 * fingertip points [{x, y}] (0..1, unmirrored) of the last second; `reply`: the conducting
 * process's newest answer; `stale`: no answer for a while, everything dimmed and marked.
 */
export function drawConducting(g, w, h, frame, trail, reply, { stale = false } = {}) {
  g.clearRect(0, 0, w, h);
  if (!frame || !frame.width) return;
  const m = frameMapping(frame.width, frame.height, w, h);
  const fs = Math.max(0.6, Math.min(1.3, (frame.width * m.scale) / 1280));  // text scale
  const px = (p) => [m.x(p.x * frame.width), m.y(p.y * frame.height)];
  g.save();
  g.lineJoin = g.lineCap = "round";

  g.strokeStyle = TRAIL; g.lineWidth = 2;
  for (let i = 1; i < trail.length; i++) line(g, ...px(trail[i - 1]), ...px(trail[i]));

  frame.hands.forEach((hand, i) => {
    const pts = hand.map(px);
    for (const [a, b, color, width] of CONNECTIONS) {
      g.strokeStyle = color; g.lineWidth = width;
      line(g, ...pts[a], ...pts[b]);
    }
    pts.forEach(([x, y], k) => {
      g.fillStyle = POINT_COLORS[k];
      g.beginPath(); g.arc(x, y, 4, 0, 2 * Math.PI); g.fill();
    });
    if (frame.handedness[i]) label(g, frame.handedness[i], pts[0][0], pts[0][1], HANDEDNESS, 22 * fs);
  });

  g.globalAlpha = stale ? 0.35 : 1;
  const x = m.ox + 12 * fs, y = m.oy;
  if (reply) {
    label(g, reply.status, x, y + 30 * fs, STATUS, 20 * fs, 600);
    label(g, reply.detail, x, y + 56 * fs, "#ffffff", 16 * fs);
  }
  label(g, "F: release / stop | R: resume", x, y + 80 * fs, "#ffffff", 15 * fs);
  if (stale) {
    g.globalAlpha = 1;
    label(g, "no answer from the conducting process", w / 2 - 190 * fs, h / 2, "#ffffff", 22 * fs);
  }
  g.restore();
}

function label(g, text, x, y, color, size, weight = 500) {
  g.font = `${weight} ${size}px sans-serif`;
  g.lineWidth = Math.max(2, size / 5);
  g.strokeStyle = "rgba(0,0,0,0.75)";
  g.strokeText(text, x, y);
  g.fillStyle = color;
  g.fillText(text, x, y);
}

function line(g, x1, y1, x2, y2) {
  g.beginPath(); g.moveTo(x1, y1); g.lineTo(x2, y2); g.stroke();
}
