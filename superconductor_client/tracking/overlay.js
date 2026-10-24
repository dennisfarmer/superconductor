// Draws the client's "overlay" message (CAMERA_PROTOCOL.md) over the live video: what
// collab.py used to draw into the frame with OpenCV. Coordinates in the message are the
// frame's pixels, unmirrored; the video is shown mirrored, so x is flipped here. Text is
// never mirrored.

const GRAY = "#8c8c8c";

/** Where the frame (fw x fh) sits in a w x h element with object-fit: contain, mirrored. */
export function frameMapping(fw, fh, w, h) {
  const scale = Math.min(w / fw, h / fh);
  const ox = (w - fw * scale) / 2, oy = (h - fh * scale) / 2;
  return {
    scale, ox, oy, fw, fh,
    x: (x) => ox + (fw - x) * scale,
    y: (y) => oy + y * scale,
    // a frame box (x1, y1, x2, y2) -> canvas rect (left, top, width, height)
    rect: (b) => [ox + (fw - b[2]) * scale, oy + b[1] * scale, (b[2] - b[0]) * scale, (b[3] - b[1]) * scale],
  };
}

/**
 * Draw `ov` on the 2D context `g` of a canvas laid over the video element (`w` x `h` CSS px).
 * `stale`: no overlay for a while; everything is dimmed and marked.
 */
export function drawOverlay(g, w, h, ov, { stale = false } = {}) {
  g.clearRect(0, 0, w, h);
  if (!ov || !ov.width) return;
  const m = frameMapping(ov.width, ov.height, w, h);
  const fs = Math.max(0.6, Math.min(1.3, (ov.width * m.scale) / 1280));  // text scale
  g.save();
  g.globalAlpha = stale ? 0.35 : 1;
  g.lineJoin = "round";

  // reference point and its range
  const ref = ov.reference;
  if (ref) {
    const rx = m.x(ref.x * ov.width), ry = m.y(ref.y * ov.height);
    g.strokeStyle = "#5a5a5a"; g.lineWidth = 1;
    circle(g, rx, ry, ref.radius * ov.width * m.scale);
    g.strokeStyle = "#ffffff"; g.lineWidth = 2;
    line(g, rx - 12, ry, rx + 12, ry); line(g, rx, ry - 12, rx, ry + 12);
  }

  for (const b of ov.hands || []) box(g, m.rect(b), "#c8c8c8", 1);

  for (const p of ov.pending || []) {
    const r = m.rect(p.box);
    box(g, r, GRAY, 1, true);
    label(g, p.label, r[0], r[1] - 8, GRAY, 13 * fs);
  }

  for (const c of ov.combos || []) {
    const pts = c.centers.map(([x, y]) => [m.x(x), m.y(y)]);
    if (c.active) {
      const r = m.rect(c.box);
      box(g, r, c.color, 4);
      g.strokeStyle = c.color; g.lineWidth = 3;
      for (let i = 1; i < pts.length; i++) line(g, ...pts[i - 1], ...pts[i]);
      label(g, c.label, r[0], r[1] + r[3] + 22 * fs, c.color, 17 * fs);
    } else if (pts.length === 2) {
      g.strokeStyle = c.color; g.lineWidth = 1;
      line(g, ...pts[0], ...pts[1]);
      label(g, c.label, (pts[0][0] + pts[1][0]) / 2 - 40, (pts[0][1] + pts[1][1]) / 2 - 6, c.color, 11 * fs);
    }
  }

  for (const o of ov.objects || []) {
    const r = m.rect(o.box);
    box(g, r, o.color, 2, o.coasted);
    if (o.progress != null) {  // dwell progress while the object is being touched
      g.fillStyle = "#ffffff";
      g.fillRect(r[0], r[1] + r[3] + 4, r[2] * o.progress, 6);
    }
    if (!o.combined && ref) {
      g.strokeStyle = o.color; g.lineWidth = 1;
      line(g, m.x(ref.x * ov.width), m.y(ref.y * ov.height), m.x(o.center[0]), m.y(o.center[1]));
    }
    label(g, o.label, r[0], Math.max(r[1] - 8, 15), o.color, 14 * fs);
  }

  drawHud(g, w, h, ov, fs);
  if (stale) {
    g.globalAlpha = 1;
    g.font = `500 ${22 * fs}px sans-serif`;
    label(g, "client offline", (w - g.measureText("client offline").width) / 2, h / 2, "#ffffff", 22 * fs);
  }
  g.restore();
}

// the calibration message, the MRT2 input panel, combo banners, the log
function drawHud(g, w, h, ov, fs) {
  let y = 26 * fs;
  if (ov.message) {
    g.font = `600 ${30 * fs}px sans-serif`;
    label(g, ov.message, (w - g.measureText(ov.message).width) / 2, 70 * fs, "#ffffff", 30 * fs, 600);
  }
  const rows = ov.mrt2?.rows || [];
  if (rows.length) {
    const x = 12 * fs, bar = 150 * fs, lh = 22 * fs;
    g.fillStyle = "rgba(0,0,0,0.45)";
    g.fillRect(4 * fs, 6 * fs, 640 * fs, lh * (rows.length + 1.4) + (Object.keys(ov.mrt2.controls || {}).length ? lh : 0));
    label(g, "MRT2 input", x, y, "#ffffff", 13 * fs);
    for (const r of rows) {
      y += lh;
      label(g, r.prompt.slice(0, 30), x, y, r.color, 13 * fs, 600);
      label(g, r.source.slice(0, 16), x + 270 * fs, y, r.color, 12 * fs);
      g.strokeStyle = r.color; g.fillStyle = r.color; g.lineWidth = 1;
      g.fillRect(x + 400 * fs, y - 11 * fs, bar * Math.min(r.value, 1), 9 * fs);
      g.strokeRect(x + 400 * fs, y - 11 * fs, bar, 9 * fs);
      label(g, `${r.value.toFixed(2)}  ${(100 * r.share).toFixed(0)}%`, x + 560 * fs, y, r.color, 12 * fs);
    }
    const controls = Object.entries(ov.mrt2.controls || {});
    if (controls.length) {
      y += lh;
      label(g, controls.map(([k, v]) => `${k} ${v.toFixed(2)}`).join("  "), x, y, "#ffffff", 12 * fs);
    }
  }
  let by = Math.max(ov.message ? 120 * fs : 80 * fs, y + 50 * fs);
  for (const b of ov.banners || []) {
    g.font = `600 ${22 * fs}px sans-serif`;
    label(g, b.text, (w - g.measureText(b.text).width) / 2, by, b.color, 22 * fs, 600);
    by += 36 * fs;
  }
  const log = ov.log || [];
  let ly = h - 12 * fs - 20 * fs * log.length;
  for (const t of log) { label(g, t, 12 * fs, ly, "#ffffff", 12 * fs); ly += 20 * fs; }
}

function label(g, text, x, y, color, size, weight = 500) {
  g.font = `${weight} ${size}px sans-serif`;
  g.lineWidth = Math.max(2, size / 5);
  g.strokeStyle = "rgba(0,0,0,0.75)";
  g.strokeText(text, x, y);
  g.fillStyle = color;
  g.fillText(text, x, y);
}

function box(g, [x, y, w, h], color, width, dashed = false) {
  g.strokeStyle = color;
  g.lineWidth = width;
  g.setLineDash(dashed ? [10, 10] : []);
  g.strokeRect(x, y, w, h);
  g.setLineDash([]);
}

function line(g, x1, y1, x2, y2) {
  g.beginPath(); g.moveTo(x1, y1); g.lineTo(x2, y2); g.stroke();
}

function circle(g, x, y, r) {
  g.beginPath(); g.arc(x, y, r, 0, 2 * Math.PI); g.stroke();
}
