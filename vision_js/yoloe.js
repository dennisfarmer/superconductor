// YOLOE detection in the browser, matching what Ultralytics' predictor does around the model in
// object_tracking/tracker.py (model.track -> predict): letterbox (ultralytics/data/augment.py
// LetterBox), the exported ONNX model, NMS (ultralytics/utils/nms.py non_max_suppression, best
// class per box, class-aware) and scaling boxes back to the frame (ops.scale_boxes).
//
// The model is YOLOE-11s without its segmentation head, exported with the class prompts built in
// and a fixed input size (see export_yoloe.py). Its output is (1, 4 + nc, anchors): box center,
// size, then one score per class.

const PAD = 114 / 255;
const MAX_WH = 7680;  // class offset so NMS never suppresses across classes
const MAX_NMS = 30000;

/** Letterbox geometry for a frame of (w, h) into the model's (inW, inH), as LetterBox(auto=False). */
export function letterbox(w, h, inW, inH) {
  const r = Math.min(inH / h, inW / w);
  const newW = Math.round(w * r), newH = Math.round(h * r);
  const dw = (inW - newW) / 2, dh = (inH - newH) / 2;
  return { r, newW, newH, left: pyRound(dw - 0.1), top: pyRound(dh - 0.1) };
}

/**
 * RGBA pixels (ImageData.data) of a (w, h) frame -> the model input, CHW float32 RGB in [0, 1],
 * resized with bilinear interpolation (cv2.INTER_LINEAR's pixel-center mapping) and padded with
 * gray. `out`: Float32Array(3 * inW * inH), reused between frames.
 */
export function preprocess(rgba, w, h, inW, inH, out) {
  const lb = letterbox(w, h, inW, inH);
  const plane = inW * inH;
  out.fill(PAD);
  const sx = w / lb.newW, sy = h / lb.newH;
  for (let y = 0; y < lb.newH; y++) {
    let fy = (y + 0.5) * sy - 0.5;
    if (fy < 0) fy = 0;
    let y0 = Math.floor(fy);
    if (y0 > h - 1) y0 = h - 1;
    const y1 = y0 < h - 1 ? y0 + 1 : y0;
    const wy = fy - y0;
    const row0 = y0 * w * 4, row1 = y1 * w * 4;
    const o = (y + lb.top) * inW + lb.left;
    for (let x = 0; x < lb.newW; x++) {
      let fx = (x + 0.5) * sx - 0.5;
      if (fx < 0) fx = 0;
      let x0 = Math.floor(fx);
      if (x0 > w - 1) x0 = w - 1;
      const x1 = x0 < w - 1 ? x0 + 1 : x0;
      const wx = fx - x0;
      const a = row0 + 4 * x0, b = row0 + 4 * x1, c = row1 + 4 * x0, d = row1 + 4 * x1;
      for (let ch = 0; ch < 3; ch++) {
        const top = rgba[a + ch] + (rgba[b + ch] - rgba[a + ch]) * wx;
        const bot = rgba[c + ch] + (rgba[d + ch] - rgba[c + ch]) * wx;
        out[ch * plane + o + x] = (top + (bot - top) * wy) / 255;
      }
    }
  }
  return lb;
}

/**
 * Model output (Float32Array, (4 + nc) x anchors, row-major) -> detections in frame pixels after
 * NMS. Returns { n, boxes (x1, y1, x2, y2 flat), scores, classes }, highest score first.
 */
export function postprocess(pred, nc, anchors, lb, w, h, conf = 0.25, iou = 0.7, maxDet = 300) {
  // best class per anchor above the threshold
  const cand = [];
  for (let a = 0; a < anchors; a++) {
    let best = -Infinity, cls = 0;
    for (let c = 0; c < nc; c++) {
      const s = pred[(4 + c) * anchors + a];
      if (s > best) { best = s; cls = c; }
    }
    if (best > conf) cand.push(a, cls, best);
  }
  const m = cand.length / 3;
  let order = Array.from({ length: m }, (_, i) => i);
  order.sort((i, j) => cand[3 * j + 2] - cand[3 * i + 2]);  // stable: ties keep anchor order
  if (order.length > MAX_NMS) order = order.slice(0, MAX_NMS);

  // xywh -> xyxy (model input pixels), offset by class for NMS
  const bx = new Float32Array(4 * order.length);
  order.forEach((i, k) => {
    const a = cand[3 * i], off = cand[3 * i + 1] * MAX_WH;
    const cx = pred[a], cy = pred[anchors + a], bw = pred[2 * anchors + a], bh = pred[3 * anchors + a];
    bx[4 * k] = cx - bw / 2 + off; bx[4 * k + 1] = cy - bh / 2 + off;
    bx[4 * k + 2] = cx + bw / 2 + off; bx[4 * k + 3] = cy + bh / 2 + off;
  });

  // greedy NMS in score order (torchvision.ops.nms)
  const keep = [];
  const dead = new Uint8Array(order.length);
  for (let k = 0; k < order.length && keep.length < maxDet; k++) {
    if (dead[k]) continue;
    keep.push(k);
    const ax1 = bx[4 * k], ay1 = bx[4 * k + 1], ax2 = bx[4 * k + 2], ay2 = bx[4 * k + 3];
    const areaA = (ax2 - ax1) * (ay2 - ay1);
    for (let q = k + 1; q < order.length; q++) {
      if (dead[q]) continue;
      const iw = Math.min(ax2, bx[4 * q + 2]) - Math.max(ax1, bx[4 * q]);
      const ih = Math.min(ay2, bx[4 * q + 3]) - Math.max(ay1, bx[4 * q + 1]);
      if (iw <= 0 || ih <= 0) continue;
      const inter = iw * ih;
      const areaB = (bx[4 * q + 2] - bx[4 * q]) * (bx[4 * q + 3] - bx[4 * q + 1]);
      if (inter / (areaA + areaB - inter) > iou) dead[q] = 1;
    }
  }

  // back to frame pixels (scale_boxes, then clip)
  const n = keep.length;
  const boxes = new Float32Array(4 * n), scores = new Float32Array(n), classes = new Int32Array(n);
  keep.forEach((k, i) => {
    const c = cand[3 * order[k] + 1], off = c * MAX_WH;
    const x1 = (bx[4 * k] - off - lb.left) / lb.r, y1 = (bx[4 * k + 1] - off - lb.top) / lb.r;
    const x2 = (bx[4 * k + 2] - off - lb.left) / lb.r, y2 = (bx[4 * k + 3] - off - lb.top) / lb.r;
    boxes[4 * i] = clamp(x1, 0, w); boxes[4 * i + 1] = clamp(y1, 0, h);
    boxes[4 * i + 2] = clamp(x2, 0, w); boxes[4 * i + 3] = clamp(y2, 0, h);
    scores[i] = cand[3 * order[k] + 2];
    classes[i] = c;
  });
  return { n, boxes, scores, classes };
}

const clamp = (v, lo, hi) => (v < lo ? lo : v > hi ? hi : v);

/** Python's round(): halves to even */
function pyRound(v) {
  const f = Math.floor(v), d = v - f;
  if (d > 0.5) return f + 1;
  if (d < 0.5) return f;
  return f % 2 === 0 ? f : f + 1;
}
