// vision_js (letterbox, NMS, box scaling) against Ultralytics' predictor on the same
// pixels and the same exported model (fixtures from make_fixtures.py), run with
// onnxruntime-node: the same detections, boxes within 2 px (the resize differs slightly
// from OpenCV's), scores within 0.01. Skipped when the fixtures or onnxruntime-node are
// missing.
//
//   python vision_js/test/make_fixtures.py && (cd vision_js && npm install) && node --test vision_js/test/

import { test } from "node:test";
import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import { preprocess, postprocess } from "../yoloe.js";

const FIXTURES = new URL("./fixtures/", import.meta.url);
const have = existsSync(new URL("expected.json", FIXTURES));
let ort = null;
try { ort = (await import("onnxruntime-node")).default; } catch { /* skipped below */ }

test("vision_js detections match Ultralytics", { skip: !have ? "no fixtures (make_fixtures.py)" : !ort && "no onnxruntime-node" }, async () => {
  const exp = JSON.parse(readFileSync(new URL("expected.json", FIXTURES)));
  const session = await ort.InferenceSession.create(new URL("yoloe-test.onnx", FIXTURES).pathname);
  const [inW, inH] = exp.input;
  const input = new Float32Array(3 * inW * inH);
  for (const [name, { w, h, boxes, scores, classes }] of Object.entries(exp.images)) {
    const rgba = new Uint8Array(readFileSync(new URL(`${name}.rgba`, FIXTURES)));
    const lb = preprocess(rgba, w, h, inW, inH, input);
    const out = await session.run({ [session.inputNames[0]]: new ort.Tensor("float32", input, [1, 3, inH, inW]) });
    const o = out[session.outputNames[0]];
    const det = postprocess(o.data, o.dims[1] - 4, o.dims[2], lb, w, h, exp.conf, exp.iou);
    assert.equal(det.n, boxes.length, `${name}: number of detections`);
    const used = new Set();
    boxes.forEach((b, i) => {  // each reference detection: the closest unused one of the same class
      let best = -1, bestD = Infinity;
      for (let k = 0; k < det.n; k++) {
        if (used.has(k) || det.classes[k] !== classes[i]) continue;
        const d = Math.max(...b.map((v, j) => Math.abs(det.boxes[4 * k + j] - v)));
        if (d < bestD) { bestD = d; best = k; }
      }
      assert.ok(best >= 0 && bestD < 2, `${name} #${i}: box off by ${bestD} px`);
      assert.ok(Math.abs(det.scores[best] - scores[i]) < 0.01, `${name} #${i}: score`);
      used.add(best);
    });
  }
});
