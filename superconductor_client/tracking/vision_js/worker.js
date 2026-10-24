// The camera page's detector, in a Web Worker so the page stays smooth: YOLOE on
// WebGPU (onnxruntime-web) + BoT-SORT (tracker_js), the same two steps model.track()
// runs in the client process. Messages (see index.js):
//   in:  {type: "init", config}       config: /api/web's "detector" (model_url, ort_url, classes, conf, iou, input)
//        {type: "frame", id, bitmap}  an ImageBitmap of the frame (transferred, closed here)
//   out: {type: "ready"} | {type: "unavailable", reason}
//        {type: "result", id, objects, ms}   objects as TrackedObject fields (CAMERA_PROTOCOL.md)

import { preprocess, postprocess } from "./yoloe.js";
import { BOTSORT } from "../tracker_js/index.js";

let ort, session, tracker, cfg, input, canvas, g;

self.onmessage = async (e) => {
  const msg = e.data;
  if (msg.type === "init") {
    try {
      await init(msg.config);
      self.postMessage({ type: "ready" });
    } catch (err) {
      self.postMessage({ type: "unavailable", reason: String(err?.message || err) });
    }
  } else if (msg.type === "frame") {
    try {
      self.postMessage({ type: "result", id: msg.id, ...(await detect(msg.bitmap)) });
    } catch (err) {
      self.postMessage({ type: "result", id: msg.id, error: String(err?.message || err), objects: [] });
    } finally {
      msg.bitmap.close();
    }
  }
};

async function init(config) {
  cfg = config;
  if (!config.model_url) throw new Error("no detector model on the client (make web-model)");
  if (!config.ort_url) throw new Error("no onnxruntime-web on the client (make web-deps)");
  if (!self.navigator.gpu) throw new Error("this browser has no WebGPU");
  const adapter = await self.navigator.gpu.requestAdapter();
  if (!adapter) throw new Error("WebGPU has no GPU adapter here");
  const ortBase = new URL(config.ort_url, self.location.href).href;
  ort = await import(ortBase + "ort.webgpu.min.mjs");
  ort.env.wasm.wasmPaths = ortBase;
  session = await ort.InferenceSession.create(new URL(config.model_url, self.location.href).href,
                                              { executionProviders: ["webgpu"] });
  const [inW, inH] = config.input;
  input = new Float32Array(3 * inW * inH);
  // one run to compile the shaders before the first real frame
  await session.run({ [session.inputNames[0]]: new ort.Tensor("float32", input, [1, 3, inH, inW]) });
  tracker = new BOTSORT(config.tracker);  // the client's botsort_superconductor.yaml
}

async function detect(bitmap) {
  const t0 = performance.now();
  const w = bitmap.width, h = bitmap.height;
  if (!canvas || canvas.width !== w || canvas.height !== h) {
    canvas = new OffscreenCanvas(w, h);
    g = canvas.getContext("2d", { willReadFrequently: true });
  }
  g.drawImage(bitmap, 0, 0);
  const rgba = g.getImageData(0, 0, w, h).data;
  const [inW, inH] = cfg.input;
  const lb = preprocess(rgba, w, h, inW, inH, input);
  const out = await session.run({ [session.inputNames[0]]: new ort.Tensor("float32", input, [1, 3, inH, inW]) });
  // output0: (1, 4 + nc [+ 32 mask coefficients for the seg model], anchors); the seg
  // model's mask prototypes (output1) aren't used
  const o = out[session.outputNames[0]];
  const det = postprocess(o.data, cfg.classes.length, o.dims[2], lb, w, h, cfg.conf, cfg.iou);
  const tracks = tracker.update(det.boxes, det.scores, det.classes, det.n);
  const objects = tracks.map((t) => {
    const [x1, y1, x2, y2] = t.box;
    return {
      track_id: t.trackId, class_name: cfg.classes[t.cls] ?? String(t.cls), confidence: t.score,
      box: [x1, y1, x2, y2], center: [(x1 + x2) / 2 / w, (y1 + y2) / 2 / h],
      box_normalized: [x1 / w, y1 / h, x2 / w, y2 / h],
    };
  });
  return { objects, ms: performance.now() - t0 };
}
