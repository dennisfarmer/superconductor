# Debug: browser YOLOE detects far fewer objects than Python

## Symptom

On the Mac (setup 1, browser detection on WebGPU), objects in front of the camera often aren't **detected** at all: YOLOE in the browser doesn't return a box for them. The native Python client (YOLOE-11s-seg on the same scene) detects them.

This is the **detection** stage: frame → YOLOE → boxes. It isn't tracking (BoT-SORT) or recognition (DINOv2 / identities).

## Stage to investigate

The browser's YOLOE step, `vision_js/worker.js`, runs:
1. the camera frame drawn to an OffscreenCanvas, then `preprocess` (`vision_js/yoloe.js`: letterbox to 480×288, RGB, /255)
2. the ONNX model `superconductor_client/tracking/var/web/models/yoloe.onnx`, exported by `superconductor_client/tracking/web_model.py` (YOLOE-11s without the seg head, weights from `yoloe-11s-seg.pt`, `set_classes(["stuffed toy", "toy", "hand"])`), run with onnxruntime-web on WebGPU
3. `postprocess` (`vision_js/yoloe.js`: confidence filter at `[tracking] conf`, class-aware NMS at IoU 0.7, boxes scaled back to the frame)

## What has been verified, and what hasn't

Verified:
- YOLOE-11s without the seg head gives the same boxes as `yoloe-11s-seg` (PyTorch, 95 frames, 0.000 px).
- The ONNX export matches PyTorch (0.001 px).
- `preprocess` + `postprocess` + onnxruntime match Ultralytics' predictor: the same detections, boxes within 1.4 px. This held in Node and in Chrome on WebGPU.

**Not verified:** all of the above used the prompts "person, hand, tie" on Ultralytics' two sample photos (`bus.jpg`, `zidane.jpg`). None of it checked:
- the model exported with the real prompts ("stuffed toy, toy, hand")
- frames from the actual camera, with toys
- the page's actual input: the live video drawn at the page's resolution (default 960×540) inside the worker

## Recommended test: JS vs Python on the same live frames

Compare raw detections, before any tracking, from three detectors on the same frames, with the same classes and `conf`:

| | Detector | How |
|---|---|---|
| P | Python, YOLOE-11s-seg (what the native client uses) | `YOLOE("yoloe-11s-seg.pt")`, `set_classes(classes)`, `predict(frame, imgsz=480, conf=conf)` |
| O | Python, the exported ONNX | `YOLO("superconductor_client/tracking/var/web/models/yoloe.onnx", task="detect").predict(frame, imgsz=(288, 480), conf=conf)` |
| J | JS, the browser pipeline | `preprocess` → onnxruntime (`yoloe.onnx`) → `postprocess`, as in `vision_js/test/parity.test.js` |

Steps:
1. **Capture frames the page actually uses.** With the page running, save several frames with toys in view (e.g. from DevTools: draw `#video` to a canvas at its `videoWidth`×`videoHeight` and download it as PNG). Or save the JPEGs the client receives on `/ws/camera`. Include objects at different distances and angles.
2. **Run P and O on them** in sc_env, with `classes = ["stuffed toy", "toy", "hand"]` and `conf` from `collab.toml`. Use a low `conf` too (e.g. 0.01), to see the scores of objects that fall just under the threshold.
3. **Run J on the same pixels.** Adapt `vision_js/test/make_fixtures.py`: use these frames instead of the sample photos, and the real `yoloe.onnx` instead of the test export. Then run the parity test (`node --test vision_js/test/`), or print its detections.
4. **Compare** the number of detections, the classes and the scores per object.

How to read it:
- **P finds the object and O doesn't** (or scores it much lower): the export is the problem. Check the detection-only model built from the seg weights with these prompts (`web_model.export`), the text embeddings `set_classes` writes into it, and the 480×288 input against PyTorch's rect letterbox for the camera's aspect ratio.
- **O finds it and J doesn't:** the JS pipeline is the problem (`preprocess` / `postprocess` / onnxruntime). Repeat J in Chrome on WebGPU, not only in Node on CPU, to rule out WebGPU numerics.
- **J finds it in this test but not live:** the problem is the frame the worker gets. Check resolution, color and the ImageBitmap → OffscreenCanvas path in `worker.js`. Compare the worker's input against the saved frame.
- **All three miss it:** the prompts or `conf` are the problem, not the browser.
