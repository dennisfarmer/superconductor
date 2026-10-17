"""Reference detections for vision_js's parity test, from Ultralytics.

Exports a detection-only YOLOE (superconductor.web_model.export) with prompts that
the sample images contain ("person", "hand", "tie"; the pipeline doesn't depend on
which), runs Ultralytics' own predictor on Ultralytics' sample images, and writes
fixtures/: the model, each image's raw RGBA pixels and the expected detections. The
fixtures are large and generated, not committed (see .gitignore).

    python vision_js/test/make_fixtures.py     (in sc_env, from superconductor_client_web;
                                                needs yoloe-11s-seg.pt, downloaded if missing)
"""
import json
from pathlib import Path

import cv2
import ultralytics
from ultralytics import YOLO

from superconductor.web_model import INPUT, export

OUT = Path(__file__).parent / "fixtures"
CLASSES = ["person", "hand", "tie"]
CONF, IOU = 0.05, 0.7


def main():
    OUT.mkdir(exist_ok=True)
    model = OUT / "yoloe-test.onnx"
    export("yoloe-11s-seg.pt", CLASSES, model)
    onnx = YOLO(str(model), task="detect")
    assets = Path(ultralytics.__file__).parent / "assets"
    expected = {"classes": CLASSES, "conf": CONF, "iou": IOU, "input": [INPUT[1], INPUT[0]], "images": {}}
    for name in ("zidane", "bus"):  # 16:9, and a portrait one (padded the other way)
        img = cv2.imread(str(assets / f"{name}.jpg"))
        h, w = img.shape[:2]
        cv2.cvtColor(img, cv2.COLOR_BGR2RGBA).tofile(OUT / f"{name}.rgba")
        r = onnx.predict(img, imgsz=INPUT, conf=CONF, iou=IOU, verbose=False)[0].boxes
        expected["images"][name] = dict(w=w, h=h, boxes=r.xyxy.tolist(), scores=r.conf.tolist(),
                                        classes=r.cls.int().tolist())
        print(f"{name}: {len(r)} detections")
    (OUT / "expected.json").write_text(json.dumps(expected))


if __name__ == "__main__":
    main()
