"""YOLO detection + BoT-SORT tracking via ultralytics.

BoT-SORT keeps a stable `track_id` per physical object across frames (and,
with ReID + a long track buffer, across short occlusions), which is what lets
each object keep controlling the same music parameter.
"""
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import torch
from ultralytics import YOLO

from superconductor.object_tracking.embedder import Embedder
from superconductor.object_tracking.identity import IdentityRegistry, color_histogram

TRACKER_CONFIG = Path(__file__).parent / "botsort_superconductor.yaml"


@dataclass
class TrackedObject:
    track_id: int
    class_name: str
    confidence: float
    box: tuple  # (x1, y1, x2, y2) in pixels
    center: tuple  # (x, y) normalized to [0, 1]
    box_normalized: tuple = None  # (x1, y1, x2, y2) in [0, 1]
    hist: np.ndarray = None  # color histogram (appearance fingerprint)
    identity: int = None  # persistent id, stable across track losses (see identity.py)
    embedding: np.ndarray = None  # appearance feature, only on frames it was computed
    crop: np.ndarray = None  # image of the object (set together with `embedding`)
    coasted: bool = False  # not detected this frame: last known position (see identity.py)


class ObjectTracker:
    def __init__(self, model="yoloe-11s-seg.pt", classes=("toy",), conf=0.3,
                 imgsz=480, tracker=TRACKER_CONFIG, device=None, identity_dir=None,
                 hand_class="hand", embedder=None, embed_imgsz=128, embeds_per_frame=2,
                 identity=None):
        """
        `model`: any ultralytics detector. COCO models (yolo11n.pt, ...) have
            fixed class names ("sports ball", "teddy bear", ...). YOLOE models
            (yoloe-11s-seg.pt, ...) are open-vocabulary: `classes` are free-text
            prompts like "toy" or "green plush toy".
        `classes`: class names to keep (None = all). A single generic prompt
            ("toy") gives class-agnostic detection; identities come from appearance.
        `identity_dir`: where to persist identities across sessions (None = don't)
        `hand_class`: detections of this class are returned in `self.hands`
            (used for "touch the ..." calibration and "held" triggers) instead
            of becoming identities
        `embedder`: appearance model for re-identification (e.g. "yolo11n-cls.pt",
            None = color only). At most `embeds_per_frame` crops are embedded per
            frame (~3 ms each on CPU): new objects first, then periodic checks.
        `identity`: extra IdentityRegistry settings (thresholds, see identity.py)
        """
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.model = YOLO(model)
        self.conf = conf
        self.imgsz = imgsz
        self.tracker = str(tracker)
        self.class_ids = None
        self.embedder = Embedder(embedder, imgsz=embed_imgsz) if embedder else None
        self.embeds_per_frame = embeds_per_frame
        self.registry = IdentityRegistry(store_dir=identity_dir, embedder=self.embedder,
                                         **(identity or {}))
        self.hand_class = hand_class
        self.hands = []

        if classes:
            if "yoloe" in Path(model).name.lower():
                self.model.set_classes(list(classes))
            else:
                name_to_id = {n: i for i, n in self.model.names.items()}
                unknown = [c for c in classes if c not in name_to_id]
                if unknown:
                    raise ValueError(f"{unknown} not in {model} classes: {sorted(name_to_id)}")
                self.class_ids = [name_to_id[c] for c in classes]

    def __call__(self, frame, now=None):
        h, w = frame.shape[:2]
        result = self.model.track(
            frame, persist=True, tracker=self.tracker, classes=self.class_ids,
            conf=self.conf, imgsz=self.imgsz, device=self.device, verbose=False,
        )[0]

        objects = []
        hands = []
        boxes = result.boxes
        if boxes is not None and boxes.id is not None:
            # segmentation models (YOLOE -seg) give object outlines for cleaner color histograms
            polygons = result.masks.xy if result.masks is not None else [None] * len(boxes)
            for xyxy, track_id, cls, conf, polygon in zip(
                    boxes.xyxy.tolist(), boxes.id.int().tolist(),
                    boxes.cls.int().tolist(), boxes.conf.tolist(), polygons):
                x1, y1, x2, y2 = xyxy
                if result.names[cls] == self.hand_class:
                    hands.append(TrackedObject(track_id=track_id, class_name=self.hand_class,
                                               confidence=conf, box=(x1, y1, x2, y2),
                                               center=((x1 + x2) / 2 / w, (y1 + y2) / 2 / h)))
                    continue
                objects.append(TrackedObject(
                    track_id=track_id,
                    class_name=result.names[cls],
                    confidence=conf,
                    box=(x1, y1, x2, y2),
                    center=((x1 + x2) / 2 / w, (y1 + y2) / 2 / h),
                    box_normalized=(x1 / w, y1 / h, x2 / w, y2 / h),
                    hist=color_histogram(frame, xyxy, polygon),
                ))
        self.hands = hands
        now = time.time() if now is None else now
        if self.embedder is not None:
            if not self.embedder.ready:
                self.embedder.observe_background(frame)
            wanted = self.registry.embed_requests(objects, now)[:self.embeds_per_frame]
            crops = [_crop(frame, obj.box) for obj in wanted]
            wanted = [(obj, c) for obj, c in zip(wanted, crops) if c is not None]
            if wanted:
                for (obj, crop), feat in zip(wanted, self.embedder([c for _, c in wanted])):
                    obj.embedding, obj.crop = feat, crop
        return self.registry.update(objects, now, frame)


def _crop(frame, box):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, w), min(y2, h)
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    return frame[y1:y2, x1:x2].copy()
