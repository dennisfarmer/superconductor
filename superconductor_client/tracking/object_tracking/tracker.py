"""YOLO detection + BoT-SORT tracking via ultralytics.

BoT-SORT keeps a stable `track_id` per physical object across frames (and,
with ReID + a long track buffer, across short occlusions), which is what lets
each object keep controlling the same music parameter.

Two steps: `_detect` (YOLO + BoT-SORT, this process) and `track_detections`
(TrackedObjects, hands, DINO crops, identities). Tracking usually does the
first step itself (vision_js + tracker_js, see CAMERA_PROTOCOL.md) and sends its
detections; then only `track_detections` runs here and YOLO is never loaded.
"""
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import torch
from ultralytics import YOLO

from superconductor_client.torch_device import cuda_unavailable, resolve_device, tell
from superconductor_client.tracking.object_tracking.embedder import Embedder
from superconductor_client.tracking.object_tracking.identity import IdentityRegistry, color_histogram, mean_color

TRACKER_CONFIG = Path(__file__).parent / "botsort_superconductor.yaml"
HAND_OCCLUDED = 0.3  # fraction of an object's box covered by hands above which it counts as occluded


@dataclass
class TrackedObject:
    track_id: int
    class_name: str
    confidence: float
    box: tuple  # (x1, y1, x2, y2) in pixels
    center: tuple  # (x, y) normalized to [0, 1]
    box_normalized: tuple = None  # (x1, y1, x2, y2) in [0, 1]
    hist: np.ndarray = None  # color histogram (appearance fingerprint)
    color: np.ndarray = None  # mean BGR inside its segmentation outline (None without one)
    identity: int = None  # persistent id, stable across track losses (see identity.py)
    embedding: np.ndarray = None  # appearance feature, only on frames it was computed
    crop: np.ndarray = None  # image of the object (set together with `embedding`)
    coasted: bool = False  # not detected this frame: last known position (see identity.py)
    occluded: bool = False  # a hand covers much of it (its view isn't kept as a reference)
    crowded: bool = False  # next to another object (its crop may show both, see identity.py)


class ObjectTracker:
    def __init__(self, model="yoloe-11s-seg.pt", classes=("toy",), conf=0.3,
                 imgsz=480, tracker=TRACKER_CONFIG, device=None, identity_dir=None,
                 hand_class="hand", embedder=None, embed_imgsz=224, embed_device=None,
                 embeds_per_frame=2, identity=None):
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
        `embedder`: appearance model for re-identification ("dinov2_vits14", or
            "yolo11n-cls.pt"; None = color only), on `embed_device` (cuda, then
            mps, then cpu). At most `embeds_per_frame` crops are embedded per frame:
            new objects first, then periodic checks.
        `identity`: extra IdentityRegistry settings (thresholds, see identity.py)
        """
        self.device = resolve_device(device, "Detection")
        if "cpu" in (self.device, embed_device):
            # NNPACK can't start on CPUs without AVX2 (e.g. the amd64 container emulated on
            # a Mac) and logs "Could not initialize NNPACK!" on every conv; torch's other
            # CPU convolutions give the same results
            torch.backends.nnpack.set_flags(False)
        self.model_name = model
        self.classes = list(classes) if classes else None
        self._model = None  # loaded on first use (load_detector)
        self.conf = conf
        self.imgsz = imgsz
        self.tracker = str(tracker)
        self.class_ids = None
        self.embedder = Embedder(embedder, imgsz=embed_imgsz, device=embed_device) if embedder else None
        self.embeds_per_frame = embeds_per_frame
        self.registry = IdentityRegistry(store_dir=identity_dir, embedder=self.embedder,
                                         **(identity or {}))
        self.hand_class = hand_class
        self.hands = []

    @property
    def model(self):
        return self.load_detector()

    def load_detector(self):
        """the YOLO model, loaded the first time it is needed (frames without detections)"""
        if self._model is None:
            model = YOLO(self.model_name)
            if self.classes:
                if "yoloe" in Path(self.model_name).name.lower():
                    model.set_classes(self.classes)
                else:
                    name_to_id = {n: i for i, n in model.names.items()}
                    unknown = [c for c in self.classes if c not in name_to_id]
                    if unknown:
                        raise ValueError(f"{unknown} not in {self.model_name} classes: {sorted(name_to_id)}")
                    self.class_ids = [name_to_id[c] for c in self.classes]
            self._model = model
        return self._model

    def __call__(self, frame, now=None):
        """detect and track in this process, then everything after (track_detections)"""
        return self.track_detections(frame, self._detect(frame), now)

    def _detect(self, frame):
        """YOLO + BoT-SORT: [(track_id, class_name, confidence, (x1, y1, x2, y2), polygon or None)]"""
        while True:
            try:
                result = self.model.track(
                    frame, persist=True, tracker=self.tracker, classes=self.class_ids,
                    conf=self.conf, imgsz=self.imgsz, device=self.device, verbose=False,
                )[0]
                break
            except Exception as exc:
                if self.device == "cpu" or not cuda_unavailable(exc):
                    raise
                tell("Detection")
                self.device = "cpu"
        boxes = result.boxes
        if boxes is None or boxes.id is None:
            return []
        # segmentation models (YOLOE -seg) give object outlines for cleaner color histograms
        # and the mean color (background removed)
        polygons = result.masks.xy if result.masks is not None else [None] * len(boxes)
        return [(track_id, result.names[cls], conf, tuple(xyxy), polygon) for xyxy, track_id, cls, conf, polygon in zip(
            boxes.xyxy.tolist(), boxes.id.int().tolist(), boxes.cls.int().tolist(), boxes.conf.tolist(), polygons)]

    def track_detections(self, frame, detections, now=None):
        """Detections of one frame -> TrackedObjects with identities. `detections`:
        (track_id, class_name, confidence, (x1, y1, x2, y2) in pixels, polygon or None),
        from `_detect` or from the camera page."""
        h, w = frame.shape[:2]
        objects = []
        hands = []
        for track_id, class_name, conf, xyxy, polygon in detections:
            x1, y1, x2, y2 = xyxy
            if class_name == self.hand_class:
                hands.append(TrackedObject(track_id=track_id, class_name=self.hand_class,
                                           confidence=conf, box=(x1, y1, x2, y2),
                                           center=((x1 + x2) / 2 / w, (y1 + y2) / 2 / h)))
                continue
            objects.append(TrackedObject(
                track_id=track_id,
                class_name=class_name,
                confidence=conf,
                box=(x1, y1, x2, y2),
                center=((x1 + x2) / 2 / w, (y1 + y2) / 2 / h),
                box_normalized=(x1 / w, y1 / h, x2 / w, y2 / h),
                hist=color_histogram(frame, xyxy, polygon),
                color=mean_color(frame, xyxy, polygon),
            ))
        self.hands = hands
        for obj in objects:
            area = (obj.box[2] - obj.box[0]) * (obj.box[3] - obj.box[1])
            covered = sum(_overlap(obj.box, hand.box) for hand in hands)
            obj.occluded = area > 0 and covered / area > HAND_OCCLUDED
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


def _overlap(a, b):
    """intersection area of two (x1, y1, x2, y2) boxes"""
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return max(w, 0) * max(h, 0)


def _crop(frame, box):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, w), min(y2, h)
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    return frame[y1:y2, x1:x2].copy()
