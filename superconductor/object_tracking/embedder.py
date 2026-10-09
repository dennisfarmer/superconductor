"""Appearance embeddings for recognizing objects (and re-recognizing them when
they come back).

Default: DINOv2 ViT-S/14 (self-supervised, 384-d CLS feature, ~5.5 ms per crop
on MPS, ~26 ms on CPU). Its features separate individual objects, not just
ImageNet classes, so two different plushies of the same kind still differ.
"yolo11n-cls.pt" (a 256-d ImageNet classifier feature, ~3 ms on CPU) is still
supported for comparison (scripts/eval_reid.py).

Raw features of any two crops can be fairly similar, so with `center=True` they
are centered on the mean feature of random crops of the current scene before
comparing. Raw (uncentered) features are what gets stored in the object library.
"""
import cv2
import numpy as np

import torch

DINO_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
DINO_STD = np.array([0.229, 0.224, 0.225], np.float32)


class Embedder:
    def __init__(self, model="dinov2_vits14", imgsz=224, device=None, center=True,
                 background_crops=32, seed=0):
        """`model`: "dinov2_vits14" (torch.hub, cached after the first download)
        or an ultralytics classifier like "yolo11n-cls.pt". `imgsz`: crop size
        (a multiple of 14 for DINOv2). `center`: compare features centered on the
        scene mean (see module docstring)."""
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.imgsz = imgsz
        self.name = f"{model}@{imgsz}"
        self.dino = model.startswith("dinov2")
        if self.dino:
            self.model = torch.hub.load("facebookresearch/dinov2", model, trust_repo=True,
                                        skip_validation=True).eval().to(self.device)
        else:
            from ultralytics import YOLO
            self.model = YOLO(model)
        self.center = center
        self.background_crops = background_crops
        self._rng = np.random.default_rng(seed)
        self._background = []
        self.mean = None  # scene mean feature; None until enough background crops are seen

    @property
    def ready(self):
        return self.mean is not None or not self.center

    def __call__(self, crops):
        """(n, d) float32 raw features for a list of BGR crops"""
        if not crops:
            return np.zeros((0, 0), np.float32)
        with torch.inference_mode():
            if not self.dino:
                feats = self.model.embed(crops, imgsz=self.imgsz, device=self.device, verbose=False)
                return np.stack([f.cpu().numpy() for f in feats]).astype(np.float32)
            batch = np.stack([self._prep(c) for c in crops])
            feats = self.model(torch.from_numpy(batch).to(self.device))
            return feats.float().cpu().numpy()

    def _prep(self, crop):
        rgb = cv2.cvtColor(cv2.resize(crop, (self.imgsz, self.imgsz), interpolation=cv2.INTER_AREA),
                           cv2.COLOR_BGR2RGB)
        return ((rgb.astype(np.float32) / 255 - DINO_MEAN) / DINO_STD).transpose(2, 0, 1)

    def observe_background(self, frame, n=8):
        """Collect random crops of the scene (a few per frame, spread over the
        first frames so startup doesn't stall) until the scene mean is known."""
        if self.ready:
            return
        h, w = frame.shape[:2]
        crops = []
        for _ in range(n):
            s = int(self._rng.uniform(0.1, 0.35) * h)
            x, y = self._rng.integers(0, w - s), self._rng.integers(0, h - s)
            crops.append(frame[y:y + s, x:x + s])
        self._background.extend(self(crops))
        if len(self._background) >= self.background_crops:
            self.mean = np.mean(self._background, axis=0)
            self._background = []

    def normalize(self, feats):
        """center on the scene mean (if enabled) and L2-normalize (works on (d,) or (n, d))"""
        feats = np.asarray(feats, np.float32)
        if self.center and self.mean is not None:
            feats = feats - self.mean
        return feats / np.maximum(np.linalg.norm(feats, axis=-1, keepdims=True), 1e-8)

    def similarity(self, feat, gallery):
        """max cosine similarity of one raw feature to any raw feature in `gallery`"""
        if feat is None or gallery is None or len(gallery) == 0:
            return None
        return float(np.max(self.normalize(gallery) @ self.normalize(feat)))
