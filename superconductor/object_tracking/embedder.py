"""Cheap appearance embeddings for re-identifying objects.

A small ImageNet classifier (yolo11n-cls, ~3 ms per 128px crop on CPU) gives a
256-d feature per object crop. Raw features of any two crops are very similar
(cosine ~0.75-0.95 even for unrelated objects), so they are centered on the mean
feature of random crops of the current scene before comparing: after that,
different plushies score ~0.4 and the same plushie ~0.7-0.9, even across
sessions and lighting changes.

Raw (uncentered) features are what gets stored in the object library; they are
centered with the current scene mean at comparison time.
"""
import numpy as np

import torch
from ultralytics import YOLO


class Embedder:
    def __init__(self, model="yolo11n-cls.pt", imgsz=128, device="cpu",
                 background_crops=32, seed=0):
        self.name = f"{model}@{imgsz}"
        self.model = YOLO(model)
        self.imgsz = imgsz
        self.device = device
        self.background_crops = background_crops
        self._rng = np.random.default_rng(seed)
        self._background = []
        self.mean = None  # scene mean feature; None until enough background crops are seen

    @property
    def ready(self):
        return self.mean is not None

    def __call__(self, crops):
        """(n, d) float32 raw features for a list of BGR crops"""
        if not crops:
            return np.zeros((0, 0), np.float32)
        with torch.inference_mode():
            feats = self.model.embed(crops, imgsz=self.imgsz, device=self.device, verbose=False)
        return np.stack([f.cpu().numpy() for f in feats]).astype(np.float32)

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
        """center on the scene mean and L2-normalize (works on (d,) or (n, d))"""
        feats = np.asarray(feats, np.float32)
        if self.mean is not None:
            feats = feats - self.mean
        return feats / np.maximum(np.linalg.norm(feats, axis=-1, keepdims=True), 1e-8)

    def similarity(self, feat, gallery):
        """max cosine similarity of one raw feature to any raw feature in `gallery`"""
        if feat is None or gallery is None or len(gallery) == 0:
            return None
        return float(np.max(self.normalize(gallery) @ self.normalize(feat)))
