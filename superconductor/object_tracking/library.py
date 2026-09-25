"""Library of known objects, saved as plain files so it can be browsed, edited
and shared:

    library/
      bulbasaur/
        object.json      name, parameter mappings (editable), detector votes
        embeddings.npy   (n, d) appearance features, one per saved view
        hist.npy         color histogram fingerprint
        views/00.jpg ... what the object looked like (one crop per embedding)

A mapping says what the object does, and when:

    {"trigger": "near", "prompt": "airy wooden nature flute melody"}
    {"trigger": "held", "parameter": "fire taiko drum"}   # a [[parameters]] entry by name
    {"trigger": "visible", "kind": "temperature", "min": 0.9, "max": 1.4}

triggers: "near" (1 at the reference point, fading with distance),
"held" (a hand is on the object), "visible" (1 while in view).
"""
import json
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

TRIGGERS = ("near", "held", "visible")
MAX_VIEWS = 24


@dataclass
class LibraryEntry:
    name: str
    embeddings: np.ndarray  # (n, d) raw features
    hist: np.ndarray
    mappings: list = field(default_factory=list)
    detected_as: dict = field(default_factory=dict)
    embedder: str = None
    path: Path = None

    @property
    def views(self):
        return sorted((self.path / "views").glob("*.jpg")) if self.path else []


def _valid_name(name):
    return (bool(name) and not name.startswith((".", "#")) and not name.isdigit()
            and "/" not in name and "\\" not in name)


class ObjectLibrary:
    def __init__(self, root, embedder_name=None):
        self.root = Path(root)
        self.embedder_name = embedder_name
        self.entries = {}
        self.load()

    def load(self):
        self.entries = {}
        if not self.root.exists():
            return
        for meta_path in sorted(self.root.glob("*/object.json")):
            d = meta_path.parent
            meta = json.loads(meta_path.read_text())
            emb = np.load(d / "embeddings.npy") if (d / "embeddings.npy").exists() else np.zeros((0, 0))
            if self.embedder_name and meta.get("embedder") not in (None, self.embedder_name):
                print(f"library: {d.name} was saved with {meta.get('embedder')}, "
                      f"now using {self.embedder_name}; ignoring its embeddings (save it again)")
                emb = np.zeros((0, 0))
            hist = np.load(d / "hist.npy") if (d / "hist.npy").exists() else None
            name = meta.get("name", d.name)
            self.entries[name] = LibraryEntry(
                name=name, embeddings=emb.astype(np.float32), hist=hist,
                mappings=meta.get("mappings", []), detected_as=meta.get("detected_as", {}),
                embedder=meta.get("embedder"), path=d)
        if self.entries:
            print(f"library: {', '.join(f'{n} ({len(e.embeddings)} views)' for n, e in self.entries.items())}")

    def save(self, name, embeddings, views, hist, detected_as=None, mappings=None):
        """Add views of an object (creating it if new). `embeddings` (n, d) and
        `views` (n BGR crops) are appended to what is already saved, keeping the
        newest MAX_VIEWS. `mappings` (if given) replace the saved mappings."""
        if not _valid_name(name):
            raise ValueError(f"invalid object name {name!r}")
        entry = self.entries.get(name)
        d = self.root / name
        (d / "views").mkdir(parents=True, exist_ok=True)
        old = entry.embeddings if entry is not None and len(entry.embeddings) else None
        embeddings = np.asarray(embeddings, np.float32).reshape(len(views), -1)
        if old is None or old.shape[1] != embeddings.shape[1]:
            all_emb = embeddings  # new object, or saved with another embedder: start over
            for f in (d / "views").glob("*.jpg"):
                f.unlink()
        else:
            all_emb = np.vstack([old, embeddings])
        # views are numbered in the same order as the embedding rows
        start = len(all_emb) - len(embeddings)
        for i, crop in enumerate(views):
            cv2.imwrite(str(d / "views" / f"{start + i:02d}.jpg"), crop)
        if len(all_emb) > MAX_VIEWS:
            drop = len(all_emb) - MAX_VIEWS
            all_emb = all_emb[drop:]
            files = sorted((d / "views").glob("*.jpg"))
            for f in files[:drop]:
                f.unlink()
            for i, f in enumerate(sorted((d / "views").glob("*.jpg"))):
                f.rename(d / "views" / f"{i:02d}.jpg")
        np.save(d / "embeddings.npy", all_emb)
        if hist is not None:
            np.save(d / "hist.npy", hist)
        if mappings is None:
            mappings = entry.mappings if entry else []
        detected = dict(entry.detected_as) if entry else {}
        for k, v in (detected_as or {}).items():
            detected[k] = round(detected.get(k, 0.0) + v, 2)
        self.entries[name] = LibraryEntry(name=name, embeddings=all_emb, hist=hist, mappings=mappings,
                                          detected_as=detected, embedder=self.embedder_name, path=d)
        self._write_meta(self.entries[name])
        return self.entries[name]

    def set_mappings(self, name, mappings):
        entry = self.entries[name]
        entry.mappings = mappings
        self._write_meta(entry)

    def _write_meta(self, entry):
        meta = {"name": entry.name, "mappings": entry.mappings, "detected_as": entry.detected_as,
                "embedder": entry.embedder, "views": len(entry.embeddings),
                "saved": time.strftime("%Y-%m-%d %H:%M:%S")}
        (entry.path / "object.json").write_text(json.dumps(meta, indent=1))

    def forget(self, name):
        """Remove an object (moved to <library>/.trash, not deleted)."""
        entry = self.entries.pop(name)
        trash = self.root / ".trash"
        trash.mkdir(parents=True, exist_ok=True)
        target = trash / f"{name}-{time.strftime('%Y%m%d-%H%M%S')}"
        shutil.move(str(entry.path), str(target))
        return target

    def contact_sheet(self, name, size=120, columns=6):
        """All saved views of an object tiled into one image (for `show`)."""
        views = [cv2.imread(str(p)) for p in self.entries[name].views]
        return tile([v for v in views if v is not None], size, columns)


def tile(crops, size=120, columns=6):
    if not crops:
        return None
    thumbs = []
    for crop in crops:
        h, w = crop.shape[:2]
        s = size / max(h, w)
        thumb = cv2.resize(crop, (max(int(w * s), 1), max(int(h * s), 1)))
        pad = np.full((size, size, 3), 40, np.uint8)
        y, x = (size - thumb.shape[0]) // 2, (size - thumb.shape[1]) // 2
        pad[y:y + thumb.shape[0], x:x + thumb.shape[1]] = thumb
        thumbs.append(pad)
    while len(thumbs) % columns and len(thumbs) > columns:
        thumbs.append(np.full((size, size, 3), 40, np.uint8))
    rows = [np.hstack(thumbs[i:i + columns]) for i in range(0, len(thumbs), columns)]
    return np.vstack(rows)
