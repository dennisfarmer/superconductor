"""Object library: every object the app has seen, cached as plain files so ids,
names, descriptions and instruments survive restarts:

    library/
      3/
        object.json      {"id": 3, "name": "lamp" | null, "description": "...",
                          "instrument": "...", "trigger": "near", "embedder": ...,
                          "color": "#rrggbb" (mean color), "user_color": "#rrggbb" | null, ...}
        embeddings.npy   (n, d) appearance features, one per distinct view
        views/00.jpg ... the crop for each embedding row
        hist.npy         color histogram (used only without an embedder)
      combos.json        [{"objects": [3, 5], "prompt": "...", "distance": 0.05, "trigger": "near"}]
      settings.json      {"test_beat": false}  (page-wide switches)

Ids are permanent and never reused, even after an object is deleted (moved to
library/.trash). Folders from the old name-keyed layout (library/bulbasaur/)
are converted on load. Features saved with another embedder are recomputed
from the views (`refresh`).
"""
import json
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np


@dataclass
class LibraryEntry:
    id: int
    name: str = None
    description: str = None
    instrument: str = None
    trigger: str = "near"
    embeddings: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), np.float32))
    hist: np.ndarray = None
    detected_as: dict = field(default_factory=dict)
    embedder: str = None
    created: str = None
    color: tuple = None  # BGR, mean color of the object's pixels
    user_color: tuple = None  # BGR, picked on the objects page
    path: Path = None

    @property
    def view_paths(self):
        return sorted((self.path / "views").glob("*.jpg")) if self.path else []

    def load_views(self):
        return [cv2.imread(str(p)) for p in self.view_paths][:len(self.embeddings)]


class ObjectLibrary:
    def __init__(self, root, embedder_name=None):
        self.root = Path(root)
        self.embedder_name = embedder_name
        self.entries = {}  # id -> LibraryEntry
        self._migrate()
        self.load()

    def load(self):
        self.entries = {}
        if not self.root.exists():
            return
        for meta_path in self.root.glob("*/object.json"):
            if not meta_path.parent.name.isdigit():
                continue
            entry = self._read(meta_path.parent)
            self.entries[entry.id] = entry
        if self.entries:
            print("library: " + ", ".join(f"#{e.id} {e.name or ''} ({len(e.embeddings)} views)".replace("  ", " ")
                                          for e in sorted(self.entries.values(), key=lambda e: e.id)))

    def _read(self, d):
        meta = json.loads((d / "object.json").read_text())
        emb = np.load(d / "embeddings.npy") if (d / "embeddings.npy").exists() else np.zeros((0, 0))
        hist = np.load(d / "hist.npy") if (d / "hist.npy").exists() else None
        return LibraryEntry(id=int(meta["id"]), name=meta.get("name"), description=meta.get("description"),
                            instrument=meta.get("instrument"), trigger=meta.get("trigger", "near"),
                            embeddings=emb.astype(np.float32),
                            hist=hist, detected_as=meta.get("detected_as", {}),
                            embedder=meta.get("embedder"), created=meta.get("created"),
                            color=hex_to_bgr(meta.get("color")), user_color=hex_to_bgr(meta.get("user_color")),
                            path=d)

    def max_id(self):
        """highest id ever used, including deleted objects"""
        ids = list(self.entries)
        trash = self.root / ".trash"
        if trash.exists():
            ids += [int(p.name.split("-")[0]) for p in trash.iterdir() if p.name.split("-")[0].isdigit()]
        return max(ids, default=0)

    def refresh(self, embedder):
        """Recompute the features of objects saved with another embedder from their views."""
        for entry in self.entries.values():
            if entry.embedder == embedder.name and len(entry.embeddings):
                continue
            views = [v for v in (cv2.imread(str(p)) for p in entry.view_paths) if v is not None]
            if not views:
                continue
            print(f"library: re-embedding #{entry.id} {entry.name or ''} ({len(views)} views, "
                  f"{entry.embedder} -> {embedder.name})")
            entry.embeddings = embedder(views)
            entry.embedder = embedder.name
            np.save(entry.path / "embeddings.npy", entry.embeddings)
            self._write_meta(entry)

    def save(self, ident):
        """Write an Identity (identity.py): metadata, embeddings and views."""
        d = self.root / str(ident.id)
        (d / "views").mkdir(parents=True, exist_ok=True)
        entry = self.entries.get(ident.id) or LibraryEntry(id=ident.id, created=time.strftime("%Y-%m-%d %H:%M:%S"),
                                                           path=d)
        entry.name, entry.description, entry.instrument = ident.user_label, ident.description, ident.instrument
        entry.trigger = ident.trigger
        entry.color = None if ident.color is None else tuple(int(round(v)) for v in ident.color)
        entry.user_color = ident.user_color
        entry.detected_as = {k: round(v, 2) for k, v in ident.votes.most_common(3)}
        entry.hist = ident.hist
        if ident.gallery and all(v is not None for v in ident.views):
            for f in (d / "views").glob("*.jpg"):
                f.unlink()
            for i, crop in enumerate(ident.views):
                cv2.imwrite(str(d / "views" / f"{i:02d}.jpg"), crop)
            entry.embeddings = np.stack(ident.gallery).astype(np.float32)
            entry.embedder = self.embedder_name
            np.save(d / "embeddings.npy", entry.embeddings)
        if ident.hist is not None:
            np.save(d / "hist.npy", ident.hist)
        self.entries[ident.id] = entry
        self._write_meta(entry)
        return entry

    def save_meta(self, ident):
        """Write only name / description / instrument (cheap; for edits from the page)."""
        entry = self.entries.get(ident.id)
        if entry is None:
            return self.save(ident)
        entry.name, entry.description, entry.instrument = ident.user_label, ident.description, ident.instrument
        entry.trigger = ident.trigger
        entry.color = None if ident.color is None else tuple(int(round(v)) for v in ident.color)
        entry.user_color = ident.user_color
        self._write_meta(entry)
        return entry

    def _write_meta(self, entry):
        meta = {"id": entry.id, "name": entry.name, "description": entry.description,
                "instrument": entry.instrument, "trigger": entry.trigger, "detected_as": entry.detected_as,
                "embedder": entry.embedder, "views": len(entry.embeddings),
                "color": bgr_to_hex(entry.color), "user_color": bgr_to_hex(entry.user_color),
                "created": entry.created, "saved": time.strftime("%Y-%m-%d %H:%M:%S")}
        entry.path.mkdir(parents=True, exist_ok=True)
        (entry.path / "object.json").write_text(json.dumps(meta, indent=1))

    def load_combos(self):
        path = self.root / "combos.json"
        return json.loads(path.read_text()) if path.exists() else []

    def save_combos(self, combos):
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "combos.json").write_text(json.dumps(combos, indent=1))

    def load_settings(self):
        path = self.root / "settings.json"
        return json.loads(path.read_text()) if path.exists() else {}

    def save_settings(self, settings):
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "settings.json").write_text(json.dumps(settings, indent=1))

    def delete(self, id):
        """Remove an object (moved to <library>/.trash, not deleted)."""
        entry = self.entries.pop(id, None)
        if entry is None or not entry.path.exists():
            return None
        trash = self.root / ".trash"
        trash.mkdir(parents=True, exist_ok=True)
        target = trash / f"{id}-{time.strftime('%Y%m%d-%H%M%S')}"
        shutil.move(str(entry.path), str(target))
        return target

    def _migrate(self):
        """library/<name>/ (old layout) -> library/<id>/ with "name" set"""
        if not self.root.exists():
            return
        old = sorted(p for p in self.root.iterdir()
                     if p.is_dir() and not p.name.isdigit() and not p.name.startswith(".")
                     and (p / "object.json").exists())
        for d in old:
            self.entries = {int(p.name): None for p in self.root.iterdir() if p.name.isdigit()}
            new_id = self.max_id() + 1
            meta = json.loads((d / "object.json").read_text())
            mappings = [m for m in meta.get("mappings", []) if m.get("prompt")]
            target = self.root / str(new_id)
            shutil.move(str(d), str(target))
            new_meta = {"id": new_id, "name": meta.get("name", d.name), "description": None,
                        "instrument": mappings[0]["prompt"] if mappings else None,
                        "detected_as": meta.get("detected_as", {}), "embedder": meta.get("embedder"),
                        "created": meta.get("saved")}
            (target / "object.json").write_text(json.dumps(new_meta, indent=1))
            print(f"library: {d.name}/ -> {new_id}/ (new layout)")
        self.entries = {}


def bgr_to_hex(bgr):
    """(b, g, r) -> "#rrggbb" (None stays None)"""
    return None if bgr is None else "#{2:02x}{1:02x}{0:02x}".format(*(int(round(v)) for v in bgr))


def hex_to_bgr(text):
    """"#rrggbb" -> (b, g, r); None for None or anything malformed"""
    if not isinstance(text, str) or len(text) != 7 or text[0] != "#":
        return None
    try:
        r, g, b = (int(text[i:i + 2], 16) for i in (1, 3, 5))
    except ValueError:
        return None
    return b, g, r
