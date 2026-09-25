"""Persistent object identities on top of BoT-SORT track ids.

BoT-SORT drops a track after `track_buffer` frames without a match and gives
the object a new track id when it reappears; open-vocabulary class labels can
also flicker between frames. Neither should change which music parameter an
object controls, so each physical object gets a permanent `Identity`:

- appearance: HSV color histogram over the object's segmentation mask
  (running average), plus, with an `Embedder`, a gallery of appearance
  features (one per distinct view; see embedder.py)
- label: a name (from the object library, calibration, or `save`), otherwise
  the confidence-weighted majority vote over detector class names.

The detector can therefore be class-agnostic (e.g. YOLOE with a single "toy"
prompt): it only has to find objects, identity comes from appearance.

A new track is not trusted immediately: it collects `min_hits` frames of color
evidence and a few appearance embeddings ("classification on first
introduction"), then all such tracks are assigned jointly to the identities
that are not currently in view (Hungarian assignment), so two objects can't
swap. A running track keeps its identity, except that bound tracks are
re-checked about once per `verify_interval` seconds and unbound if they stop
looking like their identity (BoT-SORT swapped ids when objects crossed).

Two modes:
- closed set (`allow_new=False`): identities come only from calibration
  ("touch the ...", see calibration.py) or the object library; anything else
  is ignored.
- unsupervised (`allow_new=True`): a track that looks like no known identity
  (score < `new_threshold`) becomes a new identity. Between `new_threshold` and
  `match_threshold` it is ambiguous and keeps collecting evidence; after
  `max_hits` frames it goes to the best identity above `new_threshold`, or
  becomes new.

Identities are never forgotten during a session.

With `store_dir`, identities (+ a crop image each) are saved and can be reused
next session (--skip-calibration), but fingerprints from different lighting are
less reliable than recalibrating. The object library (library.py) is the
persistent store in unsupervised mode.
"""
import dataclasses
import json
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

H_BINS, S_BINS = 30, 32
MIN_VALUE = 40  # HSV brightness below which pixels are ignored


def color_histogram(frame, box, polygon=None):
    """Normalized hue/saturation histogram of the object's pixels (mask if available)."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, w), min(y2, h)
    if x2 <= x1 or y2 <= y1:
        return None
    crop = frame[y1:y2, x1:x2]
    mask = None
    if polygon is not None and len(polygon) >= 3:
        mask = np.zeros(crop.shape[:2], np.uint8)
        cv2.fillPoly(mask, [np.asarray(polygon, np.int32) - (x1, y1)], 255)
        if cv2.countNonZero(mask) < 50:
            mask = None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    # hue is unreliable for dark pixels (noise in dim light): leave them out
    bright = cv2.inRange(hsv, (0, 0, MIN_VALUE), (180, 255, 255))
    mask = bright if mask is None else cv2.bitwise_and(mask, bright)
    if cv2.countNonZero(mask) < 50:
        return None
    hist = cv2.calcHist([hsv], [0, 1], mask, [H_BINS, S_BINS], [0, 180, 0, 256])
    return cv2.normalize(hist, None, 1.0, 0, cv2.NORM_L1).astype(np.float32)


def box_gap(a, b):
    """distance between two (x1, y1, x2, y2) boxes, edge to edge (0 if they overlap)"""
    dx = max(a[0] - b[2], b[0] - a[2], 0.0)
    dy = max(a[1] - b[3], b[1] - a[3], 0.0)
    return max(dx, dy)


def similarity(a, b):
    """1 = identical color distribution, 0 = disjoint"""
    if a is None or b is None:
        return 0.0
    return 1.0 - cv2.compareHist(a, b, cv2.HISTCMP_BHATTACHARYYA)


@dataclass
class Identity:
    id: int
    hist: np.ndarray
    votes: Counter = field(default_factory=Counter)
    user_label: str = None
    track_id: int = None
    last_seen: float = 0.0
    best_confidence: float = 0.0
    library: bool = False  # loaded from / saved to the object library
    fixed: np.ndarray = None  # (n, d) embeddings from the library (not modified)
    gallery: list = field(default_factory=list)  # embeddings seen this session (distinct views)
    views: list = field(default_factory=list)  # crop per gallery entry
    last_check: float = 0.0  # last appearance verification of the bound track
    bad_checks: int = 0
    last_obj: object = None  # last detection (TrackedObject), for coasting

    @property
    def label(self):
        if self.user_label:
            return self.user_label
        return self.votes.most_common(1)[0][0] if self.votes else f"object {self.id}"

    @property
    def named(self):
        return bool(self.user_label)

    @property
    def embeddings(self):
        """all appearance features: library + this session"""
        parts = [e for e in (self.fixed,) if e is not None and len(e)]
        if self.gallery:
            parts.append(np.stack(self.gallery))
        if not parts:
            return None
        if len(parts) == 2 and parts[0].shape[1] != parts[1].shape[1]:
            return parts[1]
        return np.vstack(parts)

    def to_json(self):
        return {"id": self.id, "label": self.user_label,
                "detected_as": dict(self.votes.most_common(3)),
                "hist": self.hist.flatten().round(6).tolist()}

    @classmethod
    def from_json(cls, d):
        return cls(id=d["id"], user_label=d.get("label"),
                   votes=Counter(d.get("detected_as", {})),
                   hist=np.asarray(d["hist"], np.float32).reshape(H_BINS, S_BINS))


@dataclass
class _Pending:
    """evidence collected for a track that has no identity yet"""
    hits: int = 0
    hist: np.ndarray = None  # running mean
    embeddings: list = field(default_factory=list)
    crops: list = field(default_factory=list)
    votes: Counter = field(default_factory=Counter)
    last_seen: float = 0.0
    best_score: float = None  # best match so far (for display)
    best_label: str = None


class IdentityRegistry:
    def __init__(self, match_threshold=0.5, min_hits=8, appearance_momentum=0.02,
                 store_dir=None, save_interval=5.0, embedder=None, embed_weight=0.7,
                 new_threshold=None, max_hits=45, samples=3, sample_every=3,
                 verify_interval=1.0, unbind_threshold=0.2, gallery_size=12,
                 novelty=0.9, coast_seconds=1.0, coast_near=0.06, swap_margin=0.15,
                 reappear_gap=0.5, near_bonus=0.2):
        """
        `match_threshold`: min appearance score to bind a new track to a known
            identity. The score is color similarity, or with an `embedder`,
            `embed_weight` * embedding similarity + the rest color similarity.
        `new_threshold`: (unsupervised) max best score for a track to become a
            new identity (default: `match_threshold` - 0.1)
        `min_hits`: frames of evidence an unbound track needs before it is bound
            (or, while `allow_new`, becomes a new identity)
        `max_hits`: after this many frames, an ambiguous track is decided anyway
        `samples`, `sample_every`: embeddings taken of a new track (every
            `sample_every` frames) before deciding who it is
        `verify_interval`: seconds between embedding checks of bound tracks;
            `unbind_threshold`: embedding similarity below which (twice in a
            row) a track is considered swapped and loses its identity; also if it
            looks `swap_margin` more like another identity than its own. A track
            that comes back after `reappear_gap` seconds unseen (BoT-SORT revives
            lost tracks by position, so it can pick the wrong object) is checked
            immediately, and one bad check is enough.
        `gallery_size`, `novelty`: views kept per identity; a view is added only
            if its similarity to all kept views is below `novelty`
        `coast_seconds`, `coast_near`: an object that stops being detected keeps
            its last box ("coasting", `obj.coasted`) for `coast_seconds`, and for
            as long as that box is within `coast_near` (fraction of the frame
            width, box edge to box edge) of a detected object: when two objects
            come together the detector often sees only one of them.
        `near_bonus`: added to the score of a new track for identities whose
            last box is right there (a partly hidden object coming back into view)

        Closed set: new identities are only created while `allow_new` is True
        (during calibration, or always in unsupervised mode). Otherwise every
        detection is either one of the known objects or ignored.
        """
        self.match_threshold = match_threshold
        self.new_threshold = match_threshold - 0.1 if new_threshold is None else new_threshold
        self.min_hits = min_hits
        self.max_hits = max_hits
        self.momentum = appearance_momentum
        self.embedder = embedder
        self.embed_weight = embed_weight
        self.samples = samples
        self.sample_every = sample_every
        self.verify_interval = verify_interval
        self.unbind_threshold = unbind_threshold
        self.gallery_size = gallery_size
        self.novelty = novelty
        self.coast_seconds = coast_seconds
        self.coast_near = coast_near
        self.swap_margin = swap_margin
        self.reappear_gap = reappear_gap
        self.near_bonus = near_bonus
        self.allow_new = False
        self.identities = []
        self.pending = []  # objects of unbound tracks in the last frame (for drawing)
        self._by_track = {}  # BoT-SORT track id -> Identity
        self._pending = {}  # unbound track id -> _Pending
        self.store_dir = Path(store_dir) if store_dir else None
        self.save_interval = save_interval
        self._last_save = 0.0
        self._dirty = False
        self.load()

    # ---------- persistence ----------

    @property
    def _json_path(self):
        return self.store_dir / "identities.json"

    def load(self):
        if self.store_dir is None or not self._json_path.exists():
            return
        data = json.loads(self._json_path.read_text())
        self.identities = [Identity.from_json(d) for d in data]
        print(f"loaded {len(self.identities)} identities: "
              f"{[(i.id, i.label) for i in self.identities]}")

    def save(self):
        if self.store_dir is None or not self._dirty:
            return
        # keep labels the user edited into the file while we were running
        if self._json_path.exists():
            on_disk = {d["id"]: d.get("label") for d in json.loads(self._json_path.read_text())}
            for ident in self.identities:
                if on_disk.get(ident.id):
                    ident.user_label = on_disk[ident.id]
        self._write()

    def set_label(self, identity, label, hist=None):
        """Give `identity` the user label `label` (removing it from any other
        identity). `hist` replaces its appearance fingerprint (current lighting)."""
        for ident in self.identities:
            if ident.id == identity:
                ident.user_label = label
                if hist is not None:
                    ident.hist = hist
            elif ident.user_label == label:
                ident.user_label = None
        self._dirty = True
        self._write()

    def _write(self):
        if self.store_dir is None:
            return
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self._json_path.write_text(json.dumps([i.to_json() for i in self.identities], indent=1))
        self._dirty = False

    def _save_crop(self, ident, frame, obj):
        if self.store_dir is None or frame is None or obj.confidence <= ident.best_confidence:
            return
        ident.best_confidence = obj.confidence
        x1, y1, x2, y2 = (int(v) for v in obj.box)
        crop = frame[max(y1, 0):y2, max(x1, 0):x2]
        if crop.size:
            self.store_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(self.store_dir / f"{ident.id}.jpg"), crop)

    # ---------- identity management ----------

    def get(self, key):
        """identity by id (int, "#3", "3") or name"""
        key = str(key).strip()
        if key.lstrip("#").isdigit():
            return next((i for i in self.identities if i.id == int(key.lstrip("#"))), None)
        return next((i for i in self.identities if i.user_label == key), None)

    def _next_id(self):
        return max((i.id for i in self.identities), default=0) + 1

    def add_library(self, library):
        """Known objects from the library become (not yet seen) identities."""
        for entry in library.entries.values():
            ident = self.get(entry.name) if entry.name else None
            if ident is None:
                ident = Identity(id=self._next_id(), hist=entry.hist, user_label=entry.name)
                self.identities.append(ident)
            ident.library = True
            ident.fixed = entry.embeddings if len(entry.embeddings) else None
            if entry.hist is not None:
                ident.hist = entry.hist
            ident.votes.update(entry.detected_as)

    def close(self, labels):
        """End of calibration: keep only identities with one of `labels` (and
        library objects), create no more."""
        self.identities = [i for i in self.identities if i.user_label in labels or i.library]
        keep = {i.id for i in self.identities}
        self._by_track = {t: i for t, i in self._by_track.items() if i.id in keep}
        self.allow_new = False
        self._dirty = True
        self.save()

    def reset(self, keep_library=False):
        """Forget all identities (start of a fresh calibration); with
        `keep_library`, only the ones that aren't library objects."""
        self.identities = [i for i in self.identities if keep_library and i.library]
        keep = {i.id for i in self.identities}
        for ident in self.identities:
            ident.track_id = None
        self._by_track = {t: i for t, i in self._by_track.items() if i.id in keep}
        self._pending.clear()
        self._dirty = True

    def merge(self, src, dst):
        """Fold identity `src` into `dst` (same physical object seen twice)."""
        dst.gallery.extend(src.gallery)
        dst.views.extend(src.views)
        dst.votes.update(src.votes)
        self._trim_gallery(dst)
        if src.track_id is not None and self._by_track.get(src.track_id) is src:
            self._bind(src.track_id, dst)
        self.identities.remove(src)

    def _bind(self, track_id, ident):
        if ident.track_id is not None and ident.track_id != track_id:
            self._by_track.pop(ident.track_id, None)
        ident.track_id = track_id
        ident.bad_checks = 0
        self._by_track[track_id] = ident
        pending = self._pending.pop(track_id, None)
        if pending is not None:
            for emb, crop in zip(pending.embeddings, pending.crops):
                self._add_view(ident, emb, crop)
            ident.votes.update(pending.votes)

    def _unbind(self, ident):
        self._by_track.pop(ident.track_id, None)
        ident.track_id = None
        ident.bad_checks = 0
        ident.last_obj = None  # its last box was really another object's: don't coast it

    # ---------- appearance ----------

    def _embed_sim(self, embedding, ident):
        if self.embedder is None or embedding is None:
            return None
        return self.embedder.similarity(embedding, ident.embeddings)

    def _add_view(self, ident, embedding, crop):
        """keep `embedding` as a view of `ident` if it shows something new"""
        if embedding is None or self.embedder is None:
            return
        known = ident.embeddings
        if known is not None and len(known) and known.shape[1] == len(embedding):
            if self.embedder.similarity(embedding, known) >= self.novelty:
                return
        ident.gallery.append(embedding)
        ident.views.append(crop)
        self._trim_gallery(ident)

    def _trim_gallery(self, ident):
        while len(ident.gallery) > self.gallery_size:
            # drop the most redundant view (highest similarity to another one)
            g = self.embedder.normalize(np.stack(ident.gallery))
            s = g @ g.T
            np.fill_diagonal(s, -1)
            drop = int(np.argmax(s.max(axis=1)))
            del ident.gallery[drop]
            del ident.views[drop]

    def score(self, pending, ident, box=None):
        """how much the evidence of an unbound track looks like `ident` (0..1,
        + `near_bonus` if `box` (normalized) is where `ident` was last seen)"""
        color = similarity(pending.hist, ident.hist)
        sims = [s for s in (self._embed_sim(e, ident) for e in pending.embeddings) if s is not None]
        score = color if not sims else \
            self.embed_weight * float(np.mean(sims)) + (1 - self.embed_weight) * color
        last = ident.last_obj
        if box is not None and last is not None and last.box_normalized is not None and \
                box_gap(box, last.box_normalized) <= self.coast_near:
            score += self.near_bonus
        return score

    @property
    def _embedding(self):
        return self.embedder is not None and self.embedder.ready

    def embed_requests(self, objects, now=None):
        """Which objects need an appearance embedding this frame, most urgent
        first: new tracks being classified, then bound tracks due for a check."""
        if not self._embedding:
            return []
        now = time.time() if now is None else now
        new, check = [], []
        for obj in objects:
            ident = self._by_track.get(obj.track_id)
            if ident is None:
                p = self._pending.get(obj.track_id)
                taken = len(p.embeddings) if p else 0
                hits = p.hits if p else 0
                if taken < self.samples and hits >= taken * self.sample_every:
                    new.append((taken, obj))
            elif now - ident.last_seen > self.reappear_gap:
                new.append((-1, obj))  # came back: make sure it is the same object
            elif now - ident.last_check >= self.verify_interval:
                check.append((ident.last_check, obj))
        new.sort(key=lambda x: x[0])
        check.sort(key=lambda x: x[0])
        return [o for _, o in new] + [o for _, o in check]

    # ---------- per frame ----------

    def update(self, objects, now=None, frame=None):
        """Sets `obj.identity` and replaces `obj.class_name` with the identity's label.
        Returns only objects that have an identity; `self.pending` has the rest."""
        now = time.time() if now is None else now
        claimed = {}  # identity id -> obj
        unbound = []

        # 1. tracks already bound keep their identity, unless it no longer looks like them
        for obj in objects:
            ident = self._by_track.get(obj.track_id)
            if ident is not None and ident.id not in claimed:
                if obj.embedding is not None and self._embedding:
                    ident.last_check = now
                    sim = self._embed_sim(obj.embedding, ident)
                    other = max((s for s in (self._embed_sim(obj.embedding, i)
                                             for i in self.identities if i is not ident) if s is not None),
                                default=None)
                    wrong = sim is not None and (sim < self.unbind_threshold or
                                                 (other is not None and other - sim > self.swap_margin))
                    if wrong:
                        reappeared = now - ident.last_seen > self.reappear_gap
                        ident.bad_checks += 2 if reappeared else 1
                        if ident.bad_checks >= 2:
                            print(f"track {obj.track_id} no longer looks like #{ident.id} "
                                  f"{ident.label} ({sim:.2f}, other {other or 0:.2f}): re-identifying")
                            self._unbind(ident)
                            unbound.append(obj)
                            continue
                    else:
                        ident.bad_checks = 0
                        if sim is None or sim >= self.match_threshold:
                            self._add_view(ident, obj.embedding, obj.crop)
                claimed[ident.id] = obj
                obj.identity = ident.id
            else:
                unbound.append(obj)

        # 2. accumulate evidence for unbound tracks
        ready = []
        for obj in unbound:
            if obj.hist is None:
                continue
            p = self._pending.setdefault(obj.track_id, _Pending())
            p.hits += 1
            p.last_seen = now
            p.hist = obj.hist.copy() if p.hist is None else p.hist + (obj.hist - p.hist) / p.hits
            p.votes[obj.class_name] += obj.confidence
            if obj.embedding is not None:
                p.embeddings.append(obj.embedding)
                p.crops.append(obj.crop)
            if p.hits >= self.min_hits and (p.embeddings or not self._embedding):
                ready.append(obj)
        for track_id in [t for t, p in self._pending.items() if now - p.last_seen > 5.0]:
            del self._pending[track_id]

        # 3. jointly assign ready tracks to identities not in view (Hungarian)
        free = [i for i in self.identities if i.id not in claimed]
        if ready and free:
            score = np.array([[self.score(self._pending[o.track_id], i, o.box_normalized) for i in free]
                              for o in ready])
            rows, cols = linear_sum_assignment(-score)
            for r, c in zip(rows, cols):
                obj, ident = ready[r], free[c]
                p = self._pending[obj.track_id]
                p.best_score, p.best_label = float(score[r].max()), free[int(score[r].argmax())].label
                patient = p.hits >= self.max_hits and score[r, c] >= self.new_threshold
                if score[r, c] >= self.match_threshold or patient:
                    self._bind(obj.track_id, ident)
                    claimed[ident.id] = obj
                    obj.identity = ident.id

        # 4. unmatched tracks that look like nothing known become new identities
        if self.allow_new:
            for obj in ready:
                if obj.identity is not None:
                    continue
                p = self._pending[obj.track_id]
                # an identity in view elsewhere can't be this object, but one in view right
                # here scoring high means a duplicate box (e.g. part of it), not a new object
                candidates = [i for i in self.identities if i.id not in claimed or (
                    obj.box_normalized is not None and claimed[i.id].box_normalized is not None and
                    box_gap(obj.box_normalized, claimed[i.id].box_normalized) <= self.coast_near)]
                best = max((self.score(p, i, obj.box_normalized) for i in candidates), default=0.0)
                if best < self.new_threshold or (p.hits >= self.max_hits and best < self.match_threshold):
                    ident = Identity(id=self._next_id(), hist=p.hist)
                    self.identities.append(ident)
                    print(f"new object: identity #{ident.id} (best match {best:.2f})")
                    self._bind(obj.track_id, ident)
                    claimed[ident.id] = obj
                    obj.identity = ident.id

        out = []
        by_id = {i.id: i for i in self.identities}
        for ident_id, obj in claimed.items():
            ident = by_id[ident_id]
            # slow appearance adaptation, only from confident look-alike observations
            if obj.hist is not None:
                if ident.hist is None:
                    ident.hist = obj.hist
                elif similarity(obj.hist, ident.hist) >= self.match_threshold:
                    ident.hist = (1 - self.momentum) * ident.hist + self.momentum * obj.hist
            ident.votes[obj.class_name] += obj.confidence
            ident.last_seen = now
            self._save_crop(ident, frame, obj)
            self._dirty = True
            obj.class_name = ident.label
            ident.last_obj = obj
            out.append(obj)
        self.pending = [o for o in objects if o.identity is None]

        # 5. objects that just vanished stay at their last position for a while,
        #    and indefinitely while they are next to a detected object
        detected = [o.box_normalized for o in out if o.box_normalized is not None]
        for ident in self.identities:
            last = ident.last_obj
            if ident.id in claimed or last is None or last.box_normalized is None:
                continue
            recent = now - ident.last_seen < self.coast_seconds
            if recent or any(box_gap(last.box_normalized, b) <= self.coast_near for b in detected):
                out.append(dataclasses.replace(last, coasted=True, embedding=None, crop=None,
                                               class_name=ident.label, confidence=0.5 * last.confidence))

        if now - self._last_save > self.save_interval:
            self.save()
            self._last_save = now
        return out

    def pending_info(self, track_id):
        """(frames seen, best score, best label) of an unbound track, for drawing"""
        p = self._pending.get(track_id)
        return (p.hits, p.best_score, p.best_label) if p else (0, None, None)
