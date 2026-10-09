"""Persistent object identities on top of BoT-SORT track ids.

BoT-SORT drops a track after `track_buffer` frames without a match and gives
the object a new track id when it reappears; open-vocabulary class labels can
also flicker between frames. Neither should change which instrument an object
plays, so each physical object gets a permanent `Identity` (`#3`), kept in the
object library across restarts (library.py):

- appearance: a gallery of appearance embeddings, one per distinct view
  (DINOv2, see embedder.py). Without an embedder, a color histogram instead.
- name: optional, typed by a person on the objects page; otherwise the
  detector's majority class name ("toy").
- description / instrument: written once by the describe server (VLM), the
  instrument can be typed over on the objects page.

The detector can therefore be class-agnostic (e.g. YOLOE with a single "toy"
prompt): it only has to find objects, identity comes from appearance.

Per frame:
1. A bound track keeps its identity. About once per `verify_interval` seconds
   (and right away when it reappears after more than that) its embedding is
   checked: if it looks more like another identity than its own, and less
   than `match_threshold` like its own, the track is unbound (BoT-SORT swapped
   ids when objects crossed, or revived a lost track on the wrong object).
   Otherwise the embedding is kept as a new view if it shows a new angle
   (the gallery learns how the object looks as it is turned), as long as it
   doesn't look more like another identity and no hand covers the object.
   An identity whose object was covered by a hand, next to another object or
   not detected is "disturbed" (the tracker may have swapped it with another
   one, e.g. two plushies quickly swapped by hand): as soon as it is clear
   again it is embedded and checked, first against the other objects in view
   (if two look more like each other's identities, they are swapped back
   right away), then as above. It learns no views until that check passes, so
   a swap can't put one object's views into another's gallery.
2. A new track collects `min_hits` frames of evidence, including `samples`
   embeddings.
3. Ready tracks are matched jointly (Hungarian assignment) to the identities
   that are not in view, including every object cached from earlier sessions,
   so two objects can't swap. Score = mean over the track's embeddings of the
   best cosine similarity to the identity's gallery. Score >= `match_threshold`:
   same object.
4. Unsupervised (`allow_new=True`): score < `new_threshold` against every
   identity -> a new identity (saved to the library). In between it keeps
   collecting evidence; after `max_hits` frames it goes to the best identity
   above `new_threshold`, or becomes new.

Closed set (`allow_new=False`, calibration mode): identities come only from
calibration ("touch the ...", calibration.py) or the library.

Calibration mode persists to `store_dir/identities.json` (labels + color
histograms) instead of the library.
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
HIST_MOMENTUM = 0.02  # running average of an identity's color histogram
COLOR_MOMENTUM = 0.02  # running average of an identity's mean color, after the first 1/n samples
COLOR_SEEDED = 30  # sample count given to a color loaded from the library (so a frame can't jump it)


def color_histogram(frame, box, polygon=None):
    """Normalized hue/saturation histogram of the object's pixels (mask if available)."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, w), min(y2, h)
    if x2 <= x1 or y2 <= y1:
        return None
    crop = frame[y1:y2, x1:x2]
    mask = _polygon_mask(crop, (x1, y1), polygon)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    # hue is unreliable for dark pixels (noise in dim light): leave them out
    bright = cv2.inRange(hsv, (0, 0, MIN_VALUE), (180, 255, 255))
    mask = bright if mask is None else cv2.bitwise_and(mask, bright)
    if cv2.countNonZero(mask) < 50:
        return None
    hist = cv2.calcHist([hsv], [0, 1], mask, [H_BINS, S_BINS], [0, 180, 0, 256])
    return cv2.normalize(hist, None, 1.0, 0, cv2.NORM_L1).astype(np.float32)


def mean_color(frame, box, polygon):
    """Mean BGR color of the object's pixels inside its segmentation outline
    (background removed); None without an outline, since the box alone is
    mostly background."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, w), min(y2, h)
    if x2 <= x1 or y2 <= y1:
        return None
    crop = frame[y1:y2, x1:x2]
    mask = _polygon_mask(crop, (x1, y1), polygon)
    if mask is None:
        return None
    return np.array(cv2.mean(crop, mask)[:3], np.float32)


def _polygon_mask(crop, origin, polygon):
    """mask of a segmentation outline (frame coords) within `crop`, or None if too small"""
    if polygon is None or len(polygon) < 3:
        return None
    mask = np.zeros(crop.shape[:2], np.uint8)
    cv2.fillPoly(mask, [np.asarray(polygon, np.int32) - origin], 255)
    return mask if cv2.countNonZero(mask) >= 50 else None


def box_gap(a, b):
    """distance between two (x1, y1, x2, y2) boxes, edge to edge (0 if they overlap)"""
    dx = max(a[0] - b[2], b[0] - a[2], 0.0)
    dy = max(a[1] - b[3], b[1] - a[3], 0.0)
    return max(dx, dy)


def similarity(a, b):
    """color histogram similarity: 1 = identical distribution, 0 = disjoint"""
    if a is None or b is None:
        return 0.0
    return 1.0 - cv2.compareHist(a, b, cv2.HISTCMP_BHATTACHARYYA)


@dataclass
class Identity:
    id: int
    hist: np.ndarray = None
    votes: Counter = field(default_factory=Counter)
    user_label: str = None  # the name a person gave it
    description: str = None  # from the describe server (VLM)
    instrument: str = None  # prompt it plays (suggested from the description, or typed)
    trigger: str = "near"  # when it plays: near | held | visible (see mapper.py)
    gallery: list = field(default_factory=list)  # embeddings of distinct views
    views: list = field(default_factory=list)  # crop per gallery entry
    track_id: int = None
    last_seen: float = 0.0
    last_check: float = 0.0  # last appearance verification of the bound track
    bad_checks: int = 0
    disturbed: bool = False  # occluded / crowded / missed since its last check (possible swap)
    recent: tuple = None  # (latest embedding, time of the first) of clear views since it was disturbed
    last_obj: object = None  # last detection (TrackedObject), for coasting
    best_confidence: float = 0.0
    library: bool = False  # cached in the object library
    dirty: bool = False  # new views not yet written to the library
    color: np.ndarray = None  # running mean BGR of its (background removed) pixels
    color_n: int = 0  # samples in `color`
    user_color: tuple = None  # BGR picked on the objects page; replaces `color`

    @property
    def display_color(self):
        """BGR color it is drawn with (picked, else its mean color), or None"""
        if self.user_color is not None:
            return tuple(int(v) for v in self.user_color)
        return None if self.color is None else tuple(int(round(v)) for v in self.color)

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
        return np.stack(self.gallery) if self.gallery else None

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
    color: np.ndarray = None  # running mean BGR
    color_n: int = 0
    embeddings: list = field(default_factory=list)
    crops: list = field(default_factory=list)
    votes: Counter = field(default_factory=Counter)
    last_seen: float = 0.0
    best_score: float = None  # best match so far (for display)
    best_label: str = None


class IdentityRegistry:
    def __init__(self, match_threshold=0.5, new_threshold=None, min_hits=8, max_hits=45,
                 samples=3, sample_every=3, verify_interval=1.0, gallery_size=24, novelty=0.9,
                 coast_seconds=1.0, coast_near=0.06, switch_margin=0.05, swap_margin=0.1, swap_k=3,
                 store_dir=None, save_interval=5.0, embedder=None, library=None):
        """
        `match_threshold`: min score to recognize a new track as a known identity
        `new_threshold`: (unsupervised) max best score for a track to become a
            new identity (default: `match_threshold` - 0.1)
        `min_hits`: frames of evidence a new track needs before it is decided
        `max_hits`: after this many frames, an ambiguous track is decided anyway
        `samples`, `sample_every`: embeddings taken of a new track (every
            `sample_every` frames) before deciding who it is
        `verify_interval`: seconds between appearance checks of bound tracks
        `gallery_size`, `novelty`: views kept per identity; a view is added only
            if its similarity to all kept views is below `novelty`
        `coast_seconds`, `coast_near`: an object that stops being detected keeps
            its last box ("coasting", `obj.coasted`) for `coast_seconds`, and for
            as long as that box is within `coast_near` (fraction of the frame
            width, box edge to box edge) of a detected object: when two objects
            come together the detector often sees only one of them. Objects
            this close are also "crowded" (see the module docstring).
        `switch_margin`: a bound track moves to another identity when it looks
            more like it by this much (2 checks in a row, 1 after a disturbance);
            a new track isn't given a free identity when one in view looks more
            like it by this much
        `swap_margin`, `swap_k`: two bound tracks are swapped back when the sum
            of their similarities to each other's identities exceeds the sum to
            their own by `swap_margin`; similarity here is the mean of the
            `swap_k` best gallery matches, so one stray view can't decide it
        `library`: ObjectLibrary to load identities from and cache them in
            (unsupervised mode); `store_dir`: identities.json (calibration mode)
        """
        self.match_threshold = match_threshold
        self.new_threshold = match_threshold - 0.1 if new_threshold is None else new_threshold
        self.min_hits = min_hits
        self.max_hits = max_hits
        self.samples = samples
        self.sample_every = sample_every
        self.verify_interval = verify_interval
        self.gallery_size = gallery_size
        self.novelty = novelty
        self.coast_seconds = coast_seconds
        self.coast_near = coast_near
        self.switch_margin = switch_margin
        self.swap_margin = swap_margin
        self.swap_k = swap_k
        self.embedder = embedder
        self.allow_new = False
        self.identities = []
        self.pending = []  # objects of unbound tracks in the last frame (for drawing)
        self._by_track = {}  # BoT-SORT track id -> Identity
        self._pending = {}  # unbound track id -> _Pending
        self.on_new = None  # callback(identity) when a new identity is cached
        self.store_dir = Path(store_dir) if store_dir else None
        self.save_interval = save_interval
        self._last_save = 0.0
        self._dirty = False
        self.library = None
        self.load()
        if library is not None:
            self.attach_library(library)

    # ---------- persistence ----------

    def attach_library(self, library):
        """Every cached object becomes a (not yet seen) identity."""
        self.library = library
        for entry in library.entries.values():
            if any(i.id == entry.id for i in self.identities):
                continue
            self.identities.append(Identity(
                id=entry.id, hist=entry.hist, votes=Counter(entry.detected_as),
                user_label=entry.name, description=entry.description, instrument=entry.instrument,
                trigger=entry.trigger,
                gallery=list(entry.embeddings), views=entry.load_views(), library=True,
                color=None if entry.color is None else np.asarray(entry.color, np.float32),
                color_n=COLOR_SEEDED if entry.color is not None else 0, user_color=entry.user_color))

    def flush(self):
        """Write new views of cached identities to the library."""
        if self.library is None:
            return
        for ident in self.identities:
            if ident.library and ident.dirty:
                self.library.save(ident)
                ident.dirty = False

    def remove(self, ident):
        """Forget an identity (and move it to the library's trash)."""
        if ident.track_id is not None:
            self._by_track.pop(ident.track_id, None)
        self.identities.remove(ident)
        if self.library is not None and ident.library:
            self.library.delete(ident.id)

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
        identity). `hist` replaces its color fingerprint (current lighting)."""
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
        self._json_path.write_text(json.dumps([i.to_json() for i in self.identities if i.hist is not None],
                                              indent=1))
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
        used = max((i.id for i in self.identities), default=0)
        if self.library is not None:
            used = max(used, self.library.max_id())
        return used + 1

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

    def _bind(self, track_id, ident):
        if ident.track_id is not None and ident.track_id != track_id:
            self._by_track.pop(ident.track_id, None)
        ident.track_id = track_id
        ident.bad_checks = 0
        ident.disturbed, ident.recent = False, None  # just decided on its appearance
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

    @property
    def _embedding(self):
        return self.embedder is not None and self.embedder.ready

    def _embed_sim(self, embedding, ident, k=1):
        """best cosine similarity of `embedding` to `ident`'s gallery (with `k`
        > 1: mean of the `k` best)"""
        if not self._embedding or embedding is None:
            return None
        gallery = ident.embeddings
        if gallery is None or gallery.shape[1] != len(embedding):
            return None
        if k == 1:
            return self.embedder.similarity(embedding, gallery)
        sims = self.embedder.normalize(gallery) @ self.embedder.normalize(embedding)
        return float(np.mean(np.sort(sims)[-k:]))

    def _add_view(self, ident, embedding, crop):
        """keep `embedding` as a view of `ident` if it shows something new"""
        if embedding is None or self.embedder is None:
            return
        known = ident.embeddings
        if known is not None and known.shape[1] != len(embedding):
            ident.gallery, ident.views = [], []  # saved with another embedder: start over
            known = None
        if known is not None and self.embedder.similarity(embedding, known) >= self.novelty:
            return
        ident.gallery.append(embedding)
        ident.views.append(crop)
        ident.dirty = True
        while len(ident.gallery) > self.gallery_size:
            # drop the most redundant view (highest similarity to another one)
            g = self.embedder.normalize(np.stack(ident.gallery))
            s = g @ g.T
            np.fill_diagonal(s, -1)
            drop = int(np.argmax(s.max(axis=1)))
            del ident.gallery[drop]
            del ident.views[drop]

    def score(self, pending, ident):
        """how much the evidence of an unbound track looks like `ident` (0..1):
        appearance embeddings if both have them, else color"""
        sims = [s for s in (self._embed_sim(e, ident) for e in pending.embeddings) if s is not None]
        if sims:
            return float(np.mean(sims))
        return similarity(pending.hist, ident.hist)

    def embed_requests(self, objects, now=None):
        """Which objects need an appearance embedding this frame, most urgent
        first: new tracks being classified, tracks that just came back and
        disturbed tracks that are clear again (possible swap), then bound tracks
        due for a check."""
        if not self._embedding:
            return []
        now = time.time() if now is None else now
        self._mark_crowded(objects)
        new, check = [], []
        for obj in objects:
            ident = self._by_track.get(obj.track_id)
            if ident is None:
                p = self._pending.get(obj.track_id)
                taken = len(p.embeddings) if p else 0
                hits = p.hits if p else 0
                if taken < self.samples and hits >= taken * self.sample_every:
                    new.append((taken, obj))
            elif now - ident.last_seen > self.verify_interval:
                new.append((-1, obj))  # came back: make sure it is the same object
            elif ident.disturbed and not (obj.occluded or obj.crowded):
                new.append((-1, obj))  # clear again after a possible swap
            elif now - ident.last_check >= self.verify_interval:
                check.append((ident.last_check, obj))
        new.sort(key=lambda x: x[0])
        check.sort(key=lambda x: x[0])
        return [o for _, o in new] + [o for _, o in check]

    # ---------- per frame ----------

    def _mark_crowded(self, objects):
        """`obj.crowded`: within `coast_near` of another object (box edge to box edge)"""
        for o in objects:
            o.crowded = o.box_normalized is not None and any(
                p is not o and p.box_normalized is not None and
                box_gap(o.box_normalized, p.box_normalized) <= self.coast_near for p in objects)

    def _swap_back(self, claimed, now):
        """A disturbed track that looks more like another bound track's identity
        than its own, and vice versa (summed over both), was swapped with it by
        the tracker: exchange their identities. Returns the swapped objects."""
        swapped = set()
        for a_id, a in list(claimed.items()):
            A = self._by_track[a.track_id]
            if (not A.disturbed or a.embedding is None or a.occluded or a.crowded
                    or a.track_id in swapped):
                continue
            best, partner = self.swap_margin, None
            for b_id, b in claimed.items():
                B = self._by_track[b.track_id]
                if b is a or b.track_id in swapped or B.recent is None:
                    continue
                e = B.recent[0]
                sims = [self._embed_sim(x, I, self.swap_k) for x, I in
                        ((a.embedding, A), (e, B), (a.embedding, B), (e, A))]
                if None in sims:
                    continue
                gain = sims[2] + sims[3] - sims[0] - sims[1]
                if gain > best:
                    best, partner = gain, b
            if partner is None:
                continue
            b = partner
            B = self._by_track[b.track_id]
            print(f"tracks {a.track_id} and {b.track_id} swapped #{A.id} {A.label} and "
                  f"#{B.id} {B.label} (by {best:.2f}): swapping back")
            A.track_id, B.track_id = b.track_id, a.track_id
            self._by_track[a.track_id], self._by_track[b.track_id] = B, A
            claimed[A.id], claimed[B.id] = b, a
            a.identity, b.identity = B.id, A.id
            for I in (A, B):
                I.last_check, I.bad_checks = now, 0
                I.disturbed, I.recent = False, None  # their embeddings belong to the other one
            swapped.update((a.track_id, b.track_id))
        return swapped

    def _verify(self, obj, ident, now):
        """The identity a bound track belongs to: `ident`, or another identity it
        looks more like by `switch_margin` (the tracker swapped it, or revived it
        on the wrong object). Relative, not an absolute threshold: two plushies in
        the same scene can be as similar to each other as two views of one."""
        ident.last_check = now
        own = self._embed_sim(obj.embedding, ident)
        if own is None:
            return ident
        others = [(s, i) for i in self.identities if i is not ident
                  for s in [self._embed_sim(obj.embedding, i)] if s is not None]
        other, nearest = max(others, key=lambda x: x[0], default=(None, None))
        if other is not None and other > own + self.switch_margin:
            reappeared = now - ident.last_seen > self.verify_interval
            ident.bad_checks += 2 if reappeared or ident.disturbed else 1
            if ident.bad_checks >= 2:
                print(f"track {obj.track_id} looks like #{nearest.id} {nearest.label}, not "
                      f"#{ident.id} {ident.label} ({other:.2f} vs {own:.2f})")
                return nearest
            return ident
        ident.bad_checks = 0
        # the tracker followed this object here, so a view that looks different is a new
        # angle of it, not another object: keep it (bounded by gallery_size), unless it
        # looks more like another known object, a hand covers it, its crop may show another
        # object, or it hasn't been checked since it may have been swapped
        if (other is None or own >= other) and not (obj.occluded or obj.crowded or ident.disturbed):
            self._add_view(ident, obj.embedding, obj.crop)
        return ident

    def _new_identity(self, obj, pending):
        ident = Identity(id=self._next_id(), hist=pending.hist, color=pending.color, color_n=pending.color_n)
        self.identities.append(ident)
        self._bind(obj.track_id, ident)
        if self.library is not None:
            ident.library = True
            self.library.save(ident)
            ident.dirty = False
            if self.on_new is not None:
                self.on_new(ident)
        return ident

    def update(self, objects, now=None, frame=None):
        """Sets `obj.identity` and replaces `obj.class_name` with the identity's label.
        Returns only objects that have an identity; `self.pending` has the rest."""
        now = time.time() if now is None else now
        claimed = {}  # identity id -> obj
        unbound = []

        # 1. tracks already bound keep their identity, unless it no longer looks like them
        self._mark_crowded(objects)
        for obj in objects:
            ident = self._by_track.get(obj.track_id)
            if ident is None or ident.id in claimed:
                unbound.append(obj)
                continue
            claimed[ident.id] = obj
            obj.identity = ident.id
            if obj.occluded or obj.crowded:
                ident.disturbed, ident.recent = True, None
            elif obj.embedding is not None:
                ident.recent = (obj.embedding, now if ident.recent is None else ident.recent[1])
        for ident in self.identities:
            if ident.track_id is not None and ident.id not in claimed:
                ident.disturbed, ident.recent = True, None  # missed: may come back as another
        swapped = self._swap_back(claimed, now)
        # a disturbed object waits (a frame or so, at most `verify_interval`) for disturbed
        # ones that are clear but not embedded yet: it may have been swapped with one of them
        waiting = any(self._by_track[o.track_id].disturbed and self._by_track[o.track_id].recent is None
                      and not (o.occluded or o.crowded) for o in claimed.values())
        for ident_id, obj in list(claimed.items()):
            ident = self._by_track[obj.track_id]
            if obj.embedding is None or obj.track_id in swapped or (
                    ident.disturbed and waiting and ident.recent is not None
                    and now - ident.recent[1] < self.verify_interval):
                continue
            target = self._verify(obj, ident, now)
            if target is ident:
                if not (obj.occluded or obj.crowded):
                    ident.disturbed = False
                continue
            self._unbind(ident)
            del claimed[ident_id]
            if target.id not in claimed:
                # it is the object it looks like (which isn't in view elsewhere): move it there
                self._bind(obj.track_id, target)
                target.last_check = now
                claimed[target.id] = obj
                obj.identity = target.id
            else:
                # that one is in view too: one of the two is wrong; re-identify this one and
                # re-check the other right away
                target.disturbed, target.recent = True, None
                obj.identity = None
                unbound.append(obj)

        # 2. accumulate evidence for unbound tracks
        ready = []
        for obj in unbound:
            if obj.hist is None and obj.embedding is None:
                continue
            p = self._pending.setdefault(obj.track_id, _Pending())
            p.hits += 1
            p.last_seen = now
            if obj.hist is not None:
                p.hist = obj.hist.copy() if p.hist is None else p.hist + (obj.hist - p.hist) / p.hits
            if obj.color is not None and not obj.occluded:
                p.color_n += 1
                p.color = obj.color.copy() if p.color is None else p.color + (obj.color - p.color) / p.color_n
            p.votes[obj.class_name] += obj.confidence
            if obj.embedding is not None:
                p.embeddings.append(obj.embedding)
                p.crops.append(obj.crop)
            if p.hits >= self.min_hits and (p.embeddings or not self._embedding):
                ready.append(obj)
        for track_id in [t for t, p in self._pending.items() if now - p.last_seen > 5.0]:
            del self._pending[track_id]

        # 3. jointly assign ready tracks to identities not in view (Hungarian), but only
        #    to the identity they look most like: if that is one in view, its track may be
        #    on the wrong object, so it is re-checked first
        free = [i for i in self.identities if i.id not in claimed]
        in_view = [i for i in self.identities if i.id in claimed]
        if ready and free:
            score = np.array([[self.score(self._pending[o.track_id], i) for i in free] for o in ready])
            rows, cols = linear_sum_assignment(-score)
            for r, c in zip(rows, cols):
                obj, ident = ready[r], free[c]
                p = self._pending[obj.track_id]
                p.best_score, p.best_label = float(score[r].max()), free[int(score[r].argmax())].label
                rival = max(in_view, key=lambda i: self.score(p, i), default=None)
                if rival is not None and self.score(p, rival) > score[r, c] + self.switch_margin:
                    if now - rival.last_check >= self.verify_interval:
                        rival.disturbed, rival.recent = True, None
                    continue
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
                best = max((self.score(p, i) for i in candidates), default=0.0)
                if best < self.new_threshold or (p.hits >= self.max_hits and best < self.match_threshold):
                    ident = self._new_identity(obj, p)
                    print(f"new object: #{ident.id} (best match {best:.2f})")
                    claimed[ident.id] = obj
                    obj.identity = ident.id

        out = []
        by_id = {i.id: i for i in self.identities}
        for ident_id, obj in claimed.items():
            ident = by_id[ident_id]
            # slow color adaptation, only from look-alike observations
            if obj.hist is not None:
                if ident.hist is None:
                    ident.hist = obj.hist
                elif similarity(obj.hist, ident.hist) >= self.match_threshold:
                    ident.hist = (1 - HIST_MOMENTUM) * ident.hist + HIST_MOMENTUM * obj.hist
            # mean color: plain average of the first samples, then slow adaptation; not while
            # a hand covers it (skin would tint it)
            if obj.color is not None and not obj.occluded:
                ident.color_n += 1
                if ident.color is None:
                    ident.color = obj.color.copy()
                else:
                    ident.color = ident.color + max(1 / ident.color_n, COLOR_MOMENTUM) * (obj.color - ident.color)
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
            self.flush()
            self._last_save = now
        return out

    def pending_info(self, track_id):
        """(frames seen, best score, best label) of an unbound track, for drawing"""
        p = self._pending.get(track_id)
        return (p.hits, p.best_score, p.best_label) if p else (0, None, None)
