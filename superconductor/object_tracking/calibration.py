"""Guided labeling: "Touch the bulbasaur", "Touch the chimchar", ...

Starts from an empty identity registry. For each label in turn, waits until a
detected hand overlaps one object for `dwell` seconds (or the object is clicked
in the window), then assigns that label to the object's identity (capturing its
appearance in the current lighting), confirms on screen, and moves on.

When all labels are assigned, the set of identities is closed: unlabeled
identities are dropped (library objects are kept) and, unless
`allow_new_after`, no new ones are created.
"""
import time


def _overlap(a, b):
    """intersection area of two (x1, y1, x2, y2) boxes"""
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return max(w, 0) * max(h, 0)


class Calibration:
    def __init__(self, labels, registry, dwell=1.0, confirm_seconds=1.5, allow_new_after=False):
        """`allow_new_after`: keep creating identities for unknown objects once
        calibration is done (unsupervised mode)"""
        self.labels = list(labels)
        self.allow_new_after = allow_new_after
        self.registry = registry
        self.dwell = dwell
        self.confirm_seconds = confirm_seconds
        self.index = len(self.labels)  # inactive until start()
        self._candidate = None  # (identity, since)
        self._confirmed = None  # (text, until)

    @property
    def active(self):
        return self.index < len(self.labels) or self._confirmed is not None

    @property
    def progress(self):
        """0..1 dwell progress on the current candidate (for drawing)"""
        if self._candidate is None:
            return 0.0
        return min((time.time() - self._candidate[1]) / self.dwell, 1.0)

    @property
    def candidate(self):
        return self._candidate[0] if self._candidate else None

    def start(self):
        self.registry.reset(keep_library=True)
        self.registry.allow_new = True
        self.index = 0
        self._candidate = None
        self._confirmed = None
        print(f"calibration: touch the {self.labels[0]}")

    def _finish(self):
        self.registry.close(set(self.labels))
        self.registry.allow_new = self.allow_new_after
        print(f"calibration complete: {[(i.id, i.label) for i in self.registry.identities]}")

    def message(self):
        if self._confirmed is not None:
            return self._confirmed[0]
        if self.index < len(self.labels):
            return f"Touch the {self.labels[self.index]}"
        return None

    def _assign(self, obj, now):
        label = self.labels[self.index]
        self.registry.set_label(obj.identity, label, hist=obj.hist)
        print(f"calibration: {label} identified (#{obj.identity})")
        self.index += 1
        self._candidate = None
        done = "  -  calibration complete" if self.index == len(self.labels) else ""
        self._confirmed = (f"{label} identified (#{obj.identity}){done}", now + self.confirm_seconds)
        if done:
            self._finish()

    def click(self, point, objects):
        """point: normalized (x, y) in unflipped frame coords"""
        if self.index >= len(self.labels) or self._confirmed is not None:
            return
        for obj in objects:
            x1, y1, x2, y2 = obj.box_normalized
            if x1 <= point[0] <= x2 and y1 <= point[1] <= y2:
                self._assign(obj, time.time())
                return

    def update(self, objects, hands, now=None):
        now = time.time() if now is None else now
        if self._confirmed is not None:
            if now < self._confirmed[1]:
                return
            self._confirmed = None
            if self.index < len(self.labels):
                print(f"calibration: touch the {self.labels[self.index]}")
        if self.index >= len(self.labels):
            return

        # object with the largest hand overlap (already-labeled objects excluded)
        labeled = {i.id for i in self.registry.identities if i.user_label in self.labels[:self.index]}
        touched, best = None, 0.0
        for obj in objects:
            if obj.identity in labeled:
                continue
            area = sum(_overlap(obj.box, hand.box) for hand in hands)
            if area > best:
                touched, best = obj, area

        if touched is None:
            self._candidate = None
        elif self._candidate is None or self._candidate[0] != touched.identity:
            self._candidate = (touched.identity, now)
        elif now - self._candidate[1] >= self.dwell:
            self._assign(touched, now)
