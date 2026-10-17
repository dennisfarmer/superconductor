"""Reference outputs for the JS tracker, from Ultralytics' own BOTSORT.

Runs synthetic detection sequences through ultralytics.trackers.BOTSORT, configured from
superconductor/object_tracking/botsort_superconductor.yaml (plus per-scenario overrides), and
writes inputs and outputs to fixtures/*.json for parity.test.js. Detections are float32 (as
Ultralytics gets them), stored in their shortest decimal form: Math.fround restores them exactly. Also records lap.lapjv results on
random cost matrices for the assignment solver.

    python tracker_js/test/make_fixtures.py      (in sc_env, from superconductor_client_web)
"""
import json
from pathlib import Path
from types import SimpleNamespace

import lap
import numpy as np
import ultralytics
import yaml
from ultralytics.engine.results import Boxes
from ultralytics.trackers.bot_sort import BOTSORT
from ultralytics.trackers.utils import matching

HERE = Path(__file__).parent
OUT = HERE / "fixtures"
CONFIG = HERE.parents[1] / "superconductor" / "object_tracking" / "botsort_superconductor.yaml"
SHAPE = (720, 1280)


def tracker_args(**overrides):
    cfg = yaml.safe_load(CONFIG.read_text())
    cfg.update(overrides)
    return SimpleNamespace(**cfg)


def run(frames, args):
    """frames: list of float32 arrays (n, 6) = x1, y1, x2, y2, score, cls"""
    tracker = BOTSORT(args)
    outputs = []
    for det in frames:
        tracks = tracker.update(Boxes(det.reshape(-1, 6), SHAPE), None)
        outputs.append([[round(float(v), 3) for v in t] for t in np.asarray(tracks).reshape(-1, 8)])
    return outputs


class Scene:
    """Boxes moving with constant velocity plus jitter; detections can be missed or noisy."""

    def __init__(self, rng, n, size=(60, 160), speed=4.0):
        self.rng = rng
        self.c = rng.uniform((100, 100), (SHAPE[1] - 100, SHAPE[0] - 100), (n, 2))
        self.wh = rng.uniform(size[0], size[1], (n, 2))
        self.v = rng.normal(0, speed, (n, 2))
        self.cls = rng.integers(0, 3, n)

    def step(self):
        self.c += self.v
        for k in range(2):  # bounce off the edges
            lim = SHAPE[1 - k]
            out = (self.c[:, k] < 50) | (self.c[:, k] > lim - 50)
            self.v[out, k] *= -1
        self.wh *= self.rng.normal(1, 0.01, self.wh.shape)

    def detections(self, visible, scores, jitter=2.0):
        rows = []
        for i in np.flatnonzero(visible):
            c = self.c[i] + self.rng.normal(0, jitter, 2)
            wh = self.wh[i] * self.rng.normal(1, 0.02, 2)
            rows.append([*(c - wh / 2), *(c + wh / 2), scores[i], self.cls[i]])
        return rows


def scenario(seed, n, frames, miss=0.05, spurious=0.05, score=(0.3, 0.95), occlude=0.0,
             occlusion_len=(10, 60), speed=4.0, duplicate=0.0, zero_size=0.0, empty=0.0):
    rng = np.random.default_rng(seed)
    scene = Scene(rng, n, speed=speed)
    hidden_until = np.zeros(n, int)
    out = []
    for f in range(frames):
        scene.step()
        start = (rng.random(n) < occlude) & (hidden_until <= f)
        hidden_until[start] = f + rng.integers(*occlusion_len, start.sum())
        visible = (hidden_until <= f) & (rng.random(n) >= miss)
        rows = scene.detections(visible, rng.uniform(*score, n))
        if rng.random() < spurious:  # a false detection somewhere
            c = rng.uniform((0, 0), SHAPE[::-1])
            wh = rng.uniform(20, 120, 2)
            rows.append([*(c - wh / 2), *(c + wh / 2), rng.uniform(0.1, 0.5), rng.integers(0, 3)])
        if rows and rng.random() < duplicate:  # the same object detected twice
            r = list(rows[rng.integers(len(rows))])
            r[:4] = [v + rng.normal(0, 3) for v in r[:4]]
            r[4] = rng.uniform(0.1, 0.9)
            rows.append(r)
        if rows and rng.random() < zero_size:
            r = list(rows[rng.integers(len(rows))])
            r[2] = r[0]
            rows.append(r)
        if rng.random() < empty:
            rows = []
        rng.shuffle(rows)
        out.append(np.asarray(rows, np.float32).reshape(-1, 6))
    return out


SCENARIOS = {
    # few objects, like the webcam scenes it runs on
    "steady": (dict(seed=1, n=4, frames=400), {}),
    "occlusions": (dict(seed=2, n=5, frames=600, occlude=0.01, miss=0.1), {}),
    "fast": (dict(seed=3, n=4, frames=300, speed=15.0, miss=0.1), {}),
    # scores straddling every threshold: second association, new_track_thresh, unconfirmed tracks
    "low_scores": (dict(seed=4, n=6, frames=400, score=(0.05, 0.45), spurious=0.3, duplicate=0.2), {}),
    # short buffer: lost tracks are removed (and the one-frame removed-but-still-lost window)
    "short_buffer": (dict(seed=5, n=5, frames=500, occlude=0.03, occlusion_len=(3, 20), miss=0.15),
                     dict(track_buffer=8)),
    "degenerate": (dict(seed=6, n=3, frames=300, zero_size=0.2, empty=0.1, duplicate=0.3), {}),
    "fuse_off": (dict(seed=7, n=5, frames=300, miss=0.1, score=(0.15, 0.9)), dict(fuse_score=False)),
    # many objects, crowded and crossing
    "crowded": (dict(seed=8, n=30, frames=300, speed=8.0, miss=0.1, occlude=0.01, spurious=0.3,
                     duplicate=0.1, score=(0.1, 0.95)), {}),
}


def assignment_cases(rng, count=300):
    """lap.lapjv as Ultralytics calls it, on random IoU-like cost matrices"""
    cases = []
    for _ in range(count):
        n, m = rng.integers(0, 12, 2)
        cost = rng.uniform(0, 1, (n, m))
        cost[rng.random((n, m)) < 0.5] = 1.0  # no overlap
        thresh = float(rng.choice([0.5, 0.7, 0.8]))
        matches, ua, ub = matching.linear_assignment(cost, thresh)
        cases.append(dict(n=int(n), m=int(m), thresh=thresh, cost=cost.ravel().tolist(),
                          matches=sorted([int(a), int(b)] for a, b in matches)))
    return cases


def main():
    OUT.mkdir(exist_ok=True)
    meta = dict(ultralytics=ultralytics.__version__, lap=lap.__version__)
    for name, (scene_kw, overrides) in SCENARIOS.items():
        frames = scenario(**scene_kw)
        args = tracker_args(**overrides)
        data = dict(meta=meta, config={k: getattr(args, k) for k in (
            "track_high_thresh", "track_low_thresh", "new_track_thresh", "track_buffer", "match_thresh",
            "fuse_score")}, frames=[[[float(str(v)) for v in d] for d in f] for f in frames], outputs=run(frames, args))
        (OUT / f"{name}.json").write_text(json.dumps(data, separators=(",", ":")))
        print(f"{name}: {len(frames)} frames, {sum(len(f) for f in frames)} detections, "
              f"{len({int(t[4]) for o in data['outputs'] for t in o})} track ids")
    (OUT / "assignment.json").write_text(json.dumps(dict(meta=meta, cases=assignment_cases(
        np.random.default_rng(0))), separators=(",", ":")))


if __name__ == "__main__":
    main()
