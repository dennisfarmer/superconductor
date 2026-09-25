"""Replay a scripted scene through the full collab pipeline, without a webcam.

Takes one webcam photo with the objects in it, cuts the objects out (YOLOE
masks), inpaints the background, and re-composes frames where the objects
disappear, reappear elsewhere, swap places, come together (combo) and separate.
Every frame goes through the real YOLOE + BoT-SORT + embedder + library +
mapper + drawing code, and the checks at the end say whether ids survived.

    python scripts/simulate_tracking.py --frame photo.jpg \
        --object bulbasaur:585,253,703,417 --object chimchar:359,229,474,393 \
        --out var/sim

Writes var/sim/sim.mp4 (annotated, as the app would show it) and a few stills.
The library it builds goes to <out>/library (the real library is not touched).
"""
import shutil
import sys
from pathlib import Path

import click
import cv2
import numpy as np

from superconductor.collab import DEFAULT_CONFIG, CollabFrontend, load_config

FPS = 15


def cut_out(frame, box, model):
    """(sprite BGR, alpha mask) of the object in `box`, using a YOLOE mask if one matches"""
    x1, y1, x2, y2 = box
    r = model.predict(frame, conf=0.15, imgsz=640, device="cpu", verbose=False)[0]
    best, best_iou = None, 0.0
    if r.masks is not None:
        for b, poly in zip(r.boxes.xyxy.tolist(), r.masks.xy):
            ix = max(0, min(x2, b[2]) - max(x1, b[0])) * max(0, min(y2, b[3]) - max(y1, b[1]))
            iou = ix / ((x2 - x1) * (y2 - y1) + (b[2] - b[0]) * (b[3] - b[1]) - ix)
            if iou > best_iou:
                best, best_iou = poly, iou
    mask = np.zeros(frame.shape[:2], np.uint8)
    if best is not None and best_iou > 0.5:
        cv2.fillPoly(mask, [best.astype(np.int32)], 255)
    else:
        mask[y1:y2, x1:x2] = 255
    mask = cv2.dilate(mask, np.ones((7, 7), np.uint8))
    return frame[y1:y2, x1:x2].copy(), mask[y1:y2, x1:x2].copy(), mask


def paste(canvas, sprite, alpha, center, occlude=0.0):
    """paste `sprite` centered at normalized `center`; `occlude` hides that fraction (from the right)"""
    h, w = canvas.shape[:2]
    sh, sw = sprite.shape[:2]
    x = int(center[0] * w - sw / 2)
    y = int(center[1] * h - sh / 2)
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + sw, w), min(y + sh, h)
    if x1 <= x0 or y1 <= y0:
        return
    a = alpha[y0 - y:y1 - y, x0 - x:x1 - x].astype(np.float32)[..., None] / 255
    if occlude:
        a[:, int((1 - occlude) * a.shape[1]):] = 0
    region = canvas[y0:y1, x0:x1].astype(np.float32)
    canvas[y0:y1, x0:x1] = (a * sprite[y0 - y:y1 - y, x0 - x:x1 - x] + (1 - a) * region).astype(np.uint8)


def lerp(a, b, t):
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)


def script(names, starts):
    """list of (phase, {name: center or None}, extra) per frame"""
    a, b = names
    pa, pb = starts[a], starts[b]
    frames = []

    def hold(phase, n, pos, extra=None):
        frames.extend((phase, dict(pos), extra) for _ in range(n))

    def move(phase, n, frm, to, extra=None):
        for i in range(n):
            t = (i + 1) / n
            frames.append((phase, {k: lerp(frm[k], to[k], t) if frm.get(k) and to.get(k) else to.get(k)
                                   for k in set(frm) | set(to)}, extra))

    hold("1 both visible", 30, {a: pa, b: pb})
    hold("2 hide " + a, 45, {a: None, b: pb})
    far = (0.14, 0.74)
    hold("3 " + a + " reappears elsewhere", 35, {a: far, b: pb})
    hold("4 both hidden", 40, {a: None, b: None})
    hold("5 reappear swapped", 40, {a: pb, b: pa})
    together = {a: pb, b: (pb[0] + 0.105, pb[1])}
    move("6 come together", 30, {a: pb, b: pa}, together)
    hold("7 together", 30, together)
    hold("8 overlapping", 30, {a: pb, b: (pb[0] + 0.06, pb[1] + 0.02)}, "occlude")
    move("9 separate", 20, {a: pb, b: (pb[0] + 0.06, pb[1])}, {a: (0.2, 0.62), b: (0.5, 0.66)})
    hold("10 apart", 30, {a: (0.2, 0.62), b: (0.5, 0.66)})
    return frames


@click.command()
@click.option("--frame", "frame_path", required=True, type=click.Path(exists=True))
@click.option("--object", "objects", multiple=True, required=True,
              help="name:x1,y1,x2,y2 (exactly two)")
@click.option("--distractor/--no-distractor", default=True,
              help="add an unknown toy (the second object recolored); it should get its own id")
@click.option("--config", "config_path", default=str(DEFAULT_CONFIG))
@click.option("--out", default="var/sim")
def main(frame_path, objects, distractor, config_path, out):
    out = Path(out)
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    photo = cv2.imread(frame_path)
    h, w = photo.shape[:2]

    cfg, mapper = load_config(config_path)
    cfg["tracking"]["library_dir"] = str(out / "library")
    cfg["tracking"]["mode"] = "unsupervised"
    app = CollabFrontend(cfg, mapper, no_music=True)
    yoloe = app.tracker.model

    boxes = {}
    for spec in objects:
        name, coords = spec.split(":")
        boxes[name] = tuple(int(v) for v in coords.split(","))
    names = list(boxes)
    sprites, full_mask = {}, np.zeros((h, w), np.uint8)
    for name, box in boxes.items():
        sprite, alpha, mask = cut_out(photo, box, yoloe)
        sprites[name] = (sprite, alpha)
        full_mask |= mask
    if distractor:
        sprite, alpha = sprites[names[1]]
        hsv = cv2.cvtColor(sprite, cv2.COLOR_BGR2HSV)
        hsv[..., 0] = (hsv[..., 0].astype(int) + 90) % 180  # same shape, different colors
        sprites["distractor"] = (cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR), alpha)
    background = cv2.inpaint(photo, cv2.dilate(full_mask, np.ones((15, 15), np.uint8)), 9, cv2.INPAINT_TELEA)
    starts = {n: ((b[0] + b[2]) / 2 / w, (b[1] + b[3]) / 2 / h) for n, b in boxes.items()}

    video = cv2.VideoWriter(str(out / "sim.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h))
    log = []  # (phase, {name: identity or None}, active combos, recipe)
    ids = {}  # name -> identity id (learned in phase 1)

    def run(frames, t0, dim=1.0, tag=""):
        now = t0
        for phase, pos, extra in frames:
            canvas = background.copy()
            for name, center in pos.items():
                if center is not None:
                    sprite, alpha = sprites[name]
                    occ = 0.5 if extra == "occlude" and name == names[0] else 0.0
                    paste(canvas, sprite, alpha, center, occ)
            if dim != 1.0:
                canvas = cv2.convertScaleAbs(canvas, alpha=dim)
            now += 1 / FPS
            shown = app.step(canvas, now=now)
            video.write(shown)
            # which identity is at each object's position
            seen = {}
            for name, center in pos.items():
                if center is None:
                    continue
                hit = min(app._objects, default=None,
                          key=lambda o: np.hypot(o.center[0] - center[0], o.center[1] - center[1]))
                if hit is not None and np.hypot(hit.center[0] - center[0], hit.center[1] - center[1]) < 0.08:
                    seen[name] = (hit.identity, hit.coasted)
            combos = [c.parameter for c in app.mapper.combos if c.active]
            log.append((tag + " " + phase, pos, seen, combos, app.mapper.recipe()))
        cv2.imwrite(str(out / f"{tag}last.jpg"), shown)
        return now

    frames = script(names, starts)
    # phase 1 alone first, to learn and save the two objects
    now = run(frames[:30], 1000.0, tag="A")
    for name in names:
        at = log[-1][2].get(name)
        if at:
            ids[name] = at[0]
            print(app.commands(f"save #{at[0]} {name}"))
    now = run(frames[30:], now, tag="A")
    if "distractor" in sprites:
        extra = [("11 new object", {names[0]: (0.2, 0.62), names[1]: (0.5, 0.66), "distractor": (0.36, 0.3)},
                  None)] * 30
        now = run(extra, now, tag="A")
    print(app.commands("list"))

    # new session: a fresh app that only knows the library, in dimmer light
    cfg2, mapper2 = load_config(config_path)
    cfg2["tracking"]["library_dir"] = str(out / "library")
    app2 = CollabFrontend(cfg2, mapper2, no_music=True)
    app = app2  # run() uses `app`
    run(frames[:30], now + 100, dim=0.75, tag="B")
    video.release()

    # ---------- report ----------
    phases = []
    for phase, pos, seen, combos, recipe in log:
        if not phases or phases[-1][0] != phase:
            phases.append([phase, [], [], []])
        # None = not in the scene this frame, () = in the scene but not found
        phases[-1][1].append({n: seen.get(n, ()) if c is not None else None for n, c in pos.items()})
        phases[-1][2].append(combos)
        phases[-1][3].append(recipe)
    failures = []
    library_ids = {i.user_label: i.id for i in app2.tracker.registry.identities if i.library}
    for phase, seens, combos, recipes in phases:
        tail = seens[len(seens) // 2:]  # after things settle
        summary = {}
        for name in names + (["distractor"] if "distractor" in sprites else []):
            vals = [s.get(name) for s in tail if s.get(name) is not None]
            got = [v[0] for v in vals if v]
            coasted = sum(1 for v in vals if v and v[1])
            summary[name] = (max(set(got), key=got.count) if got else None, len(got), len(vals), coasted)
        active = sum(1 for c in combos[len(combos) // 2:] if c) / max(len(combos[len(combos) // 2:]), 1)
        prompts = sorted(recipes[-1])
        print(f"{phase:32s} " + "  ".join(f"{n}=#{i} ({k}/{t}{', coasted ' + str(c) if c else ''})"
                                           for n, (i, k, t, c) in summary.items()) +
              f"  combo {active:.0%}  recipe {prompts}")
        expected = ids if phase.startswith("A ") else library_ids
        for name in names:
            i, k, t, c = summary[name]
            if t and (i != expected.get(name) or k < 0.8 * t):
                failures.append(f"{phase}: {name} was #{i} in {k}/{t} frames, expected #{expected.get(name)}")
        if "together" in phase or "overlapping" in phase:
            if active < 0.9:
                failures.append(f"{phase}: combo active only {active:.0%}")
        if "apart" in phase and active > 0:
            failures.append(f"{phase}: combo still active")
        if "new object" in phase:
            d = summary["distractor"][0]
            if d is None or d in ids.values():
                failures.append(f"{phase}: distractor got {d}")
    print("\n".join(failures) if failures else "all checks passed")
    print(f"video: {out / 'sim.mp4'}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
