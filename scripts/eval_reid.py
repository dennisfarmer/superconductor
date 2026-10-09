"""Compare re-identification embedders on a recorded clip.

1. Record a clip (objects leave the frame and come back, cross each other, get
   covered by hands, ...):

    python scripts/eval_reid.py record var/reid.mp4 --seconds 120 [--iphone]

2. Run the tracker on it with each embedder, from an empty library (so each
   physical object should get exactly one id, and keep it when it comes back):

    python scripts/eval_reid.py eval var/reid.mp4 --objects 3

Prints, per embedder: ids created (ideally == --objects; each extra id is an
object that wasn't recognized when it came back), how many returns were
recognized, frames per id, and ms per frame. Wrong-id cases (an object getting
another object's id) show in the annotated video <clip>_<embedder>.mp4.
"""
import sys
import tempfile
import time
from pathlib import Path

import click
import cv2
import numpy as np

from superconductor.collab import DEFAULT_CONFIG, find_camera, load_config
from superconductor.object_tracking import ObjectLibrary, ObjectTracker

EMBEDDERS = {"dinov2": ("dinov2_vits14", 224), "yolo11n-cls": ("yolo11n-cls.pt", 128)}


@click.group()
def cli():
    pass


@cli.command()
@click.argument("out", type=click.Path(dir_okay=False))
@click.option("--seconds", default=120.0)
@click.option("--iphone", is_flag=True)
@click.option("--camera", default=None, type=int)
def record(out, seconds, iphone, camera):
    """Record a clip from the webcam (shown while recording; q stops)."""
    cfg, _ = load_config(DEFAULT_CONFIG)
    if camera is None:
        camera = find_camera("iPhone Camera" if iphone else cfg.get("camera", {}).get("name", "C920"))
    cap = cv2.VideoCapture(camera)
    w, h = cfg.get("camera", {}).get("resolution", (1280, 720))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    writer, start = None, time.time()
    while time.time() - start < seconds:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.01)
            continue
        if writer is None:
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), cap.get(cv2.CAP_PROP_FPS) or 30,
                                     (frame.shape[1], frame.shape[0]))
            start = time.time()
        writer.write(frame)
        cv2.imshow("recording (q stops)", cv2.flip(frame, 1))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    print(f"wrote {out}")


@cli.command("eval")
@click.argument("clip", type=click.Path(exists=True, dir_okay=False))
@click.option("--objects", default=None, type=int, help="number of physical objects in the clip")
@click.option("--embedder", "names", multiple=True, default=list(EMBEDDERS), type=click.Choice(list(EMBEDDERS)))
@click.option("--match-threshold", default=None, type=float)
@click.option("--new-threshold", default=None, type=float)
def evaluate(clip, objects, names, match_threshold, new_threshold):
    """Run the tracker on a clip with each embedder and report ids and timing."""
    cfg, _ = load_config(DEFAULT_CONFIG)
    t, identity = cfg["tracking"], dict(cfg.get("identity", {}))
    if match_threshold is not None:
        identity["match_threshold"] = match_threshold
    if new_threshold is not None:
        identity["new_threshold"] = new_threshold
    for name in names:
        model, imgsz = EMBEDDERS[name]
        with tempfile.TemporaryDirectory() as tmp:
            tracker = ObjectTracker(model=t["model"], classes=t.get("classes"), conf=t.get("conf", 0.3),
                                    imgsz=t.get("imgsz", 480), device=t.get("device"), embedder=model,
                                    embed_imgsz=imgsz, embed_device=t.get("embed_device"),
                                    embeds_per_frame=t.get("embeds_per_frame", 2), identity=identity)
            tracker.registry.attach_library(ObjectLibrary(tmp, tracker.embedder.name))
            tracker.registry.allow_new = True
            cap = cv2.VideoCapture(clip)
            fps = cap.get(cv2.CAP_PROP_FPS) or 15
            out_path = Path(clip).with_name(f"{Path(clip).stem}_{name}.mp4")
            writer, times, frame_i = None, [], 0
            seen = {}  # id -> frames in view
            returns = 0  # times an identity came back after >1 s out of view
            last = {}
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                now = frame_i / fps
                t0 = time.time()
                objs = tracker(frame, now=now)
                times.append(time.time() - t0)
                for o in objs:
                    if o.coasted:
                        continue
                    seen[o.identity] = seen.get(o.identity, 0) + 1
                    if o.identity in last and now - last[o.identity] > 1.0:
                        returns += 1
                    last[o.identity] = now
                    x1, y1, x2, y2 = map(int, o.box)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"#{o.identity}", (x1, max(y1 - 6, 14)), cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, (0, 255, 0), 2)
                cv2.putText(frame, f"{name}  t={now:.1f}s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (255, 255, 255), 2)
                if writer is None:
                    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                             (frame.shape[1], frame.shape[0]))
                writer.write(frame)
                frame_i += 1
            cap.release()
            if writer is not None:
                writer.release()
        created = len(tracker.registry.identities)
        extra = f", extra ids {created - objects}" if objects else ""
        print(f"{name}: {created} ids created{extra}; recognized on return {returns}x; "
              f"frames per id {dict(sorted(seen.items()))}; "
              f"{1000 * np.mean(times):.1f} ms/frame (p95 {1000 * np.percentile(times, 95):.1f}) "
              f"-> {out_path}", flush=True)


if __name__ == "__main__":
    sys.exit(cli())
