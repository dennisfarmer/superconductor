"""Unit tests for the web client's parts: the browser's detections in place of YOLO
(ObjectTracker.track_detections), the frame message, and the overlay the page draws.

No camera, browser, detector or MRT2 server needed. Run with
`python tests/test_web_client.py` (or pytest).
"""
import json
import struct
import tempfile
import tomllib
from pathlib import Path

import numpy as np

from superconductor_client.tracking.collab import DEFAULT_CONFIG, CollabFrontend, load_config
from superconductor_client.tracking.object_tracking import ObjectTracker
from superconductor_client.tracking.tracking_view import parse_frame_message

W, H = 960, 540


def frame(seed=0):
    return np.random.default_rng(seed).integers(0, 255, (H, W, 3), dtype=np.uint8)


def detections(t):
    """two objects moving right and a hand, as (track_id, class_name, confidence, box, polygon)"""
    return [(1, "toy", 0.8, (100 + 5 * t, 100, 200 + 5 * t, 220), None),
            (2, "stuffed toy", 0.7, (500, 300 - 3 * t, 620, 420 - 3 * t), None),
            (3, "hand", 0.9, (700, 50, 780, 130), None)]


def snapshot(objects):
    return [(o.identity, o.track_id, o.class_name, tuple(round(v, 3) for v in o.box), o.coasted, o.occluded)
            for o in objects]


def test_browser_detections_give_the_same_objects_as_detecting_here():
    here, browser = ObjectTracker(classes=["toy", "stuffed toy", "hand"]), ObjectTracker(classes=["toy"])
    here.registry.allow_new = browser.registry.allow_new = True  # unsupervised: new objects get identities
    assert here._model is None and browser._model is None  # YOLO isn't loaded until needed
    for t in range(40):
        f = frame(t)
        here._detect = lambda _frame, t=t: detections(t)  # what YOLO + BoT-SORT would find
        a = here(f, now=t * 0.05)
        b = browser.track_detections(f, detections(t), now=t * 0.05)
        assert snapshot(a) == snapshot(b), t
        assert [h.box for h in here.hands] == [h.box for h in browser.hands] == [(700, 50, 780, 130)]
    assert len({o.identity for o in a}) == 2 and browser._model is None


def message(header, jpeg):
    head = json.dumps(header).encode()
    return struct.pack("<I", len(head)) + head + jpeg


def test_frame_message():
    jpeg = b"\xff\xd8\xff\xe0 fake jpeg"
    header = {"type": "frame", "seq": 4, "session": "s", "width": W, "height": H,
              "objects": [{"track_id": 1, "class_name": "toy", "confidence": 0.5, "box": [1, 2, 3, 4]}]}
    assert parse_frame_message(message(header, jpeg)) == (jpeg, header)
    assert parse_frame_message(jpeg) == (jpeg, None)  # compatibility mode: a plain JPEG
    for bad in (b"\x01\x00", struct.pack("<I", 99) + b"{}", message({"type": "click"}, jpeg)):
        try:
            parse_frame_message(bad)
        except ValueError:
            continue
        raise AssertionError(f"accepted {bad!r}")


def client(tmp):
    """a CollabFrontend without music, browser detection, DINO or describe server"""
    cfg, mapper = load_config(DEFAULT_CONFIG)
    cfg["tracking"].update(embedder=None, library_dir=str(Path(tmp) / "library"), mode="unsupervised")
    cfg["web"] = {"port": 0, "detection": "container", "web_dir": str(Path(tmp) / "web")}
    cfg["describe"] = {"server_url": "http://127.0.0.1:9"}  # nothing listens there
    return CollabFrontend(cfg, mapper, no_music=True)


def test_overlay_has_what_the_page_draws():
    with tempfile.TemporaryDirectory() as tmp:
        c = client(tmp)
        for t in range(30):
            objects = [{"track_id": tid, "class_name": cls, "confidence": conf, "box": list(box)}
                       for tid, cls, conf, box, _ in detections(t)]
            ov = c.step(frame(t), {"type": "frame", "seq": t, "session": "page-1", "objects": objects}, now=t * 0.05)
        json.dumps(ov)  # what goes to the page
        assert ov["type"] == "overlay" and ov["seq"] == 29 and (ov["width"], ov["height"]) == (W, H)
        assert ov["status"]["detector"] == "browser"
        by_class = {o["class_name"]: o for o in ov["objects"]}
        assert set(by_class) == {"toy", "stuffed toy"}
        toy = by_class["toy"]
        assert toy["box"] == [245.0, 100.0, 345.0, 220.0] and toy["center"] == [295.0, 160.0]  # frame pixels, unmirrored
        assert toy["label"].startswith(f"#{toy['identity']} toy") and toy["color"].startswith("#")
        assert ov["hands"] == [[700.0, 50.0, 780.0, 130.0]]
        assert 0 <= ov["reference"]["x"] <= 1 and ov["reference"]["radius"] > 0
        assert ov["mrt2"]["rows"] and all(r["color"].startswith("#") for r in ov["mrt2"]["rows"])

        # a new page (session) numbers its tracks from 1 again: they must not inherit the old identities
        before = {o["identity"] for o in ov["objects"]}
        ov = c.step(frame(31), {"type": "frame", "seq": 0, "session": "page-2", "objects": [
            {"track_id": 1, "class_name": "toy", "confidence": 0.8, "box": [800, 400, 900, 500]}]}, now=31 * 0.05)
        fresh = [o for o in ov["objects"] if not o["coasted"]]
        assert all(o["track_id"] not in range(1, 4) for o in fresh)
        c.tracker.registry.save()


def test_prompts_page_puts_parameters_first():
    html = (Path(__file__).resolve().parents[1] / "superconductor_client" / "tracking" / "tracking_view.html").read_text()
    assert "Superconductor - Prompts (MusicCoCa)" in html
    assert html.index('data-panel="parameters"') < html.index('data-panel="objects"')
    assert 'aria-selected="true">Parameters' in html
    assert 'id="add-param"' in html and "Add new parameter" in html
    assert 'min="0" max="1" step="0.01"' in html
    assert html.count('data-key="name"') == 1  # objects have a name; a parameter's prompt is its name
    assert html.count('data-key="trigger"') == 1  # objects only; a parameter has a weight slider
    assert 'id="list"' in html


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    for name, fn in tests:
        fn()
        print(f"ok  {name}")
    print(f"{len(tests)} passed")
