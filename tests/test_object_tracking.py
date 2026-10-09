"""Unit tests for identities, the object library, mappings and combos.

No camera or detector needed: detections are synthetic and the embedder is a
stand-in with the same interface. Run with `python tests/test_object_tracking.py`
(or pytest).
"""
import tempfile
from pathlib import Path

import numpy as np

import cv2

from superconductor.object_tracking import (Combo, IdentityRegistry, ObjectLibrary, Parameter, ParameterMapper,
                                            TrackedObject)
from superconductor.object_tracking.identity import H_BINS, S_BINS

W, H = 1000, 1000
rng = np.random.default_rng(0)
LOOKS = {name: rng.normal(size=16).astype(np.float32) for name in ("bulbasaur", "chimchar", "pikachu")}


def hist(peak):
    h = np.full((H_BINS, S_BINS), 1e-4, np.float32)
    h[peak % H_BINS, 20] = 1.0
    return h / h.sum()


COLORS = {"bulbasaur": hist(3), "chimchar": hist(10), "pikachu": hist(15)}


class FakeEmbedder:
    name = "fake"
    ready = True

    def normalize(self, feats):
        feats = np.asarray(feats, np.float32)
        return feats / np.linalg.norm(feats, axis=-1, keepdims=True)

    def similarity(self, feat, gallery):
        if gallery is None or len(gallery) == 0:
            return None
        return float(np.max(self.normalize(gallery) @ self.normalize(feat)))


def obj(track_id, what, x, y, size=100, embed=True):
    box = (x - size / 2, y - size / 2, x + size / 2, y + size / 2)
    return TrackedObject(track_id=track_id, class_name="toy", confidence=0.6, box=box,
                         center=(x / W, y / H), box_normalized=tuple(v / W for v in box),
                         hist=COLORS[what],
                         embedding=LOOKS[what] + rng.normal(scale=0.1, size=16).astype(np.float32)
                         if embed else None,
                         crop=np.zeros((10, 10, 3), np.uint8) if embed else None)


def registry(library=None):
    reg = IdentityRegistry(embedder=FakeEmbedder(), match_threshold=0.55, new_threshold=0.45, library=library)
    reg.allow_new = True
    return reg


def run(reg, frames, t0=0.0):
    """frames: list of lists of obj(...) specs; returns the output of the last frame"""
    out, now = [], t0
    for objs in frames:
        now += 1 / 15
        out = reg.update([obj(*o) for o in objs], now)
    return out, now


def ids(out):
    return {o.track_id: o.identity for o in out if not o.coasted}


def test_new_objects_get_ids_and_come_back_after_occlusion():
    reg = registry()
    out, now = run(reg, [[(1, "bulbasaur", 300, 500), (2, "chimchar", 700, 500)]] * 10)
    first = ids(out)
    assert len(set(first.values())) == 2
    # both hidden, then back with *new* track ids in each other's places
    out, now = run(reg, [[]] * 60, now)
    out, now = run(reg, [[(7, "chimchar", 300, 500), (8, "bulbasaur", 700, 500)]] * 10, now)
    again = ids(out)
    assert again[8] == first[1] and again[7] == first[2]


def test_unknown_object_gets_new_id_known_one_does_not():
    reg = registry()
    out, now = run(reg, [[(1, "bulbasaur", 300, 500)]] * 10)
    out, now = run(reg, [[(1, "bulbasaur", 300, 500), (2, "pikachu", 700, 500)]] * 10, now)
    assert len(reg.identities) == 2
    assert len(set(ids(out).values())) == 2


def test_swapped_track_is_corrected():
    reg = registry()
    out, now = run(reg, [[(1, "bulbasaur", 300, 500), (2, "chimchar", 700, 500)]] * 10)
    first = ids(out)
    # both vanish, the tracker revives the old track ids on the wrong objects
    out, now = run(reg, [[]] * 15, now)
    out, now = run(reg, [[(1, "chimchar", 300, 500), (2, "bulbasaur", 700, 500)]] * 12, now)
    fixed = ids(out)
    assert fixed[1] == first[2] and fixed[2] == first[1]


def test_quick_swap_by_hand_is_corrected_at_once_and_learns_no_wrong_views():
    reg = registry()
    out, now = run(reg, [[(1, "bulbasaur", 300, 500), (2, "chimchar", 700, 500)]] * 10)
    first = ids(out)
    galleries = {i.id: len(i.gallery) for i in reg.identities}
    # picked up and swapped within half a second; the tracker keeps the track ids in place
    held = [obj(1, "bulbasaur", 500, 500, embed=False), obj(2, "chimchar", 560, 500, embed=False)]
    for o in held:
        o.occluded = True
    for _ in range(4):
        now += 1 / 15
        reg.update(held, now)
    out, now = run(reg, [[(1, "chimchar", 300, 500), (2, "bulbasaur", 700, 500)]], now)
    fixed = ids(out)
    assert fixed[1] == first[2] and fixed[2] == first[1]
    assert {i.id: len(i.gallery) for i in reg.identities} == galleries


def test_similar_objects_swapped_one_at_a_time_are_corrected():
    # two plushies that look alike (similarity ~0.7, above match_threshold), and the
    # detector sees only one of them at a time
    shared = rng.normal(size=16).astype(np.float32)
    look_b, look_c = LOOKS["bulbasaur"], LOOKS["chimchar"]
    LOOKS["bulbasaur"], LOOKS["chimchar"] = 2.5 * shared + look_b, 2.5 * shared + look_c
    try:
        reg = registry()
        out, now = run(reg, [[(1, "bulbasaur", 300, 500)]] * 10)
        bulb = ids(out)[1]
        out, now = run(reg, [[(1, "bulbasaur", 300, 500), (2, "chimchar", 700, 500)]] * 10, now)
        chim = ids(out)[2]
        # chimchar out of view; bulbasaur picked up, chimchar put down where it was,
        # and the tracker revives track 1 on it
        out, now = run(reg, [[(1, "bulbasaur", 300, 500)]] * 3, now)
        out, now = run(reg, [[]] * 5, now)
        out, now = run(reg, [[(1, "chimchar", 300, 500)]], now)
        assert ids(out) == {1: chim}
        # bulbasaur comes back with a new track: it gets its own id, not the free one by default
        out, now = run(reg, [[(1, "chimchar", 300, 500), (3, "bulbasaur", 700, 500)]] * 10, now)
        assert ids(out) == {1: chim, 3: bulb}
    finally:
        LOOKS["bulbasaur"], LOOKS["chimchar"] = look_b, look_c


def test_objects_passing_close_by_keep_their_ids():
    reg = registry()
    out, now = run(reg, [[(1, "bulbasaur", 300, 500), (2, "chimchar", 700, 500)]] * 10)
    first = ids(out)
    out, now = run(reg, [[(1, "bulbasaur", 480, 500), (2, "chimchar", 540, 500)]] * 10, now)
    out, now = run(reg, [[(1, "bulbasaur", 300, 500), (2, "chimchar", 700, 500)]] * 10, now)
    assert ids(out) == first
    assert not any(i.disturbed for i in reg.identities)


def test_coasting_keeps_box_while_next_to_detected_object():
    reg = registry()
    out, now = run(reg, [[(1, "bulbasaur", 300, 500), (2, "chimchar", 420, 500)]] * 10)
    # chimchar stops being detected right next to bulbasaur: it stays where it was
    out, now = run(reg, [[(1, "bulbasaur", 300, 500)]] * 60, now)
    coasted = [o for o in out if o.coasted]
    assert len(coasted) == 1 and abs(coasted[0].center[0] - 0.42) < 1e-6
    # alone and far away, it coasts only briefly
    out, now = run(reg, [[(1, "bulbasaur", 900, 900)]] * 30, now)
    assert not [o for o in out if o.coasted]


def test_turning_object_learns_new_angles_and_is_recognized_from_them():
    reg = registry()
    reg.verify_interval = 0.2
    front, back = LOOKS["bulbasaur"], rng.normal(size=16).astype(np.float32)  # back looks unrelated

    def seen(track_id, look, occluded=False):
        o = obj(track_id, "bulbasaur", 300, 500)
        o.embedding, o.occluded = look + rng.normal(scale=0.05, size=16).astype(np.float32), occluded
        return o

    now = 0.0
    for _ in range(10):
        now += 1 / 15
        reg.update([seen(1, front)], now)
    ident = reg.identities[0]
    # turned around in view (between two checks): the tracker keeps following it
    for _ in range(30):
        now += 1 / 15
        reg.update([seen(1, back)], now)
    assert len(ident.gallery) >= 2 and len(ident.gallery) <= reg.gallery_size
    # a hand on it: its views aren't kept
    n = len(ident.gallery)
    for _ in range(30):
        now += 1 / 15
        reg.update([seen(1, -front, occluded=True)], now)
    assert len(ident.gallery) == n
    # gone, then back showing only its back, with a new track id: same identity
    for _ in range(60):
        now += 1 / 15
        reg.update([], now)
    for _ in range(10):
        now += 1 / 15
        out = reg.update([seen(9, back)], now)
    assert ids(out) == {9: ident.id} and len(reg.identities) == 1


def mapper():
    params = [Parameter("nature flute", "prompt", "flute", color=(0, 255, 0)),
              Parameter("fire taiko drum", "prompt", "taiko", color=(0, 0, 255)),
              Parameter("friendly jungle beat", "prompt", "jungle", color=(0, 255, 255))]
    return ParameterMapper(parameters=params, reference=(0.5, 0.5), smoothing=1.0,
                           assignments={"bulbasaur": "nature flute", "chimchar": "fire taiko drum"},
                           combos=[Combo(("bulbasaur", "chimchar"), "friendly jungle beat", distance=0.05)])


def labeled(o, identity, label):
    o.identity, o.class_name = identity, label
    return o


def test_combo_replaces_parameters_and_releases():
    m = mapper()
    apart = [labeled(obj(1, "bulbasaur", 400, 500), 1, "bulbasaur"),
             labeled(obj(2, "chimchar", 600, 500), 2, "chimchar")]
    m.update(apart, now=1.0)
    assert set(m.recipe()) == {"flute", "taiko"}
    together = [labeled(obj(1, "bulbasaur", 450, 500), 1, "bulbasaur"),
                labeled(obj(2, "chimchar", 570, 500), 2, "chimchar")]  # 20px = 0.02 apart
    m.update(together, now=1.1)
    assert m.combos[0].active and set(m.recipe()) == {"jungle"}
    # within the release distance (0.05 * 1.5) it stays combined
    m.update([labeled(obj(1, "bulbasaur", 450, 500), 1, "bulbasaur"),
              labeled(obj(2, "chimchar", 610, 500), 2, "chimchar")], now=1.2)
    assert m.combos[0].active
    m.update(apart, now=1.3)
    assert not m.combos[0].active and set(m.recipe()) == {"flute", "taiko"}


def test_held_trigger():
    m = mapper()
    m.set_mappings("bulbasaur", [{"trigger": "held", "prompt": "epic strings"}])
    hand = TrackedObject(track_id=9, class_name="hand", confidence=0.9, box=(250, 450, 330, 530),
                         center=(0.29, 0.49))
    b = labeled(obj(1, "bulbasaur", 300, 500), 1, "bulbasaur")
    m.update([b], now=1.0)
    assert "epic strings" not in m.recipe() and "flute" in m.recipe()  # config mapping kept
    m.update([b], now=1.1, hands=[hand])
    assert m.recipe()["epic strings"] == 1.0


def test_library_keeps_ids_names_and_instruments_across_sessions():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "library"
        reg = registry(ObjectLibrary(root, "fake"))
        described = []
        reg.on_new = described.append
        run(reg, [[(1, "bulbasaur", 300, 500), (2, "pikachu", 700, 500)]] * 10)
        by_track = {i.track_id: i for i in reg.identities}
        b, p = by_track[1], by_track[2]
        assert [i.id for i in described] == [b.id, p.id]  # each new object is described once
        assert (root / str(p.id) / "views" / "00.jpg").exists()

        # what the objects page does: name one, set an instrument on the other
        b.user_label = "bulbasaur"
        reg.library.save_meta(b)
        p.description, p.instrument = "a yellow electric mouse", "electric synth"
        reg.library.save_meta(p)
        reg.flush()

        # a new session recognizes both (new track ids, swapped places), unnamed ones too
        reg2 = registry(ObjectLibrary(root, "fake"))
        out, _ = run(reg2, [[(5, "pikachu", 300, 500), (6, "bulbasaur", 700, 500)]] * 10)
        assert ids(out) == {5: p.id, 6: b.id}
        assert reg2.get(b.id).user_label == "bulbasaur" and reg2.get(b.id).instrument is None
        assert reg2.get(p.id).instrument == "electric synth"
        assert reg2.get(p.id).description == "a yellow electric mouse"

        # deleted objects go to the trash, and their id is never reused
        reg2.remove(reg2.get(p.id))
        assert not (root / str(p.id)).exists() and len(list((root / ".trash").iterdir())) == 1
        run(reg2, [[(7, "chimchar", 500, 500)]] * 10)
        assert max(i.id for i in reg2.identities) == p.id + 1


def test_old_library_layout_is_converted():
    with tempfile.TemporaryDirectory() as tmp:
        old = Path(tmp) / "library" / "chimchar"
        (old / "views").mkdir(parents=True)
        (old / "object.json").write_text('{"name": "chimchar", "mappings": '
                                         '[{"trigger": "near", "prompt": "fire taiko"}], "embedder": "fake"}')
        np.save(old / "embeddings.npy", LOOKS["chimchar"][None])
        cv2.imwrite(str(old / "views" / "00.jpg"), np.zeros((10, 10, 3), np.uint8))
        lib = ObjectLibrary(Path(tmp) / "library", "fake")
        entry = lib.entries[1]
        assert entry.name == "chimchar" and entry.instrument == "fire taiko" and len(entry.embeddings) == 1
        assert not old.exists()


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    for name, fn in tests:
        fn()
        print(f"ok  {name}")
    print(f"{len(tests)} passed")
