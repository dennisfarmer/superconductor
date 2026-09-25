"""Unit tests for identities, the object library, mappings, combos and commands.

No camera or detector needed: detections are synthetic and the embedder is a
stand-in with the same interface. Run with `python tests/test_object_tracking.py`
(or pytest). The end-to-end check with real models is scripts/simulate_tracking.py.
"""
import tempfile
from pathlib import Path

import numpy as np

from superconductor.object_tracking import (Combo, Commands, IdentityRegistry, ObjectLibrary, Parameter,
                                            ParameterMapper, TrackedObject)
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


def registry():
    reg = IdentityRegistry(embedder=FakeEmbedder(), match_threshold=0.55, new_threshold=0.45)
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


def test_library_commands_round_trip():
    with tempfile.TemporaryDirectory() as tmp:
        lib = ObjectLibrary(Path(tmp) / "library", "fake")
        reg = registry()
        m = mapper()
        cmd = Commands(reg, m, lib)
        run(reg, [[(1, "bulbasaur", 300, 500), (2, "pikachu", 700, 500)]] * 10)
        by_track = {i.track_id: i for i in reg.identities}
        b, p = by_track[1], by_track[2]

        assert "saved bulbasaur" in cmd(f"save #{b.id} bulbasaur")
        assert "pikachu: held -> electric synth" in cmd(f"when I'm holding #{p.id} play electric synth") \
            or "held -> electric synth" in cmd(f"map #{p.id} held electric synth")
        assert "saved pikachu" in cmd(f"save #{p.id} pikachu")
        assert m.get("pikachu held") is not None  # mapping moved from "#2" to the name
        assert "bulbasaur: near -> fire taiko drum" in cmd("map bulbasaur near fire taiko drum")
        assert "within 0.08" in cmd("combine bulbasaur pikachu sparkly duet distance 0.08")
        assert "0.100" in cmd("distance 0.1")
        assert (Path(tmp) / "library" / "pikachu" / "views" / "00.jpg").exists()

        # a new session recognizes both from the library, with their mappings and the combo
        lib2 = ObjectLibrary(Path(tmp) / "library", "fake")
        reg2 = registry()
        reg2.add_library(lib2)
        m2 = mapper()
        Commands(reg2, m2, lib2)
        out, _ = run(reg2, [[(5, "pikachu", 300, 500), (6, "bulbasaur", 700, 500)]] * 10)
        assert {o.class_name for o in out} == {"pikachu", "bulbasaur"}
        assert m2.get("pikachu held").prompt == "electric synth"
        assert [c.distance for c in m2.combos if "pikachu" in c.objects] == [0.1]


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    for name, fn in tests:
        fn()
        print(f"ok  {name}")
    print(f"{len(tests)} passed")
