"""server/store.py: the session file."""

import dataclasses
import json

from server import store
from server.atlas import Alt, Saved, Session, Step


def test_store_roundtrip(tmp_path, monkeypatch):
    """The session survives a save and a load; unknown keys are ignored, missing ones defaulted."""
    monkeypatch.setenv("ATLAS_DATA", str(tmp_path))
    s = Session(tonic="E♭", bpm=96.0, steps=[Step(cid="I", notes=[39, 55, 58, 63], alts=[Alt(cid="IV", symbol="Ab")])])
    s.history = [list(s.steps)]
    s.saved = [Saved(name="one", tonic="A", mode="minor", steps=list(s.steps), extended=True)]
    store.save(s)
    back = store.load(Session)
    assert dataclasses.asdict(back) == dataclasses.asdict(s)
    assert isinstance(back.steps[0].alts[0], Alt) and isinstance(back.history[0][0], Step)
    raw = json.loads(store.path().read_text(encoding="utf-8"))
    raw["no_such_field"] = {}
    del raw["bpm"]
    store.path().write_text(json.dumps(raw), encoding="utf-8")
    assert store.load(Session).bpm == Session().bpm
