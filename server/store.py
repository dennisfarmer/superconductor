"""Session persistence: the session as JSON at $ATLAS_DATA/session.json (default data/).

Written through a temp file and a rename, so a crash can't leave it half-written.
Loading ignores keys the dataclasses don't have and lets missing ones take their
defaults, so older session files still load.
"""

from __future__ import annotations

import dataclasses
import json
import os
import tempfile
from pathlib import Path
from typing import Any, TypeVar, get_args, get_origin, get_type_hints

T = TypeVar("T")


def path() -> Path:
    return Path(os.environ.get("ATLAS_DATA", "data")) / "session.json"


def _decode(tp: Any, v: Any) -> Any:
    if dataclasses.is_dataclass(tp):
        return from_dict(tp, v)
    if get_origin(tp) is list:
        (arg,) = get_args(tp)
        return [_decode(arg, x) for x in v]
    if tp is float and isinstance(v, int):
        return float(v)
    return v


def from_dict(cls: type[T], d: dict[str, Any]) -> T:
    hints = get_type_hints(cls)
    return cls(**{
        f.name: _decode(hints[f.name], d[f.name])
        for f in dataclasses.fields(cls) if f.init and f.name in d
    })


def load(cls: type[T]) -> T:
    """The saved session, or a fresh one if there is none."""
    p = path()
    if not p.exists():
        return cls()
    with open(p, encoding="utf-8") as f:
        return from_dict(cls, json.load(f))


def save(obj: Any) -> None:
    p = path()
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=p.parent, prefix=".session.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(dataclasses.asdict(obj), f, ensure_ascii=False, indent=1)
        os.replace(tmp, p)
    except BaseException:
        os.unlink(tmp)
        raise
