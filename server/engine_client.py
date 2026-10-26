"""Thin HTTP client for the Magenta audio engine (audio_engine/engine.py)."""


from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

import os
import sys
import json
import time
import subprocess
from urllib.parse import urlparse
from urllib.request import Request, urlopen

ENGINE_URL: str = os.environ.get("ATLAS_ENGINE_URL", "http://127.0.0.1:9104")
ENGINE_PRESET: str = os.environ.get("ATLAS_ENGINE_PRESET", "warm_piano")
# where the page reaches the engine (it sends timelines and reads /status itself); behind
# a gateway that only forwards one port, e.g. "/engine"
ENGINE_PUBLIC_URL: str = os.environ.get("ATLAS_ENGINE_PUBLIC_URL", ENGINE_URL)


@dataclass(eq=False)
class EngineChord:
    label: str
    symbol: str
    notes: list[int]
    scale: list[int] = field(default_factory=list)


def _post(path: str, payload: dict[str, Any]) -> bool:
    try:
        req = Request(
            ENGINE_URL + path,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        resp = urlopen(req, timeout=0.8)
        ok = resp.status == 200
        resp.close()
        return ok
    except Exception:
        return False


def engine_online() -> bool:
    try:
        resp = urlopen(ENGINE_URL + "/status", timeout=0.5)
        ok = resp.status == 200
        resp.close()
        return ok
    except Exception:
        return False


def send_timeline(role: str, chords: list[EngineChord], timeline_id: str) -> bool:
    """Send a timeline. role: idle (vamp the current chord) | preview (loop a path) | play (once)."""
    return _post(
        "/timeline",
        {
            "role": role,
            "id": timeline_id,
            "chords": [{"label": c.label, "symbol": c.symbol, "notes": c.notes, "scale": c.scale} for c in chords]
        }
    )


def release_preview() -> bool:
    return _post("/release", {})


def send_style(prompt: str) -> bool:
    return _post("/style", {"prompt": prompt})


def send_style_layers(texts: list[str], weights: list[float]) -> bool:
    """Blend several MusicCoCa prompts; the engine mixes their embeddings by weight."""
    return _post("/style", {"layers": [{"text": t, "weight": w} for (t, w) in zip(texts, weights)]})


def send_settings(settings: dict[str, Any]) -> bool:
    return _post("/settings", settings)


def stop_engine(timeout_s: float = 15.0) -> bool:
    """Ask the engine on ENGINE_URL to exit (POST /shutdown) and wait until it has.

    Only the engine this app talks to is addressed: it is the one process bound to
    that port. The engine finishes its current chunk, closes its audio stream and
    its HTTP server, and returns from main. True once it no longer answers."""
    if not engine_online():
        return True
    _post("/shutdown", {})
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if not engine_online():
            return True
        time.sleep(0.2)
    return False


def restart_engine() -> bool:
    """Stop the audio engine (by asking it to) and launch a fresh one, the same
    command as `make engine`.

    Returns once the new process is spawned; the model takes a few seconds to load.
    The client notices the engine coming back online and re-sends the session state.
    False if the running engine did not stop (then nothing is launched)."""
    if not stop_engine():
        return False
    port = str(urlparse(ENGINE_URL).port or 9104)
    os.makedirs("logs", exist_ok=True)
    os.makedirs(".run", exist_ok=True)
    log = open("logs/engine.log", "a")
    proc = subprocess.Popen(
        [sys.executable, "audio_engine/engine.py", "--preset", ENGINE_PRESET, "--port", port],
        stdout=log,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        start_new_session=True   # outlives the app, like `make engine`
    )
    with open(".run/engine.pid", "w") as f:
        f.write(str(proc.pid))
    return True
