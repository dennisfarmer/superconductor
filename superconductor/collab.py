"""Collaborative mode: people bring objects, each tracked object drives music
parameters (by its distance to a reference point, being held, or being in view).

Webcam + YOLO/BoT-SORT run in this process; Magenta RealTime 2 runs in a separate
server process, locally (`make server`) or on the cluster (--remote); see
magenta_remote.py and superconductor_server/mrt2_server.py.

Two modes ([tracking] mode, or --mode):
- unsupervised (default): every object gets a permanent id when it first appears
  and is cached in the object library (library/<id>/), so it is recognized again
  later, across restarts. The describe server (../superconductor_describe) writes
  a one-time description of each new object and suggests an instrument from it.
  Names and instruments are edited on the objects page (web_panel.py,
  http://localhost:8467/objects).
- calibration: starts with "touch the bulbasaur", ... (unless --skip-calibration);
  only those objects are tracked.

Objects close together can combine (combos, set up on the objects page): their
instruments are replaced by the combo's until they separate.

Keys: [ ] combo distance, c calibrate, q quit. Click to move the reference point
(during calibration: click an object to label it).
"""
import json
import logging
import re
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
import webbrowser
import tomllib
from pathlib import Path
from urllib.parse import urlparse

import click
import cv2
import numpy as np

from superconductor.describe_client import DescribeClient
from superconductor.magenta_remote import MagentaClient
from superconductor.object_tracking import (Calibration, Combo, ObjectLibrary, ObjectTracker, Parameter,
                                            ParameterMapper)
from superconductor.object_tracking.identity import box_gap
from superconductor.object_tracking.library import bgr_to_hex, hex_to_bgr
from superconductor.web_panel import WebPanel

DEFAULT_CONFIG = Path(__file__).parent / "collab.toml"
# prompt of the "test beat" checkbox on the objects page. MusicCoCa embeds text, so
# describe only what should be there ("no vocals" would add a pull toward vocals).
TEST_BEAT = "solo drum kit groove, isolated drums, instrumental percussion only"
COMBO_COLORS = [(60, 220, 220), (220, 220, 80), (200, 90, 220), (255, 160, 60)]
# local end of `ssh -N -L 9000:lh2300:9100 ...` (see superconductor_server/README.md)
REMOTE_URL = "ws://localhost:9000/stream"


def find_camera(name):
    """cv2 index of the first camera whose name contains `name` (case-insensitive).

    OpenCV's AVFoundation backend orders cameras by unique ID, so sort the
    system_profiler list the same way. Falls back to 0 if not found.
    """
    out = subprocess.run(["system_profiler", "SPCameraDataType"],
                         capture_output=True, text=True).stdout
    cameras = re.findall(r"^\s{4}(\S.*):\n(?:.*\n)*?\s+Unique ID: (\S+)", out, re.MULTILINE)
    cameras.sort(key=lambda c: c[1])
    for index, (camera_name, _) in enumerate(cameras):
        if name.lower() in camera_name.lower():
            print(f"using camera [{index}] {camera_name}")
            return index
    print(f"camera matching {name!r} not found in {[c[0] for c in cameras]}, using 0")
    return 0


def load_config(path):
    with open(path, "rb") as f:
        cfg = tomllib.load(f)
    # objects get their instruments, triggers and combos on the objects page; [[parameters]]
    # and [assign] are only needed for calibration mode
    parameters = [Parameter(name=p["name"], kind=p["kind"], prompt=p.get("prompt"),
                            min=p.get("min", 0.0), max=p.get("max", 1.0),
                            color=tuple(p.get("color", (255, 255, 255))))
                  for p in cfg.get("parameters", [])]
    m = cfg["mapping"]
    # config reference is in displayed (mirrored) coords; mapper works unflipped
    rx, ry = m.get("reference", (0.5, 0.5))
    mapper = ParameterMapper(parameters=parameters, assignments=cfg.get("assign", {}),
                             reference=(1 - rx, ry),
                             max_distance=m.get("max_distance", 0.6),
                             smoothing=m.get("smoothing", 0.3),
                             hold_seconds=m.get("hold_seconds", 1.5),
                             release_after=m.get("release_after"))
    return cfg, mapper


class CollabFrontend:
    def __init__(self, cfg, mapper, camera=0, no_music=False, benchmark=None,
                 skip_calibration=False, describe_server=None):
        self.cfg = cfg
        self.mapper = mapper
        self.benchmark = benchmark
        music = cfg["music"]

        self.magenta = None
        if not no_music:
            self.magenta = MagentaClient(
                server_url=music["server_url"],
                credits=music["credits"],
                midi_port=music.get("midi_port", 8470),
            )
            self.magenta.start()

        t = cfg["tracking"]
        self.mode = t.get("mode", "unsupervised")
        unsupervised = self.mode == "unsupervised"
        self.tracker = ObjectTracker(model=t["model"], classes=t.get("classes"),
                                     conf=t.get("conf", 0.3), imgsz=t.get("imgsz", 480),
                                     device=t.get("device"),
                                     # the library replaces identities.json in unsupervised mode
                                     identity_dir=None if unsupervised else t.get("identity_dir"),
                                     embedder=t.get("embedder"), embed_imgsz=t.get("embed_imgsz", 224),
                                     embed_device=t.get("embed_device"),
                                     embeds_per_frame=t.get("embeds_per_frame", 2),
                                     identity=cfg.get("identity", {}))
        registry = self.tracker.registry
        self.library = None
        if unsupervised and t.get("library_dir"):
            embedder = self.tracker.embedder
            self.library = ObjectLibrary(t["library_dir"], embedder.name if embedder else None)
            if embedder is not None:
                self.library.refresh(embedder)
            registry.attach_library(self.library)
        registry.allow_new = unsupervised
        for ident in registry.identities:
            self._apply_instrument(ident)
        self.combos = []  # [{"objects": [ids], "prompt", "distance", "trigger"}], saved in the library
        self.settings = {"test_beat": False}  # page-wide switches, saved in the library
        if self.library is not None:
            self.settings.update(self.library.load_settings())
        if self.library is not None:
            known = {i.id for i in registry.identities}
            self.combos = [c for c in self.library.load_combos() if set(c["objects"]) <= known]
            self._apply_combos()

        # one-time descriptions of new objects (and of cached ones that never got one)
        self.describer = None
        if self.library is not None:
            self.describer = DescribeClient(describe_server or cfg.get("describe", {}).get(
                "server_url", "http://localhost:9200"))
            registry.on_new = self._describe
            for ident in registry.identities:
                if not ident.description:
                    self._describe(ident)
        self._describe_error_shown = False
        self.panel = WebPanel(cfg.get("web", {}).get("port", 8467)) if self.library is not None else None
        self._panel_views = {}  # id -> its view_rev when its views were last encoded
        self._view_scores, self._view_scores_key = {}, None  # id -> IdentityRegistry.view_scores
        self._last_publish = 0.0
        self._alive = {}  # service key -> its last probe (None: no answer), for the Services tab
        self._atlas = _find_atlas({}, search=False)  # key -> (url, answer); found again on every probe
        if self.panel is not None:
            threading.Thread(target=self._probe_services, daemon=True).start()

        # "touch the <label>" for every label used in [assign]
        self.calibration = Calibration(list(mapper.assignments) if not unsupervised else [],
                                       registry, allow_new_after=unsupervised)
        known = {i.user_label for i in registry.identities}
        if not unsupervised and not (skip_calibration and set(mapper.assignments) <= known):
            self.calibration.start()
        self._objects = []
        self.messages = []  # (text, expires)
        if self.panel is not None:
            print(f"objects page: {self.panel.url}")
            webbrowser.open(self.panel.url)

        self.camera = camera
        self.webcam = None
        self._frame_shape = None
        self._last_send = 0.0
        self._fps = 0.0
        self._track_ms = 0.0
        self._fps_log = []
        self._rtf_log = []

    def _on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and self._frame_shape:
            h, w = self._frame_shape
            # window is mirrored; use unflipped coords
            point = (1 - x / w, y / h)
            if self.calibration.active:
                self.calibration.click(point, self._objects)
                return
            self.mapper.reference = point

    # ---------- objects: instruments, descriptions, the objects page ----------

    def say(self, text, seconds=6.0):
        print(text)
        for line in text.splitlines()[-8:]:
            self.messages.append((line, time.time() + seconds))
        self.messages = self.messages[-8:]

    def _apply_instrument(self, ident):
        """an object plays its instrument with its trigger; without one it is silent"""
        mappings = [{"trigger": ident.trigger, "prompt": ident.instrument}] if ident.instrument else []
        self.mapper.set_mappings(f"#{ident.id}", mappings)

    def _sync_colors(self):
        """an object's instrument is drawn in the object's color (picked, else its mean color)"""
        colors = {f"#{i.id}": i.display_color for i in self.tracker.registry.identities}
        for param in self.mapper.parameters:
            if colors.get(param.owner) is not None:
                param.color = colors[param.owner]

    def _apply_combos(self, save=False):
        """replace the mapper's combos with `self.combos`"""
        for combo in self.mapper.combos:
            param = self.mapper.get(combo.parameter)
            if param is not None and param.owner == combo.key:
                self.mapper.parameters.remove(param)
        self.mapper.combos = []
        for i, c in enumerate(self.combos):
            combo = Combo(objects=tuple(f"#{id}" for id in c["objects"]), parameter="",
                          distance=c.get("distance", 0.05), trigger=c.get("trigger", "near"))
            combo.parameter = combo.key
            self.mapper.parameters.append(Parameter(name=combo.key, kind="prompt", prompt=c["prompt"],
                                                    owner=combo.key, color=COMBO_COLORS[i % len(COMBO_COLORS)]))
            self.mapper.combos.append(combo)
        if save and self.library is not None:
            self.library.save_combos(self.combos)

    def _combo_name(self, combo):
        """"bulbasaur + #4" for a combo's objects"""
        names = []
        for key in combo.objects:
            ident = self.tracker.registry.get(key)
            names.append(ident.user_label or key if ident else key)
        return " + ".join(names)

    def _describe(self, ident):
        """send the object's sharpest view to the describe server (once)"""
        views = [v for v in ident.views if v is not None and v.size]
        if self.describer is None or not views:
            return
        sharpest = max(views, key=lambda v: cv2.Laplacian(cv2.cvtColor(v, cv2.COLOR_BGR2GRAY),
                                                         cv2.CV_64F).var() * v.shape[0] * v.shape[1])
        ok, jpeg = cv2.imencode(".jpg", sharpest)
        if ok:
            self.describer.describe(ident.id, jpeg.tobytes(), suggest=not ident.instrument,
                                    name=not ident.user_label)

    def _save_meta(self, ident):
        if self.library is not None and ident.library:
            self.library.save_meta(ident)

    def apply_describe_results(self):
        while self.describer is not None and not self.describer.results.empty():
            kind, id, text = self.describer.results.get()
            ident = self.tracker.registry.get(id)
            if kind == "error":
                if not self._describe_error_shown:
                    self.say(text)
                    self._describe_error_shown = True
                continue
            self._describe_error_shown = False
            if ident is None:
                continue
            if kind == "description":
                ident.description = text
            elif kind == "instrument":
                ident.instrument = text
                self._apply_instrument(ident)
                self.say(f"#{ident.id} {ident.label}: plays {text}")
            elif kind == "name" and not ident.user_label and text and not text.isdigit():
                # default name from the description; a typed name is never replaced
                taken = {i.user_label for i in self.tracker.registry.identities}
                name, n = text, 2
                while name in taken:
                    name, n = f"{text} {n}", n + 1
                ident.user_label = name
            self._save_meta(ident)

    def apply_panel_actions(self):
        registry = self.tracker.registry
        while self.panel is not None and not self.panel.actions.empty():
            kind, id, body = self.panel.actions.get()
            if kind.startswith("combo"):
                self._combo_action(kind, id, body)
                continue
            if kind == "learning":  # view-learning settings (this session only; collab.toml stays)
                errors = []
                for key, cast in registry.LEARNING.items():
                    if key not in body:
                        continue
                    try:
                        value = cast(body[key])
                    except (TypeError, ValueError):
                        errors.append(f"{key}: not a number")
                        continue
                    if key == "gallery_size" and value < 1 or key != "gallery_size" and not -1 <= value <= 1:
                        errors.append(f"{key}: out of range")
                        continue
                    setattr(registry, key, value)
                if registry.learn_threshold >= registry.novelty:
                    errors.append("learn_threshold >= novelty: nothing new can be learned")
                self.say("; ".join(errors) or "learning: " + ", ".join(
                    f"{k} {getattr(registry, k)}" for k in registry.LEARNING), 3.0)
                continue
            if kind == "settings":
                if "test_beat" in body:
                    self.settings["test_beat"] = bool(body["test_beat"])
                    self.say(f"test beat ({TEST_BEAT}) {'on' if self.settings['test_beat'] else 'off'}", 2.0)
                if self.library is not None:
                    self.library.save_settings(self.settings)
                continue
            if kind == "music":  # play / pause (not saved: every session starts playing)
                if "paused" in body and self.magenta is not None:
                    self.magenta.set_paused(bool(body["paused"]))
                    self.say(f"music {'paused' if self.magenta.paused else 'playing'}", 2.0)
                continue
            ident = registry.get(id)
            if ident is None:
                continue
            if kind == "edit":
                if "name" in body:
                    name = body["name"] or None
                    if name and (name.isdigit() or name.startswith("#")):
                        self.say(f"#{ident.id}: a name can't be a number")
                        continue
                    for other in registry.identities:  # names are unique
                        if name and other is not ident and other.user_label == name:
                            other.user_label = None
                            self._save_meta(other)
                    ident.user_label = name
                if "instrument" in body:
                    ident.instrument = body["instrument"] or None
                if body.get("trigger") in ("near", "held", "visible"):
                    ident.trigger = body["trigger"]
                if "color" in body:  # "#rrggbb", or null for its mean color again
                    ident.user_color = hex_to_bgr(body["color"])
                if "frozen" in body:
                    ident.frozen = bool(body["frozen"])
                    self.say(f"#{ident.id} {ident.label}: views {'frozen' if ident.frozen else 'learning'}", 2.0)
                self._apply_instrument(ident)
                self._save_meta(ident)
            elif kind == "cover":
                error = registry.set_cover(ident, body.get("index"), body.get("rev"))
                if error:
                    self.say(f"#{ident.id} {ident.label}: {error}", 2.0)
                else:
                    self._save_meta(ident)
            elif kind == "delete_view":
                error = registry.delete_view(ident, int(body.get("index", -1)), body.get("rev"))
                self.say(f"#{ident.id} {ident.label}: " + (error or f"deleted a view ({len(ident.views)} left)"), 2.0)
            elif kind == "describe":
                self._describe(ident)
            elif kind == "suggest":
                if ident.description and self.describer is not None:
                    self.describer.suggest(ident.id, ident.description)
                else:
                    self._describe(ident)
            elif kind == "delete":
                self.mapper.set_mappings(f"#{ident.id}", [])
                for param in self.mapper.parameters:
                    if param.identity == ident.id:
                        param.identity, param.value = None, 0.0
                registry.remove(ident)
                self._panel_views.pop(ident.id, None)
                if any(ident.id in c["objects"] for c in self.combos):
                    self.combos = [c for c in self.combos if ident.id not in c["objects"]]
                    self._apply_combos(save=True)
                self.say(f"deleted #{ident.id} (moved to library/.trash)")

    def _combo_action(self, kind, index, body):
        registry = self.tracker.registry
        if kind == "combo_add":
            objects = sorted({int(i) for i in body.get("objects", []) if registry.get(int(i)) is not None})
            if len(objects) < 2 or not body.get("prompt"):
                self.say("a combo needs at least two objects and a prompt")
                return
            self.combos = [c for c in self.combos if sorted(c["objects"]) != objects]
            self.combos.append({"objects": objects, "prompt": body["prompt"],
                                "distance": float(body.get("distance") or 0.05),
                                "trigger": body.get("trigger") or "near"})
        elif not 0 <= index < len(self.combos):
            return
        elif kind == "combo_edit":
            combo = self.combos[index]
            if body.get("prompt"):
                combo["prompt"] = body["prompt"]
            if body.get("distance") not in (None, ""):
                combo["distance"] = max(0.0, float(body["distance"]))
            if body.get("trigger") in ("near", "held", "visible"):
                combo["trigger"] = body["trigger"]
        elif kind == "combo_delete":
            del self.combos[index]
        self._apply_combos(save=True)

    def models(self):
        """which model does what, and where it runs (for the objects page)"""
        here = socket.gethostname()

        def where(info):
            host = info.get("host")
            return "local" if host == here else f"remote ({host})" if host else "local"

        t = self.cfg["tracking"]
        rows = [{"task": "Detection + tracking",
                 "model": f"YOLOE-11S-seg ({t['model']}, prompts: {', '.join(t.get('classes') or [])}) "
                          f"+ BoT-SORT, on {self.tracker.device}",
                 "where": "local"}]
        embedder = self.tracker.embedder
        if embedder is not None:
            model, size = embedder.name.split("@")
            label = {"dinov2_vits14": "DINOv2 ViT-S/14", "dinov2_vits14_reg": "DINOv2 ViT-S/14 (registers)"}.get(
                model, model)
            rows.append({"task": "Recognition", "model": f"{label} ({model}, {size} px, on {embedder.device})",
                         "where": "local"})
        if self.describer is not None:
            info = self.describer.info
            rows.append({"task": "Description, name + instrument suggestions",
                         "model": f"{info['model']} via {info.get('backend', 'ollama')}" if info.get("model")
                         else f"describe server not reachable ({self.describer.server_url})",
                         "where": where(info) if info else "-"})
        if self.magenta is None:
            music = {"model": "music off", "where": "-"}
        elif not self.magenta.connected:
            music = {"model": f"not connected ({self.cfg['music']['server_url']})", "where": "-"}
        else:
            info = self.magenta.info
            music = {"model": f"Magenta RealTime 2 {info.get('model')} ({info.get('backend')})",
                     "where": where(info)}
        rows.append({"task": "Music", **music})
        return rows

    # path each service is probed at (GET) for the Services tab; Atlas is found by _find_atlas
    PROBES = {"server": "/tempo/status", "client_midi": "/status", "describe": "/health",
              "midi_player": "/api/state"}

    def _service_urls(self):
        """http base URL of each service (None: not used)"""
        svc = self.cfg.get("services", {})
        midi_port = self.cfg["music"].get("midi_port", 8470)
        return {
            # ws://localhost:9000/stream (the Lighthouse tunnel) -> http://localhost:9000
            "server": re.sub(r"^ws", "http", self.cfg["music"]["server_url"]).removesuffix("/stream"),
            "objects": f"http://localhost:{self.panel.port}",
            "client_midi": f"http://localhost:{midi_port}" if midi_port and self.magenta is not None else None,
            "describe": self.describer.server_url if self.describer is not None
            else self.cfg.get("describe", {}).get("server_url", "http://localhost:9200"),
            "midi_player": svc.get("midi_player_url", "http://localhost:8475"),
            "atlas": self._atlas["atlas"][0], "atlas_engine": self._atlas["atlas_engine"][0],
        }

    def _probe_services(self):
        """every 2 s, which services answer (in a thread, so a slow one can't stall the camera loop)"""
        while True:
            self._atlas = _find_atlas(self.cfg.get("services", {}))
            urls = self._service_urls()
            alive = {key: _probe(urls[key].rstrip("/") + path)
                     for key, path in self.PROBES.items() if urls.get(key)}
            self._alive = {**alive, **{key: answer for key, (_, answer) in self._atlas.items()}}
            time.sleep(2)

    def services(self):
        """the other SuperConductor programs: URLs, endpoints, whether they answer (for the objects page)"""
        urls, alive, here = self._service_urls(), self._alive, socket.gethostname()

        def where(base, info):
            host = (info or {}).get("host")
            if not host or host == here:
                return ""
            tunnel = urlparse(base).hostname in ("localhost", "127.0.0.1")
            return f"on {host}" + (", through the SSH tunnel" if tunnel else "")

        def row(key, name, started, endpoints, note=""):
            base = urls.get(key)
            if not base:
                return None
            base = base.rstrip("/")
            up = True if key == "objects" else (alive[key] is not None) if key in alive else None
            return {"name": name, "started": started, "base": base, "up": up, "note": note,
                    "endpoints": [{"method": m, "url": u if "://" in u else base + u} for m, u in endpoints]}

        engine = alive.get("atlas_engine") or {}
        if engine.get("sink") == "superconductor":
            atlas_note = (f"sending notes/drums to {engine.get('client_midi')}"
                          + (" (connected)" if engine.get("connected") else " (client not connected)"))
        else:
            atlas_note = "standalone: plays through its own speakers" if engine else ""
        rows = [
            row("server", "Music server (mrt2_server.py)", "superconductor_server: make server",
                [("page", "/tempo"), ("page", "/conduct"), ("GET", "/tempo/status"), ("POST", "/tempo/beat"),
                 ("POST", "/tempo/target"), ("POST", "/tempo/free"), ("WS", self.cfg["music"]["server_url"])],
                where(urls["server"], self.magenta.info if self.magenta is not None else None)),
            row("objects", "Objects page (this app)", "superconductor_client: make client",
                [("page", "/objects"), ("GET", "/api/objects")]),
            row("client_midi", "client_midi (this app)", "superconductor_client: make client",
                [("GET", "/status"), ("GET", "/input"), ("POST", "/pattern")],
                "note sources (Atlas, MIDI player) POST patterns here"),
            row("describe", "Describe server", "superconductor_describe: make describe",
                [("GET", "/health"), ("POST", "/describe"), ("POST", "/instrument"), ("POST", "/name")],
                where(urls["describe"], alive.get("describe"))),
            row("midi_player", "MIDI player", "superconductor_midi: make player",
                [("page", "/"), ("GET", "/api/state"), ("POST", "/api/play"), ("POST", "/api/stop"),
                 ("POST", "/api/upload"), ("POST", "/api/settings")]),
            row("atlas", "Atlas app", "superconductor_atlas: make run", [("page", "/")]),
            row("atlas_engine", "Atlas engine", "started by Atlas's make run",
                [("GET", "/status"), ("POST", "/timeline"), ("POST", "/release"), ("POST", "/style"),
                 ("POST", "/settings"), ("POST", "/mix")], atlas_note),
        ]
        return [r for r in rows if r]

    def publish_panel(self, now):
        if self.panel is None or now - self._last_publish < 0.5:
            return
        self._last_publish = now
        in_view = {o.identity for o in self._objects if not o.coasted}
        busy = self.describer.busy if self.describer is not None else set()
        objects, views = [], {}
        registry = self.tracker.registry
        revs = tuple((i.id, i.view_rev, len(i.gallery)) for i in registry.identities)
        if revs != self._view_scores_key:  # any gallery changed: each view's scores against all of them
            self._view_scores_key = revs
            self._view_scores = {i.id: registry.view_scores(i) for i in registry.identities if i.library}
        for ident in registry.identities:
            if not ident.library:
                continue
            # its own instrument first, then config parameters it drives
            params = sorted(self.mapper.parameters_for(ident.id), key=lambda p: p.owner != f"#{ident.id}")
            if ident.views and self._panel_views.get(ident.id) != ident.view_rev:
                # one per gallery entry (same index), newest last (the thumbnail); b"" if unreadable
                jpegs = [cv2.imencode(".jpg", v)[1].tobytes() if v is not None and v.size else b""
                         for v in ident.views]
                if any(jpegs):
                    views[ident.id] = jpegs
                    self._panel_views[ident.id] = ident.view_rev
            ago = now - ident.last_seen if ident.last_seen else None
            objects.append({
                "id": ident.id, "name": ident.user_label, "label": ident.label,
                "description": ident.description, "describing": ident.id in busy,
                "instrument": ident.instrument, "trigger": ident.trigger, "in_view": ident.id in in_view,
                "color": bgr_to_hex(ident.display_color), "color_picked": ident.user_color is not None,
                "last_seen": "not seen this session" if ago is None else f"seen {_ago(ago)} ago",
                "parameter": _param_name(params[0]) if params else None,
                "value": max((p.value for p in params), default=0.0), "views": len(ident.views),
                "view_rev": ident.view_rev, "frozen": ident.frozen, "cover": ident.cover,
                "view_scores": self._view_scores.get(ident.id, [])})
        objects.sort(key=lambda o: (not o["in_view"], -o["id"]))
        combos = [{"index": i, "objects": c["objects"], "prompt": c["prompt"], "distance": c["distance"],
                   "trigger": c.get("trigger", "near"), "name": self._combo_name(mc), "active": mc.active}
                  for i, (c, mc) in enumerate(zip(self.combos, self.mapper.combos))]
        music = {"connected": bool(self.magenta and self.magenta.connected),
                 "paused": bool(self.magenta and self.magenta.paused)}
        self.panel.publish({"objects": objects, "combos": combos,
                            "settings": dict(self.settings, test_beat_prompt=TEST_BEAT), "music": music,
                            "models": self.models(), "services": self.services(),
                            "learning": {k: getattr(registry, k) for k in registry.LEARNING}}, views)

    def handle_key(self, key):
        """returns False to quit"""
        if key == 255:
            return True
        if key == ord("q"):
            return False
        if key in (ord("["), ord("]")) and self.mapper.combos:
            for combo in self.combos:
                combo["distance"] = max(0.0, round(combo["distance"] + (0.01 if key == ord("]") else -0.01), 3))
            self._apply_combos(save=True)
            self.say(f"combo distance {self.combos[0]['distance']:.2f}", 2.0)
        elif key == ord("c"):
            if not self.calibration.labels:
                self.say("calibration mode only (--mode calibration); name objects on the objects page")
            else:
                for param in self.mapper.parameters:
                    param.identity = None
                self.calibration.start()
        return True

    def recipe(self):
        """style weights sent to MRT2: base prompt, test beat, objects and combos"""
        music = self.cfg["music"]
        recipe = self.mapper.recipe(music["base_prompt"], music["base_weight"])
        if self.settings["test_beat"]:  # always-on test sound (checkbox on the objects page)
            recipe[TEST_BEAT] = recipe.get(TEST_BEAT, 0.0) + 1.0
        return recipe

    def mrt2_input(self):
        """What the overlay's "MRT2 input" panel shows, and client_midi's GET /input serves:
        every prompt in the recipe (base, test beat, objects, combos) as (prompt, source,
        value, share of the blend, BGR color), and the sampling controls. The server
        averages the prompts' embeddings by weight, so share = weight / sum of weights."""
        music = self.cfg["music"]
        rows = []  # (prompt, source, value, color)
        if music["base_prompt"] and music["base_weight"] > 0:
            rows.append((music["base_prompt"], "base", music["base_weight"], (200, 200, 200)))
        if self.settings["test_beat"]:
            rows.append((TEST_BEAT, "test beat", 1.0, (120, 200, 255)))
        for param in self.mapper.parameters:
            combo = self.mapper.combo_for(param.identity) if isinstance(param.identity, int) else None
            if param.identity is None and param.owner and param.owner.startswith("#"):
                ident = self.tracker.registry.get(param.owner)  # its object isn't in view this session
                state = (ident.user_label if ident and ident.user_label else param.owner) + " not seen"
            elif param.identity is None:
                state = "free"
            elif isinstance(param.identity, str):
                state = "combined"
            else:
                ident = self.tracker.registry.get(param.identity)
                state = (ident.user_label if ident and ident.user_label else f"#{param.identity}") + \
                    ("" if param.visible else " hidden") + (" > combo" if combo else "")
            rows.append((_param_name(param), state, param.output if param.kind == "prompt" else param.value,
                         param.color))
        recipe = self.recipe()
        total = sum(recipe.values()) or 1.0
        rows = [(prompt, source, value, recipe.get(prompt, 0.0) / total if value > 0.01 else 0.0, color)
                for prompt, source, value, color in rows]
        return rows, self.mapper.controls()

    def send(self):
        now = time.time()
        if self.magenta is None or not self.magenta.connected:
            return
        if now - self._last_send < self.cfg["music"]["update_interval"]:
            return
        self._last_send = now
        self.magenta.update_recipe(self.recipe())
        self.magenta.update_controls(self.mapper.controls())
        rows, controls = self.mrt2_input()
        self.magenta.update_input_view({
            "rows": [{"prompt": p, "source": src, "value": round(v, 3), "share": round(sh, 3), "color": bgr_to_hex(c)}
                     for p, src, v, sh, c in rows],
            "controls": {k: round(v, 3) for k, v in controls.items()}})

    def draw(self, frame, objects):
        """Draw on the unflipped frame; text goes on a separate overlay drawn after flipping."""
        h, w = frame.shape[:2]
        ref = (int(self.mapper.reference[0] * w), int(self.mapper.reference[1] * h))
        cv2.circle(frame, ref, int(self.mapper.max_distance * w), (90, 90, 90), 1)
        cv2.drawMarker(frame, ref, (255, 255, 255), cv2.MARKER_CROSS, 24, 2)
        labels = []  # (text, org in mirrored coords, color, scale)

        for hand in self.tracker.hands:
            x1, y1, x2, y2 = map(int, hand.box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (200, 200, 200), 1)

        # tracks that are still being classified
        for obj in self.tracker.registry.pending:
            x1, y1, x2, y2 = map(int, obj.box)
            _dashed_rect(frame, (x1, y1), (x2, y2), (150, 150, 150), 1)
            hits, score, like = self.tracker.registry.pending_info(obj.track_id)
            text = "? new" if score is None else f"? {like} {score:.2f}"
            labels.append((text, (w - x2, max(y1 - 8, 15)), (150, 150, 150), 0.5))

        # combos: a frame around objects that combined, and their distance while apart
        by_label = {o.class_name: o for o in objects}
        by_label.update({f"#{o.identity}": o for o in objects})
        for combo in self.mapper.combos:
            param = self.mapper.get(combo.parameter)
            members = [by_label.get(label) for label in combo.objects]
            if any(m is None for m in members):
                continue
            centers = [(int(m.center[0] * w), int(m.center[1] * h)) for m in members]
            if combo.active:
                x1, y1, x2, y2 = (int(combo.box[0] * w) - 14, int(combo.box[1] * h) - 14,
                                  int(combo.box[2] * w) + 14, int(combo.box[3] * h) + 14)
                cv2.rectangle(frame, (x1, y1), (x2, y2), param.color, 4)
                for a, b in zip(centers, centers[1:]):
                    cv2.line(frame, a, b, param.color, 3)
                labels.append((f"{self._combo_name(combo)} = {param.prompt}",
                               (w - x2, min(y2 + 26, h - 10)), param.color, 0.7))
            elif len(members) == 2:
                gap = box_gap(members[0].box_normalized, members[1].box_normalized)
                close = gap < 3 * combo.distance
                cv2.line(frame, centers[0], centers[1], param.color if close else (90, 90, 90), 1)
                mid = ((centers[0][0] + centers[1][0]) // 2, (centers[0][1] + centers[1][1]) // 2)
                labels.append((f"{gap:.2f} / {combo.distance:.2f}", (w - mid[0] - 40, mid[1] - 6),
                               param.color if close else (140, 140, 140), 0.45))

        for obj in objects:
            combo = self.mapper.combo_for(obj.identity)
            param = self.mapper.get(combo.parameter) if combo else self.mapper.parameter_for(obj.identity)
            ident = self.tracker.registry.get(obj.identity)
            color = param.color if param else (ident and ident.display_color) or (150, 150, 150)
            x1, y1, x2, y2 = map(int, obj.box)
            cx, cy = int(obj.center[0] * w), int(obj.center[1] * h)
            if obj.coasted:  # not detected right now: last known position
                _dashed_rect(frame, (x1, y1), (x2, y2), color, 2)
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            if self.calibration.active and obj.identity == self.calibration.candidate:
                # dwell progress bar while an object is being touched
                cv2.rectangle(frame, (x1, y2 + 4), (x1 + int((x2 - x1) * self.calibration.progress), y2 + 10),
                              (255, 255, 255), -1)
            if combo is None:
                cv2.line(frame, ref, (cx, cy), color, 1)
            text = f"#{obj.identity} {obj.class_name}"
            if combo is not None:
                text += " (combined)"
            else:
                params = self.mapper.parameters_for(obj.identity)
                if params:
                    text += " -> " + ", ".join(f"{_param_name(p)} {p.value:.2f}" for p in params)
            if obj.coasted:
                text += " (last seen)"
            labels.append((text, (w - x2, max(y1 - 8, 15)), color, 0.55))  # mirrored x
        return labels

    def draw_hud(self, overlay, labels):
        for text, org, color, scale in labels:
            cv2.putText(overlay, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)

        message = self.calibration.message()
        if message:
            (tw, th), _ = cv2.getTextSize(message, cv2.FONT_HERSHEY_DUPLEX, 1.4, 2)
            org = ((overlay.shape[1] - tw) // 2, 80)
            cv2.putText(overlay, message, org, cv2.FONT_HERSHEY_DUPLEX, 1.4, (255, 255, 255), 2)

        # MRT2 input (see mrt2_input)
        rows, controls = self.mrt2_input()
        y = 30
        cv2.putText(overlay, "MRT2 input", (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        for prompt, source, value, share, color in rows:
            y += 24
            cv2.putText(overlay, prompt[:30], (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            cv2.putText(overlay, source[:16], (290, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            cv2.rectangle(overlay, (420, y - 12), (420 + int(150 * min(value, 1.0)), y - 2), color, -1)
            cv2.rectangle(overlay, (420, y - 12), (570, y - 2), color, 1)
            cv2.putText(overlay, f"{value:.2f}  {100 * share:.0f}%", (580, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 1)
        if controls:
            y += 24
            cv2.putText(overlay, "  ".join(f"{k} {v:.2f}" for k, v in controls.items()), (15, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # one banner per active combo
        banner_y = max(80 if not message else 130, y + 54)
        for combo in self.mapper.combos:
            if combo.active:
                param = self.mapper.get(combo.parameter)
                text = f"{self._combo_name(combo)} combined: {param.prompt}"
                (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 1.0, 2)
                cv2.putText(overlay, text, ((overlay.shape[1] - tw) // 2, banner_y),
                            cv2.FONT_HERSHEY_DUPLEX, 1.0, param.color, 2)
                banner_y += 40

        # command line and recent command output
        now = time.time()
        self.messages = [(t, until) for t, until in self.messages if until > now]
        y = overlay.shape[0] - 45 - 22 * len(self.messages)
        for text, _ in self.messages:
            cv2.putText(overlay, text, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y += 22

        stats = self.magenta.poll_stats() if self.magenta else {}
        music = "music off" if self.magenta is None else (
            "loading MRT2..." if not self.magenta.connected else
            f"{self.magenta.info.get('model')} ({self.magenta.info.get('backend')}) "
            f"x{stats.get('rtf', 0):.2f} realtime  underruns {stats.get('underruns', 0)}"
            + ("  PAUSED" if self.magenta.paused else ""))
        cv2.putText(overlay, f"vision {self._fps:.0f} fps ({self._track_ms:.0f} ms)  |  {music}  |  "
                             f"[ ] combo distance  q quit",
                    (15, overlay.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        if stats.get("rtf"):
            self._rtf_log.append(stats["rtf"])

    def step(self, frame, now=None):
        """Track one frame, update parameters, return the annotated (mirrored) image."""
        self._frame_shape = frame.shape[:2]
        t0 = time.time()
        objects = self.tracker(frame, now=now)
        self._track_ms = 0.9 * self._track_ms + 0.1 * (time.time() - t0) * 1000
        self._objects = objects
        self._sync_colors()
        if self.calibration.active:
            self.calibration.update([o for o in objects if not o.coasted], self.tracker.hands)
        self.mapper.update(objects, now=now, hands=self.tracker.hands)
        self.send()
        self.apply_panel_actions()
        self.apply_describe_results()
        self.publish_panel(time.time() if now is None else now)

        labels = self.draw(frame, objects)
        frame = cv2.flip(frame, 1)
        overlay = np.zeros_like(frame)
        self.draw_hud(overlay, labels)
        return cv2.add(frame, overlay)

    def run(self):
        self.webcam = cv2.VideoCapture(self.camera)
        if not self.webcam.isOpened():
            raise RuntimeError("Cannot open webcam")
        width, height = self.cfg.get("camera", {}).get("resolution", (1280, 720))
        self.webcam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cv2.namedWindow("SuperConductor")
        cv2.setMouseCallback("SuperConductor", self._on_mouse)

        start = time.time()
        prev = start
        while True:
            ok, frame = self.webcam.read()
            if not ok:
                time.sleep(0.01)
                continue
            cv2.imshow("SuperConductor", self.step(frame))

            now = time.time()
            self._fps = 0.9 * self._fps + 0.1 / max(now - prev, 1e-6)
            prev = now
            if self.magenta is None or self.magenta.connected:
                self._fps_log.append(self._fps)

            if not self.handle_key(cv2.waitKey(1) & 0xFF):
                break
            if self.benchmark and now - start > self.benchmark:
                break
        self.stop()

    def stop(self):
        self.tracker.registry.save()
        self.tracker.registry.flush()
        if self.webcam is not None:
            self.webcam.release()
        cv2.destroyAllWindows()
        if self.magenta:
            self.magenta.stop()
        if self._fps_log:
            print(f"vision fps: mean {np.mean(self._fps_log):.1f}, "
                  f"min {np.min(self._fps_log[len(self._fps_log) // 10:] or self._fps_log):.1f}")
        if self._rtf_log:
            print(f"MRT2 realtime factor: mean {np.mean(self._rtf_log):.2f}, "
                  f"min {np.min(self._rtf_log):.2f} (>1.0 = keeping up)")
        if self.magenta:
            print(f"audio underruns: {self.magenta.stats.get('underruns', 0)}")


def _param_name(param):
    """instruments and combos are shown by their prompt, config parameters by name"""
    return param.prompt if param.owner else param.name


def _ago(seconds):
    if seconds < 90:
        return f"{seconds:.0f}s"
    return f"{seconds / 60:.0f} min" if seconds < 5400 else f"{seconds / 3600:.0f} h"


def _dashed_rect(img, p1, p2, color, thickness=1, dash=10):
    (x1, y1), (x2, y2) = p1, p2
    for x in range(x1, x2, 2 * dash):
        cv2.line(img, (x, y1), (min(x + dash, x2), y1), color, thickness)
        cv2.line(img, (x, y2), (min(x + dash, x2), y2), color, thickness)
    for y in range(y1, y2, 2 * dash):
        cv2.line(img, (x1, y), (x1, min(y + dash, y2)), color, thickness)
        cv2.line(img, (x2, y), (x2, min(y + dash, y2)), color, thickness)


def _probe(url):
    """the JSON object a service answers a GET of `url` with ({} if none), or None if it doesn't answer"""
    try:
        with urllib.request.urlopen(url, timeout=0.5) as r:
            body = r.read()
    except urllib.error.HTTPError:
        return {}  # an error status is still an answer
    except (OSError, ValueError):
        return None
    try:
        data = json.loads(body)
    except ValueError:
        return {}
    return data if isinstance(data, dict) else {}


def _listening_ports(marker):
    """TCP ports listened on by local processes whose command line contains `marker`"""
    try:
        ps = subprocess.run(["ps", "-axo", "pid=,command="], capture_output=True, text=True, timeout=2).stdout
        pids = [line.split(None, 1)[0] for line in ps.splitlines() if marker in line]
        if not pids:
            return []
        out = subprocess.run(["lsof", "-nP", "-a", "-p", ",".join(pids), "-iTCP", "-sTCP:LISTEN", "-Fn"],
                             capture_output=True, text=True, timeout=2).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    return sorted({int(line.rsplit(":", 1)[1]) for line in out.splitlines()
                   if line.startswith("n") and line.rsplit(":", 1)[-1].isdigit()})


def _find_atlas(svc, search=True):
    """{"atlas": (url, answer), "atlas_engine": (url, answer)}; answer None: not running.

    `jac start` takes the next free port when its own is in use (e.g. 8009 for 8000), so the
    ports Atlas's processes listen on are tried along with the configured ones, and a port
    counts only if it answers like Atlas: its functions in /openapi.json, the engine's /status
    fields. With search=False, just the configured URLs, not running."""
    found = {}
    for key, default, marker, path, check in (
            ("atlas", "http://localhost:8000", "jac start main.jac", "/openapi.json",
             lambda a: {"/function/explore", "/function/set_tempo"} <= set(a.get("paths", {}))),
            ("atlas_engine", "http://localhost:8765", "audio_engine/engine.py", "/status",
             lambda a: {"realtime_budget_ms", "style_layers"} <= set(a))):
        url = svc.get(f"{key}_url", default).rstrip("/")
        found[key] = (url, None)
        if not search:
            continue
        port = urlparse(url).port
        candidates = [url] + [f"http://localhost:{p}" for p in _listening_ports(marker) if p != port]
        for base in candidates:
            answer = _probe(base + path)
            if answer and check(answer):
                found[key] = (base, {} if key == "atlas" else answer)
                break
    return found


@click.command()
@click.option("--config", "config_path", default=str(DEFAULT_CONFIG), type=click.Path(exists=True))
@click.option("--camera", default=None, type=int,
              help="cv2 camera index (default: first camera matching [camera].name)")
@click.option("--iphone", is_flag=True, help="Use the iPhone (Continuity Camera) instead.")
@click.option("--server", default=None,
              help="Override [music].server_url (websocket of a running mrt2_server.py).")
@click.option("--describe-server", default=None,
              help="Override [describe].server_url (describe server for one-time object descriptions).")
@click.option("--remote", is_flag=True,
              help=f"Use the SSH tunnel to the cluster server ({REMOTE_URL}).")
@click.option("--device", default=None, type=click.Choice(["cpu", "mps"]),
              help="Override [tracking].device (where YOLO runs).")
@click.option("--mode", default=None, type=click.Choice(["unsupervised", "calibration"]),
              help="Override [tracking].mode.")
@click.option("--skip-calibration", is_flag=True,
              help="Reuse objects labeled in a previous session instead of 'touch the ...'.")
@click.option("--no-music", is_flag=True, help="Only run tracking (no MRT2).")
@click.option("--benchmark", default=None, type=float,
              help="Run for N seconds, then print vision fps / MRT2 realtime stats.")
@click.option("--loglevel", default="warning")
def main(config_path, camera, iphone, server, describe_server, remote, device, mode, skip_calibration,
         no_music, benchmark, loglevel):
    logging.basicConfig(level=loglevel.upper())
    cfg, mapper = load_config(config_path)
    if remote:
        cfg["music"]["server_url"] = REMOTE_URL
    if server:
        cfg["music"]["server_url"] = server
    if device:
        cfg["tracking"]["device"] = device
    if mode:
        cfg["tracking"]["mode"] = mode
    if iphone:
        camera = find_camera("iPhone Camera")
    if camera is None:
        camera = find_camera(cfg.get("camera", {}).get("name", "C920"))
    CollabFrontend(cfg, mapper, camera=camera, no_music=no_music, benchmark=benchmark,
                   skip_calibration=skip_calibration, describe_server=describe_server).run()


if __name__ == "__main__":
    main()
