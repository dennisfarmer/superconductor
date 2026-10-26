"""Collaborative mode: people bring objects, each tracked object drives music
parameters (by its distance to a reference point, being held, or being in view).

The camera, the view and the sound are in the browser (the camera page,
http://localhost:8467/, app.html), whether this process runs natively, in Docker or
in Singularity. The page detects and tracks objects itself (YOLOE + BoT-SORT on
WebGPU: vision_js, tracker_js) and sends its detections with each frame; without
WebGPU it sends plain frames and YOLOE + BoT-SORT run here (compatibility mode).
Recognition (DINOv2), identities and the music mapping always run here. What to
draw over the video goes back to the page as JSON (`overlay`, see
CAMERA_PROTOCOL.md). Magenta RealTime 2 runs in a separate server process, locally
(`make server`) or on the cluster through the SSH tunnel (`--where remote`); see
superconductor_client/client.py and superconductor_server/mrt2_server.py. This
process sends inputs on `/stream/input`. The page, the playback plugin, or
sc-audio (`audio_receiver.py`) play `/stream/playback`.

Two modes ([tracking] mode, or --mode):
- unsupervised (default): every object gets a permanent id when it first appears
  and is cached in the object library (superconductor_client/tracking/library/<id>/), so it is recognized again
  later, across restarts. The describe server (superconductor_client/describe/) writes
  a one-time description of each new object and suggests an instrument from it.
  Names and instruments are edited on the prompts page (tracking_view.py,
  http://localhost:8467/prompts).
- calibration: starts with "touch the bulbasaur", ... (unless --skip-calibration);
  only those objects are tracked.

Objects close together can combine (combos, set up on the objects page): their
instruments are replaced by the combo's until they separate.

Keys on the camera page: [ ] combo distance, c calibrate. Click to move the reference
point (during calibration: click an object to label it). Ctrl-C here quits.
"""
import json
import logging
import os
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
import tomllib
from pathlib import Path
from urllib.parse import urlparse

import click
import cv2
import numpy as np
import yaml

from superconductor_client.addresses import AddressError, load_place
from superconductor_client.tracking.describe_client import DescribeClient, default_icon_svg
from superconductor_client.client import HttpMagentaClient, MagentaClient
from superconductor_client.tracking.object_tracking import (Calibration, Combo, ObjectLibrary, ObjectTracker, Parameter,
                                              ParameterMapper)
from superconductor_client.tracking.object_tracking.identity import box_gap
from superconductor_client.tracking.object_tracking.tracker import TRACKER_CONFIG
from superconductor_client.tracking.object_tracking.library import bgr_to_hex, hex_to_bgr, service_cards
from superconductor_client.tracking.object_tracking.mapper import PALETTE
from superconductor_client.tracking.web_model import ensure_web_model
from superconductor_client.tracking.tracking_view import TrackingView

DEFAULT_CONFIG = Path(__file__).parent / "collab.toml"


def tracking_path(path):
    """A path from collab.toml. Absolute paths stay put. Relative ones are under
    superconductor_client/tracking, so library/ and var/ live next to this file."""
    if not path:
        return path
    p = Path(path)
    return p if p.is_absolute() else Path(__file__).parent / p
# prompt of the "test beat" checkbox on the objects page. MusicCoCa embeds text, so
# describe only what should be there ("no vocals" would add a pull toward vocals).
TEST_BEAT = "solo drum kit groove, isolated drums, instrumental percussion only"
COMBO_COLORS = [(60, 220, 220), (220, 220, 80), (200, 90, 220), (255, 160, 60)]


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
    def __init__(self, cfg, mapper, no_music=False, benchmark=None,
                 skip_calibration=False, describe_server=None, client_url=None):
        self.cfg = cfg
        self.mapper = mapper
        self.benchmark = benchmark
        music = cfg["music"]
        if "server_url" in music:
            raise SystemExit("[music] server_url is not read. Set where to a name in "
                             "superconductor_client/addresses.toml.")
        try:
            self.place = load_place(music.get("where", "local"),
                                    gateway_port=os.environ.get("SC_GATEWAY_PORT"))
        except AddressError as exc:
            raise SystemExit(str(exc)) from exc

        self.magenta = None
        if not no_music:
            # make client starts client.py on its own; tracking then sends recipes there.
            # On its own, tracking starts the client process itself.
            if client_url:
                self.magenta = HttpMagentaClient(client_url)
            else:
                self.magenta = MagentaClient(
                    server_url=self.place.direct.input,
                    credits=music["credits"],
                    midi_port=music.get("midi_port", 8470),
                )
            self.magenta.start()

        t = cfg["tracking"]
        self.mode = t.get("mode", "unsupervised")
        unsupervised = self.mode == "unsupervised"
        # The client stays off the GPU on Linux. The L40S is Exclusive_Process, so the
        # music server keeps it. A Mac leaves this unset and the embedder uses MPS.
        # An explicit embed_device (the container's "cuda") is kept.
        embed_device = t.get("embed_device")
        if embed_device is None and os.uname().sysname != "Darwin":
            embed_device = "cpu"
        self.tracker = ObjectTracker(model=t["model"], classes=t.get("classes"),
                                     conf=t.get("conf", 0.3), imgsz=t.get("imgsz", 480),
                                     device=t.get("device"),
                                     # the library replaces identities.json in unsupervised mode
                                     identity_dir=None if unsupervised else tracking_path(t.get("identity_dir")),
                                     embedder=t.get("embedder"), embed_imgsz=t.get("embed_imgsz", 224),
                                     embed_device=embed_device,
                                     embeds_per_frame=t.get("embeds_per_frame", 2),
                                     identity=cfg.get("identity", {}))
        registry = self.tracker.registry
        self.library = None
        if unsupervised and t.get("library_dir"):
            embedder = self.tracker.embedder
            self.library = ObjectLibrary(tracking_path(t["library_dir"]), embedder.name if embedder else None)
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
        # knobs added on the prompts page, plus icons for prompts other services send
        self.knobs = []
        self.service_icons = {}
        self._icon_errors = {}
        if self.library is not None:
            self.knobs = [k for k in self.library.load_parameters()
                          if isinstance(k, dict) and str(k.get("id", "")).isdigit()]
            for knob in self.knobs:
                knob["id"] = int(knob["id"])
            icons = self.library.load_icons()
            self.service_icons = icons if isinstance(icons, dict) else {}
        for knob in self.knobs:
            self._sync_knob(knob)

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
        web_cfg = cfg.get("web", {})
        # the page's detector: the exported model (make web-model; exported here if its
        # classes don't match [tracking] classes) and onnxruntime-web (make web-deps)
        web_dir = tracking_path(web_cfg.get("web_dir", "var/web"))
        detection = web_cfg.get("detection", "auto")
        model_ok = detection != "container" and ensure_web_model(t["model"], t.get("classes") or [], web_dir)
        # camera page plays on the browser's playback address when its sound button is pressed
        links = dict(web_cfg.get("public_urls") or {})
        links["server"] = self.place.browser.http
        page_config = {"playback_url": self.place.browser.playback,
                       "credits": music.get("credits", 3),
                       "links": links,
                       "detector": {"mode": detection if model_ok else "container",
                                    "model_url": "/web/models/yoloe.onnx" if model_ok else None,
                                    "ort_url": "/web/ort/" if (web_dir / "ort").is_dir() else None,
                                    "classes": t.get("classes") or [], "conf": t.get("conf", 0.3),
                                    "iou": 0.7, "input": [480, 288],
                                    # the page's BoT-SORT (tracker_js) uses the same settings as this one
                                    "tracker": yaml.safe_load(TRACKER_CONFIG.read_text())}}
        self.panel = TrackingView(web_cfg.get("port", 8467), web_cfg.get("host", "localhost"),
                              page_config=page_config, web_dir=web_dir)
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
        print(f"prompts page: {self.panel.url}")

        self._frame_shape = None
        self._detector = None  # "browser" | "container": who detected the last frame
        self._track_session = None  # the detector whose track ids are in _track_ids
        self._track_ids = {}  # its track_id -> the track_id used here (_track_id)
        self._track_counter = 0
        self._last_send = 0.0
        self._fps = 0.0
        self._track_ms = 0.0
        self._fps_log = []
        self._rtf_log = []

    def _click(self, x, y):
        """a click on the camera page at (x, y), 0..1 as displayed (mirrored)"""
        point = (1 - x, y)  # display is mirrored; use unflipped coords
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

    def _knob_param_name(self, knob):
        """the prompt is the parameter's name"""
        base = (knob.get("prompt") or "").strip() or f"parameter {knob['id']}"
        for param in self.mapper.parameters:
            if param.knob != knob["id"] and param.name == base:
                return f"{base} #{knob['id']}"
        return base

    def _knob_weight(self, knob):
        """slider position, 0.00–1.00"""
        try:
            value = float(knob.get("value", 0.0))
        except (TypeError, ValueError):
            value = 0.0
        value = max(0.0, min(1.0, value))
        knob["value"] = value
        return value

    def _sync_knob(self, knob):
        """Keep the mapper parameter for a prompts-page parameter in line with it.

        The prompt is its name. The slider is its weight and goes straight into
        the recipe; distance to the reference point does not change it."""
        color = hex_to_bgr(knob.get("color")) or PALETTE[(int(knob["id"]) - 1) % len(PALETTE)]
        knob["color"] = bgr_to_hex(color)
        prompt = (knob.get("prompt") or "").strip() or None
        value = self._knob_weight(knob)
        param = next((p for p in self.mapper.parameters if p.knob == knob["id"]), None)
        if param is None:
            self.mapper.parameters.append(Parameter(
                name=self._knob_param_name(knob), kind="prompt", prompt=prompt,
                color=color, knob=knob["id"], value=value))
        else:
            param.name = self._knob_param_name(knob)
            param.prompt, param.color = prompt, color
            param.value = value
            param.identity = None

    def _save_knobs(self):
        if self.library is not None:
            self.library.save_parameters(self.knobs)

    def _save_icons(self):
        if self.library is not None:
            self.library.save_icons(self.service_icons)

    def _request_icon(self, key, text):
        text = (text or "").strip()
        if not text:
            return
        if self.describer is None:
            svg = default_icon_svg()
            if svg:
                self._store_icon(key, svg)
            else:
                self._icon_errors[key] = "describe server is not running"
                self.say("describe server is not running")
            return
        self._icon_errors.pop(key, None)
        self.describer.icon(key, text)

    def _store_icon(self, key, svg):
        self._icon_errors.pop(key, None)
        if key.startswith("knob:"):
            kid = key.split(":", 1)[1]
            knob = next((k for k in self.knobs if str(k["id"]) == kid), None)
            if knob is None:
                return
            knob["svg"] = svg
            knob["svg_rev"] = int(knob.get("svg_rev") or 0) + 1
            self._save_knobs()
            self.say(f"parameter #{kid}: icon ready", 2.0)
        elif key.startswith("svc:"):
            prompt = key[4:]
            if not prompt:
                return
            self.service_icons[prompt] = svg
            self._save_icons()
            self.say(f"{prompt}: icon ready", 2.0)

    def _param_action(self, kind, id, body):
        if kind == "param_add":
            new_id = max((int(k["id"]) for k in self.knobs), default=0) + 1
            knob = {"id": new_id, "prompt": "", "value": 0.0,
                    "color": bgr_to_hex(PALETTE[(new_id - 1) % len(PALETTE)]),
                    "svg": None, "svg_rev": 0}
            self.knobs.append(knob)
            self._sync_knob(knob)
            self._save_knobs()
            self.say(f"parameter #{new_id}", 2.0)
            return
        if isinstance(id, str) and id.startswith("svc:"):
            if kind == "param_icon":
                self._request_icon(id, id[4:])
            return
        if not isinstance(id, int):
            return
        knob = next((k for k in self.knobs if k["id"] == id), None)
        if knob is None:
            return
        if kind == "param_delete":
            self.knobs = [k for k in self.knobs if k["id"] != id]
            self.mapper.parameters = [p for p in self.mapper.parameters if p.knob != id]
            self._icon_errors.pop(f"knob:{id}", None)
            self._save_knobs()
            self.say(f"deleted parameter #{id}", 2.0)
            return
        if kind == "param_edit":
            prompt_changed = False
            if "name" in body:
                knob["name"] = body["name"] or ""
            if "prompt" in body:
                prompt = body["prompt"] or ""
                prompt_changed = (knob.get("prompt") or "") != prompt
                knob["prompt"] = prompt
            if "value" in body:
                knob["value"] = body["value"]
            if "color" in body:
                if body["color"] is None:
                    knob["color"] = bgr_to_hex(PALETTE[(id - 1) % len(PALETTE)])
                elif hex_to_bgr(body["color"]):
                    knob["color"] = body["color"]
            self._sync_knob(knob)
            self._save_knobs()
            if prompt_changed:
                self._request_icon(f"knob:{id}", knob.get("prompt") or "")
            return
        if kind == "param_icon":
            self._request_icon(f"knob:{id}", knob.get("prompt") or "")

    def _parameter_cards(self, busy, in_view):
        """prompts-page knobs, then prompts arriving from other services (Atlas drums, ...)"""
        cards = []
        for knob in self.knobs:
            param = next((p for p in self.mapper.parameters if p.knob == knob["id"]), None)
            if param is None:
                self._sync_knob(knob)
                param = next(p for p in self.mapper.parameters if p.knob == knob["id"])
            key = f"knob:{knob['id']}"
            svg = knob.get("svg") if isinstance(knob.get("svg"), str) else None
            cards.append({
                "id": knob["id"], "prompt": knob.get("prompt") or "",
                "color": bgr_to_hex(param.color), "value": round(float(param.value), 2),
                "origin": "knob", "editable": True, "badge": "", "live": False,
                "svg": svg, "svg_rev": knob.get("svg_rev") or 0,
                "drawing": key in busy, "error": self._icon_errors.get(key),
            })
        cards += service_cards(self.atlas_prompts(), self.service_icons, busy, self._icon_errors)
        return cards

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
            if isinstance(id, str) and (id.startswith("knob:") or id.startswith("svc:")):
                if kind == "svg":
                    self._store_icon(id, text)
                elif kind == "error":
                    self._icon_errors[id] = text
                    self.say(text)
                continue
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
        while not self.panel.actions.empty():
            kind, id, body = self.panel.actions.get()
            if kind.startswith("combo"):
                self._combo_action(kind, id, body)
                continue
            if kind.startswith("param"):
                self._param_action(kind, id, body)
                continue
            if kind == "click":  # camera page
                self._click(body["x"], body["y"])
                continue
            if kind == "key":  # camera page
                self.handle_key(body["key"])
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
                if "base_enabled" in body:
                    self.settings["base_enabled"] = bool(body["base_enabled"])
                    prompt = self.cfg["music"].get("base_prompt") or "base prompt"
                    self.say(f"{prompt} {'in the recipe' if self.settings['base_enabled'] else 'out of the recipe'}", 2.0)
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
                self.say(f"deleted #{ident.id} (moved to superconductor_client/tracking/library/.trash)")

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
        prompts = ", ".join(t.get("classes") or [])
        if self._detector == "browser":
            detection = {"model": f"YOLOE-11s (prompts: {prompts}) + BoT-SORT, in the browser (WebGPU)",
                         "where": "browser"}
        else:
            detection = {"model": f"YOLOE-11S-seg ({t['model']}, prompts: {prompts}) + BoT-SORT, "
                                  f"on {self.tracker.device}" + ("" if self._detector else " (no frames yet)"),
                         "where": "local"}
        rows = [{"task": "Detection + tracking", **detection}]
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
            if info.get("svg_model"):
                backend = info.get("svg_backend") or "unset"
                state = "ready" if info.get("svg_ready") else (info.get("svg_error") or "loads on the first icon")
                rows.append({"task": "Parameter icons",
                             "model": f"{info['svg_model']} via {backend} ({state})",
                             "where": where(info) if info else "-"})
        if self.magenta is None:
            music = {"model": "music off", "where": "-"}
        elif not self.magenta.connected:
            music = {"model": f"not connected ({self.place.direct.input})", "where": "-"}
        else:
            info = self.magenta.info
            music = {"model": f"Magenta RealTime 2 {info.get('model')} ({info.get('backend')})",
                     "where": where(info)}
        rows.append({"task": "Music", **music})
        return rows

    # path each service is probed at (GET) for the Services tab; Atlas is found by _find_atlas
    PROBES = {"server": "/tempo/status", "client_midi": "/status", "describe": "/health",
              "midi_player": "/api/state", "conduct": "/api/stats"}

    def _service_urls(self):
        """http base URL of each service (None: not used)"""
        svc = self.cfg.get("services", {})
        midi_port = self.cfg["music"].get("midi_port", 8470)
        return {
            # probed from this process, so the direct address, not the browser's
            "server": self.place.direct.http,
            # tracking and the objects page are this app, on one port
            "tracking": f"http://localhost:{self.panel.port}",
            "objects": f"http://localhost:{self.panel.port}",
            "client_midi": f"http://localhost:{midi_port}" if midi_port and self.magenta is not None else None,
            "describe": self.describer.server_url if self.describer is not None
            else self.cfg.get("describe", {}).get("server_url", "http://localhost:9200"),
            "midi_player": svc.get("midi_player_url", "http://localhost:8475"),
            "conduct": svc.get("conduct_url"),
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

        # [web] public_urls: where the browser reaches each service, when that isn't its
        # own URL (in the container: through the gateway, e.g. http://midi.localhost:8080)
        public = self.cfg.get("web", {}).get("public_urls", {})

        def row(key, name, started, endpoints, note=""):
            base = urls.get(key)
            if not base:
                return None
            base = base.rstrip("/")
            up = True if key in ("tracking", "objects") else (alive[key] is not None) if key in alive else None
            # same port as the objects page, so the same address the browser uses for it
            if key == "tracking" and key not in public and "objects" in public:
                shown = public["objects"].rstrip("/")
            else:
                shown = public.get(key, base).rstrip("/")
            return {"name": name, "started": started, "base": shown, "up": up, "note": note,
                    "endpoints": [{"method": m, "url": u if "://" in u else shown + u} for m, u in endpoints]}

        engine = alive.get("atlas_engine") or {}
        if engine.get("sink") == "superconductor":
            atlas_note = (f"sending notes/drums to {engine.get('client_midi')}"
                          + (" (connected)" if engine.get("connected") else " (client not connected)"))
        else:
            atlas_note = "standalone: plays through its own speakers" if engine else ""
        located = where(self.place.direct.http, self.magenta.info if self.magenta is not None else None)
        server_note = self.place.description + (f" {located}." if located else "")
        rows = [
            row("server", "Music server (mrt2_server.py)", f"addresses.toml · {self.place.name}",
                self.place.service_endpoints(), server_note),
            row("tracking", "Tracking page (this app)", "superconductor_client: make client",
                [("page", "/")]),
            row("objects", "Prompts page (this app)", "superconductor_client: make client",
                [("page", "/prompts"), ("GET", "/api/objects")]),
            row("client_midi", "client_midi (this app)", "superconductor_client: make client",
                [("GET", "/status"), ("GET", "/input"), ("POST", "/pattern")],
                "note sources (Atlas, MIDI player) POST patterns here"),
            row("describe", "Describe server", "superconductor_client/describe/: make describe",
                [("GET", "/health"), ("POST", "/describe"), ("POST", "/instrument"), ("POST", "/name"),
                 ("POST", "/svg")],
                where(urls["describe"], alive.get("describe"))),
            row("conduct", "Conducting page (MediaPipe hands)",
                "superconductor_client/conducting/: make conduct",
                [("page", "/"), ("WS", "/ws/hands"), ("WS", "/ws/hands/monitor"), ("GET", "/api/stats")],
                "hand movement sets the music server's tempo; the music plays on the camera page"),
            row("midi_player", "MIDI player", "superconductor_client/midi/: make player",
                [("page", "/"), ("GET", "/api/state"), ("POST", "/api/play"), ("POST", "/api/stop"),
                 ("POST", "/api/upload"), ("POST", "/api/settings")]),
            row("atlas", "Atlas app", "superconductor_atlas: make run", [("page", "/")]),
            row("atlas_engine", "Atlas engine", "started by Atlas's make run",
                [("GET", "/status"), ("POST", "/timeline"), ("POST", "/release"), ("POST", "/style"),
                 ("POST", "/settings"), ("POST", "/mix")], atlas_note),
        ]
        return [r for r in rows if r]

    def publish_panel(self, now):
        if now - self._last_publish < 0.5:
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
                            "parameters": self._parameter_cards(busy, in_view),
                            "other": self._other_cards(),
                            "settings": dict(self.settings, test_beat_prompt=TEST_BEAT), "music": music,
                            "models": self.models(), "services": self.services(),
                            "learning": {k: getattr(registry, k) for k in registry.LEARNING}}, views)

    def handle_key(self, key):
        """a key pressed on the camera page"""
        if key in ("[", "]") and self.mapper.combos:
            for combo in self.combos:
                combo["distance"] = max(0.0, round(combo["distance"] + (0.01 if key == "]" else -0.01), 3))
            self._apply_combos(save=True)
            self.say(f"combo distance {self.combos[0]['distance']:.2f}", 2.0)
        elif key == "c":
            if not self.calibration.labels:
                self.say("calibration mode only (--mode calibration); name objects on the objects page")
            else:
                for param in self.mapper.parameters:
                    param.identity = None
                self.calibration.start()

    def atlas_prompts(self):
        """style prompts set by SuperConductor Atlas (client_midi POST /prompts, e.g. the
        Drums page's {"drums": 0..1})"""
        return self.magenta.prompts if self.magenta else {}

    def _base_on(self):
        """the config base prompt is in the recipe unless the Other tab switches it off"""
        return bool(self.settings.get("base_enabled", True))

    def _other_cards(self):
        music = self.cfg["music"]
        prompt = (music.get("base_prompt") or "").strip()
        if not prompt:
            return []
        try:
            weight = float(music.get("base_weight") or 0.0)
        except (TypeError, ValueError):
            weight = 0.0
        return [{"id": "base", "prompt": prompt, "value": weight, "enabled": self._base_on()}]

    def recipe(self):
        """style weights sent to MRT2: base prompt, test beat, objects, combos and Atlas"""
        music = self.cfg["music"]
        weight = music["base_weight"] if self._base_on() else 0.0
        recipe = self.mapper.recipe(music["base_prompt"], weight)
        if self.settings["test_beat"]:  # always-on test sound (checkbox on the objects page)
            recipe[TEST_BEAT] = recipe.get(TEST_BEAT, 0.0) + 1.0
        for prompt, weight in self.atlas_prompts().items():
            recipe[prompt] = recipe.get(prompt, 0.0) + weight
        return recipe

    def mrt2_input(self):
        """What the overlay's "MRT2 input" panel shows, and client_midi's GET /input serves:
        every prompt in the recipe (base, test beat, objects, combos) as (prompt, source,
        value, share of the blend, BGR color), and the sampling controls. The server
        averages the prompts' embeddings by weight, so share = weight / sum of weights."""
        music = self.cfg["music"]
        rows = []  # (prompt, source, value, color)
        if self._base_on() and music["base_prompt"] and music["base_weight"] > 0:
            rows.append((music["base_prompt"], "base", music["base_weight"], (200, 200, 200)))
        if self.settings["test_beat"]:
            rows.append((TEST_BEAT, "test beat", 1.0, (120, 200, 255)))
        for prompt, weight in self.atlas_prompts().items():
            rows.append((prompt, "atlas", weight, (80, 180, 255)))
        for param in self.mapper.parameters:
            combo = self.mapper.combo_for(param.identity) if isinstance(param.identity, int) else None
            if param.identity is None and param.owner and param.owner.startswith("#"):
                ident = self.tracker.registry.get(param.owner)  # its object isn't in view this session
                state = (ident.user_label if ident and ident.user_label else param.owner) + " not seen"
            elif param.knob is not None:
                state = "parameter"
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

    def overlay(self, objects):
        """What the camera page draws over the video for this frame: the objects, hands,
        tracks still being classified, combos and the reference point (see
        CAMERA_PROTOCOL.md). Boxes and centers are in the frame's pixels, unmirrored;
        label text and colors are worked out here so the page only draws."""
        h, w = self._frame_shape
        px = lambda box: [round(float(v), 1) for v in box]
        out = {"reference": {"x": self.mapper.reference[0], "y": self.mapper.reference[1],
                             "radius": self.mapper.max_distance},
               "hands": [px(hand.box) for hand in self.tracker.hands],
               "pending": [], "combos": [], "objects": []}

        # tracks that are still being classified
        for obj in self.tracker.registry.pending:
            hits, score, like = self.tracker.registry.pending_info(obj.track_id)
            out["pending"].append({"box": px(obj.box),
                                   "label": "? new" if score is None else f"? {like} {score:.2f}"})

        # combos: a frame around objects that combined, and their distance while apart
        by_label = {o.class_name: o for o in objects}
        by_label.update({f"#{o.identity}": o for o in objects})
        for combo in self.mapper.combos:
            param = self.mapper.get(combo.parameter)
            members = [by_label.get(label) for label in combo.objects]
            if any(m is None for m in members):
                continue
            centers = [[round(m.center[0] * w, 1), round(m.center[1] * h, 1)] for m in members]
            if combo.active:
                out["combos"].append({
                    "active": True, "centers": centers, "color": bgr_to_hex(param.color),
                    "box": [combo.box[0] * w - 14, combo.box[1] * h - 14, combo.box[2] * w + 14, combo.box[3] * h + 14],
                    "label": f"{self._combo_name(combo)} = {param.prompt}"})
            elif len(members) == 2:
                gap = box_gap(members[0].box_normalized, members[1].box_normalized)
                close = gap < 3 * combo.distance
                out["combos"].append({"active": False, "centers": centers,
                                      "color": bgr_to_hex(param.color if close else (140, 140, 140)),
                                      "label": f"{gap:.2f} / {combo.distance:.2f}"})

        for obj in objects:
            combo = self.mapper.combo_for(obj.identity)
            param = self.mapper.get(combo.parameter) if combo else self.mapper.parameter_for(obj.identity)
            ident = self.tracker.registry.get(obj.identity)
            color = param.color if param else (ident and ident.display_color) or (150, 150, 150)
            text = f"#{obj.identity} {obj.class_name}"
            if combo is not None:
                text += " (combined)"
            else:
                params = self.mapper.parameters_for(obj.identity)
                if params:
                    text += " -> " + ", ".join(f"{_param_name(p)} {p.value:.2f}" for p in params)
            if obj.coasted:
                text += " (last seen)"
            touching = self.calibration.active and obj.identity == self.calibration.candidate
            out["objects"].append({
                "identity": obj.identity, "track_id": obj.track_id, "class_name": obj.class_name,
                "box": px(obj.box), "center": [round(obj.center[0] * w, 1), round(obj.center[1] * h, 1)],
                "color": bgr_to_hex(color), "coasted": bool(obj.coasted), "combined": combo is not None,
                "progress": round(self.calibration.progress, 3) if touching else None, "label": text})
        return out

    def hud(self):
        """The rest of what the page shows: the calibration message, the MRT2 input panel,
        combo banners, recent messages and the status line."""
        rows, controls = self.mrt2_input()
        now = time.time()
        self.messages = [(t, until) for t, until in self.messages if until > now]
        stats = self.magenta.poll_stats() if self.magenta else {}
        music = "music off" if self.magenta is None else (
            "loading MRT2..." if not self.magenta.connected else
            f"{self.magenta.info.get('model')} ({self.magenta.info.get('backend')}) "
            f"x{stats.get('rtf', 0):.2f} realtime  underruns {stats.get('underruns', 0)}"
            + ("  PAUSED" if self.magenta.paused else ""))
        if stats.get("rtf"):
            self._rtf_log.append(stats["rtf"])
        banners = []
        for combo in self.mapper.combos:
            if combo.active:
                param = self.mapper.get(combo.parameter)
                banners.append({"text": f"{self._combo_name(combo)} combined: {param.prompt}",
                                "color": bgr_to_hex(param.color)})
        return {"message": self.calibration.message() or None,
                "mrt2": {"rows": [{"prompt": p, "source": src, "value": round(v, 3), "share": round(sh, 3),
                                   "color": bgr_to_hex(c)} for p, src, v, sh, c in rows],
                         "controls": {k: round(v, 3) for k, v in controls.items()}},
                "banners": banners, "log": [t for t, _ in self.messages],
                "status": {"vision_fps": round(self._fps, 1), "track_ms": round(self._track_ms, 1),
                           "detector": self._detector, "music": music}}

    def _track_id(self, session, track_id):
        """A detector's track id -> the one used here. Each detector (a page load, or this
        process's own YOLO) numbers its tracks from 1, and the registry binds identities
        to track ids, so they are made unique; a new detector's tracks start fresh."""
        if session != self._track_session:
            self._track_session = session
            self._track_ids = {}
        key = int(track_id)
        if key not in self._track_ids:
            self._track_counter += 1
            self._track_ids[key] = self._track_counter
        return self._track_ids[key]

    def step(self, frame, header=None, now=None):
        """Track one frame, update parameters; returns the overlay for the page.
        `header`: the page's frame header. With `objects` in it the page has detected;
        without (or no header) YOLO + BoT-SORT run here (compatibility mode)."""
        self._frame_shape = frame.shape[:2]
        t0 = time.time()
        if header is not None and header.get("objects") is not None:
            self._detector = "browser"
            session = str(header.get("session", "browser"))
            detections = [(self._track_id(session, o["track_id"]), str(o["class_name"]), float(o["confidence"]),
                           tuple(float(v) for v in o["box"]), None) for o in header["objects"]]
        else:
            self._detector = "container"
            detections = [(self._track_id("container", tid), *rest) for tid, *rest in self.tracker._detect(frame)]
        t_det = time.time()  # DEBUG(offline): stage timing
        objects = self.tracker.track_detections(frame, detections, now=now)
        t_track = time.time()
        self._track_ms = 0.9 * self._track_ms + 0.1 * (time.time() - t0) * 1000
        self._objects = objects
        self._sync_colors()
        if self.calibration.active:
            self.calibration.update([o for o in objects if not o.coasted], self.tracker.hands)
        self.mapper.update(objects, now=now, hands=self.tracker.hands)
        t_map = time.time()
        self.send()
        t_send = time.time()
        self.apply_panel_actions()
        self.apply_describe_results()
        self.publish_panel(time.time() if now is None else now)
        t_panel = time.time()
        h, w = self._frame_shape
        out = {"type": "overlay", "seq": header.get("seq") if header else None, "width": w, "height": h,
               **self.overlay(objects), **self.hud()}
        t_end = time.time()
        if t_end - t0 > 0.3:  # DEBUG(offline)
            print(f"[slow step] {1000 * (t_end - t0):.0f} ms: detect {1000 * (t_det - t0):.0f}, "
                  f"track {1000 * (t_track - t_det):.0f}, mapper {1000 * (t_map - t_track):.0f}, "
                  f"send {1000 * (t_send - t_map):.0f}, panel {1000 * (t_panel - t_send):.0f}, "
                  f"overlay+hud {1000 * (t_end - t_panel):.0f}; {len(detections)} detections", flush=True)
        return out

    def run(self):
        """Frames come from the camera page; what to draw goes back to it. Only the newest
        frame is tracked; frames that arrive meanwhile are skipped."""
        print(f"camera page: http://localhost:{self.panel.port}/ (open it in a browser and press Start)")
        start = time.time()
        prev = start
        seq = 0
        last_frame_at = last_overlay_at = time.time()  # DEBUG(offline)
        try:
            while True:
                latest = self.panel.latest_frame(seq)
                if latest is None:
                    if self.panel.browser_connected and time.time() - last_frame_at > 1.0 and last_overlay_at >= last_frame_at:
                        print(f"[no frame] answered the last frame, none new for {time.time() - last_frame_at:.1f} s "
                              "(page side)", flush=True)
                        last_overlay_at = last_frame_at - 1  # print once per gap
                    # no new frame (no camera page, or one that stopped sending: a hidden
                    # tab, a slow detector); keep page edits, descriptions and the music
                    # status flowing
                    if self.magenta is not None:
                        self.magenta.poll_stats()
                        self.send()  # Atlas's prompts reach MRT2 without a camera page too
                    self.apply_panel_actions()
                    self.apply_describe_results()
                    self.publish_panel(time.time())
                    time.sleep(0.005)
                    continue
                seq, jpeg, header = latest
                gap = time.time() - last_frame_at
                last_frame_at = time.time()
                frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    print(f"[lost] frame {header.get('seq') if header else '?'} did not decode: no overlay", flush=True)
                    continue
                self.panel.send_overlay(self.step(frame, header))
                last_overlay_at = time.time()
                if gap > 1.0:
                    print(f"[frame gap] {gap:.1f} s since the previous frame", flush=True)

                now = time.time()
                self._fps = 0.9 * self._fps + 0.1 / max(now - prev, 1e-6)
                prev = now
                if self.magenta is None or self.magenta.connected:
                    self._fps_log.append(self._fps)
                if self.benchmark and now - start > self.benchmark:
                    break
        except KeyboardInterrupt:
            pass
        self.stop()

    def stop(self):
        self.tracker.registry.save()
        self.tracker.registry.flush()
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

    `jac start` takes the next free port when its own is in use (e.g. 8769 for 8764), so the
    ports Atlas's processes listen on are tried along with the configured ones, and a port
    counts only if it answers like Atlas: its functions in /openapi.json, the engine's /status
    fields. With search=False, just the configured URLs, not running."""
    found = {}
    for key, default, marker, path, check in (
            ("atlas", "http://localhost:8764", "jac start main.jac", "/openapi.json",
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
@click.option("--where", default=None,
              help="Which music server, a name in superconductor_client/addresses.toml. "
                   "Overrides [music] where.")
@click.option("--describe-server", default=None,
              help="Override [describe].server_url (describe server for one-time object descriptions).")
@click.option("--device", default=None, type=click.Choice(["cpu", "mps", "cuda"]),
              help="Override [tracking].device (where YOLO runs in compatibility mode).")
@click.option("--detection", default=None, type=click.Choice(["auto", "browser", "container"]),
              help="Override [web].detection (where YOLOE + BoT-SORT run; auto: the browser if it has WebGPU).")
@click.option("--mode", default=None, type=click.Choice(["unsupervised", "calibration"]),
              help="Override [tracking].mode.")
@click.option("--skip-calibration", is_flag=True,
              help="Reuse objects labeled in a previous session instead of 'touch the ...'.")
@click.option("--client-url", default=None,
              help="client_midi of an already running client (make client). "
                   "Tracking sends recipes there instead of starting the client itself.")
@click.option("--no-music", is_flag=True, help="Only run tracking (no MRT2).")
@click.option("--benchmark", default=None, type=float,
              help="Run for N seconds, then print vision fps / MRT2 realtime stats.")
@click.option("--loglevel", default="warning")
def main(config_path, where, describe_server, device, detection, mode, skip_calibration,
         client_url, no_music, benchmark, loglevel):
    logging.basicConfig(level=loglevel.upper())
    cfg, mapper = load_config(config_path)
    if where:
        cfg["music"]["where"] = where
    if device:
        cfg["tracking"]["device"] = device
    if detection:
        cfg.setdefault("web", {})["detection"] = detection
    if mode:
        cfg["tracking"]["mode"] = mode
    CollabFrontend(cfg, mapper, no_music=no_music, benchmark=benchmark,
                   skip_calibration=skip_calibration, describe_server=describe_server,
                   client_url=client_url).run()


if __name__ == "__main__":
    main()
