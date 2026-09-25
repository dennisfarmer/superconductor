"""Collaborative mode: people bring objects, each tracked object drives music
parameters (by its distance to a reference point, being held, or being in view).

Webcam + YOLO/BoT-SORT run in this process; Magenta RealTime 2 runs locally in
a separate process (see magenta_local.py).

Two modes ([tracking] mode, or --mode):
- unsupervised (default): every object gets its own id when it first appears,
  known objects are recognized from the object library (library/), and anything
  can be saved / mapped with typed commands (see object_tracking/commands.py).
- calibration: starts with "touch the bulbasaur", ... (unless --skip-calibration);
  only those objects are tracked.

Objects close together can combine ([[combos]]): their parameters are replaced by
the combo's until they separate.

Keys: / or Enter type a command, s save the selected object, [ ] combo distance,
c calibrate, q quit. Click an object to select it (during calibration: to label
it), click anywhere else to move the reference point.
"""
import logging
import queue
import re
import subprocess
import sys
import threading
import time
import tomllib
from pathlib import Path

import click
import cv2
import numpy as np

from superconductor.magenta_local import LocalMagentaClient
from superconductor.object_tracking import (Calibration, Combo, Commands, ObjectLibrary, ObjectTracker,
                                            Parameter, ParameterMapper)
from superconductor.object_tracking.identity import box_gap

DEFAULT_CONFIG = Path(__file__).parent / "collab.toml"


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
    parameters = [Parameter(name=p["name"], kind=p["kind"], prompt=p.get("prompt"),
                            min=p.get("min", 0.0), max=p.get("max", 1.0),
                            color=tuple(p.get("color", (255, 255, 255))))
                  for p in cfg["parameters"]]
    combos = []
    for c in cfg.get("combos", []):
        combo = Combo(objects=tuple(c["objects"]), parameter=c.get("parameter", ""),
                      distance=c.get("distance", 0.05), release_factor=c.get("release_factor", 1.5),
                      trigger=c.get("trigger", "near"))
        if not combo.parameter:  # inline prompt instead of a [[parameters]] name
            combo.parameter = " + ".join(combo.objects)
            parameters.append(Parameter(name=combo.parameter, kind="prompt", prompt=c["prompt"],
                                        color=tuple(c.get("color", (60, 220, 220))), owner=combo.key))
        combos.append(combo)
    m = cfg["mapping"]
    # config reference is in displayed (mirrored) coords; mapper works unflipped
    rx, ry = m.get("reference", (0.5, 0.5))
    mapper = ParameterMapper(parameters=parameters, assignments=cfg.get("assign", {}), combos=combos,
                             reference=(1 - rx, ry),
                             max_distance=m.get("max_distance", 0.6),
                             smoothing=m.get("smoothing", 0.3),
                             hold_seconds=m.get("hold_seconds", 1.5),
                             release_after=m.get("release_after"))
    return cfg, mapper


class CollabFrontend:
    def __init__(self, cfg, mapper, camera=0, no_music=False, benchmark=None,
                 skip_calibration=False):
        self.cfg = cfg
        self.mapper = mapper
        self.benchmark = benchmark
        music = cfg["music"]

        self.magenta = None
        if not no_music:
            self.magenta = LocalMagentaClient(
                model=music["model"],
                frames_per_block=music["frames_per_block"],
                max_buffered_blocks=music["max_buffered_blocks"],
                model_dir=music.get("model_dir"),
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
                                     embedder=t.get("embedder"), embed_imgsz=t.get("embed_imgsz", 128),
                                     embeds_per_frame=t.get("embeds_per_frame", 2),
                                     identity=cfg.get("identity", {}))
        registry = self.tracker.registry
        self.library = None
        if t.get("library_dir"):
            self.library = ObjectLibrary(t["library_dir"], self.tracker.embedder.name
                                         if self.tracker.embedder else None)
            registry.add_library(self.library)
        registry.allow_new = unsupervised
        self.commands = Commands(registry, mapper, self.library)

        # "touch the <label>" for every label used in [assign]
        self.calibration = Calibration(list(mapper.assignments) if not unsupervised else [],
                                       registry, allow_new_after=unsupervised)
        known = {i.user_label for i in registry.identities}
        if not unsupervised and not (skip_calibration and set(mapper.assignments) <= known):
            self.calibration.start()
        self._objects = []
        self.selected = None  # identity id
        self.typing = None  # command line being typed in the window (None = not typing)
        self.messages = []  # (text, expires)
        self._stdin = queue.Queue()
        if sys.stdin is not None and sys.stdin.isatty():
            threading.Thread(target=self._read_stdin, daemon=True).start()
        print("type 'help' (here or after pressing / in the window) for commands")

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
            hit = next((o for o in self._objects if o.box_normalized and
                        o.box_normalized[0] <= point[0] <= o.box_normalized[2] and
                        o.box_normalized[1] <= point[1] <= o.box_normalized[3]), None)
            if hit is not None:
                self.selected = hit.identity
                self.say(f"selected #{hit.identity} {hit.class_name} (s: save, / map ...)")
            else:
                self.selected = None
                self.mapper.reference = point

    # ---------- commands ----------

    def _read_stdin(self):
        for line in sys.stdin:
            self._stdin.put(line)

    def say(self, text, seconds=6.0):
        print(text)
        for line in text.splitlines()[-8:]:
            self.messages.append((line, time.time() + seconds))
        self.messages = self.messages[-8:]

    def run_command(self, line):
        self.commands._last_objects = self._objects
        result = self.commands(line)
        if result:
            self.say(result, seconds=6.0 + 0.5 * result.count("\n"))
        if self.commands.show_request is not None:
            title, image = self.commands.show_request
            self.commands.show_request = None
            cv2.imshow(title, image)

    def save_selected(self):
        visible = [o for o in self._objects if not o.coasted]
        if self.selected is None and len(visible) == 1:
            self.selected = visible[0].identity
        ident = self.tracker.registry.get(self.selected) if self.selected is not None else None
        if ident is None:
            self.say("click an object first, then press s")
        elif ident.named:
            self.run_command(f"save {ident.user_label}")
        else:
            self.typing = f"save #{ident.id} "  # the user types the name

    def handle_key(self, key):
        """returns False to quit"""
        if key == 255:
            return True
        if self.typing is not None:
            if key in (13, 10):
                line, self.typing = self.typing, None
                self.run_command(line)
            elif key == 27:
                self.typing = None
            elif key in (8, 127):
                self.typing = self.typing[:-1]
            elif 32 <= key < 127:
                self.typing += chr(key)
            return True
        if key == ord("q"):
            return False
        if key in (ord("/"), 13, 10):
            self.typing = ""
        elif key == ord("s"):
            self.save_selected()
        elif key in (ord("["), ord("]")) and self.mapper.combos:
            d = self.commands.default_distance() + (0.01 if key == ord("]") else -0.01)
            self.commands.set_distance(d)
            self.say(f"combo distance {self.commands.default_distance():.2f}", 2.0)
        elif key == ord("c"):
            if not self.calibration.labels:
                self.say("calibration mode only (--mode calibration); use: save #<id> <name>")
            else:
                for param in self.mapper.parameters:
                    param.identity = None
                self.calibration.start()
        return True

    def send(self):
        now = time.time()
        if self.magenta is None or not self.magenta.connected:
            return
        if now - self._last_send < self.cfg["music"]["update_interval"]:
            return
        self._last_send = now
        music = self.cfg["music"]
        self.magenta.update_recipe(self.mapper.recipe(music["base_prompt"], music["base_weight"]))
        self.magenta.update_controls(self.mapper.controls())

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
                labels.append((f"{' + '.join(combo.objects)} = {param.name}",
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
            color = param.color if param else (150, 150, 150)
            x1, y1, x2, y2 = map(int, obj.box)
            cx, cy = int(obj.center[0] * w), int(obj.center[1] * h)
            if obj.coasted:  # not detected right now: last known position
                _dashed_rect(frame, (x1, y1), (x2, y2), color, 2)
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            if obj.identity == self.selected:
                cv2.rectangle(frame, (x1 - 5, y1 - 5), (x2 + 5, y2 + 5), (255, 255, 255), 1)
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
                    text += " -> " + ", ".join(f"{p.name} {p.value:.2f}" for p in params)
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

        # one banner per active combo
        banner_y = max(80 if not message else 130, 30 + 24 * len(self.mapper.parameters) + 30)
        for combo in self.mapper.combos:
            if combo.active:
                param = self.mapper.get(combo.parameter)
                text = f"{' + '.join(combo.objects)} combined: {param.name}"
                (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 1.0, 2)
                cv2.putText(overlay, text, ((overlay.shape[1] - tw) // 2, banner_y),
                            cv2.FONT_HERSHEY_DUPLEX, 1.0, param.color, 2)
                banner_y += 40

        y = 30
        for param in self.mapper.parameters:
            combo = self.mapper.combo_for(param.identity) if isinstance(param.identity, int) else None
            if param.identity is None:
                state = "free"
            elif isinstance(param.identity, str):
                state = "combined"
            else:
                state = f"#{param.identity}" + ("" if param.visible else " hidden") + \
                        (" > combo" if combo else "")
            cv2.putText(overlay, f"{param.name}"[:24], (15, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, param.color, 2)
            cv2.putText(overlay, state, (230, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, param.color, 1)
            cv2.rectangle(overlay, (350, y - 12), (350 + int(150 * param.value), y - 2), param.color, -1)
            cv2.rectangle(overlay, (350, y - 12), (500, y - 2), param.color, 1)
            y += 24

        # command line and recent command output
        now = time.time()
        self.messages = [(t, until) for t, until in self.messages if until > now]
        y = overlay.shape[0] - 45 - 22 * len(self.messages)
        for text, _ in self.messages:
            cv2.putText(overlay, text, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y += 22
        if self.typing is not None:
            cv2.putText(overlay, "> " + self.typing + "_", (15, overlay.shape[0] - 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (120, 255, 255), 2)

        stats = self.magenta.poll_stats() if self.magenta else {}
        music = "music off" if self.magenta is None else (
            "loading MRT2..." if not self.magenta.connected else
            f"MRT2 x{stats.get('rtf', 0):.2f} realtime  underruns {stats.get('underruns', 0)}")
        cv2.putText(overlay, f"vision {self._fps:.0f} fps ({self._track_ms:.0f} ms)  |  {music}  |  "
                             f"/ command  s save  [ ] combo distance  q quit",
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
        if self.calibration.active:
            self.calibration.update([o for o in objects if not o.coasted], self.tracker.hands)
        self.mapper.update(objects, now=now, hands=self.tracker.hands)
        self.send()
        while not self._stdin.empty():
            self.run_command(self._stdin.get())

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


def _dashed_rect(img, p1, p2, color, thickness=1, dash=10):
    (x1, y1), (x2, y2) = p1, p2
    for x in range(x1, x2, 2 * dash):
        cv2.line(img, (x, y1), (min(x + dash, x2), y1), color, thickness)
        cv2.line(img, (x, y2), (min(x + dash, x2), y2), color, thickness)
    for y in range(y1, y2, 2 * dash):
        cv2.line(img, (x1, y), (x1, min(y + dash, y2)), color, thickness)
        cv2.line(img, (x2, y), (x2, min(y + dash, y2)), color, thickness)


@click.command()
@click.option("--config", "config_path", default=str(DEFAULT_CONFIG), type=click.Path(exists=True))
@click.option("--camera", default=None, type=int,
              help="cv2 camera index (default: first camera matching [camera].name)")
@click.option("--model", default=None, type=click.Choice(["mrt2_base", "mrt2_small"]),
              help="Override [music].model from the config.")
@click.option("--model-dir", default=None, type=click.Path(exists=True, file_okay=False),
              help="Load <model-dir>/<model>/<model>.mlxfn, e.g. quantized/ for the 4-bit mrt2_base.")
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
def main(config_path, camera, model, model_dir, device, mode, skip_calibration, no_music, benchmark,
         loglevel):
    logging.basicConfig(level=loglevel.upper())
    cfg, mapper = load_config(config_path)
    if model:
        cfg["music"]["model"] = model
    if model_dir:
        cfg["music"]["model_dir"] = model_dir
    if device:
        cfg["tracking"]["device"] = device
    if mode:
        cfg["tracking"]["mode"] = mode
    if camera is None:
        camera = find_camera(cfg.get("camera", {}).get("name", "C920"))
    CollabFrontend(cfg, mapper, camera=camera, no_music=no_music, benchmark=benchmark,
                   skip_calibration=skip_calibration).run()


if __name__ == "__main__":
    main()
