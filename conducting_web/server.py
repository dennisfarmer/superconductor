"""The conducting page's process: hand landmarks from the browser in, tempo to the
MRT2 server out. What live_conductor.py does with OpenCV and Python MediaPipe, with
the camera, MediaPipe and the preview moved into the browser (app.html):

    http://localhost:8480/      the conducting page: camera, hand tracking (MediaPipe in a
                                Worker), and the trail / hand / status drawn over the video
    WS  /ws/hands               page -> here: {"type": "hands", "seq", "t_ms", "width",
                                "height", "hands": [[{x, y, z} x 21], ...], "handedness",
                                "detect_ms"}, or {"type": "key", "key": "f" | "r"}
                                here -> page: {"type": "conduct", "seq", ...} per frame
    WS  /ws/hands/monitor       listen-only (e.g. TouchDesigner): each frame's "hands"
                                message as the page sent it, plus "conduct": the reply
    GET /api/web                where the page finds MediaPipe, the model, and the server
    GET /api/stats              frames received, fps, age of the last frame, latencies

Each frame goes through Wenjue's code unchanged: StopGestureDetector (stop_gesture.py),
CycleTempoEstimator (cycle_tempo.py), and ServerTempoPlayer (server_playback.py), which
POSTs /tempo/target and /tempo/free to the server (CONDUCTING_INTEGRATION.md on the
server branch).

    python conducting_web/server.py --port 8480 --server-url http://localhost:9100
"""
import argparse
import asyncio
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from aiohttp import web

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from cycle_tempo import CycleTempoEstimator  # noqa: E402
from server_playback import ServerTempoPlayer  # noqa: E402
from stop_gesture import StopGestureDetector  # noqa: E402

WEB_DIR = ROOT / "var" / "web"  # make web-deps: mediapipe/ and models/


def smooth(old, new):
    """the camera page's smoothing of its status numbers (client-web app.html)"""
    return new if old is None else 0.7 * old + 0.3 * new


class TimedTempoPlayer(ServerTempoPlayer):
    """ServerTempoPlayer, also timing its requests: `post_ms`, the HTTP round trip of a
    /tempo/target or /tempo/free, and `to_server_ms`, from when the request was queued (the
    frame that set the target) until the server answered it, including the player's rate
    limit (at most one request per 0.25 s) and retries."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.post_ms = self.to_server_ms = None
        self.post_log, self.to_server_log = [], []
        self._queued_at = {}  # request body (by identity) -> when it was queued

    def _queue(self, path, body):
        super()._queue(path, body)
        now = time.monotonic()
        # replaced targets are never sent; forget them after a while
        self._queued_at = {k: v for k, v in self._queued_at.items() if now - v[1] < 10}
        self._queued_at[id(body)] = (body, now)

    def _request(self, path, body=None):
        if path == "/tempo/status":
            return super()._request(path, body)
        start = time.monotonic()
        status = super()._request(path, body)
        now = time.monotonic()
        with self.condition:
            self.post_ms = smooth(self.post_ms, (now - start) * 1000)
            self.post_log.append((now - start) * 1000)
            queued = self._queued_at.pop(id(body), None)
            if queued is not None and queued[0] is body:
                self.to_server_ms = smooth(self.to_server_ms, (now - queued[1]) * 1000)
                self.to_server_log.append((now - queued[1]) * 1000)
        return status


class Conductor:
    """One page's conducting: the per-frame body of live_conductor._run_live, given the
    hand landmarks the page found instead of a camera frame. `step` returns what the
    page draws (the same text as the OpenCV preview) and the timings for its status line."""

    def __init__(self, player, tempo_window=5.0):
        self.player = player
        self.tempo_window = tempo_window
        self.reset()

    def reset(self):
        self.estimator = CycleTempoEstimator(window_seconds=self.tempo_window)
        self.stop_detector = StopGestureDetector()
        self.last_t = None
        self.step_ms = None

    def step(self, hands, t_ns, aspect_ratio):
        start = time.perf_counter()
        if self.last_t is not None and t_ns <= self.last_t:
            # a new camera, or the page reloaded: the estimator needs increasing timestamps
            self.reset()
        self.last_t = t_ns
        hands = [[SimpleNamespace(x=p["x"], y=p["y"], z=p.get("z", 0.0)) for p in hand] for hand in hands]
        player = self.player
        reason = self.stop_detector.update(hands, t_ns, self.estimator.confidence, aspect_ratio)
        if reason:
            if not player.paused:
                player.set_intentional_stop()
            bpm = None
        else:
            x, y = (1 - hands[0][12].x, hands[0][12].y) if hands else (None, None)
            bpm = self.estimator.update(x, y, t_ns)
            player.set_tempo(bpm)
        status = (f"{bpm:.1f} BPM | {player.speed:.2f}x" if bpm is not None
                  else f"Holding {player.speed:.2f}x - waiting for motion")
        if reason:
            status = f"Released: {reason} - R to resume"
        started = self.estimator.started
        warmup = None if started is None else max(0.0, self.tempo_window - (t_ns * 1e-9 - started))
        self.step_ms = smooth(self.step_ms, (time.perf_counter() - start) * 1000)
        with player.condition:
            server = {k: player.status.get(k) for k in ("mode", "speed", "target_bpm", "heard_bpm", "model_bpm")}
            error = player.error
            post_ms, to_server_ms = player.post_ms, player.to_server_ms
        return {"bpm": bpm, "speed": player.speed, "released": bool(reason), "status": status,
                "detail": player.display_status(), "warmup": warmup, "step_ms": self.step_ms,
                "post_ms": post_ms, "to_server_ms": to_server_ms, "server": server, "error": error}

    def key(self, key):
        """F releases tempo control, R resumes with a fresh estimate (as in the OpenCV preview)"""
        if key == "f":
            self.stop_detector.reason = "Manual release"
            self.player.set_intentional_stop()
        elif key == "r" and self.stop_detector.reason:
            self.stop_detector = StopGestureDetector()
            self.estimator = CycleTempoEstimator(window_seconds=self.tempo_window)
            self.player.set_intentional_stop(False)


def parse_hands(message):
    """the landmarks of a "hands" message -> [[{x, y, z} x 21], ...]; ValueError if malformed"""
    hands = message.get("hands")
    if not isinstance(hands, list):
        raise ValueError("hands must be a list")
    for hand in hands:
        if not isinstance(hand, list) or len(hand) != 21:
            raise ValueError("each hand must have 21 landmarks")
        for p in hand:
            if not all(isinstance(p.get(k), (int, float)) and math.isfinite(p[k]) for k in ("x", "y")):
                raise ValueError("landmarks need finite x and y")
    return hands


class ConductingServer:
    def __init__(self, player, server_url, tempo_window=5.0):
        self.player = player
        self.server_url = server_url
        self.tempo_window = tempo_window
        self.frames = 0
        self.last_frame = None  # time.monotonic() of the newest frame
        self.fps = 0.0
        self._fps_count, self._fps_since = 0, time.monotonic()
        self.connections = 0
        self.monitors = set()  # open /ws/hands/monitor sockets
        self._sends = set()  # their pending sends (kept so they aren't garbage-collected)

    def app(self):
        app = web.Application()
        app.router.add_get("/", self._home)
        app.router.add_get("/ws/hands", self._hands)
        app.router.add_get("/ws/hands/monitor", self._monitor)
        app.router.add_get("/api/web", self._web_config)
        app.router.add_get("/api/stats", self._stats)
        for sub in ("mediapipe", "models"):
            if (WEB_DIR / sub).is_dir():
                app.router.add_static(f"/web/{sub}", WEB_DIR / sub)
        app.router.add_static("/web", HERE)
        return app

    async def _home(self, request):
        return web.FileResponse(HERE / "app.html")

    async def _web_config(self, request):
        return web.json_response({"bundle_url": "/web/mediapipe/vision_bundle.mjs",
                                  "wasm_url": "/web/mediapipe/wasm",
                                  "model_url": "/web/models/hand_landmarker.task",
                                  "server_url": self.server_url})

    async def _stats(self, request):
        now = time.monotonic()
        with self.player.condition:
            post_ms, to_server_ms = self.player.post_ms, self.player.to_server_ms
        return web.json_response({"frames": self.frames, "fps": round(self.fps, 1), "connections": self.connections,
                                  "last_frame_age_s": None if self.last_frame is None else round(now - self.last_frame, 3),
                                  "post_ms": post_ms, "to_server_ms": to_server_ms})

    def _count_frame(self):
        now = time.monotonic()
        self.frames += 1
        self.last_frame = now
        self._fps_count += 1
        if now - self._fps_since >= 1.0:
            self.fps = smooth(self.fps, self._fps_count / (now - self._fps_since))
            self._fps_count, self._fps_since = 0, now

    async def _hands(self, request):
        ws = web.WebSocketResponse(heartbeat=20)
        await ws.prepare(request)
        conductor = Conductor(self.player, self.tempo_window)
        self.connections += 1
        try:
            async for msg in ws:
                if msg.type != web.WSMsgType.TEXT:
                    continue
                try:
                    m = json.loads(msg.data)
                except ValueError:
                    continue
                if m.get("type") == "key" and m.get("key") in ("f", "r"):
                    conductor.key(m["key"])
                elif m.get("type") == "hands":
                    try:
                        hands = parse_hands(m)
                        t_ns = int(float(m["t_ms"]) * 1e6)
                        aspect = float(m["width"]) / float(m["height"])
                    except (KeyError, TypeError, ValueError, ZeroDivisionError) as e:
                        await ws.send_str(json.dumps({"type": "error", "seq": m.get("seq"), "error": str(e)}))
                        continue
                    self._count_frame()
                    reply = conductor.step(hands, t_ns, aspect)
                    await ws.send_str(json.dumps({"type": "conduct", "seq": m.get("seq"), **reply}))
                    if self.monitors:
                        self._to_monitors(json.dumps({**m, "conduct": reply}))
        finally:
            self.connections -= 1
        return ws

    def _to_monitors(self, text):
        """send without holding up the page's reply; a slow listener only delays itself"""
        async def send(ws):
            try:
                await ws.send_str(text)
            except ConnectionError:
                pass
        for ws in list(self.monitors):
            if not ws.closed:
                task = asyncio.ensure_future(send(ws))
                self._sends.add(task)
                task.add_done_callback(self._sends.discard)

    async def _monitor(self, request):
        """listen-only: every hands frame and the conductor's reply; anything sent here
        is ignored"""
        ws = web.WebSocketResponse(heartbeat=20)
        await ws.prepare(request)
        self.monitors.add(ws)
        try:
            async for _ in ws:
                pass
        finally:
            self.monitors.discard(ws)
        return ws


def summary(name, values):
    return f"{name}: mean {np.mean(values):.0f} ms, max {np.max(values):.0f} ms ({len(values)} requests)" \
        if values else f"{name}: no requests"


def main(argv=None):
    parser = argparse.ArgumentParser(description="Conduct the MRT2 server's tempo from the browser.")
    parser.add_argument("--port", type=int, default=8480)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--server-url", default="http://localhost:9100",
                        help="MRT2 server; http://localhost:9000 through the Lighthouse tunnel")
    parser.add_argument("--tempo-window", type=float, default=5.0,
                        help="Cycle history in seconds, at least 2 (default: 5)")
    args = parser.parse_args(argv)
    if not math.isfinite(args.tempo_window) or args.tempo_window < 2:
        parser.error("--tempo-window must be finite and at least 2 seconds")
    if not (WEB_DIR / "mediapipe").is_dir():
        print(f"{WEB_DIR / 'mediapipe'} is missing: run make web-deps", file=sys.stderr)
    player = TimedTempoPlayer(args.server_url)
    with player:
        server = ConductingServer(player, args.server_url, args.tempo_window)
        print(f"Conducting page: http://{args.host}:{args.port}  (tempo to {args.server_url})", flush=True)
        web.run_app(server.app(), host=args.host, port=args.port, print=None, access_log=None)
    print(summary("POST /tempo/*", player.post_log))
    print(summary("frame -> server", player.to_server_log))


if __name__ == "__main__":
    main()
