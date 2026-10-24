"""The app's web pages, all on one port under different paths (later pages,
e.g. /conduct, go here too):

    http://localhost:8467/          the camera page (app.html): the browser's camera,
                                    its detector (vision_js + tracker_js), the view
                                    with the overlay drawn on it, and the audio
    http://localhost:8467/objects   the prompts page

The prompts page shows parameters (knobs added there, and prompts other
services send, such as Atlas drums) and every object the app knows, with its
picture, id, name, description, instrument and trigger, plus the combos
(objects that play something else together). All of it is edited there;
changes apply immediately and are saved to the library.

Runs in a daemon thread. The camera loop owns all object state: it publishes a
snapshot with `publish()`, and edits from the page are queued in `actions` and
applied by the camera loop (`CollabFrontend.apply_panel_actions`).

The camera page sends its frames over WS /ws/camera, with the objects it detected
(see CAMERA_PROTOCOL.md). Only the latest frame is kept (`latest_frame()`), so frames
never queue up behind a slow tracker or a slow tunnel. The camera loop sends back
what to draw over the video with `send_overlay()` (JSON, never pictures). Clicks and
keys from the page are queued in `actions` as ("click", -1, {"x", "y"}) and
("key", -1, {"key"}). WS /ws/tracking gets every overlay too, listen-only (e.g.
TouchDesigner), without changing what the page gets.

Static files for the page: /web/vision_js/*, /web/tracker_js/* (from the repository),
/web/overlay.js, and /web/ort/* (onnxruntime-web) and /web/models/* (the exported
detector) from `web_dir` (make web-deps / make web-model).
"""
import asyncio
import json
import queue
import struct
import threading
from pathlib import Path

from aiohttp import web

HERE = Path(__file__).parent  # superconductor_client/tracking: the camera page, its detector and its tracker
PAGE = HERE / "tracking_view.html"
APP_PAGE = HERE / "app.html"
OVERLAY_JS = HERE / "overlay.js"


def parse_frame_message(data):
    """A binary /ws/camera message -> (jpeg bytes, header dict or None).
    [uint32 LE header length][header JSON][JPEG] (the page detected), or a plain JPEG
    (compatibility mode: this process detects). See CAMERA_PROTOCOL.md."""
    if data[:2] == b"\xff\xd8":  # JPEG start-of-image marker: no header
        return data, None
    if len(data) < 4:
        raise ValueError("frame message too short")
    (n,) = struct.unpack_from("<I", data)
    if 4 + n > len(data):
        raise ValueError("frame header longer than the message")
    header = json.loads(data[4:4 + n].decode("utf-8"))
    if not isinstance(header, dict) or header.get("type") != "frame":
        raise ValueError("frame header is not a frame")
    return data[4 + n:], header


class TrackingView:
    def __init__(self, port=8467, host="localhost", page_config=None, web_dir=None):
        """`page_config`: what the camera page reads from GET /api/web (where its audio
        player connects, the detector's classes and thresholds, its links).
        `web_dir`: holds ort/ and models/ for the page's detector."""
        self.port = port
        self.host = host
        self.page_config = page_config or {}
        self.web_dir = Path(web_dir) if web_dir else None
        self.actions = queue.Queue()  # (kind, id, body) for the camera loop
        self._state = {"objects": [], "combos": [], "parameters": []}  # replaced by publish()
        self._views = {}  # id -> [jpeg bytes per view], newest last
        self._lock = threading.Lock()
        self._frame = None  # (seq, jpeg bytes, header or None): the newest frame from the browser
        self._frame_seq = 0
        self._cameras = set()  # open /ws/camera sockets
        self._trackers = set()  # open /ws/tracking sockets (listen-only)
        self._loop = None
        threading.Thread(target=self._run, daemon=True).start()

    @property
    def url(self):
        return f"http://localhost:{self.port}/objects"

    def publish(self, state, views):
        """`state`: {"objects": [...], "combos": [...]} (see CollabFrontend.publish_panel);
        `views`: id -> [jpeg bytes per gallery view] (only objects whose views changed)"""
        with self._lock:
            self._state = state
            self._views.update(views)

    def latest_frame(self, after=0):
        """(seq, jpeg, header) of the newest browser frame if it is newer than `after`,
        else None. `header`: the page's frame header (its detections), None for a plain JPEG."""
        with self._lock:
            if self._frame is not None and self._frame[0] > after:
                return self._frame
        return None

    @property
    def browser_connected(self):
        return bool(self._cameras)

    def send_overlay(self, overlay):
        """send what to draw over the video (a JSON-able dict, see CAMERA_PROTOCOL.md)
        to every open camera page, and the same text to every /ws/tracking listener"""
        if self._loop is None or not (self._cameras or self._trackers):
            return
        text = json.dumps(overlay, separators=(",", ":"))
        for ws in list(self._cameras) + list(self._trackers):
            asyncio.run_coroutine_threadsafe(self._send_text(ws, text), self._loop)

    @staticmethod
    async def _send_text(ws, text):
        if ws.closed:
            return
        try:
            await ws.send_str(text)
        except ConnectionError:
            pass

    def _run(self):
        app = web.Application(client_max_size=16 * 1024 ** 2)
        app.router.add_get("/", self._home)
        app.router.add_get("/ws/camera", self._camera)
        app.router.add_get("/ws/tracking", self._tracking)
        app.router.add_get("/api/web", self._web_config)
        app.router.add_get("/web/overlay.js", self._overlay_js)
        app.router.add_static("/web/vision_js", HERE / "vision_js")
        app.router.add_static("/web/tracker_js", HERE / "tracker_js")
        for sub in ("ort", "models"):
            if self.web_dir is not None and (self.web_dir / sub).is_dir():
                app.router.add_static(f"/web/{sub}", self.web_dir / sub)
        app.router.add_get("/objects", self._page)
        app.router.add_get("/api/objects", self._list)
        app.router.add_get("/api/objects/{id}/view.jpg", self._view)
        app.router.add_get("/api/objects/{id}/views/{index}.jpg", self._view)
        app.router.add_delete("/api/objects/{id}/views/{index}", self._delete_view)
        app.router.add_post("/api/objects/{id}/cover", self._cover)
        app.router.add_post("/api/objects/{id}", self._edit)
        app.router.add_post("/api/objects/{id}/describe", self._describe)
        app.router.add_post("/api/objects/{id}/suggest", self._suggest)
        app.router.add_delete("/api/objects/{id}", self._delete)
        app.router.add_post("/api/settings", self._settings)
        app.router.add_post("/api/learning", self._learning)
        app.router.add_post("/api/music", self._music)
        app.router.add_post("/api/combos", self._combo_add)
        app.router.add_post("/api/combos/{id}", self._combo_edit)
        app.router.add_delete("/api/combos/{id}", self._combo_delete)
        app.router.add_post("/api/parameters", self._param_add)
        app.router.add_post("/api/parameters/{id}", self._param_edit)
        app.router.add_post("/api/parameters/{id}/icon", self._param_icon)
        app.router.add_delete("/api/parameters/{id}", self._param_delete)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        runner = web.AppRunner(app, access_log=None)
        loop.run_until_complete(runner.setup())
        try:
            loop.run_until_complete(web.TCPSite(runner, self.host, self.port).start())
        except OSError as e:
            print(f"objects page: port {self.port} unavailable ({e})")
            return
        loop.run_forever()

    async def _home(self, request):
        return web.FileResponse(APP_PAGE)

    async def _overlay_js(self, request):
        return web.FileResponse(OVERLAY_JS)

    async def _web_config(self, request):
        return web.json_response(self.page_config)

    async def _camera(self, request):
        """binary messages: frames (parse_frame_message); text: {"type": "click", "x", "y"}
        (0..1, as displayed) or {"type": "key", "key"}. Sends back overlays (send_overlay)."""
        ws = web.WebSocketResponse(max_msg_size=8 * 1024 ** 2, heartbeat=20)
        await ws.prepare(request)
        self._cameras.add(ws)
        try:
            async for msg in ws:
                if msg.type == web.WSMsgType.BINARY:
                    try:
                        jpeg, header = parse_frame_message(msg.data)
                    except (ValueError, UnicodeDecodeError) as e:
                        await ws.send_str(json.dumps({"type": "error", "error": f"frame: {e}"}))
                        continue
                    with self._lock:
                        self._frame_seq += 1
                        self._frame = (self._frame_seq, jpeg, header)
                elif msg.type == web.WSMsgType.TEXT:
                    try:
                        event = json.loads(msg.data)
                    except ValueError:
                        continue
                    if event.get("type") == "click":
                        self.actions.put(("click", -1, {"x": float(event["x"]), "y": float(event["y"])}))
                    elif event.get("type") == "key" and isinstance(event.get("key"), str) and len(event["key"]) == 1:
                        self.actions.put(("key", -1, {"key": event["key"]}))
        finally:
            self._cameras.discard(ws)
        return ws

    async def _tracking(self, request):
        """listen-only: every overlay the camera page gets (send_overlay); anything sent
        here is ignored"""
        ws = web.WebSocketResponse(heartbeat=20)
        await ws.prepare(request)
        self._trackers.add(ws)
        try:
            async for _ in ws:
                pass
        finally:
            self._trackers.discard(ws)
        return ws

    async def _page(self, request):
        return web.FileResponse(PAGE)

    async def _list(self, request):
        with self._lock:
            return web.json_response(self._state)

    async def _view(self, request):
        """the object's thumbnail (its newest view), or view `index` of its gallery"""
        with self._lock:
            jpegs = self._views.get(int(request.match_info["id"])) or []
        if "index" in request.match_info:
            index = int(request.match_info["index"])
            jpeg = jpegs[index] if 0 <= index < len(jpegs) else None
        else:
            jpeg = next((j for j in reversed(jpegs) if j), None)
        if not jpeg:
            raise web.HTTPNotFound()
        return web.Response(body=jpeg, content_type="image/jpeg", headers={"Cache-Control": "no-cache"})

    def _queue(self, kind, request, body=None):
        self.actions.put((kind, int(request.match_info.get("id", -1)), body or {}))
        return web.json_response({"ok": True}, status=202)

    def _put(self, kind, key, body=None):
        self.actions.put((kind, key, body or {}))
        return web.json_response({"ok": True}, status=202)

    @staticmethod
    def _key(request):
        raw = request.match_info["id"]
        return int(raw) if raw.isdigit() else raw

    async def _edit(self, request):
        body = await request.json()
        return self._queue("edit", request, {k: body[k] for k in ("name", "instrument", "trigger", "color", "frozen")
                                             if k in body})

    async def _describe(self, request):
        return self._queue("describe", request)

    async def _suggest(self, request):
        return self._queue("suggest", request)

    async def _delete(self, request):
        return self._queue("delete", request)

    async def _cover(self, request):
        """{"index": <view index> | null (its newest view), "rev": <view_rev the page saw>}"""
        body = await request.json()
        index, rev = body.get("index"), body.get("rev")
        return self._queue("cover", request, {"index": None if index is None else int(index),
                                              "rev": None if rev is None else int(rev)})

    async def _delete_view(self, request):
        """?rev=<view_rev the page saw>, so a gallery that changed since loses no other view"""
        rev = request.query.get("rev")
        return self._queue("delete_view", request, {"index": int(request.match_info["index"]),
                                                    "rev": int(rev) if rev is not None else None})

    async def _settings(self, request):
        return self._queue("settings", request, await request.json())

    async def _learning(self, request):
        return self._queue("learning", request, await request.json())

    async def _music(self, request):
        return self._queue("music", request, await request.json())

    async def _combo_add(self, request):
        return self._queue("combo_add", request, await request.json())

    async def _combo_edit(self, request):
        return self._queue("combo_edit", request, await request.json())

    async def _combo_delete(self, request):
        return self._queue("combo_delete", request)

    async def _param_add(self, request):
        return self._put("param_add", -1)

    async def _param_edit(self, request):
        return self._put("param_edit", self._key(request), await request.json())

    async def _param_icon(self, request):
        return self._put("param_icon", self._key(request))

    async def _param_delete(self, request):
        return self._put("param_delete", self._key(request))
