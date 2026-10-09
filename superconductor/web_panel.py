"""The app's web pages, all on one port under different paths (later pages,
e.g. /conduct, go here too):

    http://localhost:8467/objects   the objects page

The objects page shows every
object the app knows, with its picture, id, name, description, instrument and
trigger, plus the combos (objects that play something else together). All of
it is edited there; changes apply immediately and are saved to the library.

Runs in a daemon thread. The camera loop owns all object state: it publishes a
snapshot with `publish()`, and edits from the page are queued in `actions` and
applied by the camera loop (`CollabFrontend.apply_panel_actions`).
"""
import asyncio
import queue
import threading
from pathlib import Path

from aiohttp import web

PAGE = Path(__file__).parent / "web_panel.html"


class WebPanel:
    def __init__(self, port=8467):
        self.port = port
        self.actions = queue.Queue()  # (kind, id, body) for the camera loop
        self._state = {"objects": [], "combos": []}  # replaced by publish()
        self._views = {}  # id -> jpeg bytes
        self._lock = threading.Lock()
        threading.Thread(target=self._run, daemon=True).start()

    @property
    def url(self):
        return f"http://localhost:{self.port}/objects"

    def publish(self, state, views):
        """`state`: {"objects": [...], "combos": [...]} (see CollabFrontend.publish_panel);
        `views`: id -> jpeg bytes"""
        with self._lock:
            self._state = state
            self._views.update(views)

    def _run(self):
        app = web.Application()
        app.router.add_get("/", self._home)
        app.router.add_get("/objects", self._page)
        app.router.add_get("/api/objects", self._list)
        app.router.add_get("/api/objects/{id}/view.jpg", self._view)
        app.router.add_post("/api/objects/{id}", self._edit)
        app.router.add_post("/api/objects/{id}/describe", self._describe)
        app.router.add_post("/api/objects/{id}/suggest", self._suggest)
        app.router.add_delete("/api/objects/{id}", self._delete)
        app.router.add_post("/api/settings", self._settings)
        app.router.add_post("/api/music", self._music)
        app.router.add_post("/api/combos", self._combo_add)
        app.router.add_post("/api/combos/{id}", self._combo_edit)
        app.router.add_delete("/api/combos/{id}", self._combo_delete)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        runner = web.AppRunner(app, access_log=None)
        loop.run_until_complete(runner.setup())
        try:
            loop.run_until_complete(web.TCPSite(runner, "localhost", self.port).start())
        except OSError as e:
            print(f"objects page: port {self.port} unavailable ({e})")
            return
        loop.run_forever()

    async def _home(self, request):
        raise web.HTTPFound("/objects")

    async def _page(self, request):
        return web.FileResponse(PAGE)

    async def _list(self, request):
        with self._lock:
            return web.json_response(self._state)

    async def _view(self, request):
        with self._lock:
            jpeg = self._views.get(int(request.match_info["id"]))
        if jpeg is None:
            raise web.HTTPNotFound()
        return web.Response(body=jpeg, content_type="image/jpeg", headers={"Cache-Control": "no-cache"})

    def _queue(self, kind, request, body=None):
        self.actions.put((kind, int(request.match_info.get("id", -1)), body or {}))
        return web.json_response({"ok": True}, status=202)

    async def _edit(self, request):
        body = await request.json()
        return self._queue("edit", request, {k: body[k] for k in ("name", "instrument", "trigger", "color") if k in body})

    async def _describe(self, request):
        return self._queue("describe", request)

    async def _suggest(self, request):
        return self._queue("suggest", request)

    async def _delete(self, request):
        return self._queue("delete", request)

    async def _settings(self, request):
        return self._queue("settings", request, await request.json())

    async def _music(self, request):
        return self._queue("music", request, await request.json())

    async def _combo_add(self, request):
        return self._queue("combo_add", request, await request.json())

    async def _combo_edit(self, request):
        return self._queue("combo_edit", request, await request.json())

    async def _combo_delete(self, request):
        return self._queue("combo_delete", request)
