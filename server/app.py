"""The HTTP server: every endpoint of server.atlas as POST /function/<name>.

Each takes a JSON body of its arguments and returns {"ok": true, "data": {"result": ...}}.
The endpoints are plain `def`, so FastAPI runs them in its threadpool and a beam search
doesn't block the event loop; one lock serves them one at a time, and the session is
saved after each.

    python -m server.app --port 9103 [--local]
"""

import argparse
import inspect
import signal
import threading
from pathlib import Path
from typing import Any, Callable, Generic, TypeVar, get_type_hints

import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, create_model

from server import atlas
from server.engine_client import stop_engine

DIST = Path(__file__).resolve().parent.parent / "client" / "dist"

T = TypeVar("T")


class Result(BaseModel, Generic[T]):
    result: T


class Envelope(BaseModel, Generic[T]):
    ok: bool = True
    data: Result[T]


app = FastAPI(title="SuperConductor Atlas")
_lock = threading.Lock()
_server: uvicorn.Server | None = None


def shutdown_app() -> dict[str, Any]:
    """Quit SuperConductor Atlas from the page: ask the audio engine to exit and wait for it,
    then ask this app server to exit. Nothing is killed: the engine winds down through
    its own /shutdown, and the server through uvicorn's own exit path. The server stops
    a moment after this reply, so the page receives it."""
    engine_stopped = stop_engine()
    threading.Timer(0.4, _stop_server).start()
    return {"engine_stopped": engine_stopped}


def _stop_server() -> None:
    """Tell this process's uvicorn server to shut down gracefully, as Ctrl-C would."""
    if _server is not None:
        _server.should_exit = True
    else:
        # started by the uvicorn CLI (make dev): its own SIGINT handler exits gracefully
        signal.raise_signal(signal.SIGINT)


ENDPOINTS: list[Callable[..., Any]] = [
    atlas.get_state, atlas.set_extended, atlas.set_key, atlas.set_tempo, atlas.set_texture,
    atlas.set_drums, atlas.set_llm, atlas.set_style_layers, atlas.refresh_options, atlas.explore,
    atlas.add_path, atlas.undo, atlas.clear_progression, atlas.describe, atlas.add_chords,
    atlas.restart_engine, shutdown_app, atlas.set_loop, atlas.add_option, atlas.commit_progression,
    atlas.rename_saved, atlas.delete_saved, atlas.resume_saved, atlas.remove_step,
    atlas.replace_step, atlas.swap_step,
]


def route(fn: Callable[..., Any]) -> None:
    """POST /function/<name>: a request model from fn's arguments, the envelope around its result."""
    hints = get_type_hints(fn)
    params = inspect.signature(fn).parameters
    camel = "".join(w.capitalize() for w in fn.__name__.split("_"))
    Args = create_model(
        f"{camel}Args",
        **{n: (hints[n], ... if p.default is p.empty else p.default) for n, p in params.items()},
    )

    def endpoint(args: Args) -> Envelope[hints["return"]]:  # type: ignore[valid-type]
        with _lock:
            try:
                result = fn(**{n: getattr(args, n) for n in params})
            finally:
                atlas.save_session()
        return {"ok": True, "data": {"result": result}}

    if not any(p.default is p.empty for p in params.values()):
        # nothing required: the body may be left out
        endpoint.__defaults__ = (Args(),)
    endpoint.__name__ = fn.__name__
    app.post(f"/function/{fn.__name__}", summary=(fn.__doc__ or "").split("\n")[0] or None)(endpoint)


for _fn in ENDPOINTS:
    route(_fn)


@app.exception_handler(Exception)
def failed(request: Any, exc: Exception) -> JSONResponse:
    return JSONResponse({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, status_code=500)


@app.get("/healthz")
def healthz() -> dict[str, bool]:
    return {"ok": True}


if DIST.is_dir():
    app.mount("/", StaticFiles(directory=DIST, html=True), name="client")


def main() -> None:
    global _server
    ap = argparse.ArgumentParser(description="SuperConductor Atlas server")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=9103)
    ap.add_argument("--local", action="store_true",
                    help="local mode: the engine runs Magenta itself (make run-local); default: SuperConductor sink")
    a = ap.parse_args()
    atlas.LOCAL = a.local
    # 0.0.0.0 is the bind address; the page you open is the loopback one.
    shown = "127.0.0.1" if a.host in ("0.0.0.0", "::") else a.host
    print(f"Atlas at http://{shown}:{a.port}", flush=True)
    _server = uvicorn.Server(uvicorn.Config(app, host=a.host, port=a.port))
    _server.run()


if __name__ == "__main__":
    main()
