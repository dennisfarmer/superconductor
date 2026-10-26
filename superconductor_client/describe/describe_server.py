"""SuperConductor describe server: one-time rich descriptions of objects with a VLM.

The client (superconductor_client) sends each new object's picture here once.
This server asks a Qwen VLM for a rich text description of it, and for an
instrument prompt that fits that description. Everything about the VLM (model,
prompts, backend) lives in this folder, so the server can run on the laptop
or on a cluster GPU node behind an SSH tunnel; only the client's URL changes.

The VLM runs in Ollama (https://ollama.com), called over its HTTP API, so the
description endpoints only need aiohttp. Requests are handled one at a time.

HTTP API:
  POST /describe    {"image": "<base64 jpeg>"}  -> {"description": "..."}
  POST /instrument  {"description": "..."}      -> {"instrument": "fiery thundering taiko drums"}
  POST /name        {"description": "..."}      -> {"name": "green plush dinosaur"}
  POST /svg         {"prompt": "..."}           -> {"svg": "<svg ...>...</svg>"}  (default.svg, a wand)
  GET  /health                                  -> {"model": ..., "backend": "ollama", "host": ...,
                                                    "svg_model": ..., "svg_backend": ..., "svg_ready": ...}

/svg returns describe/default.svg. It does not load a model, so it does not take a GPU.
"""
import argparse
import asyncio
import socket
import time
from pathlib import Path

import aiohttp
from aiohttp import web

if __package__:
    from .svg_model import SVG_MODEL, SvgError, SvgModel
else:
    # `python describe_server.py` from this folder (make describe)
    from svg_model import SVG_MODEL, SvgError, SvgModel

DESCRIBE_PROMPT = (
    "Describe this object in detail. Say what it is (name it if it is a known "
    "character, toy or product), its colors, materials and textures, its shape, "
    "and the mood, personality or theme it suggests. Write 3 to 5 sentences, "
    "with no preamble."
)

INSTRUMENT_PROMPT = (
    "Here is a description of an object:\n\n{description}\n\n"
    "Choose a musical instrument or sound that matches this object's character. "
    "Answer with a short music-generation prompt of 3 to 8 words: the instrument "
    "plus one to three descriptive words, for example \"fiery thundering taiko "
    "drums\" or \"airy wooden nature flute melody\". Reply with the prompt only."
)


NAME_PROMPT = (
    "Here is a description of an object:\n\n{description}\n\n"
    "Give it a short name of 1 to 3 words that someone would use to point it out, "
    "for example \"bulbasaur plush\" or \"red coffee mug\". Use the character or "
    "product name if the description gives one. Reply with the name only, in lowercase."
)


def _clean_instrument(text):
    """first line, without quotes, labels or a trailing period"""
    line = next((l for l in text.strip().splitlines() if l.strip()), "")
    if ":" in line and len(line.split(":", 1)[0].split()) <= 2:  # "Prompt: ..."
        line = line.split(":", 1)[1]
    return line.strip().strip("\"'`*").rstrip(".").strip()


class DescribeServer:
    def __init__(self, ollama_url, model, port, keep_alive):
        self._ollama_url = ollama_url.rstrip("/")
        self._model = model
        self._port = port
        self._keep_alive = keep_alive
        self._lock = asyncio.Lock()  # one VLM request at a time
        self._session = None
        self._app = web.Application(client_max_size=16 * 1024 * 1024)
        self._svg = SvgModel()
        self._svg_lock = asyncio.Lock()
        self._app.router.add_post("/describe", self._handle_describe)
        self._app.router.add_post("/instrument", self._handle_instrument)
        self._app.router.add_post("/name", self._handle_name)
        self._app.router.add_post("/svg", self._handle_svg)
        self._app.router.add_get("/health", self._handle_health)
        self._app.on_startup.append(self._start)
        self._app.on_cleanup.append(self._stop)

    def run(self):
        web.run_app(self._app, port=self._port)

    async def _start(self, app):
        self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=300))
        try:
            async with self._session.get(f"{self._ollama_url}/api/tags") as r:
                names = [m["name"] for m in (await r.json()).get("models", [])]
        except aiohttp.ClientError as e:
            raise SystemExit(f"Ollama is not running at {self._ollama_url} ({e})")
        if self._model not in names:
            raise SystemExit(f"model {self._model} not pulled in Ollama: run `make pull MODEL={self._model}`")
        print(f"Serving {self._model} (ollama) on port {self._port}", flush=True)
        if self._svg.backend == "transformers":
            from superconductor_client.torch_device import DESCRIBE_NEEDS_GPU, cuda_state_detached
            # "absent" means this machine has no CUDA device. "busy" means another
            # process already holds the card (often Ollama, which is a separate
            # process). Icons then fail on request. Descriptions still go to Ollama.
            if cuda_state_detached() == "absent":
                raise SystemExit(DESCRIBE_NEEDS_GPU)
        icon = Path(__file__).parent / "default.svg"
        print(f"SVG icons: {icon.name} (no model)", flush=True)

    async def _stop(self, app):
        await self._session.close()

    async def _generate(self, prompt, image=None, temperature=0.2):
        body = {"model": self._model, "prompt": prompt, "stream": False,
                "keep_alive": self._keep_alive, "options": {"temperature": temperature}}
        if image:
            body["images"] = [image]
        async with self._lock:
            t0 = time.time()
            async with self._session.post(f"{self._ollama_url}/api/generate", json=body) as r:
                data = await r.json()
                if r.status != 200:
                    raise web.HTTPBadGateway(text=f"ollama: {data.get('error', r.status)}")
        print(f"{'describe' if image else 'text'}: {time.time() - t0:.1f}s", flush=True)
        return data["response"].strip()

    async def _handle_describe(self, request):
        data = await request.json()
        if not data.get("image"):
            raise web.HTTPBadRequest(text="missing 'image' (base64 jpeg)")
        description = await self._generate(DESCRIBE_PROMPT, image=data["image"])
        return web.json_response({"description": description})

    async def _handle_instrument(self, request):
        data = await request.json()
        if not data.get("description"):
            raise web.HTTPBadRequest(text="missing 'description'")
        text = await self._generate(INSTRUMENT_PROMPT.format(description=data["description"]),
                                    temperature=0.4)
        return web.json_response({"instrument": _clean_instrument(text)})

    async def _handle_name(self, request):
        data = await request.json()
        if not data.get("description"):
            raise web.HTTPBadRequest(text="missing 'description'")
        text = await self._generate(NAME_PROMPT.format(description=data["description"]), temperature=0.2)
        name = _clean_instrument(text).lower()
        return web.json_response({"name": " ".join(name.split()[:4])})

    async def _handle_svg(self, request):
        """{"prompt": "..."} -> the wand in default.svg.

        Icons do not load a model, so this request does not take a GPU.
        """
        try:
            data = await request.json()
        except ValueError:
            raise web.HTTPBadRequest(text="body must be JSON")
        prompt = str((data or {}).get("prompt") or "").strip()
        if not prompt:
            raise web.HTTPBadRequest(text="missing 'prompt'")
        path = Path(__file__).parent / "default.svg"
        try:
            svg = path.read_text(encoding="utf-8").strip()
        except OSError as e:
            raise web.HTTPServiceUnavailable(text=f"default icon is missing: {e}")
        if not svg.lower().lstrip().startswith("<svg"):
            raise web.HTTPServiceUnavailable(text="default.svg is not an SVG")
        return web.json_response({"svg": svg})

    async def _handle_health(self, request):
        icon = Path(__file__).parent / "default.svg"
        ready = icon.is_file()
        return web.json_response({
            "model": self._model, "backend": "ollama", "host": socket.gethostname(),
            "svg_model": icon.name, "svg_backend": "file",
            "svg_ready": ready, "svg_error": None if ready else "default.svg is missing",
        })


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="qwen2.5vl:3b", help="Ollama vision model")
    parser.add_argument("--port", type=int, default=9106)
    parser.add_argument("--ollama_url", default="http://localhost:11434")
    parser.add_argument("--keep_alive", default="10m",
                        help="how long Ollama keeps the model loaded after a request "
                             "(e.g. 0 to free the GPU right away, -1 forever)")
    parser.add_argument("--svg-model", default=SVG_MODEL,
                        help="local text-to-SVG model (mlx-lm on macOS, transformers on the cluster)")
    parser.add_argument("--svg-backend", default=None, choices=["mlx", "transformers"],
                        help="override the SVG backend (default: mlx on macOS, transformers elsewhere)")
    args = parser.parse_args()
    server = DescribeServer(args.ollama_url, args.model, args.port, args.keep_alive)
    server._svg = SvgModel(args.svg_model, args.svg_backend)
    server.run()


if __name__ == "__main__":
    main()
