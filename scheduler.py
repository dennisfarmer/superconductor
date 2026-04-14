"""Chunk scheduler sitting between the laptop client and the Magenta server.

The scheduler maintains a linked list of generated audio chunks. Each node
holds the audio samples, the post-generation state (used to generate the next
chunk), and a unique id. A background worker keeps the queue full ahead of
playback by querying ``new_server.StatefulServer``'s ``/generate_chunk``
endpoint. The state-passing endpoint lets us fork the queue: when the recipe
changes, we pick a fork point far enough ahead of the playback bar that there
is at least one chunk-length of slack to regenerate audio with the new style,
truncate the queue past that point, and let the worker refill it.
"""

import asyncio
import base64
import json
import pickle
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Optional


import numpy as np
import requests
from aiohttp import web


chunk_length_seconds = 2.0
default_buffer_chunks = 5


@dataclass
class Chunk:
    id: str
    audio: np.ndarray              # generated samples, shape (num_samples, channels)
    state: object                  # MagentaRTState after this chunk was generated
    recipe: dict                   # recipe used to generate this chunk
    style_embedding: np.ndarray    # style embedding used to generate this chunk
    prev_id: Optional[str] = None
    next_id: Optional[str] = None


class Scheduler:
    """Linked-list queue of generated chunks with fork-on-recipe-change."""

    def __init__(
        self,
        server_url: str = "http://localhost:8000",
        buffer_chunks: int = default_buffer_chunks,
        request_timeout: float = 30.0,
    ):
        self._server_url = server_url.rstrip("/")
        self._buffer_chunks = buffer_chunks
        self._request_timeout = request_timeout

        self._lock = threading.RLock()
        self._chunks: dict[str, Chunk] = {}
        self._head_id: Optional[str] = None    # oldest chunk still in the queue
        self._tail_id: Optional[str] = None    # most recently generated chunk

        # The chunk the client is currently playing and how far into it (sec).
        # The playing chunk is kept in `_chunks` until the next one is popped,
        # so fork-point selection can walk forward from it.
        self._playing_id: Optional[str] = None
        self._playing_offset: float = 0.0

        # Recipe / style currently used by the worker for new chunks.
        self._current_recipe: Optional[dict] = None
        self._current_style: Optional[np.ndarray] = None

        # Bumped on every recipe update so in-flight generations can detect
        # that they are stale and discard their result.
        self._generation_id: int = 0

        self._stop = threading.Event()
        self._wake = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self, initial_recipe: dict, initial_style: np.ndarray):
        with self._lock:
            self._current_recipe = dict(initial_recipe)
            self._current_style = np.array(initial_style, dtype=np.float32)
            self._generation_id += 1
        self._stop.clear()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._wake.set()
        if self._thread:
            self._thread.join(timeout=5.0)

    def pop_next_chunk(self, wait: bool = True, timeout: float = 5.0) -> Optional[Chunk]:
        """Return the next chunk for the client to play.

        Drops the previously-playing chunk from the queue and advances the
        head pointer. Optionally blocks until a chunk is available.
        """
        deadline = time.monotonic() + timeout
        while True:
            with self._lock:
                next_id = self._next_id_locked()
                if next_id is not None:
                    # Drop the previously-playing chunk, if any.
                    if self._playing_id is not None and self._playing_id in self._chunks:
                        old = self._chunks.pop(self._playing_id)
                        if old.next_id is not None and old.next_id in self._chunks:
                            self._chunks[old.next_id].prev_id = None
                        if self._head_id == self._playing_id:
                            self._head_id = old.next_id
                    chunk = self._chunks[next_id]
                    self._playing_id = next_id
                    self._playing_offset = 0.0
                    self._wake.set()
                    return chunk
            if not wait or time.monotonic() >= deadline:
                return None
            time.sleep(0.05)

    def report_playback_progress(self, chunk_id: str, offset_seconds: float):
        """Client tells the scheduler where the playback bar currently is."""
        with self._lock:
            self._playing_id = chunk_id
            self._playing_offset = max(0.0, min(chunk_length_seconds, offset_seconds))

    def update_recipe(self, recipe: dict, style_embedding: np.ndarray):
        """Switch the recipe and schedule a fork in the queue.

        The fork point is chosen so that there is at least one chunk-length
        of slack between the playback bar and the first regenerated chunk,
        which is the worst-case generation time.
        """
        new_style = np.array(style_embedding, dtype=np.float32)
        with self._lock:
            self._current_recipe = dict(recipe)
            self._current_style = new_style
            self._generation_id += 1
            keep_id = self._choose_fork_point_locked()
            self._truncate_after_locked(keep_id)
        self._wake.set()

    def get_chunk(self, chunk_id: str) -> Optional[Chunk]:
        with self._lock:
            return self._chunks.get(chunk_id)

    def queue_length(self) -> int:
        """Number of chunks ahead of (and including) the currently-playing one."""
        with self._lock:
            return len(self._chunks)

    def _next_id_locked(self) -> Optional[str]:
        """The id of the next chunk to play (after the currently-playing one)."""
        if self._playing_id is None:
            return self._head_id
        cur = self._chunks.get(self._playing_id)
        if cur is None:
            return self._head_id
        return cur.next_id

    def _choose_fork_point_locked(self) -> Optional[str]:
        """Pick the last chunk to keep when forking; None means clear the queue.

        We need at least CHUNK_LENGTH_SECONDS of audio between the playback
        bar and the first regenerated chunk. The currently-playing chunk has
        ``CHUNK_LENGTH_SECONDS - playing_offset`` seconds of audio left, and
        every subsequent kept chunk adds another CHUNK_LENGTH_SECONDS.
        """
        if self._playing_id is None or self._playing_id not in self._chunks:
            return self._head_id

        remaining = chunk_length_seconds - self._playing_offset
        deficit = chunk_length_seconds - remaining
        if deficit <= 0:
            keep_after_current = 0
        else:
            keep_after_current = int(np.ceil(deficit / chunk_length_seconds))
        keep_after_current += 1

        node = self._chunks[self._playing_id]
        for _ in range(keep_after_current):
            if node.next_id is None or node.next_id not in self._chunks:
                break
            node = self._chunks[node.next_id]
        return node.id

    def _truncate_after_locked(self, keep_id: Optional[str]):
        """Drop every chunk after ``keep_id``; if None, drop everything."""
        if keep_id is None:
            self._chunks.clear()
            self._head_id = None
            self._tail_id = None
            return
        keep = self._chunks.get(keep_id)
        if keep is None:
            return
        cur_id = keep.next_id
        keep.next_id = None
        self._tail_id = keep.id
        while cur_id is not None:
            nxt = self._chunks[cur_id].next_id if cur_id in self._chunks else None
            self._chunks.pop(cur_id, None)
            cur_id = nxt

    def _worker(self):
        while not self._stop.is_set():
            with self._lock:
                queue_full = len(self._chunks) >= self._buffer_chunks
                tail_id = self._tail_id
                tail = self._chunks.get(tail_id) if tail_id else None
                style = self._current_style
                recipe = self._current_recipe
                gen_id = self._generation_id

            if style is None or queue_full:
                self._wake.wait(timeout=0.1)
                self._wake.clear()
                continue

            prev_state = tail.state if tail is not None else None
            try:
                audio, new_state = self._request_chunk(prev_state, style)
            except Exception as e:
                print(f"Scheduler: chunk generation failed: {e}")
                time.sleep(0.5)
                continue

            new_chunk = Chunk(
                id=str(uuid.uuid4()),
                audio=audio,
                state=new_state,
                recipe=dict(recipe) if recipe else {},
                style_embedding=np.array(style, dtype=np.float32),
            )

            with self._lock:
                # Discard if a fork happened during generation: either the
                # recipe changed or the tail we generated from no longer
                # matches the current tail.
                if gen_id != self._generation_id:
                    continue
                cur_tail_id = self._tail_id
                if tail_id is None:
                    if cur_tail_id is not None:
                        continue
                    self._head_id = new_chunk.id
                else:
                    if cur_tail_id != tail_id:
                        continue
                    cur_tail = self._chunks[cur_tail_id]
                    new_chunk.prev_id = cur_tail.id
                    cur_tail.next_id = new_chunk.id
                self._chunks[new_chunk.id] = new_chunk
                self._tail_id = new_chunk.id

    def _request_chunk(self, state, style: np.ndarray):
        body = {"style": [float(x) for x in np.asarray(style).reshape(-1)]}
        if state is not None:
            body["state"] = base64.b64encode(pickle.dumps(state)).decode("ascii")
        r = requests.post(
            f"{self._server_url}/generate_chunk",
            json=body,
            timeout=self._request_timeout,
        )
        r.raise_for_status()
        data = r.json()
        audio_bytes = base64.b64decode(data["audio"].encode("ascii"))
        audio = np.frombuffer(audio_bytes, dtype=np.float32)
        shape = data.get("shape")
        if shape:
            audio = audio.reshape(shape)
        new_state = pickle.loads(base64.b64decode(data["state"].encode("ascii")))
        return audio, new_state


class SchedulerServer:
    """Base front-end for a ``Scheduler``.

    Owns the aiohttp application, the recipe->embedding cache, and the
    methods that drive the underlying ``Scheduler``. Subclasses register
    their own routes (HTTP endpoints or a websocket) and translate
    wire-protocol messages into calls on the protected methods defined
    here. All embedding creation lives in this class so it always runs on
    the cluster — laptop clients only ever send recipes, never embeddings,
    via the new endpoints. (The legacy ``UpdateControl`` websocket message
    that ships a pre-computed embedding is still accepted for backwards
    compatibility, but new code should use the recipe-based path.)

    Runs on the compute cluster. The laptop client reaches it through an
    SSH tunnel and only ever sees audio samples and chunk ids — never the
    raw MagentaRT state. Recipes (weighted prompt dicts) are converted to
    style embeddings here by talking to the Magenta server's ``/style``
    endpoint, so the client does not need its own link to the Magenta
    server.
    """

    def __init__(
        self,
        scheduler: "Scheduler",
        magenta_server_url: str = "http://localhost:8000",
        port: int = 9100,
    ):
        self._scheduler = scheduler
        self._magenta_url = magenta_server_url.rstrip("/")
        self._port = port
        self._embedding_cache: dict[str, np.ndarray] = {}
        self._started = False

        self._app = web.Application()
        self._register_routes()

    # ---------- subclass hook ----------

    def _register_routes(self):
        """Subclasses must register their endpoints on ``self._app``."""
        raise NotImplementedError

    def run(self):
        web.run_app(self._app, port=self._port)

    # ---------- recipe -> embedding ----------

    def _get_cached_embedding(self, prompt: str) -> np.ndarray:
        if prompt not in self._embedding_cache:
            r = requests.post(
                f"{self._magenta_url}/style", data=prompt, timeout=30
            )
            r.raise_for_status()
            self._embedding_cache[prompt] = np.array(r.json(), dtype=np.float32)
        return self._embedding_cache[prompt]

    def _recipe_to_embedding(self, recipe: dict) -> Optional[np.ndarray]:
        embeddings = []
        weights = []
        for prompt, weight in recipe.items():
            if weight <= 0:
                continue
            embeddings.append(self._get_cached_embedding(prompt))
            weights.append(float(weight))
        if not embeddings:
            return None
        stacked = np.stack(embeddings)
        w = np.array(weights, dtype=np.float32)
        w = w / w.sum()
        return np.sum(stacked * w[:, None], axis=0)

    async def _embed_recipe(self, recipe: dict) -> Optional[np.ndarray]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._recipe_to_embedding, recipe)

    # ---------- scheduler operations ----------

    @property
    def is_started(self) -> bool:
        return self._started

    def _start_scheduler(self, recipe: dict, style: np.ndarray):
        self._scheduler.start(recipe, style)
        self._started = True

    def _stop_scheduler(self):
        if self._started:
            self._scheduler.stop()
            self._started = False

    def _update_recipe(self, recipe: dict, style: np.ndarray):
        self._scheduler.update_recipe(recipe, style)

    def _start_or_update(self, recipe: dict, style: np.ndarray):
        """Start the scheduler if it isn't running, otherwise live-update."""
        if self._started:
            self._update_recipe(recipe, style)
        else:
            self._start_scheduler(recipe, style)

    def _report_progress(self, chunk_id: str, offset: float):
        self._scheduler.report_playback_progress(chunk_id, offset)

    async def _pop_next_chunk_async(
        self, timeout: float = 30.0
    ) -> Optional[Chunk]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: self._scheduler.pop_next_chunk(wait=True, timeout=timeout),
        )


class SchedulerServerHTTP(SchedulerServer):
    """JSON-over-HTTP front-end for a ``Scheduler``."""

    def _register_routes(self):
        self._app.router.add_post("/start", self._handle_start)
        self._app.router.add_post("/next_chunk", self._handle_next_chunk)
        self._app.router.add_post("/report_progress", self._handle_report_progress)
        self._app.router.add_post("/update_recipe", self._handle_update_recipe)
        self._app.router.add_post("/stop", self._handle_stop)

    # ---------- handlers ----------

    async def _handle_start(self, request: web.Request) -> web.Response:
        data = await request.json()
        recipe = data.get("recipe") or {}
        emb = await self._embed_recipe(recipe)
        if emb is None:
            return web.json_response({"error": "empty recipe"}, status=400)
        self._start_or_update(recipe, emb)
        return web.json_response({"ok": True})

    async def _handle_next_chunk(self, request: web.Request) -> web.Response:
        del request
        chunk = await self._pop_next_chunk_async()
        if chunk is None:
            return web.json_response({"error": "timeout"}, status=504)
        samples = np.asarray(chunk.audio, dtype=np.float32)
        return web.json_response(
            {
                "id": chunk.id,
                "audio": base64.b64encode(samples.tobytes()).decode("ascii"),
                "shape": list(samples.shape),
            }
        )

    async def _handle_report_progress(self, request: web.Request) -> web.Response:
        data = await request.json()
        chunk_id = data.get("chunk_id")
        offset = float(data.get("offset_seconds", 0.0))
        if chunk_id:
            self._report_progress(chunk_id, offset)
        return web.json_response({"ok": True})

    async def _handle_update_recipe(self, request: web.Request) -> web.Response:
        data = await request.json()
        recipe = data.get("recipe") or {}
        chunk_id = data.get("chunk_id")
        offset = data.get("offset_seconds")
        if chunk_id is not None and offset is not None:
            self._report_progress(chunk_id, float(offset))
        emb = await self._embed_recipe(recipe)
        if emb is None:
            return web.json_response({"error": "empty recipe"}, status=400)
        self._update_recipe(recipe, emb)
        return web.json_response({"ok": True})

    async def _handle_stop(self, request: web.Request) -> web.Response:
        del request
        self._stop_scheduler()
        return web.json_response({"ok": True})


class SchedulerServerWebsocket(SchedulerServer):
    """Websocket front-end for a ``Scheduler``.

    Speaks a JSON-over-websocket protocol modeled on the original Magenta
    RealTime server, but with all style-embedding logic moved server-side:
    clients send recipes (weighted prompt dicts), the scheduler builds the
    embedding on the cluster, and the client never needs its own link to
    the Magenta server.

    The client drives generation with a pull model:

      * ``StartSession``    — reset any prior session state
      * ``UpdateRecipe``    — ``body == {recipe, chunk_id?, offset_seconds?}``,
                              identical semantics to the HTTP
                              ``/update_recipe`` endpoint. The scheduler
                              builds the style embedding from the recipe
                              and forks the queue.
      * ``UpdateProgress``  — ``body == {chunk_id, offset_seconds}``.
                              Push-only playback-position update; same
                              chunk_id/offset_seconds fields as
                              ``UpdateRecipe`` but without the recipe,
                              for clients that want to keep the fork
                              point accurate between recipe changes.
      * ``UpdatePlayback``  — ``body.state == "PLAYING"`` boots the
                              scheduler worker and immediately sends the
                              first audio chunk.
      * ``ReceivedChunk``   — "please send one more chunk"; the server
                              pops the next chunk from the scheduler and
                              sends it as a binary frame.
      * ``EndSession``      — tear down the scheduler worker.

    Each audio frame starts with the 16-byte binary UUID of the chunk
    (``uuid.UUID(chunk.id).bytes``), followed by raw little-endian
    float32 samples shaped ``(num_samples, num_channels)`` row-major.
    The client decodes with
    ``cid = uuid.UUID(bytes=msg[:16])`` and
    ``np.frombuffer(msg[16:], dtype="<f4").reshape(-1, 2)``.
    """

    def __init__(
        self,
        scheduler: "Scheduler",
        magenta_server_url: str = "http://localhost:8000",
        port: int = 9100,
    ):
        # Session state. There is only ever one active client at a time,
        # matching the original Magenta server's assumption.
        self._session_ws: Optional[web.WebSocketResponse] = None
        self._pending_recipe: Optional[dict] = None
        self._style_embedding: Optional[np.ndarray] = None
        self._session_lock = asyncio.Lock()
        super().__init__(
            scheduler=scheduler,
            magenta_server_url=magenta_server_url,
            port=port,
        )

    def _register_routes(self):
        self._app.router.add_get("/stream", self._handle_ws)

    # ---------- websocket ----------

    async def _send_next_chunk(self, ws: web.WebSocketResponse):
        """Pop one chunk from the scheduler and send it as a binary frame.

        The frame layout is ``[16-byte UUID][float32 LE samples...]`` — the
        first 16 bytes are the raw binary form of the chunk's UUID, and the
        rest is the audio as row-major ``(num_samples, num_channels)``. The
        client uses the id to report playback progress back via
        ``UpdateProgress`` / ``UpdateRecipe`` so the scheduler can pick fork
        points relative to the real playback bar.
        """
        chunk = await self._pop_next_chunk_async()
        if chunk is None:
            return
        samples = np.asarray(chunk.audio, dtype=np.float32)
        # Ensure little-endian float32, row-major (num_samples, channels).
        if samples.dtype.byteorder == ">":
            samples = samples.astype("<f4")
        id_bytes = uuid.UUID(chunk.id).bytes
        buf = id_bytes + np.ascontiguousarray(samples).tobytes()
        if not ws.closed:
            try:
                await ws.send_bytes(buf)
            except Exception as e:  # pylint: disable=broad-exception-caught
                print(f"send_bytes failed: {e}")

    async def _handle_start_session(self):
        async with self._session_lock:
            self._stop_scheduler()
            self._pending_recipe = None
            self._style_embedding = None

    async def _handle_update_progress(self, body: dict):
        """Report the playback bar without changing the recipe.

        Mirrors the ``chunk_id`` / ``offset_seconds`` portion of
        ``UpdateRecipe``, but for clients that just want to push progress
        updates periodically so future fork points are accurate.
        """
        if not body:
            return
        chunk_id = body.get("chunk_id")
        offset = body.get("offset_seconds")
        if chunk_id is None or offset is None:
            return
        self._report_progress(chunk_id, float(offset))

    async def _handle_update_recipe(self, body: dict):
        """Caller supplies a recipe; embedding is built on the cluster.

        Mirrors the HTTP ``/update_recipe`` endpoint, including the
        optional ``chunk_id`` / ``offset_seconds`` playback-position fields
        used to pick the fork point.
        """
        if not body:
            return
        recipe = body.get("recipe") or {}
        chunk_id = body.get("chunk_id")
        offset = body.get("offset_seconds")
        if chunk_id is not None and offset is not None:
            self._report_progress(chunk_id, float(offset))
        emb = await self._embed_recipe(recipe)
        if emb is None:
            print("UpdateRecipe: empty recipe; ignoring")
            return
        async with self._session_lock:
            self._pending_recipe = dict(recipe)
            self._style_embedding = emb
            if self._started:
                self._update_recipe(recipe, emb)

    async def _handle_update_playback(
        self, body: dict, ws: web.WebSocketResponse
    ):
        if not body:
            return
        state = body.get("state")
        if state != "PLAYING":
            return
        async with self._session_lock:
            if self._style_embedding is None:
                print("UpdatePlayback PLAYING without a recipe; ignoring")
                return
            if not self._started:
                self._start_scheduler(
                    self._pending_recipe or {}, self._style_embedding
                )
        # Send the first chunk immediately so the client's ring buffer has
        # something to play while its audio callback primes.
        await self._send_next_chunk(ws)

    async def _handle_end_session(self):
        async with self._session_lock:
            self._stop_scheduler()
            self._pending_recipe = None
            self._style_embedding = None

    async def _handle_ws(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)

        async with self._session_lock:
            if self._session_ws is not None and not self._session_ws.closed:
                await ws.close(message=b"Session already active")
                return ws
            self._session_ws = ws

        client_id = f"{request.remote}"
        print(f"WS connected: {client_id}")

        try:
            async for msg in ws:
                if msg.type != web.WSMsgType.TEXT:
                    if msg.type == web.WSMsgType.ERROR:
                        print(f"WS error from {client_id}: {ws.exception()}")
                        break
                    continue
                try:
                    data = json.loads(msg.data)
                except json.JSONDecodeError:
                    print(f"WS bad json: {msg.data!r}")
                    continue

                msg_type = data.get("type")
                body = data.get("body") or {}

                if msg_type == "StartSession":
                    await self._handle_start_session()
                elif msg_type == "UpdateRecipe":
                    await self._handle_update_recipe(body)
                elif msg_type == "UpdateProgress":
                    await self._handle_update_progress(body)
                elif msg_type == "UpdatePlayback":
                    await self._handle_update_playback(body, ws)
                elif msg_type == "ReceivedChunk":
                    await self._send_next_chunk(ws)
                elif msg_type == "EndSession":
                    await self._handle_end_session()
                    break
                else:
                    print(f"WS unknown message type: {msg_type}")
        finally:
            async with self._session_lock:
                self._stop_scheduler()
                self._pending_recipe = None
                self._style_embedding = None
                if self._session_ws is ws:
                    self._session_ws = None
            if not ws.closed:
                await ws.close()
            print(f"WS disconnected: {client_id}")

        return ws



SERVER_TYPES = {
    "http": SchedulerServerHTTP,
    "websocket": SchedulerServerWebsocket,
}
DEFAULT_PORTS = {"http": 9100, "websocket": 9100}


def main():
    from absl import app as absl_app
    from absl import flags

    magenta_url = flags.DEFINE_string(
        "magenta_url", "http://localhost:8000", "URL of the Magenta RT server."
    )
    server_type = flags.DEFINE_enum(
        "server_type",
        "websocket",
        list(SERVER_TYPES.keys()),
        "Front-end protocol: 'http' for the JSON HTTP API, 'websocket' for "
        "the Magenta-compatible streaming protocol.",
    )
    port = flags.DEFINE_integer(
        "port",
        -1,
        "Port to listen on. -1 picks the default of 9100.",
    )
    buffer_chunks = flags.DEFINE_integer(
        "buffer_chunks", default_buffer_chunks, "Chunks to keep generated ahead."
    )

    def _run(_):
        scheduler = Scheduler(
            server_url=magenta_url.value, buffer_chunks=buffer_chunks.value
        )
        server_cls = SERVER_TYPES[server_type.value]
        chosen_port = (
            port.value if port.value >= 0 else DEFAULT_PORTS[server_type.value]
        )
        server = server_cls(
            scheduler=scheduler,
            magenta_server_url=magenta_url.value,
            port=chosen_port,
        )
        print(
            f"Starting {server_type.value} scheduler server on port {chosen_port}"
        )
        server.run()

    absl_app.run(_run)


if __name__ == "__main__":
    main()
