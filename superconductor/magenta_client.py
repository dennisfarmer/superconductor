"""Laptop-side Magenta client.

Drop-in replacement for the original websocket-based ``MagentaClient``. The
laptop no longer talks to the Magenta server directly — it talks to a
``SchedulerServer`` running on the compute cluster (reached via an SSH
tunnel). The scheduler owns all the generation state, the chunk queue, and
the recipe → style embedding conversion; this client only ever receives
audio samples and chunk ids.

The public surface mirrors the original client so ``laptop.py`` can use it
without changes:

    client = MagentaClient()
    client.start()
    client.update_recipe({"Rock": 0.6, "Guitar": 0.8, "Jazz": 0.3})
    client.stop()

"""

import base64
"""Laptop-side Magenta client.

Drop-in replacement for the original websocket-based ``MagentaClient``. The
laptop no longer talks to the Magenta server directly — it talks to a
``SchedulerServer`` running on the compute cluster (reached via an SSH
tunnel). The scheduler owns all the generation state, the chunk queue, and
the recipe → style embedding conversion; this client only ever receives
audio samples and chunk ids.

The public surface mirrors the original client so ``laptop.py`` can use it
without changes:

    client = MagentaClient()
    client.start()
    client.update_recipe({"Rock": 0.6, "Guitar": 0.8, "Jazz": 0.3})
    client.stop()

"""

import base64
import threading
import time
import time

import numpy as np
import requests
import sounddevice as sd


SAMPLE_RATE = 48000
CHANNELS = 2
CHUNK_LENGTH_SECONDS = 2.0
CHUNK_LENGTH_SECONDS = 2.0



class MagentaClient:

    def __init__(self, uri="http://localhost:9000"):
        # Preserved attributes from the original client. ``ws`` and ``loop``
        # no longer back a real websocket / asyncio loop, but they exist so
        # that any caller that touches them keeps working.
    def __init__(self, uri="http://localhost:9000"):
        # Preserved attributes from the original client. ``ws`` and ``loop``
        # no longer back a real websocket / asyncio loop, but they exist so
        # that any caller that touches them keeps working.
        self.uri = uri
        self.ws = None
        self.loop = None
        self.thread = None
        self.connected = False

        # cache embeddings so we don't repeatedly call /style
        self.embedding_cache = {}

        # New internals for the scheduler-backed implementation.
        self._session = requests.Session()
        self._running = False
        self._state_lock = threading.Lock()
        self._current_chunk_id = None
        self._chunk_started_at = None

        # New internals for the scheduler-backed implementation.
        self._session = requests.Session()
        self._running = False
        self._state_lock = threading.Lock()
        self._current_chunk_id = None
        self._chunk_started_at = None

    # ---------- public API ----------

    def start(self):
        """Start the session and begin streaming audio in a background thread."""
        """Start the session and begin streaming audio in a background thread."""
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()

    def update_recipe(self, recipe):
        """
        recipe example:
        {"Rock":0.6,"Guitar":0.8,"Jazz":0.3}
        """
        if not self.connected:
            print("MagentaClient not connected yet")
            return

        chunk_id, offset = self._current_playback_position()
        body = {"recipe": recipe}
        if chunk_id is not None:
            body["chunk_id"] = chunk_id
            body["offset_seconds"] = offset
        try:
            r = self._session.post(
                f"{self.uri}/update_recipe", json=body, timeout=30
            )
            r.raise_for_status()
        except Exception as e:
            print(f"update_recipe failed: {e}")

    def stop(self):
        """Stop playback and tear down the session.

        Synchronous (the original was declared ``async`` but ``laptop.py``
        called it without ``await``; this version is callable from sync code
        and still works if a caller happens to await it via ``asyncio.run``).
        """
        print("Stopping Magenta client...")
        self._running = False
        self.connected = False

        try:
            self._session.post(f"{self.uri}/stop", timeout=10)
        except Exception as e:
            print("Send stop error:", e)

        try:
            sd.stop()
        except Exception:
            pass

        if self.thread and self.thread.is_alive() and self.thread is not threading.current_thread():
            self.thread.join(timeout=5.0)
        chunk_id, offset = self._current_playback_position()
        body = {"recipe": recipe}
        if chunk_id is not None:
            body["chunk_id"] = chunk_id
            body["offset_seconds"] = offset
        try:
            r = self._session.post(
                f"{self.uri}/update_recipe", json=body, timeout=30
            )
            r.raise_for_status()
        except Exception as e:
            print(f"update_recipe failed: {e}")

    def stop(self):
        """Stop playback and tear down the session.

        Synchronous (the original was declared ``async`` but ``laptop.py``
        called it without ``await``; this version is callable from sync code
        and still works if a caller happens to await it via ``asyncio.run``).
        """
        print("Stopping Magenta client...")
        self._running = False
        self.connected = False

        try:
            self._session.post(f"{self.uri}/stop", timeout=10)
        except Exception as e:
            print("Send stop error:", e)

        try:
            sd.stop()
        except Exception:
            pass

        if self.thread and self.thread.is_alive() and self.thread is not threading.current_thread():
            self.thread.join(timeout=5.0)

    # ---------- internal ----------

    def _run_loop(self):
        """Background-thread entry point: open the session, then stream chunks."""
        try:
            r = self._session.post(
                f"{self.uri}/start",
                json={"recipe": {"jazz": 1.0}},
                timeout=30,
            )
            r.raise_for_status()
        except Exception as e:
            print(f"Failed to connect to scheduler at {self.uri}: {e}")
            return

        self.connected = True
        self._running = True
        print("Connected to MagentaRT scheduler")

        try:
            self._receive_audio()
        finally:
            self.connected = False

    def _current_playback_position(self):
        with self._state_lock:
            if self._current_chunk_id is None or self._chunk_started_at is None:
                return None, 0.0
            offset = time.monotonic() - self._chunk_started_at
            offset = max(0.0, min(CHUNK_LENGTH_SECONDS, offset))
            return self._current_chunk_id, offset


    def _fetch_next(self):
        """Blocking HTTP fetch for the next chunk. Returns (id, samples) or None."""
        try:
            r = self._session.post(f"{self.uri}/next_chunk", timeout=60)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            print(f"next_chunk failed: {e}")
            return None
        audio_bytes = base64.b64decode(data["audio"].encode("ascii"))
        audio = np.frombuffer(audio_bytes, dtype=np.float32)
        shape = data.get("shape")
        if shape:
            audio = audio.reshape(shape)
        elif audio.ndim == 1 and CHANNELS > 1:
            audio = audio.reshape(-1, CHANNELS)
        return data["id"], audio

    def _receive_audio(self):
        """Fetch chunks from the scheduler and play them back-to-back.

        Keeps at most one chunk in-flight beyond the currently playing one,
        so that recipe forks on the scheduler side don't invalidate locally
        buffered audio past the scheduler's safety margin.
        """
        pending = self._fetch_next()
        while self._running and pending is not None:
            chunk_id, audio = pending

            with self._state_lock:
                self._current_chunk_id = chunk_id
                self._chunk_started_at = time.monotonic()

            try:
                sd.play(audio, SAMPLE_RATE, blocking=False)
            except Exception as e:
                print(f"sd.play failed: {e}")
                break

            next_pending = self._fetch_next() if self._running else None

            try:
                sd.wait()
            except Exception:
                pass

            pending = next_pending

        with self._state_lock:
            self._current_chunk_id = None
            self._chunk_started_at = None

    
    # ----------------------------------------------
    # other things from original client
    # ----------------------------------------------
        """Background-thread entry point: open the session, then stream chunks."""
        try:
            r = self._session.post(
                f"{self.uri}/start",
                json={"recipe": {"jazz": 1.0}},
                timeout=30,
            )
            r.raise_for_status()
        except Exception as e:
            print(f"Failed to connect to scheduler at {self.uri}: {e}")
            return

        self.connected = True
        self._running = True
        print("Connected to MagentaRT scheduler")

        try:
            self._receive_audio()
        finally:
            self.connected = False

    def _current_playback_position(self):
        with self._state_lock:
            if self._current_chunk_id is None or self._chunk_started_at is None:
                return None, 0.0
            offset = time.monotonic() - self._chunk_started_at
            offset = max(0.0, min(CHUNK_LENGTH_SECONDS, offset))
            return self._current_chunk_id, offset


    def _fetch_next(self):
        """Blocking HTTP fetch for the next chunk. Returns (id, samples) or None."""
        try:
            r = self._session.post(f"{self.uri}/next_chunk", timeout=60)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            print(f"next_chunk failed: {e}")
            return None
        audio_bytes = base64.b64decode(data["audio"].encode("ascii"))
        audio = np.frombuffer(audio_bytes, dtype=np.float32)
        shape = data.get("shape")
        if shape:
            audio = audio.reshape(shape)
        elif audio.ndim == 1 and CHANNELS > 1:
            audio = audio.reshape(-1, CHANNELS)
        return data["id"], audio

    def _receive_audio(self):
        """Fetch chunks from the scheduler and play them back-to-back.

        Keeps at most one chunk in-flight beyond the currently playing one,
        so that recipe forks on the scheduler side don't invalidate locally
        buffered audio past the scheduler's safety margin.
        """
        pending = self._fetch_next()
        while self._running and pending is not None:
            chunk_id, audio = pending

            with self._state_lock:
                self._current_chunk_id = chunk_id
                self._chunk_started_at = time.monotonic()

            try:
                sd.play(audio, SAMPLE_RATE, blocking=False)
            except Exception as e:
                print(f"sd.play failed: {e}")
                break

            next_pending = self._fetch_next() if self._running else None

            try:
                sd.wait()
            except Exception:
                pass

            pending = next_pending

        with self._state_lock:
            self._current_chunk_id = None
            self._chunk_started_at = None

    
    # ----------------------------------------------
    # other things from original client
    # ----------------------------------------------

    def _get_cached_embedding(self, prompt):
        """Preserved from the original client.

        The scheduler now does recipe → embedding conversion on the cluster,
        so this is mostly a compatibility shim. It still works on its own if
        a caller wants embeddings on the laptop.
        """

        def get_style_embedding(text):
            r = requests.post("http://localhost:9000/style", data=text)
            return r.json()
        """Preserved from the original client.

        The scheduler now does recipe → embedding conversion on the cluster,
        so this is mostly a compatibility shim. It still works on its own if
        a caller wants embeddings on the laptop.
        """

        def get_style_embedding(text):
            r = requests.post("http://localhost:9000/style", data=text)
            return r.json()

        if prompt not in self.embedding_cache:
            emb = np.array(get_style_embedding(prompt), dtype=np.float32)
            self.embedding_cache[prompt] = emb
        return self.embedding_cache[prompt]


    def _recipe_to_embedding(self, recipe):
        """Preserved from the original client (see ``_get_cached_embedding``)."""
        """Preserved from the original client (see ``_get_cached_embedding``)."""
        embeddings = []
        weights = []
        for prompt, weight in recipe.items():
            if weight <= 0:
                continue
            embeddings.append(self._get_cached_embedding(prompt))
            embeddings.append(self._get_cached_embedding(prompt))
            weights.append(weight)
        if not embeddings:
            return None
        embeddings = np.stack(embeddings)
        weights = np.array(weights, dtype=np.float32)
        weights = weights / weights.sum()
        return np.sum(embeddings * weights[:, None], axis=0)