"""Client for the describe server (../superconductor_describe/describe_server.py).

Requests run on a background thread, so the camera loop never waits for the
VLM. Results are queued in `results` as (kind, id, text) and applied by the
camera loop:
    ("description", id, text)
    ("instrument", id, text)   suggested from the description
    ("name", id, text)         default name, suggested from the description
    ("error", id, message)
`info` is the server's /health answer (model, backend, host), {} until known.
"""
import base64
import json
import queue
import threading
import urllib.error
import urllib.request


class DescribeClient:
    def __init__(self, server_url="http://localhost:9200", timeout=120.0):
        self.server_url = server_url.rstrip("/")
        self.timeout = timeout
        self.results = queue.Queue()
        self.busy = set()  # ids with a request queued or running
        self.info = {}
        self._jobs = queue.Queue()
        threading.Thread(target=self._run, daemon=True).start()

    def describe(self, id, jpeg, suggest=True, name=True):
        """Describe an object from its picture, then suggest an instrument (if
        `suggest`) and a name (if `name`) from the description."""
        self.busy.add(id)
        self._jobs.put(("describe", id, jpeg, suggest, name))

    def suggest(self, id, description):
        """Suggest an instrument from an existing description."""
        self.busy.add(id)
        self._jobs.put(("suggest", id, description, True, False))

    def _post(self, path, body):
        req = urllib.request.Request(f"{self.server_url}{path}", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read())

    def _health(self):
        try:
            with urllib.request.urlopen(f"{self.server_url}/health", timeout=5) as r:
                self.info = json.loads(r.read())
        except (urllib.error.URLError, OSError, ValueError):
            self.info = {}

    def _run(self):
        self._health()
        while True:
            try:
                kind, id, data, suggest, name = self._jobs.get(timeout=10)
            except queue.Empty:
                self._health()  # notice the server starting or stopping
                continue
            try:
                if kind == "describe":
                    description = self._post("/describe", {"image": base64.b64encode(data).decode()})["description"]
                    self.results.put(("description", id, description))
                else:
                    description = data
                if name:
                    self.results.put(("name", id, self._post("/name", {"description": description})["name"]))
                if suggest:
                    instrument = self._post("/instrument", {"description": description})["instrument"]
                    self.results.put(("instrument", id, instrument))
            except (urllib.error.URLError, OSError, KeyError, ValueError) as e:
                self.results.put(("error", id, f"describe server {self.server_url}: {e}"))
            finally:
                self.busy.discard(id)
