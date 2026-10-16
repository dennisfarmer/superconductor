"""Nonblocking exact-BPM client for the server branch's tempo HTTP API."""
import json
import math
import threading
import time
import warnings
from urllib.request import Request, urlopen
from urllib.parse import urlsplit


class ServerTempoPlayer:
    def __init__(self, server_url='http://localhost:9100', timeout=0.5):
        address = urlsplit(server_url)
        if address.scheme not in ('http', 'https') or not address.hostname or address.query or address.fragment:
            raise ValueError('Server URL must be an http(s) base URL without a query or fragment')
        self.server_url = server_url.rstrip('/')
        self.timeout = timeout
        self.condition = threading.Condition()
        self.paused = False  # Means conducting is released, not that server audio is muted.
        self.status = {}
        self.error = None
        self.pending = None
        self.version = 0
        self.last_target = None
        self.last_queued_at = 0.0
        self.next_send = 0.0
        self.closing = False
        self.worker = None
        self.release_confirmed = False

    @property
    def speed(self):
        with self.condition:
            return self.status.get('speed', 1.0)

    def __enter__(self):
        self.worker = threading.Thread(target=self._run, name='conductor-http', daemon=True)
        self.worker.start()
        return self

    def _queue(self, path, body):
        self.version += 1
        self.pending = (self.version, path, body)
        self.condition.notify_all()

    def set_tempo(self, bpm):
        if bpm is None:
            return  # Tracking gaps hold the server's target, never send /free.
        bpm = float(bpm)
        if not math.isfinite(bpm) or bpm <= 0:
            raise ValueError('Target BPM must be finite and positive')
        with self.condition:
            if self.paused or self.closing:
                return
            now = time.monotonic()
            if self.last_target is None or abs(bpm - self.last_target) >= 2.0 or now - self.last_queued_at >= 2.0:
                self.last_target = bpm
                self.last_queued_at = now
                self._queue('/tempo/target', {'bpm': bpm})

    def set_intentional_stop(self, stopped=True):
        with self.condition:
            if self.closing:
                return
            self.paused = stopped
            self.last_target = None
            if stopped:
                self.next_send = 0.0
                self._queue('/tempo/free', None)

    def _request(self, path, body=None):
        data = None if path == '/tempo/status' else json.dumps(body, allow_nan=False).encode()
        request = Request(self.server_url + path, data=data,
                          headers={'Content-Type': 'application/json'},
                          method='GET' if data is None else 'POST')
        with urlopen(request, timeout=self.timeout) as response:
            status = json.load(response)
        if not isinstance(status, dict):
            raise ValueError('Expected a JSON object from the tempo server')
        speed = status.get('speed')
        if not isinstance(speed, (float, int)) or not math.isfinite(speed) or speed <= 0:
            raise ValueError('Invalid speed in tempo server response')
        return status

    def _run(self):
        next_poll = 0.0
        while True:
            with self.condition:
                now = time.monotonic()
                if self.closing and now >= self.close_deadline:
                    return
                item = self.pending if now >= self.next_send else None
                if item is None and (self.closing or now < next_poll):
                    self.condition.wait(0.05)
                    continue
            path, body = (item[1], item[2]) if item else ('/tempo/status', None)
            try:
                status = self._request(path, body)
            except (OSError, ValueError) as exc:
                with self.condition:
                    self.error = str(exc)
                    if item:
                        self.next_send = time.monotonic() + (0.2 if self.closing else 1.0)
                next_poll = time.monotonic() + 1.0
                continue
            with self.condition:
                self.status = status
                # A successful status poll must not hide an unsent failed command.
                if item or self.pending is None:
                    self.error = None
                if item and self.pending and self.pending[0] == item[0]:
                    self.pending = None
                if item:
                    self.next_send = time.monotonic() + 0.25
                if self.closing and path == '/tempo/free':
                    self.release_confirmed = True
                    return
            next_poll = time.monotonic() + 1.0

    def display_status(self):
        with self.condition:
            if self.error:
                return 'Server unavailable / request failed - retrying'
            if not self.status:
                return 'Connecting to tempo server...'
            if self.status.get('model_bpm') is None:
                return 'Server calibrating - connect an audio client'
            return f"Server: {self.status.get('mode', '?')} | {self.status['speed']:.2f}x"

    def close(self):
        if self.worker is None or not self.worker.is_alive():
            return
        with self.condition:
            self.closing = True
            self.close_deadline = time.monotonic() + 1.2
            self.next_send = 0.0
            self._queue('/tempo/free', None)
        self.worker.join(timeout=2.5)
        if not self.release_confirmed:
            warnings.warn('Could not confirm /tempo/free on exit; server may retain the last target', RuntimeWarning)

    def __exit__(self, *exc):
        self.close()
