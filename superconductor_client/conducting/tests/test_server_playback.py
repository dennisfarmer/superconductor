"""Contract tests against an in-process HTTP server; no production server is used."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
import time
import unittest
from unittest.mock import patch, Mock

from live.server_playback import ServerTempoPlayer


def wait_for(predicate, timeout=3):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError('Background request did not complete')


class ServerTests(unittest.TestCase):
    def setUp(self):
        self.received = []
        self.failures = 0
        self.entered = threading.Event()
        self.release = threading.Event()
        self.release.set()
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                self.respond()

            def do_POST(self):
                raw = self.rfile.read(int(self.headers['Content-Length']))
                owner.received.append((self.path, json.loads(raw)))
                if self.path == '/tempo/target':
                    owner.entered.set()
                    owner.release.wait(2)
                    if owner.failures:
                        owner.failures -= 1
                        self.send_error(503)
                        return
                self.respond()

            def respond(self):
                payload = json.dumps({'speed': 1.1, 'mode': 'target', 'model_bpm': 100,
                                      'heard_bpm': 110, 'conductor_bpm': None}).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.player = ServerTempoPlayer(f'http://127.0.0.1:{self.server.server_port}')

    def tearDown(self):
        self.release.set()
        self.player.close()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def test_exact_target_status_hold_and_exit_release(self):
        with self.player:
            self.player.set_tempo(120)
            wait_for(lambda: self.player.pending is None and self.player.status)
            self.assertIn(('/tempo/target', {'bpm': 120.0}), self.received)
            self.assertEqual(self.player.speed, 1.1)
            before = list(self.received)
            self.player.set_tempo(None)
            self.player.set_tempo(121)
            time.sleep(0.1)
            self.assertEqual(self.received, before)
        self.assertEqual(self.received[-1], ('/tempo/free', None))
        self.assertTrue(self.player.release_confirmed)
        self.assertFalse(self.player.worker.is_alive())

    def test_latest_target_coalesces_and_free_cancels_pending_target(self):
        self.release.clear()
        with self.player:
            self.player.set_tempo(100)
            self.assertTrue(self.entered.wait(1))
            started = time.monotonic()
            for bpm in range(110, 180):
                self.player.set_tempo(bpm)
            self.assertLess(time.monotonic() - started, 0.1)
            self.player.set_intentional_stop()
            self.player.set_tempo(200)
            self.release.set()
            wait_for(lambda: ('/tempo/free', None) in self.received)
            self.assertEqual([r for r in self.received if r[0] == '/tempo/target'],
                             [('/tempo/target', {'bpm': 100.0})])
            self.player.set_intentional_stop(False)
            self.player.set_tempo(130)
            wait_for(lambda: ('/tempo/target', {'bpm': 130.0}) in self.received)

    def test_failed_target_retries_and_recovers(self):
        self.failures = 1
        with self.player:
            self.player.set_tempo(120)
            wait_for(lambda: self.player.error is not None)
            self.assertIn('retrying', self.player.display_status())
            wait_for(lambda: self.player.pending is None and self.player.error is None)
            self.assertEqual(self.received.count(('/tempo/target', {'bpm': 120.0})), 2)

    def test_shutdown_failure_is_reported(self):
        with patch.object(self.player, '_request', side_effect=OSError('offline')):
            self.player.__enter__()
            with self.assertWarnsRegex(RuntimeWarning, 'Could not confirm'):
                self.player.close()
            self.assertFalse(self.player.worker.is_alive())

    def test_validation_and_null_body(self):
        for bpm in (0, -1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                self.player.set_tempo(bpm)
        with self.assertRaises(ValueError):
            ServerTempoPlayer('file:///tmp/music')


class RoutingTests(unittest.TestCase):
    def test_default_server_mode_does_not_require_music_or_bungee(self):
        from live.live_conductor import run_live_conductor
        with patch('live.live_conductor.ServerTempoPlayer') as remote, \
             patch('live.live_conductor.resolve_music') as resolve, \
             patch('live.live_conductor.MusicPlayer') as local, \
             patch('live.live_conductor._run_live') as run:
            run_live_conductor(0, '/missing/music.mp3', server_url='http://localhost:9000')
            remote.assert_called_once_with('http://localhost:9000')
            resolve.assert_not_called()
            local.assert_not_called()
            self.assertEqual(run.call_args.args[3], 'server')

    def test_cli_defaults_and_local_override(self):
        from new_tempo_detector import main
        with patch('live.live_conductor.run_live_conductor') as run:
            main([])
            self.assertEqual(run.call_args.kwargs['playback'], 'server')
            main(['--server-url', 'http://localhost:9000'])
            self.assertEqual(run.call_args.kwargs['server_url'], 'http://localhost:9000')
            main(['--playback', 'local'])
            self.assertEqual(run.call_args.kwargs['playback'], 'local')
