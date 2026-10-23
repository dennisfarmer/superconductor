"""The browser conducting glue (conducting_web/server.py): landmarks from the page in,
tempo requests out, with a fake tempo server in place of the MRT2 server."""
import asyncio
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import unittest

import aiohttp
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "conducting_web"))
from server import Conductor, ConductingServer, TimedTempoPlayer, parse_hands  # noqa: E402
from test_cycle_tempo import motion  # noqa: E402


def landmarks(x=0.0, y=0.0, fist=False):
    """a page's hand: 21 {x, y, z} dicts, the middle fingertip (12) at the mirrored (x, y)"""
    points = [{"x": 0.5 + x, "y": 0.5 + y, "z": 0.0} for _ in range(21)]
    for (mcp, pip, dip, tip), offset in zip(((5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20)),
                                            (-0.04, 0, 0.03, 0.06)):
        for index, dy in ((mcp, 0.10), (pip, 0.16), (dip, 0.13 if fist else 0.20), (tip, 0.09 if fist else 0.24)):
            points[index] = {"x": 0.5 + x + offset, "y": 0.5 + y + dy, "z": 0.0}
    return points


def conducting(bpm, seconds=12):
    """frames of a hand drawing a 4/4 pattern at `bpm`: (t_ns, [hand])"""
    for t_ns, x, y in motion(bpm, seconds):
        hand = landmarks(fist=False)
        # the page sends unmirrored landmarks; the conductor mirrors x back (1 - x)
        hand[12] = {"x": 1 - x, "y": y, "z": 0.0}
        yield t_ns, [hand]


class FakePlayer:
    def __init__(self):
        self.condition = threading.Condition()
        self.paused = False
        self.status, self.error = {}, None
        self.post_ms = self.to_server_ms = None
        self.targets, self.stops = [], []

    speed = 1.0

    def set_tempo(self, bpm):
        if bpm is not None and not self.paused:
            self.targets.append(bpm)

    def set_intentional_stop(self, stopped=True):
        self.paused = stopped
        self.stops.append(stopped)

    def display_status(self):
        return "Connecting to tempo server..."


class ConductorTests(unittest.TestCase):
    def test_conducting_sets_the_target(self):
        player = FakePlayer()
        conductor = Conductor(player)
        replies = [conductor.step(hands, t, 16 / 9) for t, hands in conducting(120)]
        self.assertIsNone(replies[30]["bpm"])
        self.assertGreater(replies[30]["warmup"], 0)
        self.assertLess(abs(np.median(player.targets) - 120), 2)
        self.assertIn("BPM", replies[-1]["status"])
        self.assertIsNotNone(replies[-1]["step_ms"])

    def test_fist_releases_and_r_resumes(self):
        player = FakePlayer()
        conductor = Conductor(player)
        reply = None
        for i in range(8):
            reply = conductor.step([landmarks(fist=True)], i * 40_000_000, 16 / 9)
        self.assertTrue(reply["released"])
        self.assertEqual(player.stops, [True])
        self.assertTrue(reply["status"].startswith("Released: Closed fist"))
        conductor.key("r")
        self.assertEqual(player.stops, [True, False])
        self.assertFalse(conductor.step([landmarks()], 400_000_000, 16 / 9)["released"])

    def test_f_releases(self):
        player = FakePlayer()
        conductor = Conductor(player)
        conductor.key("f")
        reply = conductor.step([landmarks()], 1, 16 / 9)
        self.assertEqual(reply["status"], "Released: Manual release - R to resume")
        self.assertEqual(player.stops, [True])

    def test_timestamps_going_back_start_over(self):
        conductor = Conductor(FakePlayer())
        conductor.step([landmarks()], 5_000_000_000, 16 / 9)
        conductor.step([landmarks()], 1_000_000_000, 16 / 9)  # a reloaded page's clock: no ValueError
        self.assertEqual(conductor.estimator.started, 1.0)

    def test_malformed_hands(self):
        with self.assertRaises(ValueError):
            parse_hands({"hands": [[{"x": 0, "y": 0}] * 20]})
        with self.assertRaises(ValueError):
            parse_hands({"hands": [[{"x": float("nan"), "y": 0}] * 21]})
        self.assertEqual(len(parse_hands({"hands": [landmarks()]})), 1)


class FakeTempoServer(BaseHTTPRequestHandler):
    requests = []

    def do_GET(self):
        self._answer()

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        FakeTempoServer.requests.append((self.path, json.loads(self.rfile.read(length) or b"null")))
        self._answer()

    def _answer(self):
        body = json.dumps({"mode": "target", "speed": 1.0, "model_bpm": 120.0, "target_bpm": None}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


class EndToEndTests(unittest.TestCase):
    """page -> /ws/hands -> conductor -> HTTP POST /tempo/target on a fake tempo server"""

    def test_websocket_to_tempo_server(self):
        FakeTempoServer.requests = []
        http = ThreadingHTTPServer(("localhost", 0), FakeTempoServer)
        threading.Thread(target=http.serve_forever, daemon=True).start()
        url = f"http://localhost:{http.server_address[1]}"
        try:
            with TimedTempoPlayer(url) as player:
                asyncio.run(self._conduct(ConductingServer(player, url)))
                deadline = time.monotonic() + 2
                while player.to_server_ms is None and time.monotonic() < deadline:
                    time.sleep(0.05)
                self.assertIsNotNone(player.post_ms)
                self.assertIsNotNone(player.to_server_ms)
        finally:
            http.shutdown()
            http.server_close()
        targets = [body["bpm"] for path, body in FakeTempoServer.requests if path == "/tempo/target"]
        self.assertTrue(targets)
        self.assertLess(abs(targets[-1] - 120), 2)
        self.assertEqual(FakeTempoServer.requests[-1], ("/tempo/free", None))  # released on exit

    async def _conduct(self, server):
        from aiohttp.test_utils import TestClient, TestServer
        async with TestClient(TestServer(server.app())) as client:
            config = await (await client.get("/api/web")).json()
            self.assertEqual(config["server_url"], server.server_url)
            monitor = await client.ws_connect("/ws/hands/monitor")
            ws = await client.ws_connect("/ws/hands")
            last = None
            for seq, (t_ns, hands) in enumerate(conducting(120, seconds=8)):
                await ws.send_str(json.dumps({"type": "hands", "seq": seq, "t_ms": t_ns / 1e6,
                                              "width": 960, "height": 540, "hands": hands}))
                last = json.loads((await ws.receive()).data)
                self.assertEqual(last["seq"], seq)
            self.assertIsNotNone(last["bpm"])
            # the listener got every frame as the page sent it, with the conductor's reply
            for seq in range(240):
                seen = json.loads((await monitor.receive(timeout=2)).data)
                self.assertEqual((seen["type"], seen["seq"], len(seen["hands"][0])), ("hands", seq, 21))
            self.assertIn("bpm", seen["conduct"])
            await monitor.close()
            await ws.send_str(json.dumps({"type": "hands", "seq": -1, "hands": "nope"}))
            self.assertEqual(json.loads((await ws.receive()).data)["type"], "error")
            stats = await (await client.get("/api/stats")).json()
            self.assertEqual(stats["frames"], 240)
            await ws.close()


if __name__ == "__main__":
    unittest.main()
