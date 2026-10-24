"""addresses.toml is the only list of music-server addresses."""
import tempfile
import unittest
from pathlib import Path

from superconductor_client.addresses import AddressError, choose_place, describe_machine, load_place

ROOT = Path(__file__).resolve().parents[1] / "superconductor_client" / "addresses.toml"


class AddressesTest(unittest.TestCase):
    def test_shipped_places(self):
        local = load_place("local", ROOT)
        self.assertEqual(local.direct, local.browser)
        self.assertEqual(local.direct.input, "ws://localhost:9100/stream/input")
        self.assertEqual(local.browser.playback, "ws://localhost:9100/stream/playback")
        self.assertEqual(local.direct.http, "http://localhost:9100")

        remote = load_place("remote", ROOT)
        self.assertEqual(remote.browser.playback, "ws://localhost:9000/stream/playback")
        self.assertEqual(remote.direct.http, "http://localhost:9000")

        cluster = load_place("cluster", ROOT)
        self.assertEqual(cluster.direct.input, "ws://localhost:9100/stream/input")
        self.assertEqual(cluster.direct.http, "http://localhost:9100")
        self.assertEqual(cluster.browser.playback, "ws://server.localhost:8080/stream/playback")
        self.assertEqual(cluster.browser.http, "http://server.localhost:8080")
        self.assertIn(("WS playback", cluster.browser.playback), cluster.service_endpoints())
        self.assertIn(("WS input", cluster.direct.input), cluster.service_endpoints())
        self.assertIn(("page", "http://server.localhost:8080/tempo"), cluster.service_endpoints())

        moved = load_place("cluster", ROOT, gateway_port="9090")
        self.assertEqual(moved.browser.http, "http://server.localhost:9090")
        self.assertEqual(moved.direct.http, "http://localhost:9100")

        docker = load_place("docker", ROOT)
        self.assertEqual(docker.direct.input, "ws://host.docker.internal:9000/stream/input")
        self.assertEqual(docker.browser.playback, "ws://localhost:9000/stream/playback")

    def test_unknown_name_lists_the_file(self):
        with self.assertRaises(AddressError) as caught:
            load_place("third-machine", ROOT)
        self.assertIn("cluster", str(caught.exception))
        self.assertIn("local", str(caught.exception))

    def test_the_prompt_follows_the_machine(self):
        kind, choices = describe_machine("Denniss-MacBook-Pro.local")
        self.assertEqual(kind, "laptop")
        self.assertEqual([name for name, _ in choices], ["local", "remote"])

        kind, choices = describe_machine("lh2300.arc-ts.umich.edu")
        self.assertEqual(kind, "cluster node")
        self.assertEqual([name for name, _ in choices], ["local"])

        answers = iter(["nope", "2"])
        picked = choose_place("mac.local", True, lambda _prompt: next(answers))
        self.assertEqual(picked, "remote")
        self.assertEqual(choose_place("mac.local", True, lambda _prompt: ""), "local")

        def fail(_prompt):
            raise AssertionError("a single option should not ask")

        self.assertEqual(choose_place("lh2300", False, fail), "local")

    def test_a_bad_address_is_refused(self):
        text = """
[local]
[local.direct]
input = "ws://localhost:9100/stream"
playback = "ws://localhost:9100/stream/playback"
http = "http://localhost:9100"
"""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "addresses.toml"
            path.write_text(text)
            with self.assertRaises(AddressError):
                load_place("local", path)


if __name__ == "__main__":
    unittest.main()
