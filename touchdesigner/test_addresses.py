"""make visualize asks for the client and the server, then uses addresses.toml."""
import unittest

from addresses import choose_locations, load_place, place_for, visualize_plan


class VisualizeAddressesTest(unittest.TestCase):
    def test_enter_means_both_on_this_computer(self):
        answers = iter(["", ""])
        self.assertEqual(choose_locations("mac.local", True, lambda _prompt: next(answers)), "local")
        self.assertEqual(place_for("here", "remote"), "remote")
        self.assertEqual(place_for("gateway", "cluster"), "cluster")

    def test_the_container_has_one_server_choice(self):
        def read(_prompt):
            return "2"

        self.assertEqual(choose_locations("mac.local", True, read), "cluster")

    def test_a_cluster_node_does_not_ask(self):
        def fail(_prompt):
            raise AssertionError("a single option should not ask")

        self.assertEqual(choose_locations("lh2300", False, fail), "local")

    def test_plans_use_the_file(self):
        local = load_place("local")
        remote = load_place("remote")
        cluster = load_place("cluster")
        self.assertIn("client-local", visualize_plan(local))
        self.assertIn("ws://localhost:9000/stream/playback", visualize_plan(remote))
        self.assertEqual(cluster.client.tracking, "ws://localhost:8080/ws/tracking")
        self.assertEqual(cluster.client.hands, "ws://localhost:8080/conduct/ws/hands/monitor")
        self.assertIn("CHECK_PORT=8080", visualize_plan(cluster))


if __name__ == "__main__":
    unittest.main()
