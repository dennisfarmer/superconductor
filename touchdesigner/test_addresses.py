"""make visualize reads its addresses from addresses.toml."""
import unittest

from addresses import load, visualize_plan


class VisualizeAddressesTest(unittest.TestCase):
    def test_plan_uses_the_file(self):
        server, client = load()
        plan = visualize_plan(server, client)
        self.assertIn("CHECK_PORTS='9000 9100 9102'", plan)
        self.assertIn(f"TRACKING_URL={client.tracking}", plan)
        self.assertIn(f"PLAYBACK_URL={server.playback}", plan)


if __name__ == "__main__":
    unittest.main()
