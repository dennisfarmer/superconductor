"""The laptop server refuses to start when the Lighthouse SSH tunnel holds its port."""
import unittest

from mrt2_server import is_lighthouse_tunnel


class TunnelCommandTests(unittest.TestCase):
    def test_readme_tunnel(self):
        command = "ssh -N -L 9000:lh2300:9000 uniqname@lighthouse.arc-ts.umich.edu"
        self.assertTrue(is_lighthouse_tunnel(command, 9000))
        self.assertFalse(is_lighthouse_tunnel(command, 9101))

    def test_two_hop_tunnel(self):
        command = (
            "ssh -t -L 9000:127.0.0.1:19100 uniqname@lighthouse.arc-ts.umich.edu "
            "ssh -N -L 19100:127.0.0.1:9000 lh2300"
        )
        self.assertTrue(is_lighthouse_tunnel(command, 9000))

    def test_forward_without_a_space(self):
        command = "/usr/bin/ssh -N -L9000:lh2300:9000 uniqname@lh2300"
        self.assertTrue(is_lighthouse_tunnel(command, 9000))

    def test_other_ssh_forward_is_not_the_lighthouse_server(self):
        command = "ssh -N -L 9000:example.com:80 user@example.com"
        self.assertFalse(is_lighthouse_tunnel(command, 9000))

    def test_local_server_process_is_not_a_tunnel(self):
        self.assertFalse(is_lighthouse_tunnel("python mrt2_server.py --port 9000", 9000))
        self.assertFalse(is_lighthouse_tunnel("", 9000))


if __name__ == "__main__":
    unittest.main()
