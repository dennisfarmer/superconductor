"""Start the client and the processes that send it messages.

    python -m superconductor_client              local music server
    python -m superconductor_client --remote     the server through the SSH tunnel
    python -m superconductor_client -- --mode calibration
"""
import argparse
import os
import signal
import subprocess
import sys
import time

CLIENT_URL = "http://127.0.0.1:8470"
LOCAL_SERVER = "ws://localhost:9100/stream"
REMOTE_SERVER = "ws://localhost:9000/stream"
# these two keep the session; the others are started with them and may exit on their own
REQUIRED = ("client", "tracking")


def _http_base(websocket_url):
    http = websocket_url.replace("wss://", "https://").replace("ws://", "http://")
    if http.endswith("/stream"):
        http = http[: -len("/stream")]
    return http


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--remote", action="store_true",
                        help=f"music server at {REMOTE_SERVER} (the SSH tunnel)")
    parser.add_argument("--server", default=None, help="MRT2 server websocket")
    args, tracking_args = parser.parse_known_args()
    server = args.server or (REMOTE_SERVER if args.remote else LOCAL_SERVER)
    py = sys.executable
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    commands = [
        ("client", [py, "-m", "superconductor_client.client", "--server", server]),
        ("describe", [py, "-m", "superconductor_client.describe.describe_server"]),
        ("midi", [py, "-m", "superconductor_client.midi.midi_player"]),
        ("conducting", [py, "-m", "superconductor_client.conducting.conducting_web.server",
                        "--server-url", _http_base(server)]),
        ("tracking", [py, "-m", "superconductor_client.tracking.collab",
                      "--client-url", CLIENT_URL, "--server", server, *tracking_args]),
    ]
    procs = []
    shutting_down = False

    def stop(*_):
        nonlocal shutting_down
        if shutting_down:
            return
        shutting_down = True
        for proc in procs:
            if proc.poll() is None:
                try:
                    os.killpg(proc.pid, signal.SIGINT)
                except OSError:
                    pass
        deadline = time.time() + 5
        while time.time() < deadline and any(p.poll() is None for p in procs):
            time.sleep(0.1)
        for proc in procs:
            if proc.poll() is None:
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except OSError:
                    pass

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    try:
        for name, cmd in commands:
            print(f"starting {name}", flush=True)
            procs.append(subprocess.Popen(cmd, env=env, start_new_session=True))
        announced = set()
        while not shutting_down:
            for (name, _), proc in zip(commands, procs):
                code = proc.poll()
                if code is None or name in announced:
                    continue
                announced.add(name)
                if name in REQUIRED:
                    print(f"{name} exited ({code}); stopping the others", flush=True)
                    stop()
                    return code
                print(f"{name} exited ({code})", flush=True)
            time.sleep(0.3)
    finally:
        stop()
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
