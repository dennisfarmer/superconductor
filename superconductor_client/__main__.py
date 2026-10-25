"""Start the client and the processes that send it messages.

    python -m superconductor_client                     ask where the music server is
    python -m superconductor_client --where local       server on this computer
    python -m superconductor_client --where remote      server through the SSH tunnel
    python -m superconductor_client --where cluster
    python -m superconductor_client --describe        also start the describe server
    python -m superconductor_client -- --mode calibration

The name is an entry in superconductor_client/addresses.toml. That file has the
input socket, the playback socket, and the http address. This script does not
build one from another.
"""
import argparse
import os
import signal
import socket
import subprocess
import sys
import time

from superconductor_client.addresses import AddressError, choose_place, load_place
from superconductor_client.torch_device import DESCRIBE_NEEDS_GPU, cuda_state_detached

CLIENT_URL = "http://127.0.0.1:8470"
# these two keep the session; the others are started with them and may exit on their own
REQUIRED = ("client", "tracking")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--where", dest="place", default=None, metavar="NAME",
                        help="which music server, a name in addresses.toml "
                             "(local, remote, cluster). Omit to be asked.")
    parser.add_argument("--describe", action="store_true",
                        help="also start the describe server (needs Ollama already running)")
    args, tracking_args = parser.parse_known_args()
    place_name = args.place
    if place_name is None:
        place_name = choose_place(socket.gethostname(), sys.stdin.isatty(), input)
        if place_name is None:
            return 2
    try:
        place = load_place(place_name)
    except AddressError as exc:
        parser.error(str(exc))
    py = sys.executable
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    commands = [
        ("client", [py, "-m", "superconductor_client.client", "--server", place.direct.input]),
    ]
    if args.describe:
        # macOS runs the icon model with MLX. Linux needs a CUDA device this process
        # can open. Without one, skip describe and keep the client.
        gpu_ok = sys.platform == "darwin" or cuda_state_detached() == "ok"
        if gpu_ok:
            commands.append(("describe", [py, "-m", "superconductor_client.describe.describe_server"]))
        else:
            print(f"{DESCRIBE_NEEDS_GPU} Continuing without it.", flush=True)
    commands += [
        ("midi", [py, "-m", "superconductor_client.midi.midi_player"]),
        ("conducting", [py, "-m", "superconductor_client.conducting.conducting_web.server",
                        "--server-url", place.direct.http,
                        "--page-server-url", place.browser.http]),
        ("tracking", [py, "-m", "superconductor_client.tracking.collab",
                      "--client-url", CLIENT_URL, "--where", place.name, *tracking_args]),
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
