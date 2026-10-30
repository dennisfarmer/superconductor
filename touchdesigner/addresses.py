"""The music-server and client addresses in addresses.toml.

Callers use the fields as written. They do not turn an input socket into a
playback socket, or a websocket into an http address.
"""
import argparse
import shlex
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

FILE = Path(__file__).with_name("addresses.toml")
_SERVER = ("input", "playback", "http")
_CLIENT = ("tracking", "hands", "tracking_page", "conduct_page")


class AddressError(ValueError):
    """addresses.toml is missing an address, or an address in it is not usable."""


@dataclass(frozen=True)
class Server:
    input: str
    playback: str
    http: str


@dataclass(frozen=True)
class Client:
    """What this computer opens to reach the client: tracking, conducting, and the two pages."""
    tracking: str
    hands: str
    tracking_page: str
    conduct_page: str


def load(path=None):
    """(Server, Client) from the file."""
    with open(path or FILE, "rb") as f:
        table = tomllib.load(f)
    server = _table(table, "server", _SERVER)
    _check_socket(server["input"], "/stream/input", "[server] input")
    _check_socket(server["playback"], "/stream/playback", "[server] playback")
    _check_http(server["http"], "[server] http")
    client = _table(table, "client", _CLIENT)
    _check_client_socket(client["tracking"], "[client] tracking")
    _check_client_socket(client["hands"], "[client] hands")
    _check_page(client["tracking_page"], "[client] tracking_page")
    _check_page(client["conduct_page"], "[client] conduct_page")
    server["http"] = server["http"].rstrip("/")
    return Server(**server), Client(**client)


def _table(table, name, fields):
    data = table.get(name)
    if not isinstance(data, dict):
        raise AddressError(f"[{name}] is missing")
    missing = [field for field in fields if field not in data]
    if missing:
        raise AddressError(f"[{name}] is missing {', '.join(missing)}")
    return {field: str(data[field]) for field in fields}


def _check_socket(url, path, label):
    parsed = urlparse(url)
    if parsed.scheme not in ("ws", "wss") or not parsed.hostname or parsed.path != path or parsed.query or parsed.fragment:
        raise AddressError(f"{label} must be a websocket address ending in {path}, got {url}")


def _check_client_socket(url, label):
    parsed = urlparse(url)
    if parsed.scheme not in ("ws", "wss") or not parsed.hostname or not parsed.port or not parsed.path:
        raise AddressError(f"{label} must be a websocket address with a port and a path, got {url}")


def _check_page(url, label):
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname or not parsed.port:
        raise AddressError(f"{label} must be an http address, got {url}")


def _check_http(url, label):
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.path not in ("", "/") or parsed.query or parsed.fragment:
        raise AddressError(f"{label} must be an http address with no path, got {url}")


def _port(url):
    parsed = urlparse(url)
    return str(parsed.port or (443 if parsed.scheme in ("https", "wss") else 80))


def visualize_plan(server, client):
    """Shell assignments run_visualizer.sh evals."""
    ports = []
    for url in (server.playback, client.tracking, client.hands):
        if _port(url) not in ports:
            ports.append(_port(url))
    fields = {
        "CHECK_PORTS": " ".join(ports),
        "TRACKING_PAGE": client.tracking_page,
        "CONDUCT_PAGE": client.conduct_page,
        "TRACKING_URL": client.tracking,
        "HANDS_URL": client.hands,
        "PLAYBACK_URL": server.playback,
        "SERVER_HTTP": server.http,
    }
    return "\n".join(f"{key}={shlex.quote(value)}" for key, value in fields.items())


def main(argv=None):
    parser = argparse.ArgumentParser(description="Music-server and client addresses for make visualize.")
    parser.add_argument("--plan", action="store_true",
                        help="print the shell assignments run_visualizer.sh evals")
    args = parser.parse_args(argv)
    try:
        server, client = load()
    except (AddressError, OSError) as exc:
        print(exc, file=sys.stderr)
        return 1
    if args.plan:
        print(visualize_plan(server, client))
    else:
        print(f"client          tracking {client.tracking}")
        print(f"client          hands {client.hands}")
        print(f"music server    playback {server.playback}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
