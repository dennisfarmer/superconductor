"""The music-server addresses in addresses.toml.

Callers use the fields as written. They do not turn an input socket into a
playback socket, or a websocket into an http address.
"""
import argparse
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

FILE = Path(__file__).with_name("addresses.toml")
_FIELDS = ("input", "playback", "http")


class AddressError(ValueError):
    """addresses.toml is missing an address, or an address in it is not usable."""


@dataclass(frozen=True)
class Server:
    input: str
    playback: str
    http: str


def load_server(path=None):
    """The [server] entry."""
    with open(path or FILE, "rb") as f:
        data = tomllib.load(f).get("server")
    if not isinstance(data, dict):
        raise AddressError("[server] is missing")
    missing = [field for field in _FIELDS if field not in data]
    if missing:
        raise AddressError(f"[server] is missing {', '.join(missing)}")
    values = {field: str(data[field]) for field in _FIELDS}
    _check_socket(values["input"], "/stream/input", "[server] input")
    _check_socket(values["playback"], "/stream/playback", "[server] playback")
    _check_http(values["http"], "[server] http")
    return Server(values["input"], values["playback"], values["http"].rstrip("/"))


def _check_socket(url, path, label):
    parsed = urlparse(url)
    if parsed.scheme not in ("ws", "wss") or not parsed.hostname or parsed.path != path or parsed.query or parsed.fragment:
        raise AddressError(f"{label} must be a websocket address ending in {path}, got {url}")


def _check_http(url, label):
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.path not in ("", "/") or parsed.query or parsed.fragment:
        raise AddressError(f"{label} must be an http address with no path, got {url}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Print the music-server addresses in addresses.toml.")
    parser.add_argument("--field", choices=_FIELDS, help="print that one address")
    args = parser.parse_args(argv)
    try:
        server = load_server()
    except (AddressError, OSError) as exc:
        print(exc, file=sys.stderr)
        return 1
    if args.field:
        print(getattr(server, args.field))
    else:
        for field in _FIELDS:
            print(f"{field:9} {getattr(server, field)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
