"""The music-server addresses in addresses.toml.

A start command picks a name. `load_place` returns that entry. Callers use the
fields as written. They do not turn an input socket into a playback socket, or
a websocket into an http address.
"""
import argparse
import os
import re
import shlex
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

FILE = Path(__file__).with_name("addresses.toml")
_FIELDS = ("input", "playback", "http")


class AddressError(ValueError):
    """addresses.toml is missing a name, or an address in it is not usable."""


@dataclass(frozen=True)
class Side:
    """One audience: programs on the client machine, or the laptop browser."""
    input: str
    playback: str
    http: str


@dataclass(frozen=True)
class Place:
    name: str
    description: str
    direct: Side
    browser: Side

    def service_endpoints(self):
        """What the prompts page lists for the music server.

        Pages and tempo requests use the browser's http address. The two sockets
        are the browser's playback address and the client process's input address.
        """
        base = self.browser.http
        return [
            ("page", f"{base}/tempo"),
            ("page", f"{base}/conduct"),
            ("GET", f"{base}/tempo/status"),
            ("POST", f"{base}/tempo/beat"),
            ("POST", f"{base}/tempo/target"),
            ("POST", f"{base}/tempo/free"),
            ("WS playback", self.browser.playback),
            ("WS input", self.direct.input),
        ]


def names(path=None):
    """The place names in the file, in file order."""
    with open(path or FILE, "rb") as f:
        return [key for key, value in tomllib.load(f).items() if isinstance(value, dict) and "direct" in value]


def load_place(name, path=None, gateway_port=None):
    """The place `name`. `gateway_port` rewrites a browser address's :8080.

    The container's gateway is 8080 in the file. SC_GATEWAY_PORT can move it.
    That rewrite is only the port, and only on the browser side.
    """
    file = path or FILE
    with open(file, "rb") as f:
        table = tomllib.load(f)
    known = [key for key, value in table.items() if isinstance(value, dict) and "direct" in value]
    entry = table.get(name)
    if not isinstance(entry, dict) or "direct" not in entry:
        raise AddressError(f"unknown place {name!r}. addresses.toml has: {', '.join(known)}")
    direct = _side(name, "direct", entry.get("direct"))
    browser = _side(name, "browser", _browser_fields(direct, entry.get("browser")), gateway_port)
    return Place(name, str(entry.get("description") or ""), direct, browser)


def _browser_fields(direct, raw):
    """Keys the browser block leaves out are the direct addresses."""
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise AddressError("browser must be a table")
    return {field: raw.get(field, getattr(direct, field)) for field in _FIELDS}


def _side(place, which, data, gateway_port=None):
    if not isinstance(data, dict):
        raise AddressError(f"[{place}.{which}] is missing")
    missing = [field for field in _FIELDS if field not in data]
    if missing:
        raise AddressError(f"[{place}.{which}] is missing {', '.join(missing)}")
    values = {field: _rewrite_gateway(str(data[field]), gateway_port) if which == "browser" else str(data[field])
              for field in _FIELDS}
    _check_socket(values["input"], "/stream/input", f"[{place}.{which}] input")
    _check_socket(values["playback"], "/stream/playback", f"[{place}.{which}] playback")
    _check_http(values["http"], f"[{place}.{which}] http")
    return Side(values["input"], values["playback"], values["http"].rstrip("/"))


def _rewrite_gateway(url, port):
    if port is None or port == "" or str(port) == "8080":
        return url
    if not str(port).isdigit():
        raise AddressError(f"gateway port must be a number, got {port!r}")
    return url.replace(":8080", f":{port}", 1)


def _check_socket(url, path, label):
    parsed = urlparse(url)
    if parsed.scheme not in ("ws", "wss") or not parsed.hostname or parsed.path != path or parsed.query or parsed.fragment:
        raise AddressError(f"{label} must be a websocket address ending in {path}, got {url}")


def _check_http(url, label):
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.path not in ("", "/") or parsed.query or parsed.fragment:
        raise AddressError(f"{label} must be an http address with no path, got {url}")


def describe_machine(hostname):
    """(kind, choices) for where this client process is running.

    A Lighthouse compute node is named lh and digits (lh2300). Anything else is
    the laptop. Each choice is (place name, sentence shown in the prompt).

    Not offered: the server on the laptop while this client is on a compute
    node, a client on a third machine, and the container (make client-singularity
    already picks cluster).
    """
    short = hostname.split(".")[0]
    if re.fullmatch(r"lh\d+", short):
        return "cluster node", (("local", "on this node (port 9000)"),)
    return "laptop", (
        ("local", "on this computer (port 9000)"),
        ("remote", "on the cluster, through the SSH tunnel (port 9000)"),
    )


def choose_place(hostname, isatty, read_line):
    """Ask where the music server is. None when there is no answer.

    The menu is printed either way. One choice is taken without asking. Enter
    takes the first choice (on a laptop, the server on this computer).
    `read_line` is input() in normal use. `--where NAME` skips this.
    """
    short = hostname.split(".")[0]
    kind, choices = describe_machine(hostname)
    print(f"Running on {short} ({kind}).", flush=True)
    print("Where is the music server?", flush=True)
    for number, (_, text) in enumerate(choices, 1):
        print(f"  {number}  {text}", flush=True)
    if len(choices) == 1:
        name = choices[0][0]
        print(f"Using {name}.", flush=True)
        return name
    if not isatty:
        print("Pass --where NAME.", file=sys.stderr)
        return None
    while True:
        try:
            answer = read_line("Choice [1]: ").strip()
        except EOFError:
            print(file=sys.stderr)
            return None
        if answer == "":
            answer = "1"
        if answer.isdigit() and 1 <= int(answer) <= len(choices):
            name = choices[int(answer) - 1][0]
            print(f"Using {name}.", flush=True)
            return name
        print(f"Enter a number from 1 to {len(choices)}, or press enter for 1.", flush=True)


def _shell(place):
    fields = {
        "SC_WHERE": place.name,
        "SC_INPUT": place.direct.input,
        "SC_PLAYBACK": place.direct.playback,
        "SC_HTTP": place.direct.http,
        "SC_BROWSER_PLAYBACK": place.browser.playback,
        "SC_BROWSER_HTTP": place.browser.http,
    }
    return "\n".join(f"{key}={shlex.quote(value)}" for key, value in fields.items())


def main(argv=None):
    parser = argparse.ArgumentParser(description="Print one place from addresses.toml.")
    parser.add_argument("--where", required=True, help="a name in addresses.toml")
    parser.add_argument("--shell", action="store_true",
                        help="print SC_INPUT, SC_HTTP, SC_BROWSER_PLAYBACK, ... for a shell to eval")
    parser.add_argument("--field", choices=("input", "playback", "http", "browser_playback", "browser_http"),
                        help="print that one address")
    args = parser.parse_args(argv)
    if args.shell and args.field:
        parser.error("pass --shell or --field, not both")
    try:
        place = load_place(args.where, gateway_port=os.environ.get("SC_GATEWAY_PORT"))
    except (AddressError, OSError) as exc:
        print(exc, file=sys.stderr)
        return 1
    if args.shell:
        print(_shell(place))
    elif args.field:
        side = place.browser if args.field.startswith("browser_") else place.direct
        print(getattr(side, args.field.removeprefix("browser_")))
    else:
        print(f"{place.name}: {place.description}")
        print(f"  client process  input {place.direct.input}")
        print(f"  client process  playback {place.direct.playback}")
        print(f"  client process  http {place.direct.http}")
        if place.browser != place.direct:
            print(f"  browser         playback {place.browser.playback}")
            print(f"  browser         http {place.browser.http}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
