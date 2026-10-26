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
class ClientView:
    """What this computer opens to reach the client: tracking, conducting, and the two pages."""
    tracking: str
    hands: str
    tracking_page: str
    conduct_page: str


@dataclass(frozen=True)
class Place:
    name: str
    description: str
    direct: Side
    browser: Side
    client: ClientView = None

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
    client = _client(name, entry.get("client"), gateway_port)
    return Place(name, str(entry.get("description") or ""), direct, browser, client)


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


def _client(place, data, gateway_port):
    if data is None:
        return None
    if not isinstance(data, dict):
        raise AddressError(f"[{place}.client] must be a table")
    fields = ("tracking", "hands", "tracking_page", "conduct_page")
    missing = [field for field in fields if field not in data]
    if missing:
        raise AddressError(f"[{place}.client] is missing {', '.join(missing)}")
    values = {field: _rewrite_gateway(str(data[field]), gateway_port) for field in fields}
    _check_ws_path(values["tracking"], f"[{place}.client] tracking")
    _check_ws_path(values["hands"], f"[{place}.client] hands")
    _check_page(values["tracking_page"], f"[{place}.client] tracking_page")
    _check_page(values["conduct_page"], f"[{place}.client] conduct_page")
    return ClientView(values["tracking"], values["hands"], values["tracking_page"], values["conduct_page"])


def _check_ws_path(url, label):
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
        return "cluster node", (("local", "on this node (port 9100)"),)
    return "laptop", (
        ("local", "on this computer (port 9100)"),
        ("remote", "on the cluster, through the SSH tunnel (port 9100)"),
    )


def _say(text):
    print(text, file=sys.stderr, flush=True)


def _menu(title, choices, isatty, read_line):
    """Print the menu. One choice is taken without asking. Enter takes the first.

    Returns the choice's id, or None when nobody can answer.
    """
    _say(title)
    for number, (_, text) in enumerate(choices, 1):
        _say(f"  {number}  {text}")
    if len(choices) == 1:
        return choices[0][0]
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
            return choices[int(answer) - 1][0]
        _say(f"Enter a number from 1 to {len(choices)}, or press enter for 1.")


def client_choices(kind):
    """Where the client process is, as seen from this computer."""
    if kind == "cluster node":
        return (("here", "on this node (ports 8467 and 8480)"),)
    return (
        ("here", "on this computer (ports 8467 and 8480)"),
        ("gateway", "in the container, through the gateway (port 8080)"),
    )


def server_choices(kind, client_choice):
    """Where the music server is. The container case has only the node it runs on."""
    if client_choice == "gateway":
        return (("cluster", "on that same cluster node"),)
    if kind == "cluster node":
        return (("local", "on this node (port 9100)"),)
    return (
        ("local", "on this computer (port 9100)"),
        ("remote", "on the cluster, through the SSH tunnel (port 9100)"),
    )


def place_for(client_choice, server_choice):
    """The addresses.toml name for one pair of answers."""
    if client_choice == "gateway":
        return "cluster"
    if server_choice == "remote":
        return "remote"
    return "local"


def choose_locations(hostname, isatty, read_line):
    """Ask where the client is, then where the music server is. A place name, or None."""
    short = hostname.split(".")[0]
    kind, _server_only = describe_machine(hostname)
    _say(f"Running on {short} ({kind}).")
    client = _menu("Where is the client?", client_choices(kind), isatty, read_line)
    if client is None:
        return None
    server = _menu("Where is the music server?", server_choices(kind, client), isatty, read_line)
    if server is None:
        return None
    name = place_for(client, server)
    _say(f"Using {name}.")
    return name


def visualize_plan(place):
    """Shell assignments run_visualizer.sh evals. URLs come from the place, not from here."""
    if place.client is None:
        raise AddressError(f"{place.name} has no client addresses")
    if place.name == "local":
        start_server, client_target, conduct_target = "1", "client-local", "conduct"
        wait, check = "9100 8470 8467 8480", ""
    elif place.name == "remote":
        start_server, client_target, conduct_target = "0", "client-remote", "conduct-remote"
        wait, check = "8470 8467 8480", "9100"
    elif place.name == "cluster":
        start_server, client_target, conduct_target = "0", "", ""
        wait, check = "8080", "8080"
    else:
        raise AddressError(f"{place.name} is not started by make visualize")
    fields = {
        "START_SERVER": start_server,
        "CLIENT_TARGET": client_target,
        "CONDUCT_TARGET": conduct_target,
        "WAIT_PORTS": wait,
        "CHECK_PORT": check,
        "TRACKING_PAGE": place.client.tracking_page,
        "CONDUCT_PAGE": place.client.conduct_page,
        "TRACKING_URL": place.client.tracking,
        "HANDS_URL": place.client.hands,
        "PLAYBACK_URL": place.browser.playback,
        "SERVER_HTTP": place.browser.http,
    }
    return "\n".join(f"{key}={shlex.quote(value)}" for key, value in fields.items())


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
    parser = argparse.ArgumentParser(description="Music-server and client addresses for make visualize.")
    parser.add_argument("--where", default=None, metavar="NAME", help="a name in addresses.toml")
    parser.add_argument("--ask", action="store_true",
                        help="ask where the client is, then where the music server is; print the place name")
    parser.add_argument("--plan", action="store_true",
                        help="print the shell assignments run_visualizer.sh evals")
    parser.add_argument("--shell", action="store_true",
                        help="print SC_INPUT, SC_HTTP, SC_BROWSER_PLAYBACK, ... for a shell to eval")
    parser.add_argument("--field", choices=("input", "playback", "http", "browser_playback", "browser_http"),
                        help="print that one address")
    args = parser.parse_args(argv)
    if args.ask and args.where:
        parser.error("pass --ask or --where, not both")
    try:
        if args.ask:
            import socket
            name = choose_locations(socket.gethostname(), sys.stdin.isatty(), input)
            if name is None:
                return 2
            print(name)
            return 0
        if not args.where:
            parser.error("pass --where NAME, or --ask")
        place = load_place(args.where, gateway_port=os.environ.get("SC_GATEWAY_PORT"))
    except (AddressError, OSError) as exc:
        print(exc, file=sys.stderr)
        return 1
    if args.plan:
        try:
            print(visualize_plan(place))
        except AddressError as exc:
            print(exc, file=sys.stderr)
            return 1
    elif args.shell:
        print(_shell(place))
    elif args.field:
        side = place.browser if args.field.startswith("browser_") else place.direct
        print(getattr(side, args.field.removeprefix("browser_")))
    else:
        print(f"{place.name}: {place.description}")
        if place.client is not None:
            print(f"  client          tracking {place.client.tracking}")
            print(f"  client          hands {place.client.hands}")
        print(f"  music server    playback {place.browser.playback}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
