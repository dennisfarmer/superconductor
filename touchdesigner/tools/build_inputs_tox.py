"""Rebuilds superconductor_inputs.tox from build_inputs.py, without the Textport (make touchdesigner-build).

Runs build_inputs.py inside TouchDesigner through the MCP component every visualizer
contains (port 9981), then saves /inputs to touchdesigner/superconductor_inputs.tox.
If TouchDesigner is already open with a visualizer, that one is used and left open.
Otherwise this opens the first visualizer, builds, and quits TouchDesigner without
saving the .toe (it loads /inputs from the .tox anyway).

    python3 touchdesigner/tools/build_inputs_tox.py
"""
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent          # .../touchdesigner
TOX = HERE / "superconductor_inputs.tox"
BUILD = HERE / "build_inputs.py"
MCP = "http://localhost:9981"
WAIT_S = 180


def _up():
    try:
        urllib.request.urlopen(f"{MCP}/api/nodes/errors", timeout=2)
        return True
    except urllib.error.HTTPError:
        return True
    except OSError:
        return False


def _exec(script):
    body = json.dumps({"script": script}).encode()
    request = urllib.request.Request(f"{MCP}/api/td/server/exec", body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=60) as response:
        reply = json.load(response)
    if not reply.get("success"):
        raise SystemExit(f"TouchDesigner: {reply.get('error')}")
    return reply["data"]


def main():
    if sys.platform != "darwin":
        raise SystemExit("make touchdesigner-build opens TouchDesigner with macOS `open`; on Windows, "
                         "run build_inputs.py from the Textport (TOUCHDESIGNER_SETUP.md)")
    opened = False
    if not _up():
        toes = sorted(HERE.glob("visualizers/*/*.toe"))
        toes = [t for t in toes if t.stem == t.parent.name]
        if not toes:
            raise SystemExit(f"no visualizer in {HERE / 'visualizers'}")
        print(f"opening TouchDesigner ({toes[0].name})")
        try:
            subprocess.run(["open", "-n", "-a", "TouchDesigner", str(toes[0])], check=True)
        except subprocess.CalledProcessError:
            raise SystemExit("could not open TouchDesigner. Is it installed (derivative.ca)?") from None
        opened = True
        deadline = time.time() + WAIT_S
        while not _up():
            if time.time() > deadline:
                raise SystemExit(f"TouchDesigner's MCP component did not answer on {MCP} after {WAIT_S}s. "
                                 "Is /mcp_webserver_base Active in the visualizer?")
            time.sleep(2)

    data = _exec(f"exec(open({str(BUILD)!r}).read(), globals())\n"
                 f"op('/inputs').save({str(TOX)!r})\n"
                 "result = op('/inputs').path")
    print((data.get("stdout") or "").strip())
    print(f"saved {TOX}")

    if opened:
        _exec("run('project.quit(force=True)', delayFrames=10)")
        print("closed TouchDesigner")
    return 0


if __name__ == "__main__":
    sys.exit(main())
