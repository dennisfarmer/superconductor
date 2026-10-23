"""Check the SuperConductor Playback VST3 Plugin's OSC without TouchDesigner: listens for a while, then
prints how often each address arrived and its last value (playback/docs/osc.md).

    python touchdesigner/tools/osc_listen.py [seconds=10] [port=7000]

Needs python-osc (pip install python-osc; it's in sc_env)."""
import sys, time, collections
from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import ThreadingOSCUDPServer
import threading
counts, last, onsets = collections.Counter(), {}, 0
def handler(addr, *args):
    global onsets
    counts[addr] += 1; last[addr] = args
    if addr == "/sc/onset" and args[0] > 0: onsets += 1
d = Dispatcher(); d.set_default_handler(handler)
port = int(sys.argv[2]) if len(sys.argv) > 2 else 7000
srv = ThreadingOSCUDPServer(("127.0.0.1", port), d)
threading.Thread(target=srv.serve_forever, daemon=True).start()
secs = float(sys.argv[1]) if len(sys.argv) > 1 else 10
time.sleep(secs); srv.shutdown()
for a in sorted(counts):
    v = ", ".join(f"{x:.3f}" if isinstance(x, float) else str(x) for x in last[a])
    print(f"{a:26s} {counts[a]/secs:6.1f}/s  last: {v}")
print(f"onsets: {onsets} in {secs:.0f}s")
