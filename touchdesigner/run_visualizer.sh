#!/usr/bin/env bash
# Runs the SuperConductor stack and one visualizer, in a window showing only its output (macOS):
#
#   make visualize                              # overlay_example_1
#   make visualize VISUALIZER=<name>            # visualizers/<name>/<name>.toe
#
# 1. starts the MRT2 server, the client and the conducting server (make server / client /
#    conduct in superconductor_client_web), unless they're already running;
# 2. starts the SC Playback app (playback branch), which plays the music on this Mac's
#    current sound output (speakers / headphones) and sets the pace (the clock);
# 3. opens the Tracking page (as a monitor, so it stays quiet: ?audio=monitor) and the
#    Conducting page in the browser;
# 4. opens the visualizer in a new TouchDesigner, straight into perform mode: the editor is
#    hidden and only the visualizer's output window shows (resize it, or set the window
#    component's Size to Fill for full screen);
# 5. when you quit TouchDesigner (Cmd+Q; Esc first shows the editor), stops the services
#    and the app it started. Anything that was already running is left alone.
#
# Logs: var/logs/<server|client|conduct>.log. CLIENT_WEB_DIR and PLAYBACK_APP override
# where superconductor_client_web and the SC Playback app are (default: next to this
# worktree; the app is built by `make playback-build` in superconductor_playback).
set -euo pipefail

name="${1:-overlay_example_1}"
here="$(cd "$(dirname "$0")" && pwd)"                       # .../touchdesigner
toe="$here/visualizers/$name/$name.toe"
client_web="${CLIENT_WEB_DIR:-$(cd "$here/../.." && pwd)/superconductor_client_web}"
app="${PLAYBACK_APP:-$(cd "$here/../.." && pwd)/superconductor_playback/playback/build/plugin/SCPlayback_artefacts/Release/Standalone/SC Playback.app}"
app_id=edu.umich.superconductor.playback
logs="$(cd "$here/.." && pwd)/var/logs"

[ -f "$toe" ] || { echo "no visualizer at $toe"; exit 1; }
[ -d "$client_web" ] || { echo "superconductor_client_web not found at $client_web (set CLIENT_WEB_DIR)"; exit 1; }
[ -d "$app" ] || { echo "SC Playback app not found at $app: run make playback-build in superconductor_playback (or set PLAYBACK_APP)"; exit 1; }
mkdir -p "$logs"

listening() { lsof -nP -iTCP:"$1" -sTCP:LISTEN >/dev/null 2>&1; }

started_ports=()
started_app=0
stop_started() {
    if [ "$started_app" = 1 ]; then
        osascript -e "tell application id \"$app_id\" to quit" >/dev/null 2>&1 || true
        started_app=0
    fi
    for port in "${started_ports[@]+"${started_ports[@]}"}"; do
        lsof -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null | xargs kill 2>/dev/null || true
    done
    [ ${#started_ports[@]} -eq 0 ] || echo "stopped the services this started"
    started_ports=()
}
trap stop_started EXIT INT TERM

# make target, port to wait for, other ports it opens (stopped with it)
start() {
    local target=$1 port=$2; shift 2
    if listening "$port"; then
        echo "$target: already running (:$port)"
        return
    fi
    echo "$target: starting (log: var/logs/$target.log)"
    (cd "$client_web" && make "$target") > "$logs/$target.log" 2>&1 &
    started_ports+=("$port" "$@")
}

start server 9100
start client 8470 8467
start conduct 8480

echo "waiting for the services..."
for port in 9100 8470 8467 8480; do
    for _ in $(seq 300); do
        listening "$port" && break
        sleep 1
    done
    listening "$port" || { echo "nothing listening on :$port after 5 minutes; see var/logs"; exit 1; }
done

if pgrep -f "SC Playback.app/Contents/MacOS" >/dev/null; then
    echo "SC Playback app: already running"
else
    echo "SC Playback app: starting (the music plays on this Mac's sound output)"
    open -n "$app"
    started_app=1
fi

open "http://localhost:8467/?audio=monitor"    # Tracking page: object tracking (press Start); quiet
open "http://localhost:8480"                   # Conducting page: MediaPipe hands

echo "opening $name (output window only). Press Start on the Tracking page for the object tracking."
echo "Esc shows the editor, F1 the output window again; quit TouchDesigner (Cmd+Q) to stop everything."
open -n -W -a TouchDesigner --env SC_PERFORM=1 "$toe"
