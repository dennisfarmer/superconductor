#!/usr/bin/env bash
# Runs the SuperConductor stack and one visualizer, in a window showing only its output (macOS):
#
#   make visualize                    ask where the client is, then where the music server is
#   make visualize-local              client and music server on this computer
#   make visualize-remote             client on this computer, server through the tunnel (port 9000)
#   make visualize-cluster            both on the cluster; this computer uses the gateway (:8080)
#   touchdesigner/run_visualizer.sh [--where NAME] [visualizer]
#
# 1. starts whatever of that stack is supposed to run here, unless it's already up;
# 2. starts the SuperConductor Playback VST3 Plugin app (playback branch), which plays the music on this Mac's
#    current sound output (speakers / headphones) and sets the pace (the clock);
# 3. opens the Tracking page (as a monitor, so it stays quiet: ?audio=monitor) and the
#    Conducting page in the browser;
# 4. opens the visualizer in a new TouchDesigner, straight into perform mode: the editor is
#    hidden and only the visualizer's output window shows (resize it, or set the window
#    component's Size to Fill for full screen);
# 5. when you quit TouchDesigner (Cmd+Q; Esc first shows the editor), stops the services
#    and the app it started. Anything that was already running is left alone.
#
# Logs: var/logs/<server|client|conduct>.log. CLIENT_DIR and PLAYBACK_APP override
# where superconductor_client and the SuperConductor Playback VST3 Plugin app are (default: next to this
# worktree; the app is built by `make playback-build` in superconductor_playback).
set -euo pipefail

where=""
name="overlay_example_1"
while [ $# -gt 0 ]; do
    case "$1" in
        --where) where="${2:?--where needs a name}"; shift 2 ;;
        *) name="$1"; shift ;;
    esac
done
here="$(cd "$(dirname "$0")" && pwd)"                       # .../touchdesigner
if [ -z "$where" ]; then
    where="$(python3 "$here/addresses.py" --ask)" || exit $?
fi
# shellcheck disable=SC1090
eval "$(python3 "$here/addresses.py" --where "$where" --plan)"
toe="$here/visualizers/$name/$name.toe"
root="$(cd "$here/../.." && pwd)"
client_dir="${CLIENT_DIR:-${CLIENT_WEB_DIR:-$root/superconductor_client}}"
app="${PLAYBACK_APP:-$(cd "$here/../.." && pwd)/superconductor_playback/playback/build/plugin/SCPlayback_artefacts/Release/Standalone/SuperConductor Playback VST3 Plugin.app}"
app_id=edu.umich.superconductor.playback
logs="$(cd "$here/.." && pwd)/var/logs"

[ -f "$toe" ] || { echo "no visualizer at $toe"; exit 1; }
[ -d "$client_dir" ] || { echo "superconductor_client not found at $client_dir (set CLIENT_DIR)"; exit 1; }
[ -d "$app" ] || { echo "SuperConductor Playback VST3 Plugin app not found at $app: run make playback-build in superconductor_playback (or set PLAYBACK_APP)"; exit 1; }
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
    (cd "$client_dir" && make "$target") > "$logs/$target.log" 2>&1 &
    started_ports+=("$port" "$@")
}

if [ -n "$CHECK_PORT" ] && ! listening "$CHECK_PORT"; then
    if [ "$where" = "remote" ]; then
        echo "nothing is listening on port $CHECK_PORT. Open the tunnel first: ssh -N -L 9000:lh2300:9100"
    else
        echo "nothing is listening on port $CHECK_PORT. Start the container and the tunnel: ssh -N -L 8080:lh2300:8080"
    fi
    exit 1
fi
if [ "$where" != "local" ]; then
    echo "set the plugin stream to $PLAYBACK_URL (tempo address $SERVER_HTTP)"
fi
[ "$START_SERVER" = 1 ] && start server 9100
[ -n "$CLIENT_TARGET" ] && start "$CLIENT_TARGET" 8470 8467
[ -n "$CONDUCT_TARGET" ] && start "$CONDUCT_TARGET" 8480

echo "waiting for the services..."
for port in $WAIT_PORTS; do
    for _ in $(seq 300); do
        listening "$port" && break
        sleep 1
    done
    listening "$port" || { echo "nothing listening on :$port after 5 minutes; see var/logs"; exit 1; }
done

if pgrep -f "SuperConductor Playback VST3 Plugin.app/Contents/MacOS" >/dev/null; then
    echo "SuperConductor Playback VST3 Plugin app: already running"
else
    echo "SuperConductor Playback VST3 Plugin app: starting (the music plays on this Mac's sound output)"
    open -n "$app"
    started_app=1
fi

open "$TRACKING_PAGE"    # Tracking page: object tracking (press Start); quiet
open "$CONDUCT_PAGE"     # Conducting page: MediaPipe hands

echo "opening $name (output window only). Press Start on the Tracking page for the object tracking."
echo "Esc shows the editor, F1 the output window again; quit TouchDesigner (Cmd+Q) to stop everything."
open -n -W -a TouchDesigner --env SC_PERFORM=1 --env "SC_WHERE=$where" \
    --env "SC_TD_DIR=$here" --env "SC_TRACKING_URL=$TRACKING_URL" --env "SC_HANDS_URL=$HANDS_URL" "$toe"
