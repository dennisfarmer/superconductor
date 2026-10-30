#!/usr/bin/env bash
# Runs one visualizer against the SuperConductor stack, in a window showing only its output (macOS):
#
#   make visualize                    overlay_example_1
#   make visualize VISUALIZER=<name>  visualizers/<name>/<name>.toe
#   touchdesigner/run_visualizer.sh [visualizer]
#
# The music server and the client run on the cluster; the SSH tunnel (superconductor_client
# README) puts them on localhost:9000, :9100 and :9102. Running them on this computer uses the
# same ports. The addresses are addresses.toml.
#
# 1. checks that the music server and the client are listening;
# 2. starts the SuperConductor Playback VST3 Plugin app (playback branch), which plays the music on this Mac's
#    current sound output (speakers / headphones) and sets the pace (the clock);
# 3. opens the Tracking page (as a monitor, so it stays quiet: ?audio=monitor) and the
#    Conducting page in the browser;
# 4. opens the visualizer in a new TouchDesigner, straight into perform mode: the editor is
#    hidden and only the visualizer's output window shows (resize it, or set the window
#    component's Size to Fill for full screen);
# 5. when you quit TouchDesigner (Cmd+Q; Esc first shows the editor), quits the app if it started it.
#
# PLAYBACK_APP overrides where the SuperConductor Playback VST3 Plugin app is (default: next to this
# worktree; the app is built by `make playback-build` in superconductor_playback).
set -euo pipefail

name="${1:-overlay_example_1}"
here="$(cd "$(dirname "$0")" && pwd)"                       # .../touchdesigner
# shellcheck disable=SC1090
eval "$(python3 "$here/addresses.py" --plan)"
toe="$here/visualizers/$name/$name.toe"
app="${PLAYBACK_APP:-$(cd "$here/../.." && pwd)/superconductor_playback/playback/build/plugin/SCPlayback_artefacts/Release/Standalone/SuperConductor Playback VST3 Plugin.app}"
app_id=edu.umich.superconductor.playback

[ -f "$toe" ] || { echo "no visualizer at $toe"; exit 1; }
[ -d "$app" ] || { echo "SuperConductor Playback VST3 Plugin app not found at $app: run make playback-build in superconductor_playback (or set PLAYBACK_APP)"; exit 1; }

listening() { lsof -nP -iTCP:"$1" -sTCP:LISTEN >/dev/null 2>&1; }

for port in $CHECK_PORTS; do
    if ! listening "$port"; then
        echo "nothing is listening on port $port. Start make server and make client on the cluster, and open the SSH tunnel (superconductor_client README)."
        exit 1
    fi
done

started_app=0
stop_started() {
    if [ "$started_app" = 1 ]; then
        osascript -e "tell application id \"$app_id\" to quit" >/dev/null 2>&1 || true
        started_app=0
    fi
}
trap stop_started EXIT INT TERM

if pgrep -f "SuperConductor Playback VST3 Plugin.app/Contents/MacOS" >/dev/null; then
    echo "SuperConductor Playback VST3 Plugin app: already running"
else
    echo "SuperConductor Playback VST3 Plugin app: starting (the music plays on this Mac's sound output)"
    open -n --env "SC_PLAYBACK_URL=$PLAYBACK_URL" --env "SC_SERVER_HTTP=$SERVER_HTTP" "$app"
    started_app=1
fi

open "$TRACKING_PAGE"    # Tracking page: object tracking (press Start); quiet
open "$CONDUCT_PAGE"     # Conducting page: MediaPipe hands

echo "opening $name (output window only). Press Start on the Tracking page for the object tracking."
echo "Esc shows the editor, F1 the output window again; quit TouchDesigner (Cmd+Q) to stop the app."
open -n -W -a TouchDesigner --env SC_PERFORM=1 \
    --env "SC_TD_DIR=$here" --env "SC_TRACKING_URL=$TRACKING_URL" --env "SC_HANDS_URL=$HANDS_URL" "$toe"
