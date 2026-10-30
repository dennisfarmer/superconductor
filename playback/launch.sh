#!/usr/bin/env bash
# Opens the Standalone app pointed at the music server in addresses.toml
# (localhost:9000: the server on this computer, or the cluster through the SSH tunnel).
set -euo pipefail

here="$(cd "$(dirname "$0")/.." && pwd)"
url="$(python3 "$here/addresses.py" --field playback)"
http="$(python3 "$here/addresses.py" --field http)"
app="$here/playback/build/plugin/SCPlayback_artefacts/Release/Standalone/SuperConductor Playback VST3 Plugin.app"
bin="$here/playback/build/plugin/SCPlayback_artefacts/Release/Standalone/SuperConductor Playback VST3 Plugin"

echo "stream $url" >&2
echo "tempo  $http" >&2

if [ "$(uname)" = "Darwin" ]; then
    open --env "SC_PLAYBACK_URL=$url" --env "SC_SERVER_HTTP=$http" "$app"
else
    SC_PLAYBACK_URL="$url" SC_SERVER_HTTP="$http" "$bin"
fi
