#!/usr/bin/env bash
# Opens the Standalone app pointed at one music server.
#
#   playback/launch.sh                 ask where the server is
#   playback/launch.sh --where local
#   playback/launch.sh --where remote
#   playback/launch.sh --where cluster
#
# The menu is the same one `make client` and `make audio` use. This app does not
# talk to the client, so it is not asked where the client is. cluster is the
# gateway address (server.localhost:8080), for a player on this computer.
set -euo pipefail

where=""
while [ $# -gt 0 ]; do
    case "$1" in
        --where) where="${2:?--where needs a name}"; shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 1 ;;
    esac
done

here="$(cd "$(dirname "$0")/.." && pwd)"
if [ -z "$where" ]; then
    where="$(python3 "$here/addresses.py" --ask)" || exit $?
fi
url="$(python3 "$here/addresses.py" --where "$where" --field browser_playback)"
http="$(python3 "$here/addresses.py" --where "$where" --field browser_http)"
app="$here/playback/build/plugin/SCPlayback_artefacts/Release/Standalone/SuperConductor Playback VST3 Plugin.app"
bin="$here/playback/build/plugin/SCPlayback_artefacts/Release/Standalone/SuperConductor Playback VST3 Plugin"

echo "stream $url" >&2
echo "tempo  $http" >&2

if [ "$(uname)" = "Darwin" ]; then
    open --env "SC_PLAYBACK_URL=$url" --env "SC_SERVER_HTTP=$http" "$app"
else
    SC_PLAYBACK_URL="$url" SC_SERVER_HTTP="$http" "$bin"
fi
