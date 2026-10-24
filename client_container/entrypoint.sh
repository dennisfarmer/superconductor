#!/bin/bash
# Starts the client-side services, each as its own process on its usual port
# (127.0.0.1 only), then Caddy, the single port the browser uses (see Caddyfile).
# If any of them exits, the others are stopped and the container exits.
#
# Environment:
#   SC_SERVER_URL     MRT2 server websocket (default: the laptop's tunnel, through Docker's host)
#   SC_SERVICES       which services to run (default "client describe midi atlas conduct")
#   OLLAMA_URL        use this Ollama instead of the bundled one (e.g. http://host.docker.internal:11434)
#   SC_OLLAMA_PORT    port of the bundled Ollama (default 11434)
#   SC_GATEWAY_BIND   gateway address (default 0.0.0.0)
#   SC_GATEWAY_PORT   gateway port (default 8080)
#   DESCRIBE_MODEL    describe server's vision model (default qwen2.5vl:3b)
#
# Everything written goes under /data (a Docker volume, or a bind mount on the cluster).
set -Eeuo pipefail

export SC_SERVER_URL=${SC_SERVER_URL:-ws://host.docker.internal:9000/stream}
SC_SERVICES=${SC_SERVICES:-client describe midi atlas conduct}
SC_OLLAMA_PORT=${SC_OLLAMA_PORT:-11434}
export SC_GATEWAY_BIND=${SC_GATEWAY_BIND:-0.0.0.0}
export SC_GATEWAY_PORT=${SC_GATEWAY_PORT:-8080}
DESCRIBE_MODEL=${DESCRIBE_MODEL:-qwen2.5vl:3b}
SC=/opt/superconductor

if ! mkdir -p /data/client /data/midi/songs /data/ollama /data/ultralytics /data/run /data/atlas /data/web 2>/dev/null; then
    echo "/data is not writable: mount a volume (docker run -v sc-data:/data ...)" \
         "or bind a directory (singularity run --bind <dir>:/data ...)" >&2
    exit 1
fi

has() { [[ " $SC_SERVICES " == *" $1 "* ]]; }

# the MRT2 server's http base, for the gateway's server.localhost (its /tempo and /conduct pages)
SC_SERVER_HTTP=${SC_SERVER_URL/#ws/http}
export SC_SERVER_HTTP=${SC_SERVER_HTTP%/stream}
export OLLAMA_URL=${OLLAMA_URL:-}

pids=()      # every service, stopped together
watched=()   # those whose exit stops the container (not Atlas: its Quit button stops just Atlas)
stop_all() {
    trap - EXIT INT TERM
    # SIGINT first: the client saves the object library on KeyboardInterrupt
    for pid in "${pids[@]}"; do kill -INT "$pid" 2>/dev/null || true; done
    for _ in $(seq 1 50); do
        alive=0
        for pid in "${pids[@]}"; do kill -0 "$pid" 2>/dev/null && alive=1; done
        [ $alive = 0 ] && break
        sleep 0.2
    done
    for pid in "${pids[@]}"; do kill -TERM "$pid" 2>/dev/null || true; done
}
trap stop_all EXIT INT TERM

wait_for() {  # url, name
    for _ in $(seq 1 120); do curl -s -o /dev/null "$1" && return 0; sleep 0.5; done
    echo "$2 did not come up at $1" >&2
    return 1
}

if has describe; then
    if [ -z "$OLLAMA_URL" ]; then
        OLLAMA_URL=http://127.0.0.1:$SC_OLLAMA_PORT
        OLLAMA_HOST=127.0.0.1:$SC_OLLAMA_PORT ollama serve > /data/run/ollama.log 2>&1 &
        pids+=($!); watched+=($!)
        wait_for "$OLLAMA_URL/api/tags" Ollama
        if ! OLLAMA_HOST=127.0.0.1:$SC_OLLAMA_PORT ollama show "$DESCRIBE_MODEL" > /dev/null 2>&1; then
            echo "pulling $DESCRIBE_MODEL into /data/ollama (first run only)"
            OLLAMA_HOST=127.0.0.1:$SC_OLLAMA_PORT ollama pull "$DESCRIBE_MODEL"
        fi
    fi
    python "$SC/client/superconductor_client/describe/describe_server.py" --model "$DESCRIBE_MODEL" --port 9200 --ollama_url "$OLLAMA_URL" &
    pids+=($!); watched+=($!)
fi

if has client; then
    # the container's config, with the gateway's port in its links, on cpu without a GPU
    config=/data/run/collab.toml
    sed "s/localhost:8080/localhost:$SC_GATEWAY_PORT/" "$SC/client/superconductor_client/tracking/collab.container.toml" > "$config"
    # a real CUDA operation: is_available() is also true for a GPU this torch has no kernels for
    if ! python -c "import torch; torch.ones(1, device='cuda').add_(1).item()" > /dev/null 2>&1; then
        echo "no usable GPU (docker run --gpus all / singularity run --nv): recognition on cpu"
        sed -i 's/^device = "cuda"/device = "cpu"/; s/^embed_device = "cuda"/embed_device = "cpu"/' "$config"
    fi
    # the camera page's detector, built into the image; the client exports it again
    # into /data/web if [tracking] classes change
    if [ "$(cat /data/web/ort/VERSION 2>/dev/null)" != "$(cat /opt/web/ort/VERSION)" ]; then
        mkdir -p /data/web/ort && cp -f /opt/web/ort/* /data/web/ort/
    fi
    if [ ! -f /data/web/models/yoloe.onnx ] || ! cmp -s /opt/web/models/yoloe.json /data/web/models/yoloe.json; then
        mkdir -p /data/web/models && cp -f /opt/web/models/* /data/web/models/
    fi
    # the client holds the music-server connection; tracking sends it recipes
    sc-client --server "$SC_SERVER_URL" > /data/run/client.log 2>&1 &
    pids+=($!); watched+=($!)
    # run in /opt/models, where the YOLOE weights are; everything it writes is under /data
    (cd /opt/models && exec sc-tracking --config "$config" --server "$SC_SERVER_URL" \
        --client-url http://127.0.0.1:8470) &
    pids+=($!); watched+=($!)
fi

# MediaPipe runs in the browser. This process turns landmarks into POST /tempo/target
# on the MRT2 server. 127.0.0.1:8480 is the direct port (TouchDesigner on this machine);
# the page the browser opens is the gateway's /conduct/, because the camera needs
# http://localhost.
if has conduct; then
    python "$SC/client/superconductor_client/conducting/conducting_web/server.py" --host 127.0.0.1 --port 8480 \
        --server-url "$SC_SERVER_HTTP" \
        --page-server-url "http://server.localhost:$SC_GATEWAY_PORT" \
        > /data/run/conduct.log 2>&1 &
    pids+=($!); watched+=($!)
fi

if has midi; then
    python "$SC/client/superconductor_client/midi/midi_player.py" --port 8475 --client_url http://localhost:8470 \
        --songs_dir /data/midi/songs &
    pids+=($!); watched+=($!)
fi

# SuperConductor Atlas in SuperConductor mode: the engine is a sink (notes and drums to
# client_midi), the app serves the page; the page reaches the engine through the
# gateway at /engine. Both run in the Atlas folder (walker.toml, the presets).
if has atlas; then
    (cd "$SC/atlas" && exec python audio_engine/engine.py --sink superconductor \
        --client-midi http://localhost:8470 --sc-server "$SC_SERVER_HTTP" --preset warm_piano \
        --port 8765 > /data/run/atlas-engine.log 2>&1) &
    pids+=($!)
    (cd "$SC/atlas" && ATLAS_DATA=/data/atlas ATLAS_ENGINE_PUBLIC_URL=/engine \
        exec python -m server.app --host 127.0.0.1 --port 8764 > /data/run/atlas.log 2>&1) &
    pids+=($!)
fi

# Caddy keeps its state under $XDG_CONFIG_HOME/caddy and $XDG_DATA_HOME/caddy (else ~/.config, ~/.local)
XDG_CONFIG_HOME=/data/run XDG_DATA_HOME=/data/run \
    caddy run --config "$SC/client_container/Caddyfile" --adapter caddyfile > /data/run/caddy.log 2>&1 &
pids+=($!); watched+=($!)

echo
echo "  SuperConductor: open http://localhost:$SC_GATEWAY_PORT in Chrome, Edge or Firefox"
if has conduct; then
    echo "  Conducting (MediaPipe hands): http://localhost:$SC_GATEWAY_PORT/conduct/"
fi
echo "  (MRT2 server: $SC_SERVER_URL)"
echo

wait -n "${watched[@]}"
echo "a service exited; stopping the others" >&2
