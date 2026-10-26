#!/bin/bash
# Starts the client-side services, each as its own process on its usual port
# (127.0.0.1 only), then Caddy, the single port the browser uses (see Caddyfile).
# If any of them exits, the others are stopped and the container exits.
#
# Environment:
#   SC_WHERE          which music server, a name in addresses.toml
#                     (default cluster; make client-docker passes docker)
#   SC_SERVICES       which services to run (default "client midi atlas conduct";
#                     add describe to also start Ollama and the describe server)
#   OLLAMA_URL        use this Ollama instead of the bundled one (e.g. http://host.docker.internal:11434)
#   SC_OLLAMA_PORT    port of the bundled Ollama (default 11434)
#   SC_GATEWAY_BIND   gateway address (default 0.0.0.0)
#   SC_GATEWAY_PORT   gateway port (default 8080)
#   DESCRIBE_MODEL    describe server's vision model (default qwen2.5vl:3b)
#   OLLAMA_MODELS     where Ollama keeps weights (default: the copy in the image,
#                     or /data/ollama once that directory has its own models)
#
# Everything written goes under /data (a Docker volume, or a bind mount on the cluster).
set -Eeuo pipefail

if [ -n "${SC_SERVER_URL:-}" ]; then
    echo "SC_SERVER_URL is no longer used. Set SC_WHERE to a name in addresses.toml" \
         "(local, remote, cluster, docker, docker-local)." >&2
    exit 1
fi
SC_WHERE=${SC_WHERE:-cluster}
SC_SERVICES=${SC_SERVICES:-client midi atlas conduct}
SC_OLLAMA_PORT=${SC_OLLAMA_PORT:-11434}
export SC_GATEWAY_BIND=${SC_GATEWAY_BIND:-0.0.0.0}
export SC_GATEWAY_PORT=${SC_GATEWAY_PORT:-8080}
DESCRIBE_MODEL=${DESCRIBE_MODEL:-qwen2.5vl:3b}
SC=/opt/superconductor

if ! mkdir -p /data/client /data/midi/songs /data/ollama /data/huggingface /data/ultralytics /data/run /data/atlas /data/web 2>/dev/null; then
    echo "/data is not writable: mount a volume (docker run -v sc-data:/data ...)" \
         "or bind a directory (singularity run --bind <dir>:/data ...)" >&2
    exit 1
fi

has() { [[ " $SC_SERVICES " == *" $1 "* ]]; }

# addresses.toml, the place SC_WHERE names. SC_HTTP is what programs in this
# container open. SC_BROWSER_HTTP is what the laptop browser opens. The gateway
# forwards server.localhost to SC_HTTP.
eval "$(python -m superconductor_client.addresses --where "$SC_WHERE" --shell)"
export SC_SERVER_HTTP=$SC_HTTP
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

# Describe needs a CUDA device. Exclusive_Process counts as none available when
# another process already holds the card. Skip it and keep the rest of the client.
describe_ok=0
if has describe; then
    if [ -n "${SC_DESCRIBE_GPU:-}" ]; then
        if CUDA_VISIBLE_DEVICES="$SC_DESCRIBE_GPU" \
                python -c "import torch; torch.ones(1, device='cuda').add_(1).item()" \
                >/dev/null 2>&1; then
            describe_ok=1
        fi
    elif python -c "import torch; torch.ones(1, device='cuda').add_(1).item()" \
            >/dev/null 2>&1; then
        describe_ok=1
    fi
    if [ "$describe_ok" = 0 ]; then
        echo "The describe server needs a GPU, and none is available. Continuing without it." >&2
    fi
fi
if [ "$describe_ok" = 1 ]; then
    if [ -z "$OLLAMA_URL" ]; then
        # The image already has qwen2.5vl:3b. Use that copy when /data/ollama
        # has no models of its own. A different DESCRIBE_MODEL is pulled into
        # /data/ollama, which is writable.
        name=${DESCRIBE_MODEL%%:*}
        tag=${DESCRIBE_MODEL#*:}
        baked="/opt/models/ollama/manifests/registry.ollama.ai/library/${name}/${tag}"
        if [ "${OLLAMA_MODELS:-/data/ollama}" = "/data/ollama" ] \
                && [ ! -d /data/ollama/manifests ] && [ -f "$baked" ]; then
            export OLLAMA_MODELS=/opt/models/ollama
        fi
        OLLAMA_URL=http://127.0.0.1:$SC_OLLAMA_PORT
        # SC_DESCRIBE_GPU=1 puts Ollama on the second GPU. Exclusive_Process is per
        # card, so this process can own that L40S while the music server owns the first.
        # An empty CUDA_VISIBLE_DEVICES would hide every GPU, so set it only when asked.
        if [ -n "${SC_DESCRIBE_GPU:-}" ]; then export CUDA_VISIBLE_DEVICES=$SC_DESCRIBE_GPU; fi
        OLLAMA_HOST=127.0.0.1:$SC_OLLAMA_PORT ollama serve > /data/run/ollama.log 2>&1 &
        unset CUDA_VISIBLE_DEVICES
        pids+=($!)
        wait_for "$OLLAMA_URL/api/tags" Ollama
        if ! OLLAMA_HOST=127.0.0.1:$SC_OLLAMA_PORT ollama show "$DESCRIBE_MODEL" > /dev/null 2>&1; then
            echo "pulling $DESCRIBE_MODEL into $OLLAMA_MODELS"
            OLLAMA_HOST=127.0.0.1:$SC_OLLAMA_PORT ollama pull "$DESCRIBE_MODEL"
        fi
    fi
    # SVGen is baked at /opt/models/svgen. keep_alive 0 drops the VLM off the
    # L40S once a description finishes, so the icon model can use that GPU.
    svg_args=()
    if [ -d /opt/models/svgen ]; then
        svg_args=(--svg-model /opt/models/svgen --svg-backend transformers)
    fi
    # Same card as Ollama. Unset SC_DESCRIBE_GPU leaves both on the default GPU.
    if [ -n "${SC_DESCRIBE_GPU:-}" ]; then export CUDA_VISIBLE_DEVICES=$SC_DESCRIBE_GPU; fi
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME=/data/huggingface \
        python "$SC/client/superconductor_client/describe/describe_server.py" \
        --model "$DESCRIBE_MODEL" --port 9106 --ollama_url "$OLLAMA_URL" \
        --keep_alive 0 "${svg_args[@]}" &
    unset CUDA_VISIBLE_DEVICES
    pids+=($!)
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
    sc-client --server "$SC_INPUT" > /data/run/client.log 2>&1 &
    pids+=($!); watched+=($!)
    # run in /opt/models, where the YOLOE weights are; everything it writes is under /data
    (cd /opt/models && exec sc-tracking --config "$config" --where "$SC_WHERE" \
        --client-url http://127.0.0.1:9105) &
    pids+=($!); watched+=($!)
fi

# MediaPipe runs in the browser. This process turns landmarks into POST /tempo/target
# on the MRT2 server. 127.0.0.1:9102 is the direct port (TouchDesigner on this machine);
# the page the browser opens is the gateway's /conduct/, because the camera needs
# http://localhost.
if has conduct; then
    python "$SC/client/superconductor_client/conducting/conducting_web/server.py" --host 0.0.0.0 --port 9102 \
        --server-url "$SC_HTTP" \
        --page-server-url "$SC_BROWSER_HTTP" \
        --playback-url "$SC_BROWSER_PLAYBACK" \
        --objects-url "http://localhost:$SC_GATEWAY_PORT/prompts" \
        > /data/run/conduct.log 2>&1 &
    pids+=($!); watched+=($!)
fi

if has midi; then
    python "$SC/client/superconductor_client/midi/midi_player.py" --port 9101 --client_url http://localhost:9105 \
        --songs_dir /data/midi/songs &
    pids+=($!); watched+=($!)
fi

# SuperConductor Atlas in SuperConductor mode: the engine is a sink (notes and drums to
# client_midi), the app serves the page; the page reaches the engine through the
# gateway at /engine. Both run in the Atlas folder (walker.toml, the presets).
if has atlas; then
    (cd "$SC/atlas" && exec python audio_engine/engine.py --sink superconductor \
        --client-midi http://localhost:9105 --sc-server "$SC_HTTP" --preset warm_piano \
        --port 9104 > /data/run/atlas-engine.log 2>&1) &
    pids+=($!)
    (cd "$SC/atlas" && ATLAS_DATA=/data/atlas ATLAS_ENGINE_PUBLIC_URL=/engine \
        exec python -m server.app --host 0.0.0.0 --port 9103 > /data/run/atlas.log 2>&1) &
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
echo "  (music server: $SC_WHERE, programs open $SC_HTTP, the browser opens $SC_BROWSER_HTTP)"
echo

wait -n "${watched[@]}"
echo "a service exited; stopping the others" >&2
