# syntax=docker/dockerfile:1.7
#
# SuperConductor client-side services in one image, used entirely from a browser
# (see client_container/README.md): the client process, tracking, the describe server
# with Ollama, the MIDI player, conducting, SuperConductor Atlas, and Caddy as the
# single-port gateway. The MRT2 server is not in here; the client and conducting
# connect to it by the place named in SC_WHERE (addresses.toml).
#
# tracking, conducting, describe and midi are inside superconductor_client/. Atlas still comes from its
# branch, as a named build context:
#   docker buildx build --platform linux/amd64 \
#     --build-context atlas=../superconductor_atlas \
#     -f client_container/Dockerfile .
# (`make image` does this; CI checks the atlas branch out instead.)

FROM caddy:2.10.2 AS caddy
FROM ollama/ollama:0.12.6 AS ollama

# Atlas's page (React + TypeScript), built into client/dist
FROM node:22.12.0-bookworm-slim AS atlas-client
WORKDIR /client
COPY --from=atlas client/package.json client/package-lock.json ./
RUN npm ci
COPY --from=atlas client/index.html client/tsconfig.json client/vite.config.ts ./
COPY --from=atlas client/src ./src
RUN npm run build

FROM python:3.12-slim-bookworm

# libgl1 / libglib2.0-0: opencv; procps / lsof: the Services tab looks for running
# programs with ps / lsof (collab._listening_ports)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 ca-certificates curl procps lsof \
    && rm -rf /var/lib/apt/lists/*

COPY --from=caddy /usr/bin/caddy /usr/bin/caddy
COPY --from=ollama /usr/bin/ollama /usr/bin/ollama
COPY --from=ollama /usr/lib/ollama /usr/lib/ollama

ENV VIRTUAL_ENV=/opt/sc-venv \
    PATH=/opt/sc-venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    TORCH_HOME=/opt/models/torch \
    YOLO_CONFIG_DIR=/data/ultralytics \
    OLLAMA_MODELS=/data/ollama

RUN python -m venv /opt/sc-venv
COPY client_container/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
COPY client_container/requirements-atlas.txt /tmp/requirements-atlas.txt
RUN pip install --no-cache-dir -r /tmp/requirements-atlas.txt

# superconductor_client/: client.py, plus tracking/, conducting/, describe/ and midi/.
COPY pyproject.toml /opt/superconductor/client/
COPY superconductor_client /opt/superconductor/client/superconductor_client
RUN pip install --no-cache-dir --no-deps -e /opt/superconductor/client
# MediaPipe for the conducting page. server.py reads conducting/var/web; the .sif is read-only.
RUN bash /opt/superconductor/client/superconductor_client/conducting/scripts/fetch_mediapipe.sh \
        /opt/superconductor/client/superconductor_client/conducting/var/web
# SuperConductor Atlas: app server, sink engine (and its presets), walker settings, page
COPY --from=atlas server /opt/superconductor/atlas/server
COPY --from=atlas audio_engine/engine.py /opt/superconductor/atlas/audio_engine/
COPY --from=atlas audio_engine/presets /opt/superconductor/atlas/audio_engine/presets
COPY --from=atlas walker.toml /opt/superconductor/atlas/
COPY --from=atlas-client /client/dist /opt/superconductor/atlas/client/dist

# The models, so nothing downloads at startup (the .sif is read-only, and a
# compute node may have no internet): YOLOE-11s-seg and its text encoder land in
# /opt/models (the client runs there), DINOv2 in TORCH_HOME, the describe VLM in
# /opt/models/ollama, and SVGen (parameter icons) in /opt/models/svgen.
# Ultralytics writes its settings to YOLO_CONFIG_DIR, which is under /data at runtime.
WORKDIR /opt/models
RUN YOLO_CONFIG_DIR=/tmp/ultralytics python -c "\
import tomllib; \
from superconductor_client.tracking.object_tracking import ObjectTracker; \
t = tomllib.load(open('/opt/superconductor/client/superconductor_client/tracking/collab.container.toml', 'rb'))['tracking']; \
ObjectTracker(model=t['model'], classes=t['classes'], device='cpu', \
              embedder=t['embedder'], embed_imgsz=t['embed_imgsz'], embed_device='cpu').load_detector()" \
    && rm -rf /tmp/ultralytics

# The camera page's detector: onnxruntime-web and YOLOE exported to ONNX with the
# container config's classes, in /opt/web (the entrypoint copies it to /data/web, where
# the client exports it again if the classes change)
RUN bash /opt/superconductor/client/superconductor_client/tracking/fetch_ort.sh /opt/web \
    && YOLO_CONFIG_DIR=/tmp/ultralytics python -c "\
import sys, tomllib; \
from superconductor_client.tracking.web_model import ensure_web_model; \
t = tomllib.load(open('/opt/superconductor/client/superconductor_client/tracking/collab.container.toml', 'rb'))['tracking']; \
sys.exit(not ensure_web_model(t['model'], t['classes'], '/opt/web'))" \
    && rm -rf /tmp/ultralytics

# Describe: the vision model (Ollama) and SVGen's Qwen2.5-3B (transformers).
# The entrypoint uses these when /data has not already got its own copy.
RUN mkdir -p /opt/models/ollama /opt/models/svgen \
    && OLLAMA_MODELS=/opt/models/ollama bash -c '\
        set -euo pipefail; \
        ollama serve >/tmp/ollama-build.log 2>&1 & pid=$!; \
        for i in $(seq 1 60); do ollama list >/dev/null 2>&1 && break; sleep 1; done; \
        ollama pull qwen2.5vl:3b; \
        kill $pid || true; wait $pid || true' \
    && python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('gitcat-404/SVGen-Qwen2.5-3B-Instruct', local_dir='/opt/models/svgen')"

COPY client_container/entrypoint.sh client_container/Caddyfile /opt/superconductor/client_container/
RUN chmod +x /opt/superconductor/client_container/entrypoint.sh

# at runtime nothing is written outside /data (the .sif is read-only); the .pyc files
# pip compiled at install time are still used
ENV PYTHONDONTWRITEBYTECODE=1

VOLUME /data
EXPOSE 8080
ENTRYPOINT ["/opt/superconductor/client_container/entrypoint.sh"]
