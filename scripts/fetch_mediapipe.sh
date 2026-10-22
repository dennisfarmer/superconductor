#!/bin/bash
# MediaPipe for the conducting page's hand tracker (conducting_web/hands_worker.js),
# served by the conducting process itself rather than a CDN: @mediapipe/tasks-vision
# (the JS bundle and its WASM runtime) from the npm registry, and the hand landmarker
# model that hand_tracking.py:ensure_task_file downloads, each checked against a fixed
# sha256 (make web-deps). The same layout as client-web's scripts/fetch_ort.sh.
#
#   scripts/fetch_mediapipe.sh [web_dir]     (default var/web; writes <web_dir>/mediapipe/ and models/)
set -euo pipefail

VERSION=1.0.1  # the same as the Python mediapipe in sc_env
SHA256=ee318eaa3d42230aa10910d114faf2a488c577c4e4d33c7cb04126924aca505f
MODEL_URL=https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
MODEL_SHA256=fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1

web="${1:-var/web}"
check() {  # check <sha256> <file>
    echo "$1  $2" | shasum -a 256 -c - > /dev/null 2>&1 || echo "$1  $2" | sha256sum -c - > /dev/null 2>&1
}
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

out="$web/mediapipe"
if [ -f "$out/VERSION" ] && [ "$(cat "$out/VERSION")" = "$VERSION" ]; then
    echo "@mediapipe/tasks-vision $VERSION already in $out"
else
    curl -sfL "https://registry.npmjs.org/@mediapipe/tasks-vision/-/tasks-vision-$VERSION.tgz" -o "$tmp/mp.tgz"
    check "$SHA256" "$tmp/mp.tgz" || { echo "@mediapipe/tasks-vision $VERSION: checksum mismatch" >&2; exit 1; }
    tar -xzf "$tmp/mp.tgz" -C "$tmp"
    rm -rf "$out"
    mkdir -p "$out"
    cp "$tmp/package/vision_bundle.mjs" "$out/"
    cp -R "$tmp/package/wasm" "$out/"
    echo "$VERSION" > "$out/VERSION"
    echo "@mediapipe/tasks-vision $VERSION -> $out"
fi

model="$web/models/hand_landmarker.task"
if [ -f "$model" ] && check "$MODEL_SHA256" "$model"; then
    echo "hand_landmarker.task already in $web/models"
else
    mkdir -p "$web/models"
    curl -sfL "$MODEL_URL" -o "$tmp/hand_landmarker.task"
    check "$MODEL_SHA256" "$tmp/hand_landmarker.task" || { echo "hand_landmarker.task: checksum mismatch" >&2; exit 1; }
    mv "$tmp/hand_landmarker.task" "$model"
    echo "hand_landmarker.task -> $model"
fi
