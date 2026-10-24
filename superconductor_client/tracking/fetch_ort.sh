#!/bin/bash
# onnxruntime-web for the camera page's detector (vision_js), served by the client
# itself rather than a CDN: the WebGPU build and its WASM runtime, from the npm
# registry, checked against a fixed sha256 (make web-deps).
#
#   superconductor_client/tracking/fetch_ort.sh [web_dir]
#   default: var/web next to this script
set -euo pipefail

VERSION=1.30.0
SHA256=d2228df7e4616bc3348bf504ee888f3bec43789a273f0a63f3e68d203ce3bf71
FILES="ort.webgpu.min.mjs ort-wasm-simd-threaded.asyncify.mjs ort-wasm-simd-threaded.asyncify.wasm"

here="$(cd "$(dirname "$0")" && pwd)"
out="${1:-$here/var/web}/ort"
if [ -f "$out/VERSION" ] && [ "$(cat "$out/VERSION")" = "$VERSION" ]; then
    echo "onnxruntime-web $VERSION already in $out"
    exit 0
fi
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
curl -sfL "https://registry.npmjs.org/onnxruntime-web/-/onnxruntime-web-$VERSION.tgz" -o "$tmp/ort.tgz"
echo "$SHA256  $tmp/ort.tgz" | shasum -a 256 -c - > /dev/null 2>&1 \
    || echo "$SHA256  $tmp/ort.tgz" | sha256sum -c - > /dev/null \
    || { echo "onnxruntime-web $VERSION: checksum mismatch" >&2; exit 1; }
tar -xzf "$tmp/ort.tgz" -C "$tmp"
mkdir -p "$out"
for f in $FILES; do cp "$tmp/package/dist/$f" "$out/"; done
echo "$VERSION" > "$out/VERSION"
echo "onnxruntime-web $VERSION -> $out"
