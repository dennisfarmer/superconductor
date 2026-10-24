#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
build_dir="$project_dir/.build"
revision=8cb6977d0c1a1b411ac320493b3c7f5182ed2d22
mkdir -p "$build_dir"
if [ ! -f "$build_dir/tools/cmake/__init__.py" ]; then
    python3 -m pip install --target "$build_dir/tools" cmake==3.31.6
fi
if [ ! -d "$build_dir/bungee/.git" ]; then
    git clone https://github.com/bungee-audio-stretch/bungee.git "$build_dir/bungee"
fi
git -C "$build_dir/bungee" checkout --detach "$revision"
git -C "$build_dir/bungee" submodule update --init --recursive
export PYTHONPATH="$build_dir/tools${PYTHONPATH:+:$PYTHONPATH}"
python3 -m cmake -S "$project_dir/native" -B "$build_dir/native" \
    -DBUNGEE_SOURCE="$build_dir/bungee" -DCMAKE_BUILD_TYPE=Release
python3 -m cmake --build "$build_dir/native" --target conductor_bungee --parallel 2
