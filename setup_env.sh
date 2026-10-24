#!/bin/bash
# Create / update the sc_env conda environment on the laptop.
#
# magenta-rt (Magenta RealTime 2) and recurrentgemma are installed with
# --no-deps because recurrentgemma pins absl-py<2 while mediapipe>=0.10.30
# requires absl-py~=2.3; their real dependencies are listed in environment.yml.
set -Eeuo pipefail
cd "$(dirname "$0")"

if conda env list | grep -qE '^sc_env\s'; then
  conda env update -n sc_env -f environment.yml --prune
else
  conda env create -f environment.yml
fi

conda run -n sc_env python -m pip install --no-deps "magenta-rt[mlx]==2.0.3" "recurrentgemma==1.0.1"
conda run -n sc_env python -m pip install --no-deps -e .

# Music weights are downloaded by `make setup` in the server checkout.
# Parameter-icon weights (SVGen) and the Ollama vision model are downloaded by
# `make setup DESCRIBE=1` in this checkout.
