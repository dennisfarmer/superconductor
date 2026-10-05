# SuperConductor MRT2 server shortcuts (mrt2_server.py). Self-contained, so it
# also works from a checkout of only the server branch (e.g. on Lighthouse).
#
#   make env          build / update the sc_env conda env from environment.yml (Mac, MLX)
#   make models       download the weights for MODEL / BACKEND
#   make server       mrt2_small on this Mac (MLX), listening on port 9100
#   make server BACKEND=jax MODEL=mrt2_base   mrt2_base on a GPU node (Lighthouse)
#   make server PORT=9000 ARGS="--frames_per_block 4"   other mrt2_server.py flags
#
# Uses the sc_env conda env when it exists, without needing `conda activate`;
# otherwise the python on PATH (e.g. an activated venv on Lighthouse).
# environment.yml is a verbatim copy of superconductor_client/environment.yml.

ENV_NAME ?= sc_env
ENV_PREFIX := $(shell conda info --base 2>/dev/null)/envs/$(ENV_NAME)
ifneq ($(wildcard $(ENV_PREFIX)/bin/python),)
# run the env's binaries directly
IN_ENV := PATH="$(ENV_PREFIX)/bin:$$PATH" CONDA_PREFIX="$(ENV_PREFIX)"
endif
BACKEND ?= mlx
MODEL ?= mrt2_small
PORT ?= 9100
ARGS ?=

.PHONY: server env models check-env

server: check-env
	$(IN_ENV) python mrt2_server.py --backend $(BACKEND) --model $(MODEL) --port $(PORT) $(ARGS)

# magenta-rt and recurrentgemma are installed with --no-deps: recurrentgemma pins
# absl-py<2, which conflicts with mediapipe (see environment.yml)
env:
	if conda env list | grep -qE '^$(ENV_NAME)\s'; then \
		conda env update -n $(ENV_NAME) -f environment.yml --prune; \
	else \
		conda env create -n $(ENV_NAME) -f environment.yml; \
	fi
	conda run -n $(ENV_NAME) python -m pip install --no-deps "magenta-rt[mlx]==2.0.3" "recurrentgemma==1.0.1"

# mlx loads exported models (models/), jax loads raw checkpoints (checkpoints/);
# both need the shared resources. Set MAGENTA_HOME to download somewhere other
# than ~/Documents/Magenta (e.g. scratch on Lighthouse).
models: check-env
	$(IN_ENV) mrt models init
ifeq ($(BACKEND),jax)
	$(IN_ENV) mrt checkpoints download $(MODEL)
else
	$(IN_ENV) mrt models download $(MODEL)
endif

check-env:
	@$(IN_ENV) python -c "import importlib.util as u, sys; sys.exit(not (u.find_spec('magenta_rt') and u.find_spec('aiohttp')))" || { \
		echo "python with magenta_rt + aiohttp not found: run 'make env' (Mac) or activate the server venv (see README.md)"; \
		exit 1; }
