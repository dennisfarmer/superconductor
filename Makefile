# SuperConductor MRT2 server shortcuts (mrt2_server.py). Self-contained, so it
# also works from a checkout of only the server branch (e.g. on Lighthouse).
#
#   make setup        install the env and download any missing weights.
#                     macOS: sc_env and mrt2_small. lh2300: .venv and both JAX checkpoints.
#                     This always runs the installer. `make server` skips it when
#                     .cache/setup-stamp still matches.
#   make env          build / update the sc_env conda env from environment_macbook.yml (the laptop, MLX)
#   make server       listen on port 9100: mrt2_small (MLX) on the laptop,
#                     mrt2_base (JAX) on Linux, e.g. a Lighthouse GPU node.
#                     Runs `make setup` when the environment is not already there.
#   make server BACKEND=mlx MODEL=mrt2_base   override the defaults
#   make server PORT=9000 ARGS="--frames_per_block 4"   other mrt2_server.py flags
#   make image        build the linux/amd64 dev image; make image-push sends it to :server-dev
#   make sif          on lh2300: pull the release image again (:server)
#   make sif-dev      on lh2300: pull the dev image again (:server-dev)
#   make server-singularity       run the release image, pulling it if the .sif is absent
#   make server-singularity-dev   run the dev image, pulling it if the .sif is absent
#                     Checkpoints come from MAGENTA_HOME on scratch. These do not run `make setup`.
#
# Uses the sc_env conda env when it exists, without needing `conda activate`.
# On lh2300, where that env is not installed, it uses .venv from `make setup`.
# environment_macbook.yml is a verbatim copy of superconductor_client/environment.yml.

ENV_NAME ?= sc_env
ENV_PREFIX := $(shell conda info --base 2>/dev/null)/envs/$(ENV_NAME)
VENV_PY := $(CURDIR)/.venv/bin/python
ifneq ($(wildcard $(ENV_PREFIX)/bin/python),)
# run the env's binaries directly
IN_ENV := PATH="$(ENV_PREFIX)/bin:$$PATH" CONDA_PREFIX="$(ENV_PREFIX)"
else ifneq ($(wildcard $(VENV_PY)),)
IN_ENV := PATH="$(CURDIR)/.venv/bin:$$PATH"
endif
# defaults: mrt2_base on JAX on Linux (Lighthouse), mrt2_small on MLX otherwise (the laptop)
ifeq ($(shell uname -s),Linux)
BACKEND ?= jax
MODEL ?= mrt2_base
else
BACKEND ?= mlx
MODEL ?= mrt2_small
endif
# magenta_rt looks for weights in $MAGENTA_HOME (default ~/Documents/Magenta).
# On Lighthouse (jax) they live on shared scratch; exported so both `make setup`
# and `make server` use it. Override with MAGENTA_HOME=... if needed.
ifeq ($(BACKEND),jax)
MAGENTA_HOME ?= /scratch/aimusic_project_root/aimusic_project/shared_data
export MAGENTA_HOME
endif
PORT ?= 9100
ARGS ?=
# Release tags move when someone runs Actions → release on the client branch.
# Dev tags move on `make image-push` from a laptop.
IMAGE_RELEASE ?= ghcr.io/dennisfarmer/superconductor_server:server
IMAGE_DEV ?= ghcr.io/dennisfarmer/superconductor_server:server-dev
SIF ?= superconductor_server.sif
SIF_DEV ?= superconductor_server-dev.sif
# `singularity pull` cache. The default is ~/.singularity, on the home quota.
SINGULARITY_CACHEDIR ?= /scratch/aimusic_project_root/aimusic_project/shared_data/.singularity_cache

.PHONY: server env setup setup-stamp ensure-setup setup-weights check-env image image-push sif sif-dev server-singularity server-singularity-dev

# Where `mrt` writes. JAX uses $MAGENTA_HOME; MLX uses the default under $HOME.
WEIGHTS_ROOT := $(if $(MAGENTA_HOME),$(MAGENTA_HOME),$(HOME)/Documents/Magenta)/magenta-rt-v2
# macOS runs mrt2_small. The base MLX weights are large and usually unused there.
ifeq ($(shell uname -s),Darwin)
MLX_MODELS := mrt2_small
else
MLX_MODELS := mrt2_base mrt2_small
endif
ifeq ($(shell uname -s),Linux)
SETUP_MRT := $(CURDIR)/.venv/bin/mrt
else
SETUP_MRT := $(ENV_PREFIX)/bin/mrt
endif

server: ensure-setup
	@if [ -x "$(ENV_PREFIX)/bin/python" ]; then \
	  PATH="$(ENV_PREFIX)/bin:$$PATH" CONDA_PREFIX="$(ENV_PREFIX)" python mrt2_server.py --backend $(BACKEND) --model $(MODEL) --port $(PORT) $(ARGS); \
	elif [ -x "$(VENV_PY)" ]; then \
	  PATH="$(CURDIR)/.venv/bin:$$PATH" python mrt2_server.py --backend $(BACKEND) --model $(MODEL) --port $(PORT) $(ARGS); \
	else \
	  echo "python with magenta_rt + aiohttp not found: run 'make setup'"; exit 1; \
	fi

# magenta-rt and recurrentgemma are installed with --no-deps: recurrentgemma pins
# absl-py<2, which conflicts with mediapipe (see environment_macbook.yml)
env:
	if conda env list | grep -qE '^$(ENV_NAME)\s'; then \
		conda env update -n $(ENV_NAME) -f environment_macbook.yml --prune; \
	else \
		conda env create -n $(ENV_NAME) -f environment_macbook.yml; \
	fi
	conda run -n $(ENV_NAME) python -m pip install --no-deps "magenta-rt[mlx]==2.0.3" "recurrentgemma==1.0.1"

# Laptop: sc_env, if it is not there yet. lh2300: .venv and requirements_lighthouse.txt.
# Then the shared resources and the models this machine runs. macOS skips mrt2_base.
# Files that are already on disk are left alone. `make setup` always runs this.
# `make server` calls ensure-setup, which stops after a few file tests when the stamp matches.
SETUP_STAMP := $(CURDIR)/.cache/setup-stamp
HASH_CMD := $(shell command -v sha256sum >/dev/null 2>&1 && echo sha256sum || echo shasum -a 256)
ifeq ($(shell uname -s),Linux)
SETUP_HASH_FILES := requirements_lighthouse.txt
else
SETUP_HASH_FILES := environment_macbook.yml
endif

setup:
ifeq ($(shell uname -s),Linux)
	@test -x .venv/bin/python || python3.12 -m venv .venv
	.venv/bin/python -m pip install -r requirements_lighthouse.txt
else
	@if [ ! -x "$(SETUP_MRT)" ]; then $(MAKE) env; else echo "sc_env already exists"; fi
endif
	@$(MAKE) setup-weights
	@$(MAKE) setup-stamp

setup-stamp:
	@mkdir -p $(dir $(SETUP_STAMP))
	@cat $(SETUP_HASH_FILES) | $(HASH_CMD) | awk '{print $$1}' > $(SETUP_STAMP)

ensure-setup:
	@root="$(WEIGHTS_ROOT)"; ok=1; \
	[ -x "$(SETUP_MRT)" ] || ok=0; \
	[ -f "$$root/resources/musiccoca/text_encoder.tflite" ] || ok=0; \
	[ -f "$$root/resources/spectrostream/spectrostream_encoder.mlxfn" ] || ok=0; \
	if [ "$(BACKEND)" = jax ]; then \
	  [ -f "$$root/checkpoints/mrt2_base.safetensors" ] || ok=0; \
	  [ -f "$$root/checkpoints/mrt2_small.safetensors" ] || ok=0; \
	else \
	  for name in $(MLX_MODELS); do \
	    [ -f "$$root/models/$$name/$$name.mlxfn" ] || ok=0; \
	  done; \
	fi; \
	hash=$$(cat $(SETUP_HASH_FILES) | $(HASH_CMD) | awk '{print $$1}'); \
	stamp="$(SETUP_STAMP)"; \
	if [ "$$ok" = 1 ] && { [ ! -f "$$stamp" ] || [ "$$(cat "$$stamp")" = "$$hash" ]; }; then \
	  if [ ! -f "$$stamp" ]; then mkdir -p "$$(dirname "$$stamp")" && printf '%s\n' "$$hash" > "$$stamp"; fi; \
	  echo "setup already done"; \
	else \
	  $(MAKE) setup; \
	fi

setup-weights:
	@root="$(WEIGHTS_ROOT)"; mrt="$(SETUP_MRT)"; \
	if [ -f "$$root/resources/musiccoca/text_encoder.tflite" ] \
	   && [ -f "$$root/resources/spectrostream/spectrostream_encoder.mlxfn" ]; then \
		echo "shared model resources already present"; \
	else \
		"$$mrt" models init; \
	fi; \
	if [ "$(BACKEND)" = jax ]; then \
		for name in mrt2_base mrt2_small; do \
			if [ -f "$$root/checkpoints/$$name.safetensors" ]; then \
				echo "$$name checkpoint already present"; \
			else \
				"$$mrt" checkpoints download "$$name"; \
			fi; \
		done; \
	else \
		for name in $(MLX_MODELS); do \
			if [ -f "$$root/models/$$name/$$name.mlxfn" ]; then \
				echo "$$name already present"; \
			else \
				"$$mrt" models download "$$name"; \
			fi; \
		done; \
	fi

IMAGE_BUILD := docker buildx build --platform linux/amd64 --provenance=false --sbom=false \
	-f server_container/Dockerfile -t $(IMAGE_DEV) .

image:
	$(IMAGE_BUILD) --load

image-push:
	$(IMAGE_BUILD) --push

sif:
	mkdir -p $(SINGULARITY_CACHEDIR)
	SINGULARITY_CACHEDIR=$(SINGULARITY_CACHEDIR) singularity pull --force $(SIF) docker://$(IMAGE_RELEASE)

sif-dev:
	mkdir -p $(SINGULARITY_CACHEDIR)
	SINGULARITY_CACHEDIR=$(SINGULARITY_CACHEDIR) singularity pull --force $(SIF_DEV) docker://$(IMAGE_DEV)

# The file rule pulls only when the .sif is absent. `make sif` still pulls --force.
$(SIF):
	$(MAKE) sif

$(SIF_DEV):
	$(MAKE) sif-dev

server-singularity: $(SIF)
	singularity run --nv --bind $(MAGENTA_HOME):$(MAGENTA_HOME) \
		--env MAGENTA_HOME=$(MAGENTA_HOME) --env MODEL=$(MODEL) --env PORT=$(PORT) \
		$(SIF)

server-singularity-dev: $(SIF_DEV)
	singularity run --nv --bind $(MAGENTA_HOME):$(MAGENTA_HOME) \
		--env MAGENTA_HOME=$(MAGENTA_HOME) --env MODEL=$(MODEL) --env PORT=$(PORT) \
		$(SIF_DEV)

check-env:
	@$(IN_ENV) python -c "import importlib.util as u, sys; sys.exit(not (u.find_spec('magenta_rt') and u.find_spec('aiohttp')))" || { \
		echo "python with magenta_rt + aiohttp not found: run 'make setup'"; \
		exit 1; }
