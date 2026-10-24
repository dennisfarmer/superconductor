# SuperConductor shortcuts. `make client` runs `make setup` when the environment
# is missing or its inputs changed. That check is a few file tests when the
# stamp in .cache/setup-stamp still matches. macOS uses the sc_env conda env.
# lh2300 uses .venv in this checkout, so the client runs there without the
# Singularity image.
#
#   make setup        install it: macOS sc_env, lh2300 .venv from
#                     client_container/requirements.txt. This always runs the
#                     installer. `make client` skips it when the stamp matches.
#                     DESCRIBE=1 also downloads the icon weights and the Ollama model.
#   make server       MRT2 server on the laptop (mrt2_small, ../superconductor_server/Makefile); keep it running, then:
#   make client       the client, plus tracking, conducting, and midi.
#                     Describe is separate: make describe, or make client DESCRIBE=1.
#                     Asks where the music server is, unless you pass
#                     --where NAME (make client ARGS="--where local").
#                     Open the camera page http://localhost:8467/ in Chrome
#   make client-local    same, without the question: server on this computer (port 9100)
#   make client-remote   same, without the question: server through the SSH tunnel (port 9000)
#   make client ARGS="--mode calibration"   extra tracking options (--detection container: no browser detection)
#   make audio        sc-audio. Asks where the music server is, same question as make client.
#   make audio-local  same, without the question: --where local
#   make audio-remote same, without the question: --where remote
#   make web-deps     onnxruntime-web for the page's detector (var/web/ort)
#   make web-model    export the page's detector (YOLOE, [tracking] classes) to var/web/models
#   make server MODEL=mrt2_base   other server options (MODEL, PORT, ARGS, ...) are
#                     passed on to ../superconductor_server/Makefile
#   make describe     describe server (superconductor_client/describe/). Ollama has to
#                     already be running. POST /svg draws parameter icons (SVGen)
#   make -C superconductor_client/describe pull-svg   download that icon model into the HF cache
#   make midi         MIDI player (superconductor_client/midi/): plays .mid files through the running client's
#                     notes input, page on http://localhost:8475
#   make atlas        SuperConductor Atlas (../superconductor_atlas, its make start): engine as the
#                     running client's note / drum source, page on http://localhost:8764; stop it
#                     with make -C ../superconductor_atlas stop
#   make worktrees    server, atlas, playback and touchdesigner into the sibling folders.
#                     describe, conducting and midi live in this checkout.
#   make conduct      conducting page (superconductor_client/conducting/): browser
#                     hand tracking sets the MRT2 server's tempo, page on http://localhost:8480;
#                     make conduct-remote: to the server through the Lighthouse tunnel (:9000)
#   make playback     SuperConductor Playback VST3 Plugin app (../superconductor_playback/Makefile;
#                     builds it if needed): plays the local server
#                     (ws://localhost:9100/stream/playback) on this machine's speakers.
#                     Through the SSH tunnel, set the plugin URL to
#                     ws://localhost:9000/stream/playback
#   make touchdesigner   a TouchDesigner visualizer (../superconductor_touchdesigner/Makefile),
#                     showing only its output window: VISUALIZER=<name> (default
#                     overlay_example_1), PERFORM=0 to open it in the editor instead
#   make vision       tracking only, no music
#   make test         unit tests: tracking / library / combos, the web client, tracker_js
#
# Container (camera, view and sound in the browser; see client_container/README.md):
#   make image        build the linux/amd64 dev image from this checkout and the atlas
#                     worktree; make image-push sends it to :client-dev
#   make sif          on lh2300: pull the release image again (:client)
#   make sif-dev      on lh2300: pull the dev image again (:client-dev)
#   make client-singularity       run the release .sif, pulling it if the file is absent.
#                     DESCRIBE=1 also starts Ollama and the describe server.
#   make client-singularity-dev   run the dev .sif, pulling it if the file is absent
#                     (SC_DATA=... for /data). These do not run `make setup`.

ENV_NAME ?= sc_env
ENV_PREFIX := $(shell conda info --base 2>/dev/null)/envs/$(ENV_NAME)
# macOS: sc_env. lh2300: .venv from `make setup` (client_container/requirements.txt).
# PYTHONPATH is this checkout, so a pull is what the next `make client` runs.
ifeq ($(shell uname -s),Darwin)
IN_ENV := PATH="$(ENV_PREFIX)/bin:$$PATH" CONDA_PREFIX="$(ENV_PREFIX)" PYTHONPATH="$(CURDIR)"
else
IN_ENV := PATH="$(CURDIR)/.venv/bin:$$PATH" PYTHONPATH="$(CURDIR)"
endif
ARGS ?=
# set DESCRIBE=1 to install and start the describe server with the client
DESCRIBE ?=
DESCRIBE_MODEL ?= qwen2.5vl:3b
SC_SERVICES ?= client midi atlas conduct
ifneq ($(DESCRIBE),)
SC_SERVICES := $(SC_SERVICES) describe
endif
SERVER_DIR := ../superconductor_server
DESCRIBE_DIR := superconductor_client/describe
MIDI_DIR := superconductor_client/midi
ATLAS_DIR := ../superconductor_atlas
CONDUCTING_DIR := ../superconductor_conducting
CONDUCT_DIR := superconductor_client/conducting
PLAYBACK_DIR := ../superconductor_playback
TOUCHDESIGNER_DIR := ../superconductor_touchdesigner
# Release tags move when someone runs Actions → release. Dev tags move on
# `make image-push` from a laptop. `make sif` / `make sif-dev` pull those.
IMAGE_RELEASE ?= ghcr.io/dennisfarmer/superconductor_client:client
IMAGE_DEV ?= ghcr.io/dennisfarmer/superconductor_client:client-dev
SIF ?= superconductor_client.sif
SIF_DEV ?= superconductor_client-dev.sif
# Which entry in superconductor_client/addresses.toml. Empty means cluster
# for client-singularity.
SC_WHERE ?=
OLLAMA_URL ?=
# Docker Desktop on a Mac has no GPU to give the container. Used by client-docker, below.
# GPUS ?= $(if $(filter Darwin,$(shell uname -s)),,--gpus all)
SC_DATA ?= /scratch/aimusic_project_root/aimusic_project/shared_data/sc_data_$(USER)
# `singularity pull` cache. The default is ~/.singularity, on the home quota.
SINGULARITY_CACHEDIR ?= /scratch/aimusic_project_root/aimusic_project/shared_data/.singularity_cache

.PHONY: setup setup-stamp ensure-setup worktrees server describe midi atlas conduct conduct-remote playback touchdesigner client client-local client-remote vision test check-env audio audio-local audio-remote web-deps web-model \
	image image-push sif sif-dev client-singularity client-singularity-dev

# macOS: sc_env from environment.yml, created if it is missing.
# lh2300: .venv and the CUDA 12.6 client requirements (the V100). No Singularity.
# `make setup DESCRIBE=1` also downloads the SVGen icon weights and pulls the
# Ollama vision model. `make client` calls ensure-setup, which stops after a
# few file tests when .cache/setup-stamp still matches.
.PHONY: setup setup-describe setup-stamp ensure-setup
MLX_PINS := mlx==0.31.2 mlx-metal==0.31.2 mlx-lm==0.31.2 transformers==5.12.1 \
	huggingface-hub==1.16.1 tokenizers==0.22.2 safetensors==0.8.0 \
	tqdm==4.70.1 PyYAML==6.0.3 typer==0.27.3
SVGEN_REPO := gitcat-404/SVGen-Qwen2.5-3B-Instruct
SETUP_STAMP := $(CURDIR)/.cache/setup-stamp
HASH_CMD := $(shell command -v sha256sum >/dev/null 2>&1 && echo sha256sum || echo shasum -a 256)
ifeq ($(shell uname -s),Darwin)
SETUP_PY := $(ENV_PREFIX)/bin/python
SETUP_HF := $(if $(HF_HOME),$(HF_HOME),$(HOME)/.cache/huggingface)
SETUP_HASH_FILES := environment.yml pyproject.toml
SETUP_HASH_EXTRA := $(MLX_PINS)
else
SETUP_PY := $(CURDIR)/.venv/bin/python
SETUP_HF := $(if $(HF_HOME),$(HF_HOME),$(CURDIR)/.cache/huggingface)
SETUP_HASH_FILES := client_container/requirements.txt pyproject.toml
SETUP_HASH_EXTRA :=
endif
SVGEN_DIR := $(SETUP_HF)/hub/models--$(subst /,--,$(SVGEN_REPO))

setup:
ifeq ($(shell uname -s),Darwin)
	@if [ -x "$(ENV_PREFIX)/bin/python" ]; then echo "sc_env already exists"; else ./setup_env.sh; fi
	@"$(ENV_PREFIX)/bin/python" -m pip install --no-deps -e .
else
	@test -x .venv/bin/python || python3.12 -m venv .venv
	.venv/bin/python -m pip install -r client_container/requirements.txt
	.venv/bin/python -m pip install --no-deps -e .
endif
ifneq ($(DESCRIBE),)
	@$(MAKE) setup-describe
endif
	@$(MAKE) setup-stamp

setup-describe:
ifeq ($(shell uname -s),Darwin)
	@"$(ENV_PREFIX)/bin/python" -m pip install --upgrade-strategy only-if-needed $(MLX_PINS)
	@"$(ENV_PREFIX)/bin/python" -c "from huggingface_hub import snapshot_download; snapshot_download('$(SVGEN_REPO)')"
else
	HF_HOME="$(SETUP_HF)" .venv/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download('$(SVGEN_REPO)')"
endif
	@command -v ollama >/dev/null || { echo "ollama is not installed"; exit 1; }
	ollama pull $(DESCRIBE_MODEL)

setup-stamp:
	@mkdir -p $(dir $(SETUP_STAMP))
	@{ printf '%s\n' '$(SETUP_HASH_EXTRA)'; cat $(SETUP_HASH_FILES); } | $(HASH_CMD) | awk '{print $$1}' > $(SETUP_STAMP)

# Fast when the interpreter and the stamp are already in place.
# With DESCRIBE=1, the SVGen weights have to be there too.
ensure-setup:
	@hash=$$({ printf '%s\n' '$(SETUP_HASH_EXTRA)'; cat $(SETUP_HASH_FILES); } | $(HASH_CMD) | awk '{print $$1}'); \
	stamp="$(SETUP_STAMP)"; \
	describe_ok=1; \
	if [ -n "$(DESCRIBE)" ] && [ ! -d "$(SVGEN_DIR)" ]; then describe_ok=0; fi; \
	if [ -x "$(SETUP_PY)" ] && [ "$$describe_ok" = 1 ] \
	   && { [ ! -f "$$stamp" ] || [ "$$(cat "$$stamp")" = "$$hash" ]; }; then \
	  if [ ! -f "$$stamp" ]; then mkdir -p "$$(dirname "$$stamp")" && printf '%s\n' "$$hash" > "$$stamp"; fi; \
	  echo "setup already done"; \
	else \
	  $(MAKE) setup; \
	fi

worktrees:
	@git fetch origin
	@test -e $(SERVER_DIR) || git worktree add $(SERVER_DIR) server
	@test -e $(ATLAS_DIR) || git worktree add $(ATLAS_DIR) atlas
	@test -e $(CONDUCTING_DIR) || git worktree add $(CONDUCTING_DIR) tempo-live-conductor
	@test -e $(PLAYBACK_DIR) || git worktree add $(PLAYBACK_DIR) playback
	@test -e $(TOUCHDESIGNER_DIR) || git worktree add $(TOUCHDESIGNER_DIR) touchdesigner
	@git config push.default upstream
	@git worktree list

# command-line variables (MODEL=..., ARGS=...) reach the sub-make through MAKEFLAGS
server:
	$(MAKE) -C $(SERVER_DIR) server ENV_NAME=$(ENV_NAME)

describe:
	$(MAKE) -C $(DESCRIBE_DIR) describe ENV_NAME=$(ENV_NAME)

midi:
	$(MAKE) -C $(MIDI_DIR) player ENV_NAME=$(ENV_NAME)

atlas:
	$(MAKE) -C $(ATLAS_DIR) start

conduct:
	$(MAKE) -C $(CONDUCT_DIR) conduct ENV_NAME=$(ENV_NAME)

conduct-remote:
	$(MAKE) -C $(CONDUCT_DIR) conduct-remote ENV_NAME=$(ENV_NAME)

playback:
	$(MAKE) -C $(PLAYBACK_DIR) playback

touchdesigner:
	$(MAKE) -C $(TOUCHDESIGNER_DIR) touchdesigner

client: ensure-setup web-deps
	$(MAKE) -C $(CONDUCT_DIR) web-deps
	$(IN_ENV) python -m superconductor_client $(if $(DESCRIBE),--describe,) $(ARGS)

client-local: ensure-setup web-deps
	$(MAKE) -C $(CONDUCT_DIR) web-deps
	$(IN_ENV) python -m superconductor_client --where local $(if $(DESCRIBE),--describe,) $(ARGS)

client-remote: ensure-setup web-deps
	$(MAKE) -C $(CONDUCT_DIR) web-deps
	$(IN_ENV) python -m superconductor_client --where remote $(if $(DESCRIBE),--describe,) $(ARGS)

vision: check-env web-deps
	$(IN_ENV) python -m superconductor_client.tracking.collab --no-music $(ARGS)

audio: check-env
	$(IN_ENV) python audio_receiver.py $(ARGS)

audio-local: check-env
	$(IN_ENV) python audio_receiver.py --where local $(ARGS)

audio-remote: check-env
	$(IN_ENV) python audio_receiver.py --where remote $(ARGS)

web-deps:
	@superconductor_client/tracking/fetch_ort.sh

web-model: check-env
	$(IN_ENV) python -m superconductor_client.tracking.web_model

test: check-env
	$(IN_ENV) python tests/test_object_tracking.py
	$(IN_ENV) python tests/test_addresses.py
	$(IN_ENV) python tests/test_web_client.py
	$(IN_ENV) python tests/test_svg_model.py
	node --test superconductor_client/tracking/tracker_js/test/*.test.js

IMAGE_BUILD := docker buildx build --platform linux/amd64 --provenance=false --sbom=false \
	--build-context atlas=$(ATLAS_DIR) \
	-f client_container/Dockerfile -t $(IMAGE_DEV) .

image:
	$(IMAGE_BUILD) --load

image-push:
	$(IMAGE_BUILD) --push

# client-docker:
# 	docker run --rm -it --name sc-client --platform linux/amd64 $(GPUS) \
# 		-p 8080:8080 -v sc-data:/data --add-host=host.docker.internal:host-gateway \
# 		-e SC_WHERE=$(if $(SC_WHERE),$(SC_WHERE),docker) -e OLLAMA_URL=$(OLLAMA_URL) $(IMAGE_DEV)
#
# from another terminal (the client saves its library first)
# client-docker-stop:
# 	docker stop sc-client

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

client-singularity: $(SIF)
	mkdir -p $(SC_DATA)
	singularity run --nv --bind $(SC_DATA):/data \
		--env SC_WHERE=$(if $(SC_WHERE),$(SC_WHERE),cluster) \
		--env SC_SERVICES="$(SC_SERVICES)" \
		$(SIF)

client-singularity-dev: $(SIF_DEV)
	mkdir -p $(SC_DATA)
	singularity run --nv --bind $(SC_DATA):/data \
		--env SC_WHERE=$(if $(SC_WHERE),$(SC_WHERE),cluster) \
		--env SC_SERVICES="$(SC_SERVICES)" \
		$(SIF_DEV)

check-env:
ifeq ($(shell uname -s),Darwin)
	@test -x "$(ENV_PREFIX)/bin/python" || { \
		echo "conda env '$(ENV_NAME)' not found at $(ENV_PREFIX); run 'make setup'"; \
		exit 1; }
else
	@test -x .venv/bin/python || { \
		echo ".venv not found; run 'make setup'"; \
		exit 1; }
endif
