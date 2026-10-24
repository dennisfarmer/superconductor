# SuperConductor shortcuts. `make setup` once (and again after a pull; it leaves
# an environment that is already there). Then `make client`.
# macOS uses the sc_env conda env. lh2300 uses .venv in this checkout, so the
# client runs there without the Singularity image.
#
#   make setup        macOS: sc_env, if it is missing. lh2300: .venv from
#                     client_container/requirements.txt, no Singularity image.
#   make server       MRT2 server on the laptop (mrt2_small, ../superconductor_server/Makefile); keep it running, then:
#   make client       the client, plus tracking, conducting, describe and midi;
#                     music from the local server. Open the camera page
#                     http://localhost:8467/ in Chrome: camera, view and sound are there
#   make client-remote   same, music from the cluster server (mrt2_base, via the SSH tunnel)
#   make client ARGS="--mode calibration"   extra tracking options (--detection container: no browser detection)
#   make audio        sc-audio: play the music on the laptop's sound device instead of the
#                     page (AUDIO_URL=ws://localhost:8080/audio through the container's gateway;
#                     ARGS="--role monitor" to play along with the page)
#   make web-deps     onnxruntime-web for the page's detector (var/web/ort)
#   make web-model    export the page's detector (YOLOE, [tracking] classes) to var/web/models
#   make server MODEL=mrt2_base   other server options (MODEL, PORT, ARGS, ...) are
#                     passed on to ../superconductor_server/Makefile
#   make describe     describe server (superconductor_client/describe/): one-time VLM descriptions of new
#                     objects; needs Ollama running. POST /svg draws parameter icons (SVGen)
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
#                     builds it if needed): plays the music on the laptop's sound output, as the
#                     clock. Open the Tracking page as http://localhost:8467/?audio=monitor then,
#                     or the page plays the music too
#   make touchdesigner   a TouchDesigner visualizer (../superconductor_touchdesigner/Makefile),
#                     showing only its output window: VISUALIZER=<name> (default
#                     overlay_example_1), PERFORM=0 to open it in the editor instead
#   make vision       tracking only, no music
#   make test         unit tests: tracking / library / combos, the web client, tracker_js
#
# Container (camera, view and sound in the browser; see client_container/README.md):
#   make image        build the linux/amd64 image from this checkout and the atlas
#                     worktree (IMAGE=...); make image-push also pushes it
#   make client-docker   run it: http://localhost:8080 (SC_SERVER_URL=..., OLLAMA_URL=...)
#   make client-docker-stop   stop it
#   make sif          on lh2300: pull the image as a Singularity .sif
#   make client-singularity   on lh2300: run the .sif (SC_DATA=... for /data)

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
AUDIO_URL ?= ws://localhost:8470/audio
SERVER_DIR := ../superconductor_server
DESCRIBE_DIR := superconductor_client/describe
MIDI_DIR := superconductor_client/midi
ATLAS_DIR := ../superconductor_atlas
CONDUCTING_DIR := ../superconductor_conducting
CONDUCT_DIR := superconductor_client/conducting
PLAYBACK_DIR := ../superconductor_playback
TOUCHDESIGNER_DIR := ../superconductor_touchdesigner
IMAGE ?= ghcr.io/dennisfarmer/superconductor:client
SIF ?= superconductor_client.sif
# where the MRT2 server is, seen from inside the container: the laptop's SSH tunnel
# (docker), or the server on the same node (lh2300)
SC_SERVER_URL ?= ws://host.docker.internal:9000/stream
OLLAMA_URL ?=
# Docker Desktop on a Mac has no GPU to give the container
GPUS ?= $(if $(filter Darwin,$(shell uname -s)),,--gpus all)
SC_DATA ?= /scratch/aimusic_project_root/aimusic_project/shared_data/sc_data_$(USER)

.PHONY: setup worktrees server describe midi atlas conduct conduct-remote playback touchdesigner client client-remote vision test check-env audio web-deps web-model \
	image image-push client-docker client-docker-stop sif client-singularity

# macOS: sc_env from environment.yml, created if it is missing, then mlx-lm and
# the SVGen weights (parameter icons). lh2300: .venv, the CUDA 12.6 client
# requirements (the V100), and the same weights via transformers. No Singularity.
# A second run leaves an environment that is already there; the icon packages
# and the weight download still run (snapshot_download skips files it has).
setup:
ifeq ($(shell uname -s),Darwin)
	@if [ -x "$(ENV_PREFIX)/bin/python" ]; then echo "sc_env already exists"; else ./setup_env.sh; fi
	@"$(ENV_PREFIX)/bin/python" -m pip install --upgrade-strategy only-if-needed \
		"mlx==0.31.2" "mlx-metal==0.31.2" "mlx-lm==0.31.2" "transformers==5.12.1" \
		"huggingface-hub==1.16.1" "tokenizers==0.22.2" "safetensors==0.8.0" \
		"tqdm==4.70.1" "PyYAML==6.0.3" "typer==0.27.3"
	@"$(ENV_PREFIX)/bin/python" -c "from huggingface_hub import snapshot_download; snapshot_download('gitcat-404/SVGen-Qwen2.5-3B-Instruct')"
	@"$(ENV_PREFIX)/bin/python" -m pip install --no-deps -e .
else
	@test -x .venv/bin/python || python3.12 -m venv .venv
	.venv/bin/python -m pip install -r client_container/requirements.txt
	HF_HOME=$${HF_HOME:-$(CURDIR)/.cache/huggingface} .venv/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download('gitcat-404/SVGen-Qwen2.5-3B-Instruct')"
	.venv/bin/python -m pip install --no-deps -e .
endif

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

client: check-env web-deps
	$(MAKE) -C $(CONDUCT_DIR) web-deps
	$(IN_ENV) python -m superconductor_client $(ARGS)

client-remote: check-env web-deps
	$(MAKE) -C $(CONDUCT_DIR) web-deps
	$(IN_ENV) python -m superconductor_client --remote $(ARGS)

vision: check-env web-deps
	$(IN_ENV) python -m superconductor_client.tracking.collab --no-music $(ARGS)

audio: check-env
	$(IN_ENV) python audio_receiver.py --url $(AUDIO_URL) $(ARGS)

web-deps:
	@superconductor_client/tracking/fetch_ort.sh

web-model: check-env
	$(IN_ENV) python -m superconductor_client.tracking.web_model

test: check-env
	$(IN_ENV) python tests/test_object_tracking.py
	$(IN_ENV) python tests/test_web_client.py
	$(IN_ENV) python tests/test_svg_model.py
	node --test superconductor_client/tracking/tracker_js/test/*.test.js

IMAGE_BUILD := docker buildx build --platform linux/amd64 --provenance=false --sbom=false \
	--build-context atlas=$(ATLAS_DIR) \
	-f client_container/Dockerfile -t $(IMAGE) .

image:
	$(IMAGE_BUILD) --load

image-push:
	$(IMAGE_BUILD) --push

client-docker:
	docker run --rm -it --name sc-client --platform linux/amd64 $(GPUS) \
		-p 8080:8080 -v sc-data:/data --add-host=host.docker.internal:host-gateway \
		-e SC_SERVER_URL=$(SC_SERVER_URL) -e OLLAMA_URL=$(OLLAMA_URL) $(IMAGE)

# from another terminal (the client saves its library first)
client-docker-stop:
	docker stop sc-client

sif:
	singularity pull --force $(SIF) docker://$(IMAGE)

client-singularity:
	mkdir -p $(SC_DATA)
	singularity run --nv --bind $(SC_DATA):/data \
		--env SC_SERVER_URL=$(if $(filter ws://host.docker.internal%,$(SC_SERVER_URL)),ws://localhost:9100/stream,$(SC_SERVER_URL)) \
		$(SIF)

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
