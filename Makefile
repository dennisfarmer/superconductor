# SuperConductor shortcuts. Uses the sc_env conda environment without needing
# `conda activate` (build it with scripts/setup_env.sh).
#
#   make server       MRT2 server on this Mac (mrt2_small, ../superconductor_server/Makefile); keep it running, then:
#   make client       the client, music from the local server; open the camera page
#                     http://localhost:8467/ in Chrome: camera, view and sound are there
#   make client-remote   same, music from the cluster server (mrt2_base, via the SSH tunnel)
#   make client ARGS="--mode calibration"   extra sc-collab options (--detection container: no browser detection)
#   make audio        sc-audio: play the music on this machine's sound device instead of the
#                     page (AUDIO_URL=ws://localhost:8080/audio through the container's gateway;
#                     ARGS="--role monitor" to play along with the page)
#   make web-deps     onnxruntime-web for the page's detector (var/web/ort)
#   make web-model    export the page's detector (YOLOE, [tracking] classes) to var/web/models
#   make server MODEL=mrt2_base   other server options (MODEL, PORT, ARGS, ...) are
#                     passed on to ../superconductor_server/Makefile
#   make describe     describe server (../superconductor_describe/Makefile): one-time
#                     VLM descriptions of new objects; needs Ollama running
#   make midi         MIDI player (../superconductor_midi/Makefile): plays .mid files through
#                     the running client's notes input, page on http://localhost:8475
#   make worktrees    server / describe / midi_experiment_dennis / client-web branches into ../superconductor_server,
#                     ../superconductor_describe, ../superconductor_midi (local branch superconductor_midi),
#                     ../superconductor_client_web
#   make vision       tracking only, no music
#   make test         unit tests: tracking / library / combos, the web client, tracker_js
#
# Container (camera, view and sound in the browser; see CLIENT_CONTAINER.md):
#   make client-web   same as make client (kept for the container docs)
#   make image        build the linux/amd64 image from this branch and the describe /
#                     midi / atlas worktrees (IMAGE=...); make image-push also pushes it
#   make client-docker   run it: http://localhost:8080 (SC_SERVER_URL=..., OLLAMA_URL=...)
#   make client-docker-stop   stop it
#   make sif          on lh2300: pull the image as a Singularity .sif
#   make client-singularity   on lh2300: run the .sif (SC_DATA=... for /data)

ENV_NAME ?= sc_env
ENV_PREFIX := $(shell conda info --base 2>/dev/null)/envs/$(ENV_NAME)
# run the env's binaries directly; PYTHONPATH: this checkout's code, not the one sc_env's
# editable install points at (the client branch)
IN_ENV := PATH="$(ENV_PREFIX)/bin:$$PATH" CONDA_PREFIX="$(ENV_PREFIX)" PYTHONPATH="$(CURDIR)"
ARGS ?=
AUDIO_URL ?= ws://localhost:8470/audio
SERVER_DIR := ../superconductor_server
DESCRIBE_DIR := ../superconductor_describe
MIDI_DIR := ../superconductor_midi
CLIENT_WEB_DIR := ../superconductor_client_web
ATLAS_DIR := ../superconductor_atlas
IMAGE ?= ghcr.io/dennisfarmer/superconductor:client-web
SIF ?= superconductor_client-web.sif
# where the MRT2 server is, seen from inside the container: the laptop's SSH tunnel
# (docker), or the server on the same node (lh2300)
SC_SERVER_URL ?= ws://host.docker.internal:9000/stream
OLLAMA_URL ?=
# Docker Desktop on a Mac has no GPU to give the container
GPUS ?= $(if $(filter Darwin,$(shell uname -s)),,--gpus all)
SC_DATA ?= /scratch/aimusic_project_root/aimusic_project/shared_data/sc_data_$(USER)

.PHONY: worktrees server describe midi client client-remote vision test check-env audio web-deps web-model \
	client-web image image-push client-docker client-docker-stop sif client-singularity

worktrees:
	@git fetch origin
	@test -e $(SERVER_DIR) || git worktree add $(SERVER_DIR) server
	@test -e $(DESCRIBE_DIR) || git worktree add $(DESCRIBE_DIR) describe
	@# local branch superconductor_midi tracks origin/midi_experiment_dennis
	@test -e $(MIDI_DIR) || { git show-ref -q --verify refs/heads/superconductor_midi \
		&& git worktree add $(MIDI_DIR) superconductor_midi \
		|| git worktree add --track -b superconductor_midi $(MIDI_DIR) origin/midi_experiment_dennis; }
	@test -e $(CLIENT_WEB_DIR) || git worktree add $(CLIENT_WEB_DIR) client-web
	@test -e $(ATLAS_DIR) || git worktree add $(ATLAS_DIR) atlas
	@git config push.default upstream
	@git worktree list

# command-line variables (MODEL=..., ARGS=...) reach the sub-make through MAKEFLAGS
server:
	$(MAKE) -C $(SERVER_DIR) server ENV_NAME=$(ENV_NAME)

describe:
	$(MAKE) -C $(DESCRIBE_DIR) describe ENV_NAME=$(ENV_NAME)

midi:
	$(MAKE) -C $(MIDI_DIR) player ENV_NAME=$(ENV_NAME)

client: check-env web-deps
	$(IN_ENV) sc-collab $(ARGS)

client-remote: check-env web-deps
	$(IN_ENV) sc-collab --remote $(ARGS)

vision: check-env web-deps
	$(IN_ENV) sc-collab --no-music $(ARGS)

audio: check-env
	$(IN_ENV) python -m superconductor.audio_receiver --url $(AUDIO_URL) $(ARGS)

web-deps:
	@scripts/fetch_ort.sh var/web

web-model: check-env
	$(IN_ENV) python -m superconductor.web_model

test: check-env
	$(IN_ENV) python tests/test_object_tracking.py
	$(IN_ENV) python tests/test_web_client.py
	node --test tracker_js/test/*.test.js

client-web: client

IMAGE_BUILD := docker buildx build --platform linux/amd64 --provenance=false --sbom=false \
	--build-context describe=$(DESCRIBE_DIR) --build-context midi=$(MIDI_DIR) \
	--build-context atlas=$(ATLAS_DIR) \
	-f container/Dockerfile -t $(IMAGE) .

image:
	$(IMAGE_BUILD) --load

image-push:
	$(IMAGE_BUILD) --push

client-docker:
	docker run --rm -it --name sc-client-web --platform linux/amd64 $(GPUS) \
		-p 8080:8080 -v sc-data:/data --add-host=host.docker.internal:host-gateway \
		-e SC_SERVER_URL=$(SC_SERVER_URL) -e OLLAMA_URL=$(OLLAMA_URL) $(IMAGE)

# from another terminal (the client saves its library first)
client-docker-stop:
	docker stop sc-client-web

sif:
	singularity pull --force $(SIF) docker://$(IMAGE)

client-singularity:
	mkdir -p $(SC_DATA)
	singularity run --nv --bind $(SC_DATA):/data \
		--env SC_SERVER_URL=$(if $(filter ws://host.docker.internal%,$(SC_SERVER_URL)),ws://localhost:9100/stream,$(SC_SERVER_URL)) \
		$(SIF)

check-env:
	@test -x "$(ENV_PREFIX)/bin/sc-collab" || { \
		echo "conda env '$(ENV_NAME)' with sc-collab not found at $(ENV_PREFIX); run scripts/setup_env.sh"; \
		exit 1; }
