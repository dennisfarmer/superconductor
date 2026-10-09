# SuperConductor shortcuts. Uses the sc_env conda environment without needing
# `conda activate` (build it with scripts/setup_env.sh).
#
#   make server       MRT2 server on this Mac (mrt2_small, ../superconductor_server/Makefile); keep it running, then:
#   make client       webcam + music from the local server
#   make client-iphone   same, with the iPhone (Continuity Camera) instead of the Logitech
#   make client-remote   webcam + music from the cluster server (mrt2_base, via the SSH tunnel)
#   make client-remote-iphone   cluster server + iPhone camera
#   make client ARGS="--mode calibration"   extra sc-collab options
#   make server MODEL=mrt2_base   other server options (MODEL, PORT, ARGS, ...) are
#                     passed on to ../superconductor_server/Makefile
#   make describe     describe server (../superconductor_describe/Makefile): one-time
#                     VLM descriptions of new objects; needs Ollama running
#   make midi         MIDI player (../superconductor_midi/Makefile): plays .mid files through
#                     the running client's notes input, page on http://localhost:8475
#   make worktrees    server / describe / superconductor_midi branches into ../superconductor_server,
#                     ../superconductor_describe, ../superconductor_midi
#   make vision       webcam tracking only, no music
#   make test         unit tests for tracking / library / combos

ENV_NAME ?= sc_env
ENV_PREFIX := $(shell conda info --base 2>/dev/null)/envs/$(ENV_NAME)
# run the env's binaries directly
IN_ENV := PATH="$(ENV_PREFIX)/bin:$$PATH" CONDA_PREFIX="$(ENV_PREFIX)"
ARGS ?=
SERVER_DIR := ../superconductor_server
DESCRIBE_DIR := ../superconductor_describe
MIDI_DIR := ../superconductor_midi

.PHONY: worktrees server describe midi client client-iphone client-remote client-remote-iphone vision test check-env

worktrees:
	@git fetch origin
	@test -e $(SERVER_DIR) || git worktree add $(SERVER_DIR) server
	@test -e $(DESCRIBE_DIR) || git worktree add $(DESCRIBE_DIR) describe
	@test -e $(MIDI_DIR) || git worktree add $(MIDI_DIR) superconductor_midi
	@git worktree list

# command-line variables (MODEL=..., ARGS=...) reach the sub-make through MAKEFLAGS
server:
	$(MAKE) -C $(SERVER_DIR) server ENV_NAME=$(ENV_NAME)

describe:
	$(MAKE) -C $(DESCRIBE_DIR) describe ENV_NAME=$(ENV_NAME)

midi:
	$(MAKE) -C $(MIDI_DIR) player ENV_NAME=$(ENV_NAME)

client: check-env
	$(IN_ENV) sc-collab $(ARGS)

client-iphone: check-env
	$(IN_ENV) sc-collab --iphone $(ARGS)

client-remote: check-env
	$(IN_ENV) sc-collab --remote $(ARGS)

client-remote-iphone: check-env
	$(IN_ENV) sc-collab --remote --iphone $(ARGS)

vision: check-env
	$(IN_ENV) sc-collab --no-music $(ARGS)

test: check-env
	$(IN_ENV) python tests/test_object_tracking.py

check-env:
	@test -x "$(ENV_PREFIX)/bin/sc-collab" || { \
		echo "conda env '$(ENV_NAME)' with sc-collab not found at $(ENV_PREFIX); run scripts/setup_env.sh"; \
		exit 1; }
