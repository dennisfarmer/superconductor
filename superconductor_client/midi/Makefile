# SuperConductor MIDI player shortcuts (midi_player.py). Self-contained, so it
# also works from a checkout of only this folder.
#
#   make player       player page on http://localhost:9101, sending to the
#                     client's client_midi (http://localhost:9105)
#   make player PORT=8476 CLIENT=http://localhost:9105 ARGS="song.mid"
#
# Uses the sc_env conda env when it exists, without needing `conda activate`;
# otherwise the python on PATH. Needs aiohttp and mido.

ENV_NAME ?= sc_env
ENV_PREFIX := $(shell conda info --base 2>/dev/null)/envs/$(ENV_NAME)
ifneq ($(wildcard $(ENV_PREFIX)/bin/python),)
IN_ENV := PATH="$(ENV_PREFIX)/bin:$$PATH" CONDA_PREFIX="$(ENV_PREFIX)"
endif
PORT ?= 9101
CLIENT ?= http://localhost:9105
ARGS ?=

.PHONY: player check-env

player: check-env
	$(IN_ENV) python midi_player.py --port $(PORT) --client_url $(CLIENT) $(ARGS)

check-env:
	@$(IN_ENV) python -c "import aiohttp, mido" || { \
		echo "python with aiohttp and mido not found: activate sc_env, then pip install mido"; \
		exit 1; }
