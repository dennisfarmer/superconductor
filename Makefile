# SuperConductor shortcuts. Uses the sc_env conda environment without needing
# `conda activate` (build it with scripts/setup_env.sh).
#
#   make run          webcam + Magenta RealTime 2 small (mrt2_small)
#   make run ARGS="--mode calibration"   extra sc-collab options
#   make vision       webcam tracking only, no music
#   make test         unit tests for tracking / library / combos

ENV_NAME ?= sc_env
ENV_PREFIX := $(shell conda info --base 2>/dev/null)/envs/$(ENV_NAME)
# run the env's binaries directly (keeps the terminal interactive for typed commands)
IN_ENV := PATH="$(ENV_PREFIX)/bin:$$PATH" CONDA_PREFIX="$(ENV_PREFIX)"
ARGS ?=

.PHONY: run vision test check-env

run: check-env
	$(IN_ENV) sc-collab --model mrt2_small $(ARGS)

vision: check-env
	$(IN_ENV) sc-collab --no-music $(ARGS)

test: check-env
	$(IN_ENV) python tests/test_object_tracking.py

check-env:
	@test -x "$(ENV_PREFIX)/bin/sc-collab" || { \
		echo "conda env '$(ENV_NAME)' with sc-collab not found at $(ENV_PREFIX); run scripts/setup_env.sh"; \
		exit 1; }
