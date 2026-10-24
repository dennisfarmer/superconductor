# SuperConductor describe server shortcuts (describe_server.py). Self-contained,
# so it also works from a checkout of only this folder (e.g. on Lighthouse).
#
#   make pull         download the VLM into Ollama (MODEL, default qwen2.5vl:3b)
#   make describe     describe server on port 9200 (needs `ollama serve` running)
#   make describe MODEL=qwen2.5vl:7b PORT=9300 ARGS="--keep_alive 0"
#
# Uses the sc_env conda env when it exists, without needing `conda activate`;
# otherwise the python on PATH (e.g. an activated venv on Lighthouse). The only
# Python dependency is aiohttp.

ENV_NAME ?= sc_env
ENV_PREFIX := $(shell conda info --base 2>/dev/null)/envs/$(ENV_NAME)
ifneq ($(wildcard $(ENV_PREFIX)/bin/python),)
IN_ENV := PATH="$(ENV_PREFIX)/bin:$$PATH" CONDA_PREFIX="$(ENV_PREFIX)"
endif
MODEL ?= qwen2.5vl:3b
PORT ?= 9200
ARGS ?=

.PHONY: describe pull check-env

describe: check-env
	$(IN_ENV) python describe_server.py --model $(MODEL) --port $(PORT) $(ARGS)

pull:
	ollama pull $(MODEL)

check-env:
	@$(IN_ENV) python -c "import aiohttp" || { \
		echo "python with aiohttp not found: activate sc_env or a venv with aiohttp (see README.md)"; \
		exit 1; }
