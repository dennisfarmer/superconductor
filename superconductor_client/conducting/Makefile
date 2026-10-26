# SuperConductor conducting shortcuts (conducting_web/server.py). Self-contained, so it
# also works from a checkout of only this folder; the client's `make conduct` runs it.
#
#   make conduct          conducting page on http://localhost:9102, sending the tempo to the
#                         MRT2 server on this machine (http://localhost:9000, make server)
#   make conduct-remote   the same, to the server on Lighthouse through the SSH tunnel
#                         (http://localhost:9000; use with the client's make client-remote)
#   make conduct PORT=8481 SERVER=http://localhost:9000 ARGS="--tempo-window 3"
#   make web-deps         MediaPipe and the hand model for the page, into var/web
#   make test             unit tests in tests/
#
# Uses the sc_env conda env when it exists, without needing `conda activate`;
# otherwise the python on PATH. Needs aiohttp, numpy and scipy.

ENV_NAME ?= sc_env
ENV_PREFIX := $(shell conda info --base 2>/dev/null)/envs/$(ENV_NAME)
ifneq ($(wildcard $(ENV_PREFIX)/bin/python),)
IN_ENV := PATH="$(ENV_PREFIX)/bin:$$PATH" CONDA_PREFIX="$(ENV_PREFIX)"
endif
PORT ?= 9102
SERVER ?= http://localhost:9000
ARGS ?=

.PHONY: conduct conduct-remote web-deps test check-env

conduct: check-env web-deps
	$(IN_ENV) python conducting_web/server.py --port $(PORT) --server-url $(SERVER) $(ARGS)

conduct-remote: SERVER = http://localhost:9000
conduct-remote: conduct

web-deps:
	@scripts/fetch_mediapipe.sh var/web

test: check-env
	$(IN_ENV) python -m unittest discover -s tests -t . -p 'test_*.py'

check-env:
	@$(IN_ENV) python -c "import aiohttp, numpy, scipy" || { \
		echo "python with aiohttp, numpy and scipy not found: activate sc_env"; \
		exit 1; }
