# SuperConductor Atlas: start/stop the audio engine and the app (FastAPI server in
# server/, React client in client/, built into client/dist when its sources change).
#
# SuperConductor (the default): the engine is a sink. It sends the notes and drums to
# the SuperConductor client (make server + make client in ../superconductor_client),
# whose prompts set the sound. No Magenta runs here.
#
#   make run           engine + app in the foreground (Ctrl-C stops both)
#   make start         engine + app in the background, wait until both are up
#
# Standalone: the engine runs Magenta RT 2 itself and plays through this machine's
# speakers. No SuperConductor client needed.
#
#   make run-local     engine + app in the foreground (Ctrl-C stops both)
#   make start-local   engine + app in the background, wait until both are up
#
# Either way:
#
#   make stop          stop both
#   make restart       stop, then start (SuperConductor)
#   make status        show what's running
#   make logs          follow both logs
#
# Development: the engine (SuperConductor sink), the API with auto-reload on
# DEV_API_PORT, and the Vite dev server on PORT (hot reload; it proxies the API).
#
#   make dev

ENV_PREFIX ?= $(shell conda info --base 2>/dev/null)/envs/sc_env_atlas
PY         := $(ENV_PREFIX)/bin/python
PORT       ?= 8000
DEV_API_PORT := 8001
ENGINE_PORT?= 8765
PRESET     ?= warm_piano
LOGS       := logs
RUN        := .run
CLIENT_MIDI?= http://localhost:8470
SC_SERVER  ?= http://localhost:9100

# what the engine is started with, per mode
SC_ENGINE    := audio_engine/engine.py --sink superconductor --client-midi $(CLIENT_MIDI) --sc-server $(SC_SERVER) --preset $(PRESET) --port $(ENGINE_PORT)
LOCAL_ENGINE := audio_engine/engine.py --preset $(PRESET) --port $(ENGINE_PORT)

.PHONY: run start check-client run-local start-local stop restart status logs app client-build dev

# Ask the engine on ENGINE_PORT to exit (POST /shutdown: it finishes its chunk, closes
# the audio stream and its server), then wait up to 15 s until it no longer answers.
# Only that engine is addressed; nothing is signalled or killed.
STOP_ENGINE = if curl -s -m 1 -o /dev/null http://127.0.0.1:$(ENGINE_PORT)/status; then \
	curl -s -m 2 -o /dev/null -X POST -H 'Content-Type: application/json' -d '{}' http://127.0.0.1:$(ENGINE_PORT)/shutdown; \
	for i in $$(seq 1 75); do curl -s -m 1 -o /dev/null http://127.0.0.1:$(ENGINE_PORT)/status || break; sleep 0.2; done; \
	fi; rm -f $(RUN)/engine.pid

# Foreground: $(1) is the engine's arguments, $(2) what to call it. Ctrl-C stops the
# app (uvicorn's own graceful exit); the engine is then asked to exit over its control
# API, which also reaches an engine relaunched from the page.
define RUN_FG
	@mkdir -p $(LOGS)
	@stop_engine() { $(STOP_ENGINE); }; \
	trap 'stop_engine; exit 130' INT TERM; trap stop_engine EXIT; \
	$(PY) $(1) > $(LOGS)/engine.log 2>&1 & E=$$!; \
	echo "$(2) starting (log: $(LOGS)/engine.log); app on http://localhost:$(PORT)"; \
	$(PY) -m server.app --port $(PORT)
endef

# Background: start the engine with $(1) unless one is already up, wait until it answers.
define START_ENGINE
	@mkdir -p $(LOGS) $(RUN)
	@if curl -s -o /dev/null http://127.0.0.1:$(ENGINE_PORT)/status; then \
		echo "audio engine already running"; \
	else \
		echo "starting $(2)..."; \
		nohup $(PY) $(1) > $(LOGS)/engine.log 2>&1 & echo $$! > $(RUN)/engine.pid; \
		for i in $$(seq 1 60); do curl -s -o /dev/null http://127.0.0.1:$(ENGINE_PORT)/status && break; sleep 0.5; done; \
		curl -s -o /dev/null http://127.0.0.1:$(ENGINE_PORT)/status || { echo "engine failed to start; see $(LOGS)/engine.log"; exit 1; }; \
	fi
endef

define STARTED
	@echo ""
	@echo "  SuperConductor Atlas is running on http://localhost:$(PORT)"
	@echo "  ($(1) on http://127.0.0.1:$(ENGINE_PORT); stop with: make stop)"
	@echo ""
endef


# ==================================================================================
# SuperConductor (default): the engine sends notes and drums to the SuperConductor
# client at CLIENT_MIDI
# ==================================================================================

run: check-client client-build
	$(call RUN_FG,$(SC_ENGINE),engine (SuperConductor sink))

start: check-client
	$(call START_ENGINE,$(SC_ENGINE),audio engine (SuperConductor sink))
	@$(MAKE) --no-print-directory app
	$(call STARTED,engine as SuperConductor sink)

# A warning, not a failure: the engine connects once the client is up.
check-client:
	@curl -s -m 1 -o /dev/null $(CLIENT_MIDI)/status || \
		echo "SuperConductor client not reachable at $(CLIENT_MIDI): start 'make server' and 'make client' in ../superconductor_client (starting anyway; it connects when they're up)"


# ==================================================================================
# Standalone: local Magenta RT 2 engine, no SuperConductor
# ==================================================================================

run-local: client-build
	$(call RUN_FG,$(LOCAL_ENGINE),engine (local Magenta))

start-local:
	$(call START_ENGINE,$(LOCAL_ENGINE),audio engine (local Magenta, preset $(PRESET)))
	@$(MAKE) --no-print-directory app
	$(call STARTED,local Magenta engine)


# ==================================================================================
# Shared by both modes
# ==================================================================================

app: client-build
	@mkdir -p $(LOGS) $(RUN)
	@if curl -s -o /dev/null http://localhost:$(PORT)/healthz; then \
		echo "app already running"; \
	else \
		echo "starting the app..."; \
		nohup $(PY) -m server.app --port $(PORT) < /dev/null > $(LOGS)/app.log 2>&1 & echo $$! > $(RUN)/app.pid; \
		for i in $$(seq 1 120); do curl -s -o /dev/null http://localhost:$(PORT)/healthz && break; sleep 1; done; \
		curl -s -o /dev/null http://localhost:$(PORT)/healthz || { echo "app failed to start; see $(LOGS)/app.log"; exit 1; }; \
	fi

stop:
	@if curl -s -o /dev/null http://localhost:$(PORT)/healthz; then \
		echo "asking the app to quit..."; \
		curl -s -m 30 -o /dev/null -X POST -H 'Content-Type: application/json' -d '{}' http://localhost:$(PORT)/function/shutdown_app; \
		for i in $$(seq 1 50); do curl -s -m 1 -o /dev/null http://localhost:$(PORT)/healthz || break; sleep 0.2; done; \
	fi
	@$(STOP_ENGINE)
	@rm -f $(RUN)/app.pid
	@echo "SuperConductor Atlas stopped"

restart: stop start

# Install the client's packages if needed, and rebuild client/dist when its sources are
# newer than the last build.
client-build:
	@[ -d client/node_modules ] || npm --prefix client ci
	@if [ ! -f client/dist/index.html ] || [ -n "$$(find client/src client/index.html client/package.json client/vite.config.ts -newer client/dist/index.html | head -1)" ]; then \
		npm --prefix client run build; \
	fi

# Engine (SuperConductor sink) + API (reloads on server/ changes) + Vite (hot reload).
# Ctrl-C stops all three.
dev: check-client
	@[ -d client/node_modules ] || npm --prefix client ci
	@mkdir -p $(LOGS)
	@stop_engine() { $(STOP_ENGINE); }; \
	trap 'kill $$A 2>/dev/null; stop_engine; exit 130' INT TERM; trap 'kill $$A 2>/dev/null; stop_engine' EXIT; \
	$(PY) $(SC_ENGINE) > $(LOGS)/engine.log 2>&1 & \
	$(PY) -m uvicorn server.app:app --reload --reload-dir server --port $(DEV_API_PORT) > $(LOGS)/app.log 2>&1 & A=$$!; \
	echo "engine log: $(LOGS)/engine.log, API log: $(LOGS)/app.log; app on http://localhost:$(PORT)"; \
	npm --prefix client run dev

status:
	@curl -s -o /dev/null http://localhost:$(PORT)/healthz && echo "app:    up   http://localhost:$(PORT)" || echo "app:    down"
	@curl -s http://127.0.0.1:$(ENGINE_PORT)/status 2>/dev/null | $(PY) -c "import json,sys; d=json.load(sys.stdin); print(f\"engine: up   {d['model']} · {d['ms_per_frame']} ms/frame · {d['underruns']} underruns · {d['style']}\")" 2>/dev/null || echo "engine: down"

logs:
	@tail -f $(LOGS)/engine.log $(LOGS)/app.log
