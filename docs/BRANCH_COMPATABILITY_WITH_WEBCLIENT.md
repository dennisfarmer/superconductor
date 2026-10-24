Other branches and the web client
=================================

The web client (camera, view and sound in the browser; container in `client_container/`, see [client_container/README.md](../client_container/README.md)) is built from the `client` branch. The other branches go into the image as they are: nothing on them was changed for it. This file lists the changes that *would* be needed or useful on those branches, and what the container does for now instead.

### `describe` (`describe_server.py`) and `midi_experiment_dennis` (`midi_player.py`)

**Change:** a `--host` argument, passed to `web.run_app(..., host=...)`, with today's behavior as the default.

**Why:** `web.run_app` listens on every interface. Under Singularity the container shares lh2300's network, so the describe server (9200) and the MIDI player (8475) can be reached by other users of the cluster, not only through the gateway. In Docker this doesn't matter (only 8080 is published).

**For now:** nothing; they listen on every interface on lh2300. The client, client_midi and the gateway's other services already listen on 127.0.0.1 only.

### `atlas`

SuperConductor Atlas (ported to Python + TypeScript) is in the image, in SuperConductor mode. Changed on the `atlas` branch for it:
- **Mode:** `python -m server.app --local` (passed by `make run-local` / `start-local`); without it, SuperConductor mode. The session view has `local`. In SuperConductor mode the page hides the engine restart button, the LLM switch and "Describe this path", and the server refuses `restart_engine`, `set_llm` and `describe` (the restart started the local Magenta engine, which isn't in the image, and wrote `logs/` and `.run/` in the app folder).
- **`ATLAS_ENGINE_PUBLIC_URL`:** where the page reaches the engine (it sends timelines and reads `/status` itself), default `ATLAS_ENGINE_URL`. The container sets `/engine`; the gateway sends `atlas.localhost:8080/engine/*` to the engine.

Already as needed: the sink imports no MLX / magenta-rt / sounddevice; the engine listens on 127.0.0.1 and the app takes `--host`; the session goes to `ATLAS_DATA` (`/data/atlas`).

### `client`

**Change:** merge the web mode back from `client`:
- `superconductor_client/tracking/collab.py`: `--web`, `run_web()`, `_click()`, `--device cuda`, and the `[web] public_urls` display URLs on the Services tab;
- `superconductor_client/tracking/tracking_view.py`: the camera page, `/ws/camera` and `/api/web`;
- `superconductor_client/client.py`: `output="browser"` and client_midi's `/audio`;
- the new `superconductor_client/tracking/app.html` and `superconductor_client/tracking/collab.container.toml`.

All of it is additive: without `--web` the client behaves as before (camera window, sounddevice). `make worktrees` already creates `../superconductor_client_web`.

### Default branch (GitHub)

**Change:** add `.github/workflows/client-image.yml` there too, with a `workflow_dispatch:` trigger.

**Why:** GitHub only shows the "Run workflow" button for workflows on the default branch. Until then, rebuilding the image after `describe` or `midi_experiment_dennis` changes takes a push to `client` (an empty commit is enough).

### `server`

Nothing needed: the MRT2 server stays outside the container. The gateway serves its pages at `http://server.localhost:8080` (`/tempo`, `/conduct`) by forwarding to `SC_SERVER_URL`.
