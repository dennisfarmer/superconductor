# SuperConductor Server

### Setup

`environment.yml` is a verbatim copy of `superconductor_client/environment.yml`.

- **Mac (MLX):** `make env` builds the `sc_env` conda env (the same one the client uses), then `make models` downloads `mrt2_small` (`make models MODEL=mrt2_base` for base).
- **Lighthouse (JAX):** in a Python 3.12 venv (create it once; activate it in every new terminal):
    ```
    cd /scratch/aimusic_project_root/aimusic_project/shared_data
    git clone -b server --single-branch https://github.com/dennisfarmer/superconductor.git superconductor_server
    cd superconductor_server
    python3.12 -m venv .venv
    source .venv/bin/activate
    pip install "magenta-rt[jax]==2.0.3" "jax[cuda12]" aiohttp
    ```
    - use `jax[cuda12]`, not cuda13: CUDA 13 dropped Volta (V100) support
    - download the weights: `make models`
    - V100: watch `gen_ms_per_frame` in the `Stats` messages, check if <40ms/frame

### Startup Sequence

- locally: `make server` (from here or from `superconductor_client`), leave it running, then (with Ollama running) `make describe` and `make client` in `superconductor_client`
- on Lighthouse:
    - activate the venv, then `make server`
    - on the laptop, start the ssh tunnel: `ssh -N -L 9000:lh2300:9100 UNIQNAME@lighthouse.arc-ts.umich.edu`
    - the describe server still runs on the laptop: with Ollama running, `make describe` in `superconductor_client` (see `superconductor_describe/README.md`)
    - then `make client-remote` in `superconductor_client` (connects to `ws://localhost:9000/stream`)

Other flags: `make server PORT=... ARGS="--frames_per_block 5"`, or `python mrt2_server.py --help`.

### What it sends to the client

MRT2 state stays in the server process. The client receives:

- one `Ready` JSON text frame after the model has loaded: `{"type": "Ready", "body": {"model", "backend", "sample_rate", "frames_per_block", "host"}}`
- a `Stats` JSON text frame about once a second: `{"type": "Stats", "body": {"rtf", "gen_ms_per_frame"}}` (`rtf` > 1 means generating faster than real time)
- audio as binary frames, one per block (`frames_per_block` × 40ms, default 5 = 200ms, 48kHz stereo):

```
[4 bytes: uint32 LE block sequence number][rest: float32 LE samples, row-major (num_samples, 2)]
```

### Websocket protocol

The client opens a websocket at `GET /stream` (one client at a time; a second one is closed with "Session already active"). Flow control is credit-based: the server only generates while it has credits, so the client's sound card paces generation. All client → server messages are JSON text frames `{"type": ..., "body": ...}`:

| `type`           | `body`                                              | Effect |
|------------------|-----------------------------------------------------|--------|
| `StartSession`   | `{"recipe"?: {...}, "controls"?: {...}, "credits": int}` | Reset the state and start generating, with `credits` blocks in flight (default recipe `{"jazz": 1.0}`). |
| `UpdateRecipe`   | `{"recipe": {"jazz": 0.6, "flute": 0.3}}`           | Weighted blend of the prompts' MusicCoCa embeddings (cached), used from the next block on. |
| `UpdateControls` | `{"temperature"?: float, "top_k"?: int, "cfg_musiccoca"?: float}` | Sampling controls, used from the next block on. |
| `ReceivedChunk`  | `null`                                              | One more block of credit (sent when the client finishes playing a block). |
| `Pause` / `Resume` | `null`                                            | Stop generating (session and state kept) / continue. |
| `Pattern`        | `{"notes"?: lane, "drums"?: lane, "cfg_notes"?, "cfg_drums"?, "id"?}` | MRT2's per-frame NOTES / DRUMS input (see below). |
| `EndSession`     | `null`                                              | Stop generating and close the session. |

### Tempo (adaptive playback speed)

`tempo.py` time-stretches every block toward a target tempo before it is sent (speed changes, pitch doesn't), so audio blocks are `200ms / speed` long. It also measures the tempo MRT2 is generating ("model bpm") from the last 8s of audio, and moves the speed gradually (time constant 1.5s, clamped to 0.75–1.3) toward `target / model bpm`, matching up to half/double time.

- **Test page:** `http://localhost:9100/tempo` (or `:9000` through the tunnel): set an exact bpm, watch the model / heard bpm, and open the conductor window.
- **Conductor window:** `http://localhost:9100/conduct`: each space bar press sends one beat. The beat period is smoothed with a 1-D Kalman filter; if no beat arrives for 3s, the current tempo is held.

| Request | Body | Effect |
|---|---|---|
| `POST /tempo/beat` | `{"t"?: seconds}` | one conductor beat (`t` on the sender's clock, so network delay doesn't affect intervals) |
| `POST /tempo/target` | `{"bpm": 120}` | exact target tempo |
| `POST /tempo/free` | `null` | no target (speed 1) |
| `GET /tempo/status` | | `mode, speed, model_bpm, heard_bpm, measured_output_bpm, target_bpm, conductor_bpm` |

### NOTES / DRUMS (`Pattern`)

Besides the style, MRT2 takes two inputs every 40 ms frame: NOTES (128 MIDI-pitch slots: -1 model chooses, 0 off, 1 held, 2 onset, 3 on) and DRUMS (one slot: -1 model chooses, 0 no hit, 1 hit). A `Pattern` sets either lane as run-length steps, looped or played once, from the next generated frame. With no pattern every slot is -1, as before. If a pattern is rejected, the server replies `{"type": "Error", "body": {"error": ...}}` and keeps generating.

Note sources don't talk to the server directly: they POST to the client's client_midi (port 8470), which forwards over this websocket. The full format, with a MIDI-file converter, is in [`superconductor_client/MIDI_MSG_PROTOCOL.md`](../superconductor_client/MIDI_MSG_PROTOCOL.md); the reference implementation is `expand_lane` / `Lane` in `mrt2_server.py`.
