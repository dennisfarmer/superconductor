# SuperConductor Server

### Setup

`environment.yml` is a verbatim copy of `superconductor_client/environment.yml`.

- **Mac (MLX):** `make env` builds the `sc_env` conda env (the same one the client uses), then `make models` downloads `mrt2_small` (`make models MODEL=mrt2_base` for base).
- **Lighthouse (JAX):** in a Python 3.12 venv (create it once; activate it in every new terminal):
    ```
    git clone https://github.com/dennisfarmer/superconductor.git
    cd superconductor
    git checkout server
    python3.12 -m venv .venv
    source .venv/bin/activate
    pip install "magenta-rt[jax]==2.0.3" "jax[cuda12]" aiohttp
    ```
    - use `jax[cuda12]`, not cuda13: CUDA 13 dropped Volta (V100) support
    - download the weights on a login node, into scratch: `MAGENTA_HOME=<scratch dir> make models BACKEND=jax MODEL=mrt2_base` (`MAGENTA_HOME` defaults to `~/Documents/Magenta`; set it the same way when starting the server)
    - this path hasn't been run yet; ms/frame on the V100 must stay under 40ms (watch `gen_ms_per_frame` in the `Stats` messages)

### Startup Sequence

- locally: `make server` (from here or from `superconductor_client`), leave it running, then `make client` in `superconductor_client`
- on Lighthouse:
    - allocate a gpu if not on a gpu session: `salloc --account=aimusic_project --partition=aimusic_project --gpus=1 --mem=64G --cpus-per-task=4 --time=00:15:00`
        - adjust `--time` based on how long you need the server; the job stops when it expires, or run `exit` to release it early
    - activate the venv, then `make server BACKEND=jax MODEL=mrt2_base` (listens on port 9100)
    - on the laptop, start the ssh tunnel: `ssh -N -L 9000:<gpu node, e.g. lh2300>:9100 YOUR_UNIQNAME@lighthouse.arc-ts.umich.edu`
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
