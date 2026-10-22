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
| `UpdateControls` | `{"temperature"?: float, "top_k"?: int, "cfg_scales"?: {"musiccoca"?, "notes"?, "drums"?}}` | MRT2's `generate()` arguments under their own names (see Controls below), used from the next block on. |
| `ReceivedChunk`  | `null`                                              | One more block of credit (sent when the client finishes playing a block). |
| `Pause` / `Resume` | `null`                                            | Stop generating (session and state kept) / continue. |
| `Pattern`        | `{"notes"?: lane, "drums"?: lane, "id"?}`          | MRT2's per-frame NOTES / DRUMS input (see below). |
| `EndSession`     | `null`                                              | Stop generating and close the session. |

### Controls (`UpdateControls`)

Everything that steers sampling goes through `UpdateControls` (and `StartSession`'s `controls`), named as MRT2's `generate(conditioning, cfg_scales, temperature, top_k, ...)` names them. A message changes only the keys it carries; `null` puts a key back to the model default. A bad message gets an `Error` reply and changes nothing.

| Key | Range | Model default | What it does |
|---|---|---|---|
| `temperature` | > 0 | 1.3 | sampling temperature: higher is more varied, lower more repetitive |
| `top_k` | ≥ 1 | 40 | sample only from the k likeliest tokens |
| `cfg_scales.musiccoca` | -1 to 7, steps of 0.2 | 3.0 | classifier-free guidance toward the style (the recipe's MusicCoCa embedding) |
| `cfg_scales.notes` | -1 to 7, steps of 0.2 | 1.0 | guidance toward the Pattern's NOTES lane |
| `cfg_scales.drums` | -1 to 7, steps of 1 | 1.0 | guidance toward the Pattern's DRUMS lane |

The ranges and steps are MRT2's (`magenta_rt/config.py`, `CFG_CONDITIONING_*`): each scale is sent to the model as a conditioning token, so values are rounded to its step. HTTP senders (Atlas, the MIDI player) reach this message through the client's client_midi: `POST /controls` relays its body unchanged.

### Tempo (adaptive playback speed)

`tempo.py` time-stretches every block toward a target tempo before it is sent (speed changes, pitch doesn't), so audio blocks are `200ms / speed` long. It also measures the tempo MRT2 is generating ("model bpm") from the last 8s of audio, and moves the speed gradually (time constant 1.5s, clamped to 0.75–1.3) toward `target / model bpm`, matching up to half/double time.

- **Test page:** `http://localhost:9100/tempo` (or `:9000` through the tunnel): set an exact bpm, watch the model / heard bpm, and open the conductor window.
- **Conductor window:** `http://localhost:9100/conduct`: each space bar press sends one beat. The beat period is smoothed with a 1-D Kalman filter; if no beat arrives for 3s, the current tempo is held.

| Request | Body | Effect |
|---|---|---|
| `POST /tempo/beat` | `{"t"?: seconds}` | one conductor beat (`t` on the sender's clock, so network delay doesn't affect intervals) |
| `POST /tempo/target` | `{"bpm": 120}` | exact target tempo |
| `POST /tempo/free` | `null` | no target (speed 1) |
| `POST /tempo/meter` | `{"beats": 3}` | pin 3 or 4 beats per bar (default: 4, i.e. 4/4); `{"beats": null}` detects it |
| `POST /tempo/downbeat` | `null` | the beat being heard now is beat one (held until cleared) |
| `POST /tempo/override/clear` | `null` | back to the default 4/4 and the detected beat one |
| `GET /tempo/status` | | `mode, speed, model_bpm, heard_bpm, measured_output_bpm, target_bpm, conductor_bpm`, and the beat fields below |

**Beat one and beats per bar.** `BeatGrid` (in `tempo.py`) runs once a second on the last 8 s of heard audio. *Beat phase*: the onset envelope is sampled once per beat period starting at each possible offset (a comb over offsets), recent beats weighted more; the loudest offset is where the beats are, smoothed by a circular median of the last 5. *Beats per bar and beat one*: each beat's accent (onset + bass onset + chord change, normalised) is compared across 3- and 4-beat groupings; each window votes, and the last 6 votes are counted. The position is the audio the client is playing now: the end of the last block it acknowledged plus the time since then. The sound card's latency is not included; the pages have a "light delay" setting for it.

| Status field | Meaning |
|---|---|
| `playing` | audio is being heard (false before the first acknowledgement and once the client's buffer has played out) |
| `beat` | index of the beat being heard within the bar, 0 = beat one; counted modulo 4 while `beats_per_bar` is null; null when not playing |
| `beat_frac` | 0–1, how far through that beat |
| `beats_per_bar` | 3 or 4 (4 by default); null only in auto while not sure yet |
| `meter_confidence` | share of the last 6 votes that agree on beats per bar and beat one (null before the first vote) |
| `meter_source` | `manual` (set, including the default 4/4), `detected` (auto), or null |
| `downbeat_source` | `detected`, `manual`, or null (beat one unknown: `beat` then counts from an arbitrary beat) |

| `stretching` | true while the heard audio is sped up or slowed down (speed ≠ 1); at speed exactly 1 the audio passes through unchanged |
| `conductor_holding` | following the conductor, but no beat for 3 s: the last speed is held |
| `inputs` | what shapes the generation now: `session` (`none` / `generating` / `paused`), `recipe` (style prompts and weights), `controls`, `notes` / `drums` (`{loop, frames}` of the Pattern lane, or null when the model chooses), `pattern_id` |

To light a beat in time: `anchor = now + latency - beat_frac * 60 / heard_bpm` is when `beat` started being heard (see `pages/tempo.html`).

### NOTES / DRUMS (`Pattern`)

Besides the style, MRT2 takes two inputs every 40 ms frame: NOTES (128 MIDI-pitch slots: -1 model chooses, 0 off, 1 held, 2 onset, 3 on) and DRUMS (one slot: -1 model chooses, 0 no hit, 1 hit). A `Pattern` sets either lane as run-length steps, looped or played once, from the next generated frame. With no pattern every slot is -1, as before. If a pattern is rejected, the server replies `{"type": "Error", "body": {"error": ...}}` and keeps generating.

Note sources don't talk to the server directly: they POST to the client's client_midi (port 8470), which forwards over this websocket. The full format, with a MIDI-file converter, is in [`superconductor_client/MIDI_MSG_PROTOCOL.md`](../superconductor_client/MIDI_MSG_PROTOCOL.md); the reference implementation is `expand_lane` / `Lane` in `mrt2_server.py`.
