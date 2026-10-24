# SuperConductor Server

### Setup

`environment_macbook.yml` is a verbatim copy of `superconductor_client/environment.yml`.

Testing on either machine is `git pull` and `make server`. `make server` runs `make setup` when the environment is missing or `environment_macbook.yml` / `requirements_lighthouse.txt` has changed. That check is a few file tests when `.cache/setup-stamp` still matches. `make setup` creates the environment if it is missing (the `sc_env` conda env on macOS, a Python 3.12 `.venv` from `requirements_lighthouse.txt` on lh2300) and downloads weights. macOS gets `mrt2_small` only. lh2300 gets the JAX checkpoints for `mrt2_base` and `mrt2_small`. Files already on disk are left alone. `make env` updates the macOS conda env from `environment_macbook.yml` when that file changes.

```bash
cd /scratch/aimusic_project_root/aimusic_project/shared_data
git clone -b server --single-branch https://github.com/dennisfarmer/superconductor.git superconductor_server
cd superconductor_server
git pull && make server
```

`requirements_lighthouse.txt` installs `magenta-rt[jax]==2.0.3` with `jax[cuda12]`. CUDA 13 dropped Volta (V100). On a V100, watch `gen_ms_per_frame` in the `Stats` messages and check that it stays under 40 ms/frame.

### Image

The release image is `ghcr.io/dennisfarmer/superconductor_server:server`. It is published with the client image by **Actions** → **release** on the `client` branch. On lh2300, `make server-singularity` sources `/scratch/aimusic_project_root/aimusic_project/shared_data/.singularity/singularity-setup.sh` (Singularity and the ghcr.io login), then runs the image. It pulls when `superconductor_server.sif` is absent. The copy of the script at the top of this checkout is only a reference. `make sif` pulls again after a new release. Checkpoints stay on scratch (`MAGENTA_HOME`). The image run does not run `make setup`.

`make image-push` on a laptop publishes the dev image, `:server-dev`. On lh2300, `make sif-dev` pulls it and `make server-singularity-dev` runs it. Notes for the next release go under `## [Unreleased]` in `CHANGELOG.md`.

### Startup Sequence

- on the Macbook Pro: `make server` (from here or from `superconductor_client`), leave it running, then `make client` in `superconductor_client`. `make client` starts Ollama if one is not already running.
- on Lighthouse:
    - `make server` (base model; uses `.venv` from `make setup`)
    - or `make server MODEL=mrt2_small` (small model)
    - on the laptop, start the ssh tunnel: `ssh -N -L 9000:lh2300:9100 UNIQNAME@lighthouse.arc-ts.umich.edu`
    - then `make client-remote` in `superconductor_client` (connects to `ws://localhost:9000/stream/input`). That starts describe on the laptop, and Ollama if one is not already running.

Other flags: `make server PORT=... ARGS="--frames_per_block 5"`, or `python mrt2_server.py --help`.

### What playback receives

MRT2 state stays in the server process. `GET /stream/playback` receives:

- one `Ready` JSON text frame when it connects: `{"type": "Ready", "body": {"model", "backend", "sample_rate", "frames_per_block", "host"}}`
- a `Stats` JSON text frame about once a second: `{"type": "Stats", "body": {"rtf", "gen_ms_per_frame"}}` (`rtf` > 1 means generating faster than real time)
- audio as binary frames, one per block (`frames_per_block` × 40ms, default 5 = 200ms, 48kHz stereo):

```
[4 bytes: uint32 LE block sequence number][rest: float32 LE samples, row-major (num_samples, 2)]
```

The input socket (`GET /stream/input`) gets `Ready` when it connects, and `Error` when a message is rejected. Audio and `Stats` are not sent there.

### Websocket protocol

Two sockets. Playback is one connection at a time; a second is closed with "Session already active". The client sends recipe, controls and pattern on the input socket. Generation waits until playback is attached, playback has granted a lead, and a recipe has been set. The lead is that many blocks; a new recipe, controls or pattern applies on the next block, and blocks already sent still play.

**`GET /stream/playback`** — the player.

| `type`           | `body`                 | Effect |
|------------------|------------------------|--------|
| `Credits`        | `{"credits": N}`       | Once per connection: set how many blocks may be in flight. |
| `ReceivedChunk`  | `null`                 | One more block of credit, sent when a block has been played. |

`UpdateRecipe`, `UpdateControls`, `Pattern`, `Pause`, `Resume` and `EndSession` on this socket get an `Error` and change nothing. When playback disconnects, generation stops and the lead is cleared. The recipe, controls, pattern and model state stay, and the block sequence keeps climbing; the next playback connection continues from there.

**`GET /stream/input`** — the client.

| `type`           | `body`                                              | Effect |
|------------------|-----------------------------------------------------|--------|
| `UpdateRecipe`   | `{"recipe": {"jazz": 0.6, "flute": 0.3}}`           | Weighted blend of the prompts' MusicCoCa embeddings (cached), used from the next block on. |
| `UpdateControls` | `{"temperature"?: float, "top_k"?: int, "cfg_scales"?: {"musiccoca"?, "notes"?, "drums"?}}` | MRT2's `generate()` arguments under their own names (see Controls below), used from the next block on. |
| `Pattern`        | `{"notes"?: lane, "drums"?: lane, "id"?}`          | MRT2's per-frame NOTES / DRUMS input (see below). |
| `Pause` / `Resume` | `null`                                            | Stop generating (session and state kept) / continue. |
| `EndSession`     | `null`                                              | Reset the model state, recipe, controls, pattern, lead and block sequence, and close playback if it is open. |

`Credits` and `ReceivedChunk` on this socket get an `Error` and do not add credit. The input socket closing does not stop playback.

### Controls (`UpdateControls`)

Everything that steers sampling goes through `UpdateControls`, named as MRT2's `generate(conditioning, cfg_scales, temperature, top_k, ...)` names them. A message changes only the keys it carries; `null` puts a key back to the model default. A bad message gets an `Error` reply and changes nothing.

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

`Pattern` is a message on the input websocket. Note sources POST to the client's client_midi (port 8470). The full format, with a MIDI-file converter, is in [`superconductor_client/MIDI_MSG_PROTOCOL.md`](../superconductor_client/MIDI_MSG_PROTOCOL.md); the reference implementation is `expand_lane` / `Lane` in `mrt2_server.py`.
