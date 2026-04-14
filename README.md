# SuperConductor Server

# First-time Setup (already done on Lighthouse)

```
git clone https://github.com/dennisfarmer/superconductor.git
cd superconductor
git checkout server

# ------------------------------------------------------------
# if fresh install, setup magenta realtime
# (remove .git so magenta-realtime/ can be gitignored and not
# added as a submodule)

git clone https://github.com/magenta/magenta-realtime.git
rm -rf magenta-realtime/.git
# (or) mv magenta-realtime/.git ../.git_magenta_realtime
# then follow installation steps in magenta-realtime/README.md
# ------------------------------------------------------------

cp scheduler.py magenta-realtime/
cp superconductor_server.py magenta-realtime/
```


### Startup Sequence:

- login to Lighthouse
- run `cd /scratch/aimusic_project_root/aimusic_project/shared_data/magenta_native/magenta-realtime`
- allocate gpu if not on a gpu session: `salloc --account=aimusic_project --partition=aimusic_project --gpus=1 --mem=64G --cpus-per-task=4 --time=00:15:00`
    - Adjust --time based on how long you need the server
    - The job will stop automatically after the time expires
    - You can also run exit to release resources early
- then, in seperate terminals, start the server and then the scheduler
    - for each terminal, activate environment with `source .venv/bin/activate`
    - `python superconductor_server.py` - runs on localhost:8000 by default
    - `python scheduler.py` - runs on localhost:9100 by default
- on client-side (laptop), start ssh tunnel:
    - `ssh -N -L 9000:lh2300:9100 YOUR_UNIQNAME@lighthouse.arc-ts.umich.edu`
    - this forwards: `localhost:9000 → lighthouse:9100`
- on client-side (laptop) in a seperate terminal, activate environment and start laptop.py
    - `conda activate sc_env`
    - `python3 superconductor/laptop.py`
    - a window will open with the camera + UI
    - **press q (while the window is focused) to exit**


# Scheduler Server

`scheduler.py` runs on the cluster between `superconductor_server.py` (the Magenta RT server) and the laptop client.

`scheduler.py` supports two wire protocols, selected with the
`--server_type` flag:

- **`websocket` (default)** — a push-based JSON/binary protocol for the streaming client `superconductor/magenta_client_stream.py`, which provides an interface very similar to the original magenta-realtime `server.py`.
- **`http`** — a request/response JSON API for `superconductor/magenta_client.py`.

Everything below focuses on the websocket variant since it is what the current client code is using.

## What it manages

- **A linked list of generated chunks.** Each node holds the audio
  samples, the MagentaRT generation state *after* that chunk, the recipe
  and style embedding used to generate it, and a UUID. The list is the
  playback queue.
- **A background generation worker.** Keeps the queue filled
  `--buffer_chunks` chunks ahead of the playback bar by POSTing to the
  Magenta server's `/generate_chunk` endpoint, passing the current tail's
  state so the next chunk continues seamlessly.
- **Recipe → style embedding on the cluster.** Clients send weighted
  prompt dicts (e.g. `{"Rock": 0.6, "Guitar": 0.8}`); the scheduler
  resolves each prompt via the Magenta server's `/style` endpoint,
  caches the result, and mixes the weighted sum. The laptop never makes
  its own call to `/style` and never handles embeddings directly.
- **Fork-on-recipe-change.** When the recipe changes, the scheduler
  picks the last chunk it can safely keep (far enough ahead of the
  playback bar that there are at least 2s of audio left to mask the
  worst-case generation time, plus a one-chunk safety margin), truncates
  the queue past that point, and lets the worker refill from the new
  style. In-flight generations that started before the fork are detected
  via a `_generation_id` counter and discarded so stale audio is never
  appended.

## What it sends to the client

Only **audio samples, tagged with a chunk UUID**. MagentaRT state stays on the cluster. Each audio frame is a single binary websocket message with the layout:

```
[16 bytes: chunk UUID (uuid.UUID.bytes)][rest: float32 LE samples, row-major (num_samples, num_channels)]
```

The client decodes it with:

```python
chunk_id = str(uuid.UUID(bytes=msg[:16]))
audio = np.frombuffer(msg[16:], dtype="<f4").reshape(-1, 2)
```

The UUID lets the client tell the scheduler which chunk it is currently playing (via `UpdateRecipe` or `UpdateProgress`) so that fork points can be picked relative to the real playback bar.

## Websocket protocol

The client opens a websocket at `GET /stream` and drives generation with
a pull model. All client → server messages are JSON text frames:

| `type`           | `body`                                                                | Effect |
|------------------|-----------------------------------------------------------------------|--------|
| `StartSession`   | `null`                                                                | Reset any prior session; arm for a new one. |
| `UpdateRecipe`   | `{"recipe": {...}, "chunk_id"?: "...", "offset_seconds"?: float}`     | Set (or change) the recipe. Before the first `UpdatePlayback PLAYING`, this arms the scheduler with an initial recipe (the `chunk_id` / `offset_seconds` fields are ignored, and should be omitted, when no audio has been played yet). After playback has started, the scheduler builds the new embedding and forks the queue using the optional `chunk_id` / `offset_seconds` fields to pick the fork point. |
| `UpdateProgress` | `{"chunk_id": "...", "offset_seconds": float}`                        | Push-only playback-position update. Same `chunk_id` / `offset_seconds` fields as `UpdateRecipe`, but without touching the recipe — for clients that want to keep the fork point accurate between recipe changes. |
| `UpdatePlayback` | `{"state": "PLAYING"}`                                                | Boots the scheduler worker and immediately sends the first audio chunk. Requires a prior `UpdateRecipe`. |
| `ReceivedChunk`  | `null`                                                                | "Send me one more chunk" — server replies with a binary audio frame. |
| `EndSession`     | `null`                                                                | Stop the scheduler and close the session. |

Server -> client messages are all binary audio frames (see above); the server never sends JSON back.

## How to run it on Lighthouse

On the cluster, pointed at the Magenta server:

```
python scheduler.py --magenta_url=http://localhost:8000 --port=9100 --buffer_chunks=5
```

On the laptop, forward port `9100` and launch `laptop.py`:

```
ssh -N -L 9000:lh2300:9100 YOUR_UNIQNAME@lighthouse.arc-ts.umich.edu
python3 superconductor/laptop.py
```

The SSH tunnel maps `localhost:9000 → lh2300:9100`, which is where the
laptop-side client expects the scheduler.

### HTTP variant

If you want the JSON request/response API instead (for debugging or for the non-streaming `magenta_client.py`), run the same binary with `--server_type=http`. It exposes `POST /start`, `POST /next_chunk`, `POST /update_recipe`, `POST /report_progress`, and `POST /stop` with JSON-encoded audio in the `/next_chunk` response. 
