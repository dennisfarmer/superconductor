# SuperConductor Server

![diagram](media/diagram.jpeg)

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

`scheduler.py` runs on the cluster between `superconductor_server.py` (the
Magenta RT server) and the laptop client. It exists so the laptop doesn't have
to hold MagentaRT state or talk to the Magenta server directly.

## What it manages

- **A linked list of generated chunks.** Each node holds the audio samples,
  the MagentaRT generation state *after* that chunk, the recipe and style
  embedding used to generate it, and a UUID. The list is the playback queue.
- **A background generation worker.** Keeps the queue filled
  `--buffer_chunks` chunks ahead of the playback bar by POSTing to the
  Magenta server's `/generate_chunk` endpoint, passing the current tail's
  state so the next chunk continues seamlessly.
- **A recipe → style embedding cache.** The scheduler converts the laptop's
  weighted prompt dicts (e.g. `{"Rock": 0.6, "Guitar": 0.8}`) into style
  embeddings by calling the Magenta server's `/style` endpoint and caching
  the result, so the laptop never needs its own connection to Magenta.
- **The current playback position.** The laptop reports which chunk it is
  playing and how far into it the playback bar has progressed. The scheduler
  uses this to choose fork points when the recipe changes.
- **Fork-on-recipe-change.** When the recipe changes, the scheduler picks
  the last chunk it can safely keep (far enough ahead of the playback bar
  that there are at least `chunk_length_seconds` (2s) of audio left to mask
  the worst-case generation time, plus a one-chunk safety margin), truncates
  the queue past that point, and lets the worker refill from the new style.
  In-flight generations that started before the fork are detected via a
  `_generation_id` counter and discarded so stale audio is never appended.

## What it sends to the client

Only **audio samples and chunk ids**. MagentaRT state stays on the cluster.
Each `POST /next_chunk` response is JSON:

```json
{
  "id": "<uuid>",
  "audio": "<base64 float32 samples>",
  "shape": [num_samples, num_channels]
}
```

## HTTP API

All endpoints are POST and accept/return JSON. Defaults to port `9100`.

| Endpoint           | Body                                                       | Purpose |
|--------------------|------------------------------------------------------------|---------|
| `/start`           | `{"recipe": {...}}`                                        | Boots the worker with an initial recipe. Calling again behaves like `/update_recipe`. |
| `/next_chunk`      | *(empty)*                                                  | Blocks until a chunk is ready, then returns `{id, audio, shape}`. |
| `/update_recipe`   | `{"recipe": {...}, "chunk_id": "...", "offset_seconds": float}` | Switches the recipe and forks the queue using the reported playback position. |
| `/report_progress` | `{"chunk_id": "...", "offset_seconds": float}`             | Pushes playback position without changing the recipe. |
| `/stop`            | *(empty)*                                                  | Stops the worker and clears the queue. |

## How to run the scheduler on Lighthouse

```
python scheduler.py --magenta_url=http://localhost:8000 --port=9100 --buffer_chunks=5
```

The laptop reaches it via the SSH tunnel set up in the startup sequence
above (`localhost:9000 -> lh2300:9100`); `superconductor/magenta_client.py`
talks to `http://localhost:9000` by default.
