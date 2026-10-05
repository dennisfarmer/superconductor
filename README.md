Setup
-----

```bash
git clone https://github.com/dennisfarmer/superconductor.git superconductor_client
cd superconductor_client
make worktrees    # server and describe branches into ../superconductor_server, ../superconductor_describe
```

### Quick test: new-object flow

With Ollama running, use three terminals: `make server`, `make describe`, `make client-iphone` (or `make client`). The objects page opens at http://localhost:8467/objects.

What should happen:
1. A new object shows a dashed "? new" box, then `#1`. A card appears on the page and `library/1/` is created.
2. After about 5–10 s the card shows a description and an instrument, and you hear it.
3. Out of frame and back again: still `#1`.
4. After quitting (`q`) and restarting: still `#1`, with the same description and instrument.
5. A second object becomes `#2`.

### Quick test: tempo / conducting

Run `make server`, then `make client` (music only plays while a client is connected).

| Link | What it shows |
|---|---|
| http://localhost:8467/objects | Objects page (names, descriptions, instruments) |
| http://localhost:9100/tempo | Tempo test page: make the music match a bpm, see the current bpm |
| http://localhost:9100/conduct | Conductor window: space bar = one beat (also opens from the tempo page) |

With `make client-remote`, use port `9000` instead of `9100` (the SSH tunnel).

What should happen:
1. After about 4 s, "Model bpm" on the tempo page shows MRT2's tempo.
2. **Match this bpm** gradually brings "Heard bpm" to that bpm, without skips.
3. Tapping steadily in the conductor window makes the music drift toward your tempo; when you stop, it holds.
4. **Stop tempo control** eases the music back to MRT2's own tempo.

Start SuperConductor
--------------------

The system consists of three parts:

1.  **MagentaRT Server and Scheduler Interface (Lighthouse GPU)**
    
2.  **SSH tunnel (connect server → local)**
    
3.  **Client (local laptop)**
    

### 1\. Start the Server (Lighthouse)

- login to Lighthouse
- run `cd /scratch/aimusic_project_root/aimusic_project/shared_data/magenta_native/magenta-realtime`
- allocate gpu if not on a gpu session: `salloc --account=aimusic_project --partition=aimusic_project --gpus=1 --mem=64G --cpus-per-task=4 --time=00:15:00`
    - Adjust --time based on how long you need the server
    - The job will stop automatically after the time expires
    - You can also run exit to release resources early
- then, in seperate terminals, start the server and then the scheduler
    - for each terminal, activate environment with `source .venv/bin/activate`
    - `python superconductor_server.py` - runs on localhost:8000 by default
    - `python scheduler.py` - runs on localhost:9100 by default, client makes requests to scheduler

### 2\. Start SSH Tunnel (on your laptop)

Open a **new local terminal** and run:

`ssh -N -L 9000:lh2300:9100 YOUR_UNIQNAME@lighthouse.arc-ts.umich.edu`

This forwards:

`localhost:9000 → lighthouse:9100`

### 3\. Start the Client (local)

In your local project:

- `conda activate sc_env`
- `python3 superconductor/laptop.py`
- A window will open with the camera + UI
- **Press q (while the window is focused) to exit**

  

## Setup

```bash
conda env create -f environment.yml
conda activate sc_env
python -m pip install -e . --no-deps
# chmod +x bin/superconductor
```

# Other Commands

todo: reimplement these after we integrate gesture / interface additions

(see `[project.scripts]` in `./pyproject.toml`)

## Set Audio Device
```bash
sc-audio-device --list
# [1] USB-C to 3.5mm Headphone Jack Adapter (out=2)
# [3] BlackHole 2ch (out=2)
# [5] MacBook Pro Speakers (out=2)
# [6] Microsoft Teams Audio (out=1)
# [7] Steam Streaming Microphone (out=2)
# [8] Steam Streaming Speakers (out=2)
# [9] ZoomAudioDevice (out=2)

sc-audio-device --select 1
# selected [1] USB-C to 3.5mm Headphone Jack Adapter (out=2)
```

# SuperConductor Frontend Diagram

![diagram](media/diagram.jpeg)