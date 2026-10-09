Setup
-----

```bash
git clone https://github.com/dennisfarmer/superconductor.git superconductor_client
cd superconductor_client
make worktrees    # server, describe and superconductor_midi branches into ../superconductor_server, ../superconductor_describe, ../superconductor_midi
scripts/setup_env.sh                       # the sc_env conda env (client, MRT2 server, MIDI player) + MRT2 weights
make -C ../superconductor_describe pull    # the describe server's VLM, into Ollama (~3 GB)
```

Running everything
------------------

Each part is its own process, so use one terminal each. Run all `make` commands from `superconductor_client`.

| Part | Command | Port | What it does | Needed? |
|---|---|---|---|---|
| MRT2 server | `make server` (or on Lighthouse, below) | 9100 (9000 through the tunnel) | generates the music | yes |
| Describe server | `make describe` | 9200 | describes new objects and suggests their instruments and names (needs Ollama running) | optional |
| Client | `make client` / `make client-iphone`, or `make client-remote` / `make client-remote-iphone` | 8467, 8470 | camera, tracking, objects page; sends the prompts and plays the audio | yes |
| MIDI player | `make midi` | 8475 | plays `.mid` files through the music model (see `../superconductor_midi/README.md`) | optional |

### Local (mrt2_small on this Mac)

1. `make server`, and wait for `Serving mrt2_small (mlx) on port 9100`. The client connects once at startup and doesn't retry, so the server has to be up first.
2. `make describe`, with Ollama running (the app, or `ollama serve`). Without it, objects still work but get no description, suggested instrument or default name.
3. `make client` (Logitech C920) or `make client-iphone`. The objects page opens by itself.
4. Optional: `make midi`, then open http://localhost:8475, pick a song and press Play.

### Lighthouse (server on lh2300)

The server runs in your Lighthouse session on lh2300; everything else runs on the laptop.

1. On Lighthouse, in your session on lh2300:
    ```bash
    cd /scratch/aimusic_project_root/aimusic_project/shared_data/superconductor_server
    source .venv/bin/activate
    git pull                  # the server branch
    make server               # mrt2_base; or make server MODEL=mrt2_small
    ```
    Wait for `Serving mrt2_base (jax) on port 9100`. First-time setup (venv, weights) is in `../superconductor_server/README.md`.
2. On the laptop, open the tunnel and leave it running:
    ```bash
    ssh -N -L 9000:lh2300:9100 YOUR_UNIQNAME@lighthouse.arc-ts.umich.edu
    ```
3. `make describe` (with Ollama running), as in the local steps.
4. `make client-remote` (Logitech C920) or `make client-remote-iphone`. It connects to `ws://localhost:9000/stream`.
5. Optional: `make midi`, as in the local steps.

To stop: `q` in the camera window, `Ctrl-C` in the other terminals.

| Link | What it is |
|---|---|
| http://localhost:8467/objects | objects page: names, instruments, combos, play / pause |
| http://localhost:8475 | MIDI player: upload `.mid` files, play them at a chosen tempo |
| http://localhost:9100/tempo | tempo test page from the server (`:9000` with Lighthouse) |
| http://localhost:9100/conduct | conductor window: space bar = one beat (`:9000` with Lighthouse) |
| http://localhost:8470/status | client_midi, where note sources like the MIDI player send patterns |

### Troubleshooting

- **The music ignores the objects (e.g. it sounds like jazz piano):** the camera loop isn't running, so the client never sends its prompts, and the server plays its default `jazz` prompt. Check that the camera window shows video and that the objects page's Music row shows `Magenta RealTime 2 ...` rather than `not connected`. If the camera window is frozen, another app (Zoom, Photo Booth, a browser tab) may be holding the C920. Quit it, or replug the camera, then restart the client. Or use the `-iphone` target instead.
- **Notes from the MIDI player or Harmonic Atlas have no effect:** the server's terminal prints `WS unknown message type: Pattern`. That server is older than the notes input, so `git pull` and restart it.
- **The client says connection failed:** start the server (and the tunnel) first, then restart the client.

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

### NOTES / DRUMS from another program (client_midi)

While the client runs, it listens on http://localhost:8470 (`midi_port` in `collab.toml`). Other programs POST patterns of MRT2 NOTES / DRUMS input there, and the client forwards them to the server. The prompts still come from the objects. The protocol is in [MIDI_MSG_PROTOCOL.md](MIDI_MSG_PROTOCOL.md). Harmonic Atlas uses it with `make run-sc` (in `../harmonic_atlas`).

```bash
curl -s localhost:8470/status
curl -s -X POST localhost:8470/pattern -d '{"drums":{"loop":true,"steps":[{"frames":25,"first":1}]}}'   # a hit every second
curl -s -X POST localhost:8470/pattern -d 'null'                                                         # back to free
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