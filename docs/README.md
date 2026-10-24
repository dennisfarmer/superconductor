Running client-web
==================

Pick one of the four setups below and follow its steps top to bottom. Each list is complete.
In every setup the browser (Chrome) is the camera, the view and the speakers.

## 1. Everything on the Mac

**One time:**
1. `git clone https://github.com/dennisfarmer/superconductor.git superconductor_client`
2. `cd superconductor_client && make worktrees` (creates `../superconductor_client_web`, `../superconductor_server`, ...)
3. `setup_env.sh` (the `sc_env` conda env and the MRT2 weights)
4. `cd ../superconductor_client_web && make web-deps` (onnxruntime-web for the page)

**Each time,** from `superconductor_client_web`:
1. Terminal 1: `make server`. Wait for `Serving mrt2_small (mlx) on port 9100`.
2. Terminal 2: `make client`. The first run also exports the page's detector model (about 30 s).
3. Optional, terminal 3: `make describe` (with Ollama running) for object descriptions and instruments.
4. Open http://localhost:8467 in Chrome and press **Start**. Allow the camera.

**Check:**
- The status line says `detecting here (WebGPU, N ms)`, and boxes appear over the video.
- Tick **compatibility mode**: the status says `detecting in the client`, and the boxes keep coming.
- Terminal 4: `make audio`. The sound moves to the Mac's output and the page says `sound plays in sc-audio`. Ctrl-C hands it back to the page.
- `make test` runs the unit tests.

**Stop:** Ctrl-C in each terminal.

## 2. Client on the Mac, music server on lh2300

**One time, on the Mac:**
1. `git clone https://github.com/dennisfarmer/superconductor.git superconductor_client`
2. `cd superconductor_client && make worktrees` (creates `../superconductor_client_web`, `../superconductor_server`, ...)
3. `setup_env.sh` (the `sc_env` conda env and the MRT2 weights)
4. `cd ../superconductor_client_web && make web-deps` (onnxruntime-web for the page)

**One time, on lh2300** (in your session there):
1. `cd /scratch/aimusic_project_root/aimusic_project/shared_data`
2. If `superconductor_server` isn't there yet: `git clone -b server --single-branch https://github.com/dennisfarmer/superconductor.git superconductor_server`
3. `cd superconductor_server && python3.12 -m venv .venv && source .venv/bin/activate`
4. `pip install "magenta-rt[jax]==2.0.3" "jax[cuda12]" aiohttp`
5. `make models`

**Each time:**
1. lh2300: `cd /scratch/aimusic_project_root/aimusic_project/shared_data/superconductor_server && source .venv/bin/activate && git pull && make server`. Wait for `Serving mrt2_base (jax) on port 9100`.
2. Mac, terminal 1: `ssh -N -L 9000:lh2300:9100 YOUR_UNIQNAME@lighthouse.arc-ts.umich.edu` (leave it running).
3. Mac, terminal 2, in `superconductor_client_web`: `make client-remote`.
4. Open http://localhost:8467 in Chrome and press **Start**.

**Check:**
- The status line says `detecting here (WebGPU, N ms)`, and boxes appear over the video.
- Tick **compatibility mode**: the status says `detecting in the client`, and the boxes keep coming.
- Another terminal, in `superconductor_client_web`: `make audio`. The sound moves to the Mac's output and the page says `sound plays in sc-audio`. Ctrl-C hands it back to the page.

**Stop:** Ctrl-C in each terminal, and on lh2300.

## 3. Client on lh2300

The client runs in a Singularity container. Build and run steps: [client_container/README.md](../client_container/README.md).

Protocols: `CAMERA_PROTOCOL.md` (page ↔ client), `AUDIO_PROTOCOL.md` (client → players).

### Troubleshooting

- **The music ignores the objects (e.g. it sounds like jazz piano):** the camera loop isn't running, so the client never sends its prompts, and the server plays its default `jazz` prompt. Check that the camera page shows video and that the objects page's Music row shows `Magenta RealTime 2 ...` rather than `not connected`. If the video is frozen, another app (Zoom, Photo Booth, another tab) may be holding the camera; quit it or pick another camera on the page.
- **Notes from the MIDI player or SuperConductor Atlas have no effect:** the server's terminal prints `WS unknown message type: Pattern`. That server is older than the notes input, so `git pull` and restart it.
- **The client says connection failed:** start the server (and the tunnel) first, then restart the client.

### Quick test: new-object flow

With Ollama running, use three terminals: `make server`, `make describe`, `make client`; open http://localhost:8467 and http://localhost:8467/objects.

What should happen:
1. A new object shows a dashed "? new" box, then `#1`. A card appears on the page and `superconductor_client/tracking/library/1/` is created.
2. After about 5–10 s the card shows a description and an instrument, and you hear it.
3. Out of frame and back again: still `#1`.
4. After quitting (Ctrl-C) and restarting: still `#1`, with the same description and instrument.
5. A second object becomes `#2`.

### Quick test: tempo / conducting

Run `make server`, then `make client` (music only plays while a client is connected).

| Link | What it shows |
|---|---|
| http://localhost:8467/objects | Objects page (names, descriptions, instruments) |
| http://localhost:9100/tempo | Tempo View page: make the music match a bpm, see the current bpm |
| http://localhost:9100/conduct | Conductor window: space bar = one beat (also opens from the tempo page) |

With the server on lh2300 (tunnel), use port `9000` instead of `9100`.

What should happen:
1. After about 4 s, "Model bpm" on the tempo page shows MRT2's tempo.
2. **Match this bpm** gradually brings "Heard bpm" to that bpm, without skips.
3. Tapping steadily in the conductor window makes the music drift toward your tempo; when you stop, it holds.
4. **Stop tempo control** eases the music back to MRT2's own tempo.

### NOTES / DRUMS from another program (client_midi)

While the client runs, it listens on http://localhost:8470 (`midi_port` in `collab.toml`). Other programs POST patterns of MRT2 NOTES / DRUMS input there, and the client forwards them to the server. The prompts still come from the objects. The protocol is in [MIDI_MSG_PROTOCOL.md](MIDI_MSG_PROTOCOL.md). SuperConductor Atlas uses it with `make run` (in `../superconductor_atlas`, the `atlas` branch).

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

