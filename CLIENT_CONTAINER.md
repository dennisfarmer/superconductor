SuperConductor in the browser (container)
=========================================

The client-side services run in one container, and everything that touches hardware is done by the browser: it captures the camera, shows the annotated view and plays the music. This works the same on Windows, Linux and the cluster. The MRT2 server is not in the container. It runs where it runs now (lh2300, or `make server` on a Mac), and the client connects to it by URL.

```
Dockerfile ──docker build──▶ image ──push──▶ ghcr.io/dennisfarmer/superconductor:client-web
                                                  │
                    laptop / Windows PC: docker run --gpus all -p 8080:8080 …
                    lh2300:              singularity pull docker://ghcr.io/… → .sif
                                         singularity run --nv …
```

What runs inside, each as its own process on its usual port (127.0.0.1 only):

| Service | Port | Started with |
|---|---|---|
| Client (`sc-collab --web`) | 8467 (pages), 8470 (client_midi, `/audio`) | `container/entrypoint.sh` |
| Describe server + Ollama | 9200, 11434 | same |
| MIDI player | 8475 | same |
| SuperConductor Atlas: app, sink engine | 8000, 8765 | same |
| Gateway (Caddy) | 8080, the only port the browser uses | same |

The gateway only forwards: it picks the service by hostname and passes requests and websockets through unchanged. Without it every service still works on its own port inside the container.

| Open | What it is |
|---|---|
| http://localhost:8080 | camera page: press **Start**, allow the camera; the view and the music come back here |
| http://localhost:8080/objects | objects page (names, instruments, combos, Services tab) |
| http://midi.localhost:8080 | MIDI player |
| http://atlas.localhost:8080 | SuperConductor Atlas (the page reaches its engine at `/engine`) |
| http://describe.localhost:8080/health | describe server |
| http://server.localhost:8080/tempo | the MRT2 server's tempo page (`/conduct` too) |

`*.localhost` works in Chrome, Edge and Firefox. Safari doesn't resolve it, so on a Mac use Chrome (or run natively, see README.md). The camera only works on `http://localhost`, so open the page through the forwarded port, not through a hostname.

### Windows / Linux laptop (Docker)

1. Open the tunnel to the MRT2 server on lh2300 and leave it running (the same tunnel as for `make client-remote`):
    ```bash
    ssh -N -L 9000:lh2300:9100 YOUR_UNIQNAME@lighthouse.arc-ts.umich.edu
    ```
2. Run the container. With an NVIDIA GPU (Windows: Docker Desktop with WSL2) keep `--gpus all`; without one, leave it out and tracking runs on the CPU.
    ```bash
    docker run --rm -it --gpus all -p 8080:8080 -v sc-data:/data \
        --add-host=host.docker.internal:host-gateway \
        ghcr.io/dennisfarmer/superconductor:client-web
    ```
    In PowerShell, put it on one line (or use a backtick instead of `\`). The first run pulls the describe model (~3 GB) into the `sc-data` volume.
3. Open http://localhost:8080 and press **Start**.

`SC_SERVER_URL` defaults to `ws://host.docker.internal:9000/stream`, the tunnel on the laptop. Pass `-e SC_SERVER_URL=...` for another server.

### lh2300 (Singularity)

On lh2300, in your Lighthouse session (or `make sif` / `make client-singularity` from a `client-web` checkout):
```bash
singularity pull --force superconductor_client-web.sif docker://ghcr.io/dennisfarmer/superconductor:client-web
mkdir -p /scratch/aimusic_project_root/aimusic_project/shared_data/sc_data_$USER
singularity run --nv \
    --bind /scratch/aimusic_project_root/aimusic_project/shared_data/sc_data_$USER:/data \
    --env SC_SERVER_URL=ws://localhost:9100/stream \
    superconductor_client-web.sif
```
`make client-singularity` connects to the MRT2 server on the same node (`ws://localhost:9100/stream`), so start `make server` in the server directory first (see README.md).

On the laptop:
```bash
ssh -N -L 8080:lh2300:8080 YOUR_UNIQNAME@lighthouse.arc-ts.umich.edu
```
and open http://localhost:8080.

Singularity shares the node's network, so the container's ports are the node's ports. Only one person at a time can run it on lh2300 with the default ports. If someone else's Ollama already has 11434, set `SINGULARITYENV_SC_OLLAMA_PORT=11435`.

### On a Mac

The image is `linux/amd64`, so Docker Desktop on an Apple Silicon Mac runs it under emulation, without a GPU. This is fine for trying it out; for real use, run natively (`make client`). To try it with the server and Ollama running natively on the Mac:
```bash
make image
make client-docker SC_SERVER_URL=ws://host.docker.internal:9100/stream OLLAMA_URL=http://host.docker.internal:11434
```
Without the container, `make client-web` runs the same camera page on http://localhost:8467/ (the music comes from client_midi on 8470).

### Settings (environment variables)

| Variable | Default | |
|---|---|---|
| `SC_SERVER_URL` | `ws://host.docker.internal:9000/stream` | MRT2 server websocket |
| `SC_SERVICES` | `client describe midi atlas` | which services to start |
| `OLLAMA_URL` | (bundled Ollama) | use another Ollama, e.g. `http://host.docker.internal:11434` |
| `DESCRIBE_MODEL` | `qwen2.5vl:3b` | describe server's vision model |
| `SC_OLLAMA_PORT` | `11434` | port of the bundled Ollama |
| `SC_GATEWAY_BIND` | `0.0.0.0` | gateway address |
| `SC_GATEWAY_PORT` | `8080` | gateway port (and the links on the pages) |

With Singularity, pass them as `--env NAME=value` or `SINGULARITYENV_NAME=value`.

Everything the services write goes under `/data`: the object library (`client/library`), Ollama's models (`ollama`), uploaded MIDI files (`midi/songs`), Atlas's session (`atlas`), logs (`run`). The image itself is read-only.

### Building for the cluster (`linux/amd64`)

The cluster has no Docker, so the image is built elsewhere and pulled from ghcr.io. The cluster is x86_64 while the Mac is ARM (arm64), so a plain `docker build` on the Mac produces an image the cluster can't run. `make image` builds for the cluster's architecture instead, from this branch and the `../superconductor_describe`, `../superconductor_midi` and `../superconductor_atlas` worktrees (`make worktrees`):

```bash
make image        # docker buildx build --platform linux/amd64 ... --load
make image-push   # the same, pushed to ghcr.io (docker login ghcr.io first)
```

On the Mac this runs under emulation (QEMU), so it is slow, especially installing torch. It's fine for an occasional build, not for every change.

### GitHub Actions

`.github/workflows/client-web-image.yml` builds the image on GitHub's x86_64 runners and pushes it to ghcr.io on every push to `client-web`, so nobody builds by hand. It takes the describe server, the MIDI player and Atlas from the `describe`, `midi_experiment_dennis` and `atlas` branches as they are. To rebuild without pushing (e.g. after those change), open the repo's **Actions** tab on GitHub, pick **client-web image**, and press **Run workflow**. Anyone with write access to the repo can do this; nobody needs Docker or a token. Then, on lh2300, `make sif` again.

ghcr.io holds only the current image. Each build moves the `client-web` tag to the new image, and the workflow then deletes the old one, which is now untagged. That's why the builds turn off provenance and SBOM attestations: with them on, the tag would point at untagged child manifests, and the cleanup would delete those too and break the image. After a `make image-push`, the old image stays until the next workflow run.

If the ghcr.io package is private, log in first with `singularity remote login --username <github user> docker://ghcr.io`, using a GitHub token with `read:packages`.

### SuperConductor Atlas

Atlas runs in SuperConductor mode: its engine is a sink that sends notes and drums to client_midi, and SuperConductor makes the sound. In this mode Atlas doesn't offer restarting the engine or the LLM path descriptions (local mode only). Its Quit button stops Atlas alone; the other services keep running. Logs: `/data/run/atlas.log`, `/data/run/atlas-engine.log`.

## Browser detection

The camera page detects and tracks objects itself (YOLOE-11s + BoT-SORT on WebGPU: `vision_js/`, `tracker_js/`). It sends the client its detections with each frame, and the client sends back what to draw over the video (`CAMERA_PROTOCOL.md`). No pictures come back down.

- **Supported browsers:** Chrome/Edge on Windows and macOS, Safari 26, Firefox 141+ on Windows.
- **Without WebGPU, or with "compatibility mode" ticked:** the page sends plain frames and the client runs YOLOE + BoT-SORT itself, on the container's GPU or CPU.
- **The page's status line** shows where detection runs and the processing delay.
- **The image** contains the detector (`/opt/web`: onnxruntime-web and the ONNX model for `[tracking] classes`). The entrypoint copies it to `/data/web`, and the client exports it again there if the classes change.
- **Playing the music elsewhere:** `sc-audio` (`make audio`, on the machine with the speakers) takes over playback from the page; `AUDIO_PROTOCOL.md` describes the protocol. Through the gateway: `make audio AUDIO_URL=ws://localhost:8080/audio`.
