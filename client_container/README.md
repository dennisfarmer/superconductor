# Client container

One image runs the client-side services. The browser is the camera, the view, and the speakers. The music server stays outside the image, and the client connects to it by URL.

Singularity runs this image on lh2300. Docker builds the image. A note on running it with Docker is at the bottom.

## Open these

The gateway is the only port the browser uses. Chrome, Edge, and Firefox resolve `*.localhost`. The camera needs `http://localhost`, so open the page through the forwarded port.

| Open | What it is |
|---|---|
| http://localhost:8080 | Camera page. Press **Start** and allow the camera. |
| http://localhost:8080/conduct/ | Conducting. MediaPipe hands set the server's tempo. The music still plays on the camera page. |
| http://localhost:8080/objects | Objects: names, instruments, combos, Services. |
| http://midi.localhost:8080 | MIDI player |
| http://atlas.localhost:8080 | Atlas. Its engine is at `/engine`. |
| http://describe.localhost:8080/health | Describe server |
| http://server.localhost:8080/tempo | The music server's tempo page, and `/conduct` |

On the machine running the container, conducting is also at http://127.0.0.1:8480. TouchDesigner can listen at `ws://127.0.0.1:8480/ws/hands/monitor`, or through the laptop tunnel at `ws://localhost:8080/conduct/ws/hands/monitor`.

## What starts

Each service listens on 127.0.0.1. Caddy on 8080 forwards by hostname. `client_container/entrypoint.sh` starts them. `SC_SERVICES` defaults to `client describe midi atlas conduct`.

| Service | Port |
|---|---|
| Client (camera page, objects, client_midi, `/audio`) | 8467, 8470 |
| Describe server, and Ollama | 9200, 11434 |
| MIDI player | 8475 |
| Atlas app, Atlas engine | 8764, 8765 |
| Conducting | 8480 |
| Gateway | 8080 |

Writable files go under `/data`: the object library (`/data/client/library`), Ollama's models, uploaded MIDI files, Atlas's session, and logs in `/data/run`. The image itself is read-only.

## Build a new image

GitHub builds `ghcr.io/dennisfarmer/superconductor:client-web`. Pushing a branch does not rebuild it.

1. On GitHub: **Actions** → **client-web image** → **Run workflow**. The workflow on the `client` branch is the one GitHub shows. It checks out `client-web` and `atlas`. The build takes about 10–20 minutes.
2. On lh2300, in this checkout: `git pull && make sif`. That pull is:

   ```bash
   singularity pull --force superconductor_client-web.sif docker://ghcr.io/dennisfarmer/superconductor:client-web
   ```

The registry keeps the current image. Each build moves the `client-web` tag and deletes the previous one.

If the package is private, log in once on lh2300:

```bash
singularity remote login --username dennisfarmer docker://ghcr.io
```

Use a GitHub token with `read:packages`.

## Run it on lh2300

**Once**, in your Lighthouse session:

```bash
cd /scratch/aimusic_project_root/aimusic_project/shared_data
git clone -b server --single-branch https://github.com/dennisfarmer/superconductor.git superconductor_server
cd superconductor_server && python3.12 -m venv .venv && source .venv/bin/activate
pip install "magenta-rt[jax]==2.0.3" "jax[cuda12]" aiohttp
make models

cd /scratch/aimusic_project_root/aimusic_project/shared_data
git clone -b client-web --single-branch https://github.com/dennisfarmer/superconductor.git superconductor_client_web
module load singularity
export SINGULARITY_CACHEDIR=/scratch/aimusic_project_root/aimusic_project/$USER/.singularity
cd superconductor_client_web && make sif
```

**Each time:**

1. On lh2300, start the music server and wait for `Serving mrt2_base (jax) on port 9100`:
   ```bash
   cd /scratch/aimusic_project_root/aimusic_project/shared_data/superconductor_server
   source .venv/bin/activate && make server
   ```
2. On lh2300, start the client:
   ```bash
   cd /scratch/aimusic_project_root/aimusic_project/shared_data/superconductor_client_web
   module load singularity && make client-singularity
   ```
3. On the laptop: `ssh -N -L 8080:lh2300:8080 YOUR_UNIQNAME@lighthouse.arc-ts.umich.edu`
4. Open http://localhost:8080 and press **Start**.

`make client-singularity` talks to the music server on the same node (`ws://localhost:9100/stream`). Singularity shares the node's network, so these ports belong to whoever starts the container. If another Ollama already has 11434, set `SINGULARITYENV_SC_OLLAMA_PORT=11435`.

After a new image is published: `git pull && make sif` on lh2300, then start it again.

To play the music on the laptop's speakers instead of the browser, from a checkout with the conda env: `make audio AUDIO_URL=ws://localhost:8080/audio`. Ctrl-C hands playback back to the page.

## Settings

| Variable | Default | |
|---|---|---|
| `SC_SERVER_URL` | `ws://host.docker.internal:9000/stream` | Music server websocket. `make client-singularity` uses `ws://localhost:9100/stream` when this default is still set. |
| `SC_SERVICES` | `client describe midi atlas conduct` | Which services to start |
| `OLLAMA_URL` | the Ollama in the image | Another Ollama, for example `http://host.docker.internal:11434` |
| `DESCRIBE_MODEL` | `qwen2.5vl:3b` | Describe server's vision model |
| `SC_OLLAMA_PORT` | `11434` | Port of the Ollama in the image |
| `SC_GATEWAY_PORT` | `8080` | Gateway port, and the links on the pages |

With Singularity, pass them as `--env NAME=value` or `SINGULARITYENV_NAME=value`.

## Docker, as a build step

Docker produces the `linux/amd64` image that Singularity pulls. The cluster has no Docker. GitHub Actions is the build we use. `make image` does the same build on a machine that has Docker (`make image-push` also sends it to ghcr.io). On an Apple Silicon Mac that build runs under emulation and is slow.

Running the app from that image with Docker is a way to look at it on a laptop, not how it runs on lh2300:

```bash
make client-docker
# music server on this Mac:
make client-docker SC_SERVER_URL=ws://host.docker.internal:9100/stream
```

Then open http://localhost:8080. `make client-docker-stop` stops it. The image is `linux/amd64`, so Docker Desktop on an Apple Silicon Mac runs it under emulation and without a GPU. For day-to-day work on a Mac, run `make client` instead. That serves the camera page at http://localhost:8467/.
