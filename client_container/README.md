# Client container

One image runs the client-side services. The browser is the camera, the view, and the speakers. The music server stays outside the image, and the client connects to it by URL.

Singularity runs this image on lh2300. Docker builds the image. A note on running it with Docker is at the bottom.

## Open these

The gateway is the only port the browser uses. Open these on the laptop, through the port 8080 tunnel. `server.localhost` is the gateway name for the music server, on that same port. `midi.localhost`, `atlas.localhost`, and `describe.localhost` are the same kind of name. Chrome, Edge, and Firefox send `*.localhost` to this machine. Tracking has to be opened as `localhost`, not a `*.localhost` name, because the browser will only share the camera with `localhost`.

| Page | On the laptop |
|---|---|
| Tracking | http://localhost:8080 |
| Prompts | http://localhost:8080/objects |
| Conducting | http://localhost:8080/conduct/ |
| MIDI player | http://midi.localhost:8080 |
| Atlas | http://atlas.localhost:8080 |
| Describe | http://describe.localhost:8080/health |
| Tempo View | http://server.localhost:8080/tempo |
| Conductor | http://server.localhost:8080/conduct |
| Playback | ws://server.localhost:8080/stream/playback |
| Conducting, on the container machine | http://127.0.0.1:8480 |
| TouchDesigner, on the container machine | ws://127.0.0.1:8480/ws/hands/monitor |
| TouchDesigner, through the laptop tunnel | ws://localhost:8080/conduct/ws/hands/monitor |

## What starts

Each service listens on 127.0.0.1. Caddy on 8080 forwards by hostname. `client_container/entrypoint.sh` starts them. `SC_SERVICES` defaults to `client midi atlas conduct`. Add `describe` to that list to start Ollama and the describe server.

| Service | Port |
|---|---|
| Client (Tracking, Prompts, client_midi) | 8467, 8470 |
| Describe server, Ollama, and the SVGen icon model | 9200, 11434 |
| MIDI player | 8475 |
| Atlas app, Atlas engine | 8764, 8765 |
| Conducting | 8480 |
| Gateway | 8080 |

Writable files go under `/data`: the object library (`/data/client/library`), uploaded MIDI files, Atlas's session, and logs in `/data/run`. The image itself is read-only. It already contains the vision weights (`qwen2.5vl:3b` under `/opt/models/ollama`) and the parameter-icon weights (SVGen's Qwen2.5-3B under `/opt/models/svgen`). A `/data/ollama` that already has models is used instead of the copy in the image. Descriptions ask Ollama to drop the vision model when they finish (`--keep_alive 0`), so an icon can use the same GPU.

## Build a new image

These are the four images. Pushing a branch does not build one.

| Image | GHCR tag | Who publishes it | Pull on lh2300 |
|---|---|---|---|
| Client release | `ghcr.io/dennisfarmer/superconductor_client:client` | **Actions** → **release** → **Run workflow** | `make sif` |
| Server release | `ghcr.io/dennisfarmer/superconductor_server:server` | the same button | `make sif` in the server checkout |
| Client dev | `ghcr.io/dennisfarmer/superconductor_client:client-dev` | `make image-push` on a laptop | `make sif-dev` |
| Server dev | `ghcr.io/dennisfarmer/superconductor_server:server-dev` | `make image-push` in the server checkout | `make sif-dev` in the server checkout |

**Release.** The workflow GitHub shows is the one on the `client` branch. It takes the current `client`, `server`, and `atlas` commits, builds both release images, and opens a GitHub Release. The tag points at the client commit. The notes and the `release.json` asset name the server commit and the atlas commit in the client image. Before pressing the button, add bullets under `## [Unreleased]` in `CHANGELOG.md` on both branches; those sections are the release notes. The version defaults to the next patch (`v0.1.0` the first time). The build takes about 10–20 minutes. On lh2300, `make sif` in each checkout pulls the new release tags.

The release list keeps every version's commits. A new release moves the two release tags. The two dev tags stay until the next `make image-push`. `make client-singularity` pulls `:client` when `superconductor_client.sif` is absent, and leaves a file that is already there. `make sif` pulls again.

**Dev.** On a machine with Docker, `make image` builds `:client-dev` and `make image-push` sends it (log in first: `docker login ghcr.io`, with a token that has `write:packages`). On lh2300, `make sif-dev` pulls that tag into `superconductor_client-dev.sif`, and `make client-singularity-dev` runs it. The server checkout has the same pair: `make image-push`, then `make sif-dev` and `make server-singularity-dev`.

On lh2300, `make sif` and `make client-singularity` source `/scratch/aimusic_project_root/aimusic_project/shared_data/.singularity/singularity-setup.sh`. That loads Singularity, reads `ghcr.token` in that directory, and logs in to ghcr.io as `dennisfarmer`. The copy of the script at the top of this checkout is only a reference.

## Run it on lh2300

**Once**, in your Lighthouse session:

```bash
cd /scratch/aimusic_project_root/aimusic_project/shared_data
git clone -b server --single-branch https://github.com/dennisfarmer/superconductor.git superconductor_server

cd /scratch/aimusic_project_root/aimusic_project/shared_data
git clone -b client --single-branch https://github.com/dennisfarmer/superconductor.git superconductor_client
```

**Each time:**

1. On lh2300, start the server image and wait for `Serving mrt2_base (jax) on port 9100`:
   ```bash
   cd /scratch/aimusic_project_root/aimusic_project/shared_data/superconductor_server
   make server-singularity
   ```
2. On lh2300, start the client image:
   ```bash
   cd /scratch/aimusic_project_root/aimusic_project/shared_data/superconductor_client
   make client-singularity
   ```
3. On the laptop: `ssh -N -L 8080:lh2300:8080 YOUR_UNIQNAME@lighthouse.arc-ts.umich.edu`
4. Open Tracking from the table above and press **Start**.

`make client-singularity` uses the `cluster` entry in `superconductor_client/addresses.toml`. Programs in the container open port 9100 on the node. The laptop browser uses the table above. Singularity shares the node's network, so these ports belong to whoever starts the container. If another Ollama already has 11434, set `SINGULARITYENV_SC_OLLAMA_PORT=11435`.

After a new release is published: `make sif` in the server checkout and `make sif` in this checkout, then start them again. For a laptop dev image, `make sif-dev` and `make client-singularity-dev` (and `make server-singularity-dev` in the server checkout).

To play the music on this machine's speakers, from a checkout with the conda env: `make audio` asks where the server is, the same question as `make client`. `make audio-local` and `make audio-remote` skip it.

## Settings

| Variable | Default | |
|---|---|---|
| `SC_WHERE` | `cluster` | Which entry in `superconductor_client/addresses.toml`. `make client-singularity` uses `cluster`. |
| `SC_SERVICES` | `client midi atlas conduct` | Which services to start. Add `describe` for Ollama and the describe server. |
| `OLLAMA_URL` | the Ollama in the image | Another Ollama, for example `http://host.docker.internal:11434` |
| `DESCRIBE_MODEL` | `qwen2.5vl:3b` | Describe server's vision model. The image already has this one. A different name is pulled into `/data/ollama` on a machine that has internet. |
| `SC_OLLAMA_PORT` | `11434` | Port of the Ollama in the image |
| `SC_GATEWAY_PORT` | `8080` | Gateway port, and the links on the pages |

With Singularity, pass them as `--env NAME=value` or `SINGULARITYENV_NAME=value`.

## Docker, as a build step

Docker produces the `linux/amd64` image that Singularity pulls. The cluster has no Docker. GitHub Actions builds the two release images. `make image` builds the dev image on a machine that has Docker, and `make image-push` sends it to `:client-dev`. On a laptop that is not linux/amd64, that build runs under emulation and is slow.
