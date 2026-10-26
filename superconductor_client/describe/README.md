# SuperConductor Describe Server

Writes a one-time rich text description of each new object the client sees, using a Qwen VLM, and suggests an instrument prompt and a default name from that description. The client (`superconductor_client`) sends each object's picture here once and stores the result in `superconductor_client/tracking/library/<id>/object.json`. Recognizing objects (DINOv2) stays in the client; this server is only called once per object, or when you press "describe again" on the objects page.

It runs as its own process so the VLM doesn't compete with the camera loop, and so it can move to a cluster GPU node later. The VLM itself runs in [Ollama](https://ollama.com); `describe_server.py` holds the prompts and talks to Ollama's HTTP API, so descriptions only need `aiohttp`. Parameter icons (`POST /svg`) are a separate local model, described below.

### Setup

- **Laptop:** `make setup DESCRIBE=1` installs mlx-lm, downloads the icon weights, and runs `ollama pull` for `qwen2.5vl:3b` (about 3 GB). Ollama itself has to be installed already.
- **Lighthouse:** the client Singularity image contains `qwen2.5vl:3b` and the SVGen weights. It starts Ollama only when `describe` is in `SC_SERVICES`. A checkout without the image needs Ollama's Linux `.tgz` from https://github.com/ollama/ollama/releases and `OLLAMA_MODELS` on scratch, then `make pull` on a login node.

### Startup

- locally: Ollama already running, then `make describe` (from here or from `superconductor_client`), listening on port 9106. `make client` does not start this. `make client DESCRIBE=1` does. The client uses `http://localhost:9106` by default.
- on Lighthouse:
    - `salloc --account=aimusic_project --partition=aimusic_project --gpus=1 --mem=32G --cpus-per-task=4 --time=01:00:00`
    - `ollama serve`, then `make describe`
    - on the laptop: `ssh -N -L 9106:lh2300:9106 YOUR_UNIQNAME@lighthouse.arc-ts.umich.edu`; the client's default URL then reaches the cluster. Use another local port with `sc-tracking --describe-server http://localhost:<port>`.

Other flags: `make describe MODEL=qwen2.5vl:7b PORT=... ARGS="--keep_alive 0"` (`--keep_alive 0` frees the GPU right after each request, which helps when the VLM shares the laptop's GPU with MRT2), or `python describe_server.py --help`.

A second GPU can be given to this server alone. `Exclusive_Process` is per card, and `CUDA_VISIBLE_DEVICES` is per process. The music server keeps the first L40S. Ollama and this process both have to see only the second, or one of them will take the music server's card:

```bash
CUDA_VISIBLE_DEVICES=1 ollama serve
CUDA_VISIBLE_DEVICES=1 make describe
```

The client stays on CPU and does not open either GPU. In the container image the same split is `SC_DESCRIBE_GPU=1`.

`make client DESCRIBE=1` and `make client-singularity DESCRIBE=1` start this server only when a CUDA device can be opened. If the L40S is already taken, or there is no GPU, they print that the describe server needs a GPU and continue as `make client` or `make client-singularity`. Ollama is a separate process from this one. Under `Exclusive_Process` only one of them can hold a given card.

### HTTP API

| Request | Body | Response |
|---|---|---|
| `POST /describe` | `{"image": "<base64 jpeg>"}` | `{"description": "..."}` |
| `POST /instrument` | `{"description": "..."}` | `{"instrument": "fiery thundering taiko drums"}` |
| `POST /name` | `{"description": "..."}` | `{"name": "green plush dinosaur"}` (the default name on the objects page) |
| `POST /svg` | `{"prompt": "..."}` | `{"svg": "<svg ...>...</svg>"}` (an icon for that prompt; optional, from the prompts page) |
| `GET /health` | | `{"model": "qwen2.5vl:3b", "backend": "ollama", "host": "<hostname>", "svg_model": "default.svg", "svg_backend": "file", "svg_ready": true}` |

Requests are handled one at a time. The prompt templates are at the top of `describe_server.py`.

### Icons (`POST /svg`)

Every prompt returns `default.svg` in this folder, a wand drawn once by SVGen. The request does not load a model and does not take a GPU. Descriptions still go to Ollama, which is the process that can hold the card.
