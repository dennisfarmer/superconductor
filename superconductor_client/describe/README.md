# SuperConductor Describe Server

Writes a one-time rich text description of each new object the client sees, using a Qwen VLM, and suggests an instrument prompt and a default name from that description. The client (`superconductor_client`) sends each object's picture here once and stores the result in `superconductor_client/tracking/library/<id>/object.json`. Recognizing objects (DINOv2) stays in the client; this server is only called once per object, or when you press "describe again" on the objects page.

It runs as its own process so the VLM doesn't compete with the camera loop, and so it can move to a cluster GPU node later. The VLM itself runs in [Ollama](https://ollama.com); `describe_server.py` holds the prompts and talks to Ollama's HTTP API, so descriptions only need `aiohttp`. Parameter icons (`POST /svg`) are a separate local model, described below.

### Setup

- **Laptop:** install Ollama, then `make pull` (downloads `qwen2.5vl:3b`, about 3 GB). `make setup` in the client checkout installs mlx-lm and downloads the icon weights.
- **Lighthouse:** the client Singularity image starts Ollama, and it already contains `qwen2.5vl:3b` and the SVGen weights. `make client-singularity` is the way to run it. A checkout without the image can still do this by hand: Ollama's Linux `.tgz` from https://github.com/ollama/ollama/releases, `OLLAMA_MODELS` on scratch, `make setup` (the `.venv` and the icon weights), then `ollama serve` and `make pull` on a login node.

### Startup

- locally: Ollama running (the app, or `ollama serve`), then `make describe` (from here or from `superconductor_client`), listening on port 9200. The client uses `http://localhost:9200` by default.
- on Lighthouse:
    - `salloc --account=aimusic_project --partition=aimusic_project --gpus=1 --mem=32G --cpus-per-task=4 --time=01:00:00`
    - `ollama serve &` then `make describe`
    - on the laptop: `ssh -N -L 9200:lh2300:9200 YOUR_UNIQNAME@lighthouse.arc-ts.umich.edu`; the client's default URL then reaches the cluster. Use another local port with `sc-tracking --describe-server http://localhost:<port>`.

Other flags: `make describe MODEL=qwen2.5vl:7b PORT=... ARGS="--keep_alive 0"` (`--keep_alive 0` frees the GPU right after each request, which helps when the VLM shares the laptop's GPU with MRT2), or `python describe_server.py --help`.

### HTTP API

| Request | Body | Response |
|---|---|---|
| `POST /describe` | `{"image": "<base64 jpeg>"}` | `{"description": "..."}` |
| `POST /instrument` | `{"description": "..."}` | `{"instrument": "fiery thundering taiko drums"}` |
| `POST /name` | `{"description": "..."}` | `{"name": "green plush dinosaur"}` (the default name on the objects page) |
| `POST /svg` | `{"prompt": "..."}` | `{"svg": "<svg ...>...</svg>"}` (an icon for that prompt; optional, from the prompts page) |
| `GET /health` | | `{"model": "qwen2.5vl:3b", "backend": "ollama", "host": "<hostname>", "svg_model": "...", "svg_backend": "mlx" or "transformers", "svg_ready": false}` |

Requests are handled one at a time. The prompt templates are at the top of `describe_server.py`.

### Icons (`POST /svg`)

Parameter icons are not the VLM and not Ollama. They come from [SVGen](https://github.com/gitcat-404/SVGen)'s Qwen2.5-3B (`gitcat-404/SVGen-Qwen2.5-3B-Instruct`), a plain Qwen2 checkpoint fine-tuned to write SVG. The same weights run two ways, and setup installs the one that matches the machine:

- **macOS:** [mlx-lm](https://github.com/ml-explore/mlx-lm) 0.31.2, pinned to the same MLX as magenta-rt. `make setup` installs it and downloads the weights.
- **Cluster:** transformers, in the image and in the Linux `.venv` from `make setup`. The image loads `/opt/models/svgen` with `--svg-backend transformers`. On a V100 that is float16 (the card has no bfloat16). The weights load on the first `POST /svg` (`/health` reports `svg_ready`).

`make pull-svg` downloads the same checkpoint into the Hugging Face cache. On lh2300, set `HF_HOME` to scratch first so it stays off your home quota. The published image has to be rebuilt (Actions → client image, then `make sif`) before a running `.sif` includes these weights.
