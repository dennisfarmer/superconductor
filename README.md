# SuperConductor Describe Server

Writes a one-time rich text description of each new object the client sees, using a Qwen VLM, and suggests an instrument prompt and a default name from that description. The client (`superconductor_client`) sends each object's picture here once and stores the result in `library/<id>/object.json`. Recognizing objects (DINOv2) stays in the client; this server is only called once per object, or when you press "describe again" on the objects page.

It runs as its own process so the VLM doesn't compete with the camera loop, and so it can move to a cluster GPU node later. The VLM itself runs in [Ollama](https://ollama.com); `describe_server.py` only holds the prompts and talks to Ollama's HTTP API, so its only Python dependency is `aiohttp`.

### Setup

- **Mac:** install Ollama (the app, or `brew install ollama`), then `make pull` (downloads `qwen2.5vl:3b`, about 3 GB).
- **Lighthouse (docs only, not run yet):**
    - Ollama ships as a user-space binary: download the Linux `.tgz` from https://github.com/ollama/ollama/releases into scratch, and set `OLLAMA_MODELS=<scratch dir>/ollama-models` so the weights stay off your home quota.
    - a Python venv with `pip install aiohttp`
    - pull the model on a login node (compute nodes may not have internet): `ollama serve &` then `make pull`

### Startup

- locally: Ollama running (the app, or `ollama serve`), then `make describe` (from here or from `superconductor_client`), listening on port 9200. The client uses `http://localhost:9200` by default.
- on Lighthouse:
    - `salloc --account=aimusic_project --partition=aimusic_project --gpus=1 --mem=32G --cpus-per-task=4 --time=01:00:00`
    - `ollama serve &` then `make describe`
    - on the laptop: `ssh -N -L 9200:<gpu node>:9200 YOUR_UNIQNAME@lighthouse.arc-ts.umich.edu`; the client's default URL then reaches the cluster. Use another local port with `sc-collab --describe-server http://localhost:<port>`.

Other flags: `make describe MODEL=qwen2.5vl:7b PORT=... ARGS="--keep_alive 0"` (`--keep_alive 0` frees the GPU right after each request, which helps when the VLM shares the Mac's GPU with MRT2), or `python describe_server.py --help`.

### HTTP API

| Request | Body | Response |
|---|---|---|
| `POST /describe` | `{"image": "<base64 jpeg>"}` | `{"description": "..."}` |
| `POST /instrument` | `{"description": "..."}` | `{"instrument": "fiery thundering taiko drums"}` |
| `POST /name` | `{"description": "..."}` | `{"name": "green plush dinosaur"}` (the default name on the objects page) |
| `GET /health` | | `{"model": "qwen2.5vl:3b", "backend": "ollama", "host": "<hostname>"}` |

Requests are handled one at a time. The prompt templates are at the top of `describe_server.py`.
