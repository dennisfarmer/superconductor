# SuperConductor

Recent years have witnessed major advances in generative AI and its success in many fields. Yet, interfaces for AI music generation models remain unsatisfactory in terms of interactability, real-timeness, and accessibility. Few music generation models allow interactive controls with real-time feedback, and existing music generation models rely on command-line interfaces, making them hard to reach to musicians, artists and the general public. This project aims at challenging the state of the art of AI music for live performances and art installations – can we create a new interface where the user can co-create lively with an AI music generation model through simple gestures or body motions?

SuperConductor will be an interactive human-AI music co-creating system where the user can control an AI music generator through hand gestures and body movements. The user will be able to control the instrumentation, dynamic, tempo, density, and tension of the generated music – as if they were “conducting” an ensemble of AI music agents. The SuperConductor explores the future of human-AI music co-creativity.

# Getting Started

If running either the client or the server on Lighthouse, you will want to first open an ssh tunnel with one of the commands listed below, depending on where the client and server processes are located. The ports are the same between running locally and via the cluster so once the tunnel is open you can run the same commands for all these situations (with `atlas` also running either on laptop or lighthouse, again due to using same ports either way):
- client and server on laptop
- client on laptop, server on lighthouse
- client and server on lighthouse

Note that client assumes MacOS or Lighthouse, but may also run via WSL just fine as long as both the `make` and `ssh` commands are ran inside of WSL terminals. server doesn't run on windows unless further modifications are made.

I assume that this would work with multiple users simultaneously on difference sessions, since perhaps the nodes are unique to the lighthouse session. Since for now the client -> server protocol covers most / all of possible input options, and tempo control is via a server endpoint rather than a client endpoint, likely all prototype development work can be done on laptop side (MacOS or WSL), with the only thing running on lighthouse being the server since the server implementation doesn't have to change very often for now.

### SSH tunnel for client + server both running on lighthouse

The processes listen on every interface of `lh2300`, so the login node connects straight to that machine. One `ssh` is enough.

```bash
# 9000 Server, Tempo View and Conductor
# 9100 Tracking and Prompts
# 9101 MIDI player
# 9102 Conducting
# 9103 Atlas
# 9104 Atlas engine
# 9105 client_midi
# 9106 Describe
ssh -N \
  -L 9000:lh2300:9000 \
  -L 9100:lh2300:9100 \
  -L 9101:lh2300:9101 \
  -L 9102:lh2300:9102 \
  -L 9103:lh2300:9103 \
  -L 9104:lh2300:9104 \
  -L 9105:lh2300:9105 \
  -L 9106:lh2300:9106 \
  YOUR_UNIQNAME@lighthouse.arc-ts.umich.edu
```

### SSH tunnel for client on laptop, server on lighthouse

```
ssh -N -L 9000:lh2300:9000 YOUR_UNIQNAME@lighthouse.arc-ts.umich.edu
```

### Setup for `client`, `server`, optional `atlas`

Note: whenever updates are made, simply run `git pull` because the `make client`, `make server`, etc. automatically detect changes and re-run the environment setup idempotently, so running setup once is the same as running it multiple times.

On laptop:
```bash
git clone -b client https://github.com/dennisfarmer/superconductor.git superconductor_client
cd superconductor_client
make worktrees
```

On lighthouse:
```bash
# client
cd /scratch/aimusic_project_root/aimusic_project/shared_data/superconductor_client
# server
cd /scratch/aimusic_project_root/aimusic_project/shared_data/superconductor_server
# atlas
cd /scratch/aimusic_project_root/aimusic_project/shared_data/superconductor_atlas
```

Start the server..
```bash
cd ../superconductor_server
# runs setup automatically
make server
```

..and the client..
```bash
# runs setup automatically, prompts to ask if server is on laptop or lighthouse
cd ../superconductor_client
make client
# The describe server (camera object auto-analysis, parameter logo generation, ...) is optional. This starts it along with the client.
# make client DESCRIBE=1
```

Optionally also start this atlas web app. For now this can control notes and drums by sending messages to client to be forwarded to server. Later this functionality will be replaced by things that are cooler than knobs and buttons!
```bash
# runs idempotent setup automatically
cd ../superconductor_atlas
make atlas
```

Here is a table of the different web application views, which can be opened in a web browser after starting the client/server/atlas stack (and opening the ssh tunnel if using lighthouse).

| Page | URL |
|---|---|
| Tracking | http://localhost:9100 |
| Prompts | http://localhost:9100/prompts |
| Conducting | http://localhost:9102 |
| MIDI player | http://localhost:9101 |
| Atlas | http://localhost:9103 |
| Tempo View | http://localhost:9000/tempo |
| Conductor | http://localhost:9000/conduct |
| Playback | ws://localhost:9000/stream/playback |

# Development Guide

## Making a release

probably ask AI to do it, but the idea is tagged commit with v0.1.0, then run the workflow to create the singularity images, then create a release using the tagged commits to `client`, `server`, and `atlas` branches, along with the images at the fixed urls.

Images this release publishes:

- `ghcr.io/dennisfarmer/superconductor_client:client`
- `ghcr.io/dennisfarmer/superconductor_server:server`

```
gh workflow run release --ref client -f bump=patch
```

### Running the published images

The published images are to maybe make running client and server ultra reproducable by hosting a singularity container via `ghcr.io` on the github repository. Hasn't been fully tested yet but the idea is that images are associated with a "release", which is a known to be working version of the SuperConductor stack.

**Actions** → **release** → **Run workflow** publishes the two release images and a GitHub Release for the current `client` and `server` commits. On lh2300, `make server-singularity` and `make client-singularity` pull those images when the `.sif` is not already in the checkout. `make sif` pulls again after a new release. `make sif-dev` does the same for a dev image from `make image-push` on a laptop. See [client_container/README.md](client_container/README.md).

**First time, on lh2300:**

```bash
cd /scratch/aimusic_project_root/aimusic_project/shared_data
git clone -b server --single-branch https://github.com/dennisfarmer/superconductor.git superconductor_server

cd /scratch/aimusic_project_root/aimusic_project/shared_data
git clone -b client --single-branch https://github.com/dennisfarmer/superconductor.git superconductor_client
```

**Each time you run it.** On lh2300, start the server image and leave that terminal open:

```bash
cd /scratch/aimusic_project_root/aimusic_project/shared_data/superconductor_server
make server-singularity
```

Wait until it prints `Serving mrt2_base (jax) on port 9000`. Then start the client image in another terminal on lh2300:

```bash
cd /scratch/aimusic_project_root/aimusic_project/shared_data/superconductor_client
make client-singularity
```

On the Macbook Pro, open a tunnel and leave it open:

```bash
ssh -N -L 8080:lh2300:8080 YOUR_UNIQNAME@lighthouse.arc-ts.umich.edu
```

A `.sif` already in the checkout is reused. When a new release is published, run `make sif` in the server checkout and in the client checkout, then start them again. For a dev image, `make sif-dev`, then `make server-singularity-dev` / `make client-singularity-dev`.

`server.localhost` is only the Singularity column. The gateway on port 8080 answers that name for the music server. The other columns do not use it. `midi.localhost` is the same kind of name, for the MIDI player. Chrome, Edge, and Firefox send `*.localhost` to this machine, so both work through the port 8080 tunnel.

For the cluster-clones column, the laptop reaches those ports through the SSH tunnel in that section.

Saved objects are `superconductor_client/tracking/library/<id>/` when you run a clone, and `/data/client/library` inside the image.

| Page | Local: client and server on the laptop | Remote: client on the laptop, server on lh2300 | Cluster: client and server on lh2300, from the clones | Singularity: images on lh2300 |
|---|---|---|---|---|
| Tracking | http://localhost:9100 | http://localhost:9100 | http://localhost:9100 | http://localhost:8080 |
| Prompts | http://localhost:9100/prompts | http://localhost:9100/prompts | http://localhost:9100/prompts | http://localhost:8080/prompts |
| Conducting | http://localhost:9102 | http://localhost:9102 | http://localhost:9102 | http://localhost:8080/conduct/ |
| MIDI player | http://localhost:9101 | http://localhost:9101 | http://localhost:9101 | http://midi.localhost:8080 |
| Tempo View | http://localhost:9000/tempo | http://localhost:9000/tempo | http://localhost:9000/tempo | http://server.localhost:8080/tempo |
| Conductor | http://localhost:9000/conduct | http://localhost:9000/conduct | http://localhost:9000/conduct | http://server.localhost:8080/conduct |
| Playback | ws://localhost:9000/stream/playback | ws://localhost:9000/stream/playback | ws://localhost:9000/stream/playback | ws://server.localhost:8080/stream/playback |

## Sound on this machine

This is a minimal example of an audio reciever that is seperate from the web browser. See also the `playback` branch, which implements a more formal plugin which can forward audio to speakers, DAW, touchdesigner, etc. by reading directly from server websocket for minimal latency.

Note that as of now only one process can connect to and read from the server playback socket at one time, since it sends clock messages to confirm that audio blocks were recieved to obtain the subsequent blocks.

`make audio` asks where the music server is, the same question as `make client`. `make audio-local` and `make audio-remote` skip it (`--where local` and `--where remote`).

```bash
python audio_receiver.py --list-devices
#    0 ASUS VG34V, Core Audio (0 in, 2 out)
# <  2 USB-C to 3.5mm Headphone Jack Adapter, Core Audio (0 in, 2 out)
#    5 BlackHole 2ch, Core Audio (2 in, 2 out)
#    7 MacBook Pro Speakers, Core Audio (0 in, 2 out)
#    8 Microsoft Teams Audio, Core Audio (1 in, 1 out)
#    9 Steam Streaming Microphone, Core Audio (2 in, 2 out)
#   10 Steam Streaming Speakers, Core Audio (2 in, 2 out)
#   11 ZoomAudioDevice, Core Audio (2 in, 2 out)

# to play audio using device x (default shown above by '<')
python audio_receiver.py --device 2
```
