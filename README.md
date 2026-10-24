# SuperConductor

Recent years have witnessed major advances in generative AI and its success in many fields. Yet, interfaces for AI music generation models remain unsatisfactory in terms of interactability, real-timeness, and accessibility. Few music generation models allow interactive controls with real-time feedback, and existing music generation models rely on command-line interfaces, making them hard to reach to musicians, artists and the general public. This project aims at challenging the state of the art of AI music for live performances and art installations – can we create a new interface where the user can co-create lively with an AI music generation model through simple gestures or body motions?

SuperConductor will be an interactive human-AI music co-creating system where the user can control an AI music generator through hand gestures and body movements. The user will be able to control the instrumentation, dynamic, tempo, density, and tension of the generated music – as if they were “conducting” an ensemble of AI music agents. The SuperConductor explores the future of human-AI music co-creativity.

# Development Guide

## Web Application Views

These are opened on the laptop to provide insight behind the messages the client sends to the server, as well as to the touchdesigner branch visualizer(s) and playback branch VST3 plugin.

| | All local (Macbook Pro) | Client local, server remote | All remote, from the clones | All remote, published images |
|---|---|---|---|---|
| Camera | http://localhost:8467 | http://localhost:8467 | http://localhost:8467 | http://localhost:8080 |
| Objects | http://localhost:8467/objects | http://localhost:8467/objects | http://localhost:8467/objects | http://localhost:8080/objects |
| Conducting | http://localhost:8480 | http://localhost:8480 | http://localhost:8480 | http://localhost:8080/conduct/ |
| MIDI player | http://localhost:8475 | http://localhost:8475 | http://localhost:8475 | http://midi.localhost:8080 |
| Tempo | http://localhost:9100/tempo | http://localhost:9000/tempo | http://localhost:9100/tempo | http://server.localhost:8080/tempo |
| Conductor (space bar = one beat) | http://localhost:9100/conduct | http://localhost:9000/conduct | http://localhost:9100/conduct | http://server.localhost:8080/conduct |

Saved objects are `superconductor_client/tracking/library/<id>/` when you run a clone, and `/data/client/library` inside the image.


## All local (Macbook Pro)

The client and the server both run on the Macbook Pro. The server is `mrt2_small` on port 9100. Use two terminals. One runs the server. The other runs the client. A new terminal starts in your home directory, so each one changes into its own checkout.

**First time.** In one terminal:

```bash
cd ~/Michigan/superconductor
git clone -b client https://github.com/dennisfarmer/superconductor.git superconductor_client
cd superconductor_client
make worktrees
make setup
```

`make worktrees` puts the server at `~/Michigan/superconductor/superconductor_server`. In a second terminal:

```bash
cd ~/Michigan/superconductor/superconductor_server
make setup
```

**Each time you run it.** Leave the server terminal open.

Server terminal:

```bash
cd ~/Michigan/superconductor/superconductor_server
git pull && make setup && make server
```

Wait until that terminal prints `Serving mrt2_small (mlx) on port 9100`.

Client terminal:

```bash
cd ~/Michigan/superconductor/superconductor_client
git pull && make setup && make client
```

## Client local, server remote

The client runs on the Macbook Pro. The server is `mrt2_base` on lh2300, port 9100. An SSH tunnel on the Macbook Pro forwards localhost port 9000 to that server.

**First time, on the Macbook Pro:**

```bash
cd ~/Michigan/superconductor
git clone -b client https://github.com/dennisfarmer/superconductor.git superconductor_client
cd superconductor_client
make worktrees
make setup
```

**First time, on lh2300:**

```bash
cd /scratch/aimusic_project_root/aimusic_project/shared_data
git clone -b server --single-branch https://github.com/dennisfarmer/superconductor.git superconductor_server
cd superconductor_server
make setup
```

**Each time you run it.** On lh2300, in the server checkout:

```bash
cd /scratch/aimusic_project_root/aimusic_project/shared_data/superconductor_server
git pull && make setup && make server
```

Wait until that terminal prints `Serving mrt2_base (jax) on port 9100`.

On the Macbook Pro, open a terminal and leave the tunnel running:

```bash
ssh -N -L 9000:lh2300:9100 YOUR_UNIQNAME@lighthouse.arc-ts.umich.edu
```

Open another terminal on the Macbook Pro for the client:

```bash
cd ~/Michigan/superconductor/superconductor_client
git pull && make setup && make client-remote
```

## All remote

The client and the server both run on lh2300. One way is the git checkouts. The other is the published images.

### Testing from the git clones

**First time, on lh2300:**

```bash
cd /scratch/aimusic_project_root/aimusic_project/shared_data
git clone -b server --single-branch https://github.com/dennisfarmer/superconductor.git superconductor_server
cd superconductor_server && make setup

cd /scratch/aimusic_project_root/aimusic_project/shared_data
git clone -b client --single-branch https://github.com/dennisfarmer/superconductor.git superconductor_client
cd superconductor_client && make setup
```

**Each time you run it.** On lh2300, start the server and leave that terminal open:

```bash
cd /scratch/aimusic_project_root/aimusic_project/shared_data/superconductor_server
git pull && make setup && make server
```

Wait until it prints `Serving mrt2_base (jax) on port 9100`. Then start the client in another terminal on lh2300:

```bash
cd /scratch/aimusic_project_root/aimusic_project/shared_data/superconductor_client
git pull && make setup && make client
```

On the Macbook Pro, open a tunnel to the node that is running `make client`. Leave it open:

```bash
ssh -N -L 8467:localhost:8467 YOUR_UNIQNAME@lh2300
```

### Running the published images

**Actions** → **release** → **Run workflow** publishes the two release images and a GitHub Release for the current `client` and `server` commits. On lh2300, `make server-singularity` and `make client-singularity` pull those images when the `.sif` is not already in the checkout. `make sif` pulls again after a new release. `make sif-dev` does the same for a dev image from `make image-push` on a laptop. See [client_container/README.md](client_container/README.md).

**First time, on lh2300:**

```bash
cd /scratch/aimusic_project_root/aimusic_project/shared_data
module load singularity
export SINGULARITY_CACHEDIR=/scratch/aimusic_project_root/aimusic_project/shared_data/.singularity_cache

git clone -b server --single-branch https://github.com/dennisfarmer/superconductor.git superconductor_server

cd /scratch/aimusic_project_root/aimusic_project/shared_data
git clone -b client --single-branch https://github.com/dennisfarmer/superconductor.git superconductor_client
```

**Each time you run it.** On lh2300, start the server image and leave that terminal open:

```bash
cd /scratch/aimusic_project_root/aimusic_project/shared_data/superconductor_server
module load singularity && make server-singularity
```

Wait until it prints `Serving mrt2_base (jax) on port 9100`. Then start the client image in another terminal on lh2300:

```bash
cd /scratch/aimusic_project_root/aimusic_project/shared_data/superconductor_client
module load singularity && make client-singularity
```

On the Macbook Pro, open a tunnel and leave it open:

```bash
ssh -N -L 8080:lh2300:8080 YOUR_UNIQNAME@lighthouse.arc-ts.umich.edu
```

A `.sif` already in the checkout is reused. When a new release is published, run `make sif` in the server checkout and in the client checkout, then start them again. For a dev image, `make sif-dev`, then `make server-singularity-dev` / `make client-singularity-dev`.

## Sound on this machine

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
