# SuperConductor: Playback + TouchDesigner

## Setup

Clone the client (main) branch. `make worktrees` then creates the other branches next to it,
including `superconductor_playback/` and `superconductor_touchdesigner/`:

```bash
git clone -b client https://github.com/dennisfarmer/superconductor.git superconductor_client
cd superconductor_client
make worktrees
```

The `playback` and `touchdesigner` branches are used together.

## 1. Start the stack (cluster)

On lh2300, in two terminals:

```bash
# server
cd /scratch/aimusic_project_root/aimusic_project/shared_data/superconductor_server
make server
```

```bash
# client
cd /scratch/aimusic_project_root/aimusic_project/shared_data/superconductor_client
make client
```

On the laptop, open the SSH tunnel and leave it open. That puts everything on `localhost`:

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

## 2. Playback plugin (`superconductor_playback/`)

```bash
make playback-build   # once, and after plugin changes; also installs the VST3/AU
make playback         # open the Standalone app on ws://localhost:9000/stream/playback
```

To use it in a DAW or TouchDesigner, load **SuperConductor Playback VST3 Plugin**. Details are in `playback/README.md`.

## 3. TouchDesigner projects (`superconductor_touchdesigner/`, macOS)

TouchDesigner has to be installed (derivative.ca), and the Playback plugin has to be built (step 2).

```bash
make touchdesigner-build            # optional: rebuild touchdesigner/superconductor_inputs.tox
make visualize                      # overlay_example_1
make visualize VISUALIZER=<name>    # touchdesigner/visualizers/<name>/<name>.toe
make touchdesigner VISUALIZER=<name> PERFORM=0   # open only the .toe, in the editor
```

`make visualize` opens the Playback app, the Tracking and Conducting pages, and the visualizer.
Quit TouchDesigner with Cmd+Q to stop. Details are in `touchdesigner/TOUCHDESIGNER_SETUP.md`.

`superconductor_inputs.tox` (the `/inputs` that every visualizer loads: plugin, OSC, tracking,
conducting) is committed already built, so `make touchdesigner-build` is optional. Run it only after
`touchdesigner/build_inputs.py` changes, for example after the client ports move. It opens
TouchDesigner, rebuilds the `.tox`, and closes it. Commit the new `.tox`. If a visualizer shows no
tracking or conducting while both pages work, the `.tox` is out of date, so run it.
