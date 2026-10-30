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

On lh2300: `make server` and `make client`. On the laptop, open the SSH tunnel from the
`superconductor_client` README. That puts everything on `localhost` (9000 server, 9100 tracking,
9102 conducting).

## 2. Playback plugin (`superconductor_playback/`)

```bash
make playback-build   # once, and after plugin changes; also installs the VST3/AU
make playback         # open the Standalone app on ws://localhost:9000/stream/playback
```

To use it in a DAW or TouchDesigner, load **SuperConductor Playback VST3 Plugin**. Details are in `playback/README.md`.

## 3. TouchDesigner projects (`superconductor_touchdesigner/`, macOS)

```bash
make visualize                      # overlay_example_1
make visualize VISUALIZER=<name>    # touchdesigner/visualizers/<name>/<name>.toe
make touchdesigner VISUALIZER=<name> PERFORM=0   # open only the .toe, in the editor
```

`make visualize` opens the Playback app, the Tracking and Conducting pages, and the visualizer.
Quit TouchDesigner with Cmd+Q to stop. Details are in `touchdesigner/TOUCHDESIGNER_SETUP.md`.
