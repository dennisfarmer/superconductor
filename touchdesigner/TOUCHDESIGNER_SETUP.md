# Running SuperConductor in TouchDesigner

How to get a visualizer running: building the plugin, starting the services, opening a visualizer, and checking that the music, the sound measurements (OSC), the tracking, the MediaPipe hands and the webcams all arrive. What the inputs contain, and how to make your own visualizer, is in `TOUCHDESIGNER_PROTOCOL_GETTING_STARTED.md`.

## The quick way: one command

From the `superconductor_touchdesigner` folder (after building the plugin once, below):
```bash
make visualize                                # overlay_example_1
make visualize VISUALIZER=<name>              # visualizers/<name>/<name>.toe
```
It:
1. starts the MRT2 server, the client and the conducting server (any that aren't already running);
2. starts the **SC Playback app** (`playback` branch), which plays the music on your Mac's current sound output (speakers or headphones) and sets the pace;
3. opens the **Tracking** page (quiet, so the music isn't played twice) and the **Conducting** page in the browser;
4. opens the visualizer showing only its output, in a window you can resize or make full screen.

Press **Start** on the Tracking page for the object tracking. **Esc** shows the editor (**F1** goes back); quitting TouchDesigner (**Cmd+Q**) stops the services and the app it started. Logs are in `var/logs/`. This is macOS only, and needs the SC Playback app built (`make playback-build` in `superconductor_playback`).

The rest of this document is the same thing step by step, and how to check each part.

## Before you start

1. **Build the plugin** (`playback` branch, once, and again after plugin changes):
   ```bash
   cd superconductor_playback && make playback-build
   ```
   This installs `~/Library/Audio/Plug-Ins/VST3/SC Playback.vst3`.
2. **Start the services** (in `superconductor_client_web`, each in its own terminal):
   ```bash
   make server    # MRT2 server, :9100
   make client    # Tracking page http://localhost:8467 + the audio stream :8470
   make conduct   # Conducting page http://localhost:8480 (MediaPipe hands)
   ```
   Open the Tracking page and the Conducting page in the browser. Their tracking only flows while they're open.

## Open a visualizer

Each visualizer is a `.toe` in its own folder under `visualizers/`. Open one by double-clicking it, or with:
```bash
open -n -a TouchDesigner touchdesigner/visualizers/overlay_example_1/overlay_example_1.toe     # -n: a new TouchDesigner window
```
Run one visualizer at a time: two would both listen on OSC port 7000, both load the plugin and both open the webcam.

When it opens, it loads the two shared components from `touchdesigner/` (`superconductor_inputs.tox` as `/inputs`, and the MCP component) and contains its own visual part (e.g. `/overlay_example_1`). Save with **Cmd+S**. TouchDesigner also writes numbered copies (`overlay_example_1.1.toe`, …) and a `Backup/` folder next to the `.toe`; git ignores them.

## The output window

A visualizer opened normally starts in the editor. **F1** switches to *perform mode*: the editor is hidden and only the visualizer's output shows, in an ordinary 1280×720 window that you can move and resize. **Esc** goes back to the editor. `make visualize` starts straight in perform mode (it opens TouchDesigner with `SC_PERFORM=1`).

Perform mode shows the visualizer's `window` component (for `overlay_example_1`, `/overlay_example_1/window`). **For full screen,** e.g. on a projector: in the editor, click that `window` component, set **Size** to **Fill** and turn **Borders** off. **Display** chooses the monitor. Save, and perform mode opens it that way from then on.

## The plugin's settings

1. Double-click `inputs` to go inside it, and click the **`audio`** operator.
2. In its parameters (press **p** if they're hidden), on the **VST** page, click **Display Plugin GUI**. In the Textport, that's `op('/inputs/audio').par.displaygui.pulse()`.
3. In the plugin window:
   - **Role.** In TouchDesigner it starts as **Monitor**: it plays along with the copy in your DAW, which is the **Clock** and sets the pace. With no DAW (and no Tracking page or SC Playback app playing), switch it to **Clock**, or nothing plays.
   - **Send OSC.** On by default in TouchDesigner, off in a DAW. Only one copy should send it. That's what fills the `osc` operator.
   - **Per second** sets how often OSC is sent (15–120, default 60). **Bands** sets the number of frequency bands (4–32).

The plugin's settings are saved with the visualizer's `.toe`.

**Sample rate:** `audio` runs at 48 kHz (its **Sample Rate** parameter), the stream's own rate. Keep it there.

**Hearing it in TouchDesigner:** the plugin's output is just channels in TouchDesigner. Normally you hear the music through the DAW. To hear it from TouchDesigner instead, add an **Audio Device Out CHOP** after `audio`.

## Webcams

`/inputs/camera1` opens the first webcam. To use another one, click `camera1` and pick it under **Device**. For more cameras, copy `camera1` (**Cmd+C**, **Cmd+V**) and give each copy its own Device. A camera can also be open in the browser's Tracking page at the same time. If one can't be shared, make a duplicate with an OBS virtual camera and use that here.

## Checking it works

Inside `inputs`:

| Look at | You should see |
|---|---|
| `audio` | two moving audio channels |
| `osc` | `sc/level`, `sc/bands1…8`, `sc/onset`, `sc/tempo/…` changing |
| `tracking_info` | `connected` 1, `seq` counting up while the Tracking page runs |
| `objects`, `hand_boxes`, `objects_chop` | rows / samples for what the Tracking page tracks |
| `conduct` | `connected` 1, `bpm` while you conduct on the Conducting page |
| `hand_points`, `hand_points_chop` | 21 rows / samples per hand |
| `camera1` | the webcam picture |

**If something's empty:**
- **`audio` says "Loading Plugin…"** It takes a few seconds the first time.
- **`audio` is silent.** Nothing is the clock: set one copy to Clock (see above), or open the Tracking page.
- **`osc` has no channels.** No copy has Send OSC on (TouchDesigner's is on unless someone switched it off).
- **`connected` is 0.** That service isn't running (`make client` for tracking, `make conduct` for hands). The connections retry by themselves once it is.

`tools/osc_listen.py` shows the OSC from the terminal, without TouchDesigner.

## Rebuilding the inputs component

`superconductor_inputs.tox` is built by `build_inputs.py`. Only rebuild it when the inputs themselves change (e.g. other ports), since every visualizer loads it:
1. Open any visualizer and its Textport.
2. Paste this line and press Return:
   ```python
   exec(open('/Users/dennisfj/Michigan/superconductor/superconductor_touchdesigner/touchdesigner/build_inputs.py').read())
   ```
   It prints `built /inputs: ...` and replaces the earlier copy, keeping its link to the `.tox` file.
3. Save it to its file: right-click `inputs` → **Save Component .tox**, onto `touchdesigner/superconductor_inputs.tox`. Or in the Textport: `op('/inputs').save('../../superconductor_inputs.tox')` (the path is relative to the visualizer's folder).

## Claude Code in TouchDesigner (MCP)

With a visualizer open, Claude Code can build, inspect and test it through the TouchDesigner MCP plugin (install: `../TOUCHDESIGNER_MCP_PLUGIN.md`; host `127.0.0.1`, port `9981`).

- **The component is pinned to v2.0.0** of touchdesigner-mcp, matching the plugin's MCP server. It's committed in `mcp/` as downloaded from the release (Python files and a `.tox`, the same on Mac and Windows). To change versions, replace `mcp/` with the matching release's `touchdesigner-mcp-td.zip`.
- **It accepts commands from any machine on the network**, with no password: it listens on port 9981 on all network interfaces and runs whatever Python it's sent. Switch it off when you don't need it, and always on shared Wi-Fi or at a show: select `/mcp_webserver_base` and turn **Active** off. Then save, so it stays off.
