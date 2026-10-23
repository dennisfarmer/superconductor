# Running SuperConductor in TouchDesigner

What to do inside TouchDesigner to get the music, the sound measurements (OSC), the tracking, the MediaPipe hands and the webcams. `README.md` lists what each operator holds.

## Before you start

1. **Build the plugin** (`playback` branch, once, and again after plugin changes):
   ```bash
   cd superconductor_playback && make playback-build
   ```
   This installs `~/Library/Audio/Plug-Ins/VST3/SC Playback.vst3`.
2. **Start the services** (in `superconductor_client_web`, each in its own terminal):
   ```bash
   make server    # MRT2 server, :9100
   make client    # camera page http://localhost:8467 + the audio stream :8470
   make conduct   # conducting page http://localhost:8480 (MediaPipe hands)
   ```
   Open the camera page and the conducting page in the browser. Their tracking only flows while they're open.

## Load the network

Use either one.

**A. Drag in the component (quickest).** Drag `touchdesigner/superconductor.tox` from Finder into the network editor, or run this in the Textport (next section):
```python
op('/project1').loadTox('/Users/dennisfj/Michigan/superconductor/superconductor_touchdesigner/touchdesigner/superconductor.tox')
```

**B. Build it from the script** (after changing `build_network.py`, e.g. other ports):
1. Open the Textport: menu **Dialogs → Textport and DATs** (or **Alt+T** / **⌥T**).
2. Paste this line and press Return:
   ```python
   exec(open('/Users/dennisfj/Michigan/superconductor/superconductor_touchdesigner/touchdesigner/build_network.py').read())
   ```
   It prints `built /project1/superconductor: ...` and replaces any earlier copy.
3. To update the saved component, run in the Textport:
   ```python
   op('/project1/superconductor').save('/Users/dennisfj/Michigan/superconductor/superconductor_touchdesigner/touchdesigner/superconductor.tox')
   ```

Either way you get a `superconductor` component. Double-click it (or press **i** with it selected) to go inside.

## The plugin's settings

1. Inside `superconductor`, click the **`audio`** operator.
2. In its parameters (press **p** if they're hidden), on the **VST** page, click **Display Plugin GUI**. In the Textport, that's `op('/project1/superconductor/audio').par.displaygui.pulse()`.
3. In the plugin window:
   - **Role.** It starts as **Monitor**: it plays along with the copy in your DAW, which is the **Clock** and sets the pace. With no DAW (and no camera page or SC Playback app playing), switch it to **Clock**, or nothing plays.
   - **Send OSC.** Turn it on in **one** copy only: here, or in the DAW's copy. That's what fills the `osc` operator.
   - **Per second** sets how often OSC is sent (15–120, default 60). **Bands** sets the number of frequency bands (4–32).

The settings are saved with your project.

**Sample rate:** `audio` runs at 48 kHz (its **Sample Rate** parameter), the stream's own rate. Keep it there.

**Hearing it in TouchDesigner:** the plugin's output is just channels in TouchDesigner. Normally you hear the music through the DAW. To hear it from TouchDesigner instead, add an **Audio Device Out CHOP** after `audio`.

## Webcams

`camera1` opens the first webcam. To use another one, click `camera1` and pick it under **Device**. For more cameras, copy `camera1` (**Cmd+C**, **Cmd+V**) and give each copy its own Device. A camera can also be open in the browser's camera page at the same time. If one can't be shared, make a duplicate with an OBS virtual camera and use that here.

## Checking it works

Inside `superconductor`:

| Look at | You should see |
|---|---|
| `audio` | two moving audio channels |
| `osc` | `sc/level`, `sc/bands1…8`, `sc/onset`, `sc/tempo/…` changing (once one copy sends OSC) |
| `tracking_info` | `connected` 1, `seq` counting up while the camera page runs |
| `objects`, `hand_boxes`, `objects_chop` | rows / samples for what the camera page tracks |
| `conduct` | `connected` 1, `bpm` while you conduct on the conducting page |
| `hand_points`, `hand_points_chop` | 21 rows / samples per hand |
| `camera1` | the webcam picture |

**If something's empty:**
- **`audio` says "Loading Plugin…"** It takes a few seconds the first time.
- **`audio` is silent.** Nothing is the clock: set one copy to Clock (see above), or open the camera page.
- **`osc` has no channels.** No copy has Send OSC on.
- **`connected` is 0.** That service isn't running (`make client` for tracking, `make conduct` for hands). The connections retry by themselves once it is.

`tools/osc_listen.py` shows the OSC from the terminal, without TouchDesigner.

## Letting Claude drive TouchDesigner (MCP)

Claude Code can build and check the network itself through the TouchDesigner MCP plugin (install: `../TOUCHDESIGNER_MCP_PLUGIN.md`). TouchDesigner needs the MCP component loaded:
```bash
open -a TouchDesigner ~/Documents/touchdesigner-mcp/mcp_webserver_base.tox
```
(`/touchdesigner:launch` in Claude Code downloads it there and does this for you.)

Things to know:
- **There's no `/project1` in that window.** Opening the component this way starts a project without one, so the network is built at `/superconductor`.
- **Don't save that project into another folder** while Claude uses the MCP. The component finds its own files relative to the project's folder, and stops answering after such a save. Save the network as `superconductor.tox` (above) instead. If it does stop answering, run the `open -a` line again.
