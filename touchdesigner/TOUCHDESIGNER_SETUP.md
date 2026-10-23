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

## Open the project

Open **`touchdesigner/superconductor.toe`**, by double-clicking it, or with:
```bash
open -n -a TouchDesigner touchdesigner/superconductor.toe     # -n: a new TouchDesigner window
```
It holds two components, each loaded from its file in this folder when the project opens:

| Component | File | What |
|---|---|---|
| `/superconductor` | `superconductor.tox` | the network: audio, OSC, tracking, hands, webcams |
| `/mcp_webserver_base` | `mcp/mcp_webserver_base.tox` | lets Claude Code work in this TouchDesigner (below) |

Both are linked by paths relative to the project, so keep the files where they are, and save the project in this folder (**Cmd+S**). TouchDesigner also writes numbered copies (`superconductor.1.toe`, …) and a `Backup/` folder; git ignores them.

**`superconductor.tox` is the network's source.** The project reloads it every time it opens, so changes you make inside `/superconductor` are lost unless you save the component back to its file: right-click it → **Save Component .tox**, or in the Textport:
```python
op('/superconductor').save('superconductor.tox')
```

### Using it in another project

Drag `superconductor.tox` into that project's network editor. Or, in its Textport:
```python
op('/project1').loadTox('/Users/dennisfj/Michigan/superconductor/superconductor_touchdesigner/touchdesigner/superconductor.tox')
```

### Rebuilding it from the script

After changing `build_network.py` (e.g. other ports):
1. Open the Textport: menu **Dialogs → Textport and DATs** (or **⌥T**).
2. Paste this line and press Return:
   ```python
   exec(open('/Users/dennisfj/Michigan/superconductor/superconductor_touchdesigner/touchdesigner/build_network.py').read())
   ```
   It prints `built /superconductor: ...` and replaces the earlier copy, keeping its link to `superconductor.tox`.
3. Save the component to its file, as above.

## The plugin's settings

1. Double-click `superconductor` to go inside it, and click the **`audio`** operator.
2. In its parameters (press **p** if they're hidden), on the **VST** page, click **Display Plugin GUI**. In the Textport, that's `op('/superconductor/audio').par.displaygui.pulse()`.
3. In the plugin window:
   - **Role.** It starts as **Monitor**: it plays along with the copy in your DAW, which is the **Clock** and sets the pace. With no DAW (and no camera page or SC Playback app playing), switch it to **Clock**, or nothing plays.
   - **Send OSC.** Turn it on in **one** copy only: here, or in the DAW's copy. That's what fills the `osc` operator.
   - **Per second** sets how often OSC is sent (15–120, default 60). **Bands** sets the number of frequency bands (4–32).

The plugin's settings are saved with the project.

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

## Claude Code in TouchDesigner (MCP)

With `superconductor.toe` open, Claude Code can build, inspect and test the network itself, through the TouchDesigner MCP plugin (install: `../TOUCHDESIGNER_MCP_PLUGIN.md`; host `127.0.0.1`, port `9981`).

- **The component is pinned to v2.0.0** of touchdesigner-mcp, matching the plugin's MCP server. It's committed in `mcp/` as downloaded from the release (Python files and a `.tox`, the same on Mac and Windows). To change versions, replace `mcp/` with the matching release's `touchdesigner-mcp-td.zip`.
- **It accepts commands from any machine on the network**, with no password: it listens on port 9981 on all network interfaces and runs whatever Python it's sent. Switch it off when you don't need it, and always on shared Wi-Fi or at a show: select `/mcp_webserver_base` and turn **Active** off. Then save, so it stays off.
