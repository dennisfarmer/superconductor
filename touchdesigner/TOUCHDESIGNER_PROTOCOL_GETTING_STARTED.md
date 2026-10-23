# SuperConductor in TouchDesigner: the protocol, and getting started

This explains how the TouchDesigner side is organised, exactly what information SuperConductor gives TouchDesigner (the protocol), and how to make a new visualizer that uses it. To get things running (building the plugin, starting the services, checking that data arrives), see `TOUCHDESIGNER_SETUP.md`.

## The idea

SuperConductor's programs (the MRT2 music server, the camera page with its object tracking, the conducting page with MediaPipe) send their information out in their own formats: an audio stream, OSC messages and JSON over websockets. One shared TouchDesigner component, `superconductor_inputs.tox`, receives all of it and turns it into things TouchDesigner works with directly: audio channels, tables, CHOP channels and webcam images.

Every **visualizer** is a separate TouchDesigner project (`.toe`) that loads that shared component and draws something with it. The component stays the same; the visualizers are where people experiment.

```
MRT2 server ──► client ──► audio stream ──► SC Playback plugin ─┐
                                    (plugin also sends OSC) ─────┤
camera page (object tracking) ──► client ──► websocket ──────────┤──► /inputs ──► a visualizer
conducting page (MediaPipe) ──► conducting server ──► websocket ─┤   (superconductor_inputs.tox)
webcams ─────────────────────────────────────────────────────────┘
```

## TouchDesigner terms used here

- **Operator:** one building block, such as a webcam input, a table or a shape. Operators come in families, named by their last letters:
  - **CHOP:** numbers that change over time (channels: audio, OSC values, positions);
  - **TOP:** images;
  - **DAT:** text and tables;
  - **SOP:** 3D shapes;
  - **COMP:** a container that holds other operators.
- **Component:** a COMP with operators inside. It can be saved on its own as a `.tox` file and loaded into any project.
- **`.toe`:** a whole TouchDesigner project, the file you open.
- **`.tox`:** one component, as a file.
- **Network:** the operators inside a component, as you see them in the **Network Editor**.
- **Textport:** TouchDesigner's Python console: menu **Dialogs → Textport and DATs** (or **⌥T**).

TouchDesigner has no compile step. A project runs while you edit it, and **Cmd+S** saves it.

## Folder layout

```
touchdesigner/
  superconductor_inputs.tox      the shared inputs component (loaded as /inputs)
  build_inputs.py                builds it (only needed when the inputs change)
  mcp/                           lets Claude Code work inside TouchDesigner (shared)
  tools/osc_listen.py            shows the OSC messages in a terminal
  visualizers/
    overlay_example_1/           one visualizer per folder
      overlay_example_1.toe      the project to open
      build_overlay_example_1.py builds its visual part (optional: a script, so it can be read and diffed)
```

A visualizer's `.toe` loads `../../superconductor_inputs.tox` and `../../mcp/mcp_webserver_base.tox`, paths relative to its own folder. So every visualizer must sit exactly two folders below `touchdesigner/`, at `visualizers/<name>/<name>.toe`.

## The protocol: what `/inputs` provides

Everything below is inside the `inputs` component. A visualizer component placed next to it reads it as `op('../inputs/<name>')`, or with the full path, `op('/inputs/<name>')`.

### Positions

All positions are fractions of the camera image, 0 to 1:
- **x** runs left to right;
- **y** runs **top to bottom** (as image rows do);
- **not mirrored.**

TouchDesigner's own y runs bottom to top, so use `1 - y` there.

### Music: `audio` (Audio VST CHOP)

Two channels, `chan1` and `chan2` (left and right), at 48 kHz: the music as it plays, from the SC Playback plugin.

### Sound measurements and tempo: `osc` (OSC In CHOP, port 7000)

One channel per value, updated 60 times a second by default. These only arrive while one copy of the plugin has **Send OSC** on (see `TOUCHDESIGNER_SETUP.md`). The measurements describe the sound as it's being heard.

| Channel | Range | Meaning |
|---|---|---|
| `sc/level` | 0–1 | Loudness over the last ~21 ms (-80 dB → 0, 0 dB → 1) |
| `sc/rms_db` | -80 … 0 | The same, in decibels |
| `sc/peak` | 0–1 | Loudest single sample over the last ~21 ms |
| `sc/bands1` … `sc/bandsN` | 0–1 each | Frequency bands, low to high, spaced evenly in pitch from 40 Hz to 16 kHz (N = 8 by default) |
| `sc/flux` | small, ≥ 0 | How much the spectrum just grew |
| `sc/onset` | ≥ 0 | Above 0 on the update where a hit was detected, otherwise 0. At most one hit every 80 ms |
| `sc/tempo/reachable` | 0 / 1 | The music server answered |
| `sc/tempo/playing` | 0 / 1 | Music is being heard |
| `sc/tempo/bpm` | | The tempo being heard (0 if unknown) |
| `sc/tempo/beat_known` | 0 / 1 | The server has found the beat. While 0, beat and beat_frac stay at 0 |
| `sc/tempo/beat` | 0 … 3 | Beat within the bar, 0 = beat one |
| `sc/tempo/beat_frac` | 0–1 | How far through that beat |
| `sc/tempo/beats_per_bar` | 3 or 4 | 4 while the server isn't sure |
| `sc/tempo/bar_phase` | 0–1 | How far through the bar |

### Object tracking from the camera page

Updated on every camera frame the client handles (around 15 a second), while the camera page is open.

| Operator | Type | Contents |
|---|---|---|
| `objects` | table | One row per object the client recognised: `identity` (stays the same for the same object), `track_id`, `class_name`, `label` (ready to show), `x1 y1 x2 y2` (its box), `cx cy` (its centre), `coasted` (1 = not seen right now; last known place), `combined` (1 = part of an active combo), `color` (`#rrggbb`) |
| `objects_chop` | CHOP | The same, one sample per object: `identity`, `cx`, `cy`, `x1`, `y1`, `x2`, `y2`, `coasted`, `combined`. Ready for instancing (one shape per object) |
| `hand_boxes` | table | One row per hand the camera page found: `x1 y1 x2 y2` |
| `combos` | table | `active` (0/1), `label`: objects that play something together |
| `tracking_info` | table | `connected` (0/1), `seq` (frame number), `width`, `height` (camera page's image size), `message` (calibration instruction), `vision_fps`, `music` (the music engine's status) |

### MediaPipe hands from the conducting page

Updated on every frame the conducting page tracks, while it's open.

| Operator | Type | Contents |
|---|---|---|
| `hand_points` | table | One row per point: `hand` (0, 1), `handedness` (`Left` / `Right`), `point` (0–20, MediaPipe's numbering: 0 wrist, 4 thumb tip, 8 index tip, 12 middle tip, 16 ring tip, 20 little-finger tip), `x`, `y`, `z` (depth, relative to the wrist) |
| `hand_points_chop` | CHOP | The same, one sample per point: `hand`, `point`, `x`, `y`, `z` |
| `conduct` | table | `connected` (0/1), `seq`, and the conductor's reply for that frame: `bpm`, `speed`, `released`, `status`, … |

### Webcams

| Operator | Type | Contents |
|---|---|---|
| `camera1` (more can be added) | TOP | The webcam image, opened by TouchDesigner itself |

## Making a new visualizer

The quickest way is to copy the example:

1. Copy the folder `visualizers/overlay_example_1` to `visualizers/<your_name>`, and rename the `.toe` inside to `<your_name>.toe`. Delete the copied `build_overlay_example_1.py` and any numbered `.toe` copies.
2. Open `<your_name>.toe`. Delete the `overlay_example_1` component, or keep it to look at while you work.
3. Make your own component next to `inputs`: in the Network Editor, press **Tab**, choose **COMP → Container**, and name it, e.g. `<your_name>`. Build inside it, reading from `../inputs/...`.
4. Save (**Cmd+S**).

**Starting from an empty project instead:**
1. **File → New**.
2. **File → Save As**, into a new folder: `visualizers/<your_name>/<your_name>.toe`.
3. In the Textport, load the two shared components:
   ```python
   for path in ('../../superconductor_inputs.tox', '../../mcp/mcp_webserver_base.tox'):
       c = op('/project1').loadTox(path)
       c.par.externaltox = path            # reload it from the file each time the project opens
       c.par.enableexternaltox = True
   ```
4. Build your component inside `/project1`, next to `inputs`, and save.

**Rules:**
- **Don't change `/inputs` inside a visualizer.** It's reloaded from `superconductor_inputs.tox` each time the project opens, so changes there are lost, and other visualizers depend on it staying the same. If the protocol itself needs to change, change `build_inputs.py` and rebuild it (`TOUCHDESIGNER_SETUP.md`), then check the other visualizers still work.
- **One person per visualizer at a time.** `.toe` and `.tox` files are binary, so git can't merge two people's changes to the same one. Separate visualizers in separate folders avoid that.
- **A build script is optional but useful.** Like `build_overlay_example_1.py`, a script that creates the component is readable and diffable in git, where the `.toe` isn't.

## Learning from `overlay_example_1`

It uses every input, so it doubles as a reference for reading them. Its build script, `build_overlay_example_1.py`, is commented. In short:
- **Data into shapes:** Script CHOPs read `inputs`' tables and OSC channels and produce one sample per shape to draw (position, size, colour).
- **Shapes:** Geometry COMPs draw one circle or rectangle per sample. This is called **instancing**.
- **Rendering:** a Render TOP with an orthographic camera draws them in the same 0–1 coordinates as the image.
- **Layering:** Over TOPs layer the shapes, a text line and a flash over `inputs/camera1`.
- **Result:** `out` is the final image, and `window` shows it full screen.

## Licence note

The free Non-Commercial TouchDesigner licence limits every image to 1280×1280 pixels (TouchDesigner warns about this on `camera1`). A 1920×1080 output, e.g. for a projector at a show, needs an Educational or Commercial licence.
