# SuperConductor in TouchDesigner

What TouchDesigner gets, and from where. Audio and data travel separately: the audio comes through the SC Playback plugin, the tracking data through two listen-only websockets, and the webcams directly.

| Operator | Type | Source |
|---|---|---|
| `audio` | Audio VST CHOP | SC Playback plugin (`playback` branch): the music, 2 channels |
| `osc` | OSC In CHOP, port 7000 | the plugin copy with **Send OSC** on: loudness, bands, hits, tempo (`playback/docs/osc.md`) |
| `tracking` | WebSocket DAT | client: `ws://localhost:8467/ws/tracking` (camera page's tracking, `CAMERA_PROTOCOL.md` on client-web) |
| `hands` | WebSocket DAT | conducting page server: `ws://localhost:8480/ws/hands/monitor` (MediaPipe) |
| `camera1`, `camera2`, … | Video Device In TOP | webcams, opened by TouchDesigner itself |

## Setup

`TOUCHDESIGNER_SETUP.md`: what to run inside TouchDesigner, step by step. In short: build the plugin, start the services, then open `superconductor.toe`. It loads the network from `superconductor.tox`, and the MCP component (`mcp/`, for Claude Code) from its own file.

## What's inside `/superconductor`

- **`audio`**: starts as a **monitor** (it plays along with the DAW's copy, which is the clock). Pulse **Display Plugin GUI** on it to switch it to Clock, or to turn on Send OSC. Its settings are saved with the project.
- **`osc`**: channels `sc/level`, `sc/bands1…`, `sc/onset`, `sc/tempo/bar_phase`, … (`playback/docs/osc.md`).
- **`tracking`** fills, on every camera frame:
  - `objects`: identity, track_id, class_name, label, x1 y1 x2 y2, cx cy, coasted, combined, color.
  - `hand_boxes`: x1 y1 x2 y2.
  - `combos`: active, label.
  - `tracking_info`: connected, seq, width, height, message, vision_fps, music.
  - `objects_chop`: one sample per object (identity, cx, cy, x1…y2, coasted, combined), for instancing.
- **`hands`** fills, on every hands frame:
  - `hand_points`: hand, handedness, point (0–20, MediaPipe's numbering), x, y, z.
  - `conduct`: connected, seq, bpm, speed, released, status, …
  - `hand_points_chop`: one sample per point (hand, point, x, y, z).
- **`camera1`**: the first webcam (AVFoundation on Mac, Media Foundation on Windows). Pick another under **Device**. For more cameras, copy it and pick each copy's Device.

**Coordinates:** all positions are 0–1 of the camera image, with x to the right and **y down** (image rows), not mirrored. Use `1 - y` for TouchDesigner's y-up space.

## Testing without TouchDesigner

`tools/osc_listen.py` prints what the plugin sends over OSC: `python touchdesigner/tools/osc_listen.py 10`.
