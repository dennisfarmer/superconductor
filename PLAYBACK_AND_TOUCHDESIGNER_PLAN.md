# Plan: playback plugin + TouchDesigner receiver

## Context

Today the music only plays in the camera page (client-web). We want a second, optional way to play it, so the web page stays as it is and the new option can be tested on its own. TouchDesigner (TD) should get the audio, measurements of the audio (loudness, frequency bands, hits, tempo), the webcam video, the object tracking and the MediaPipe hand points.

Design rule (from you): **audio and data travel separately.** The only data that travels with the audio is the OSC measurements of the audio plus tempo, because they describe the sound that is playing.

## How things flow (all on one computer)

```
                               ┌──────────── AUDIO ─────────────┐
MRT2 server :9100 ─► client ─► ws://localhost:8470/audio
                                  ├─► camera page (unchanged; goes quiet while another player is the clock)
                                  ├─► SC plugin in a DAW          ─► you hear it through the DAW
                                  ├─► SC plugin in TD (Audio VST CHOP) ─► audio channels in TD
                                  └─► SC standalone app           ─► speakers or BlackHole, no DAW
                                  One copy of the plugin also sends OSC ─► TD (loudness, bands, hits, tempo)

                               ┌──────────── DATA ──────────────┐
camera page (tab 1, YOLOE + BoT-SORT) ─► sc-collab :8467 /ws/camera ─► NEW listen-only WS /ws/tracking ─► TD
conducting page (tab 2, MediaPipe)    ─► conducting server :8480 /ws/hands ─► NEW listen-only WS /ws/hands/monitor ─► TD

                               ┌──────────── VIDEO ─────────────┐
webcams ─► TD Video Device In TOP (one per camera; TD opens them itself)
```

- **Audio:** no new process. The plugin connects to the existing `/audio` endpoint itself (see `AUDIO_PROTOCOL.md`). Nothing in client-web or the server changes for audio.
- **Data:** no new process. The two programs that already receive the tracking data each get a listen-only endpoint, similar to `/audio`. Each forwards what it already handles, and the pages don't change. TD reads them with WebSocket DATs (JSON at camera frame rate, which suits TD's Python).
- **Video:** TD opens the webcams directly. If a camera has to be shared, an OBS virtual camera can duplicate it.

## Branch `playback` (worktree `superconductor_playback`)

Everything goes in a new top-level `playback/` folder, so existing client-web files aren't touched.

```
playback/
  README.md            what each piece is, how to build, how to route on Mac / Windows / Linux
  CMakeLists.txt       fetches JUCE and IXWebSocket (the library that opens the WebSocket), pinned versions
  shared/              plain C++, used by every target
    StreamClient       connects to ws://…/audio: role (clock / monitor), ack, pause, underrun, reconnect
    AudioQueue         holds received blocks; as a monitor, adds or drops a few samples to stay in step with the clock
    Resampler          48 kHz stream → whatever rate the DAW or TD runs at
    Features           loudness, frequency bands, hit detection, measured on the samples as they are output
    TempoPoller        reads the MRT2 server's GET /tempo/status (bpm, beat, beat_frac, bar) and estimates the beat position between reads
    OscSender          sends the features + tempo to TD
  plugin/              JUCE project: VST3 (Mac/Win/Linux), AU (Mac), LV2 (Linux), Standalone app
    PluginProcessor    an instrument with no input: outputs the stream; on the audio thread it only reads the queue
    PluginEditor       stream URL, clock/monitor switch, OSC on/off + host/port, connection status, level meter
  docs/
    osc.md             every OSC address and what it holds
    routing.md         DAW setup; BlackHole (Mac) / VB-CABLE (Windows) / PipeWire (Linux) for programs that can't load plugins
```

Rules built into the plugin:
- **Timing:** only one copy is the clock (the one that acks to the server); the others are monitors. This is the existing `/audio` behaviour, chosen with a switch in the plugin.
- **OSC:** only one copy sends OSC, chosen with a switch.
- **Audio thread:** it never waits on the network. Received blocks and acks pass between threads through lock-free queues.

Reuse as reference only (not built on): `superconductor/audio_receiver.py` (the role, ack and underrun handling), and `superconductor/magenta_remote.py` (`_update_roles`, how the clock is chosen).

Make targets in the root `Makefile`: `make playback-build` (CMake build of all formats) and `make playback` (runs the Standalone app).

## Branch `touchdesigner` (worktree `superconductor_touchdesigner`)

1. **Listen-only tracking endpoint (client-web code, additive):** `superconductor/web_panel.py` gets `WS /ws/tracking`. `send_overlay()` already turns every overlay into JSON text, and that same text is also sent to every `/ws/tracking` listener (objects, hands, combos, mrt2, status, plus the frame width/height). It is documented in `CAMERA_PROTOCOL.md`. Nothing the camera page sees changes.
2. **Listen-only hands endpoint (conducting code, additive):** `conducting_web/server.py` gets `WS /ws/hands/monitor`, which receives every hands frame (21 points per hand, handedness) and the conductor's reply (tempo, gesture state). This file lives on the `conducting_sendtoserver` branch, so the change is committed there.
3. **TD project** in `touchdesigner/`:
   - `build_network.py`, run once from TD's Textport. It creates the network: an Audio VST CHOP loading the SC plugin, OSC In CHOP, two WebSocket DATs (tracking, hands) with callbacks that parse the JSON into Table DATs and CHOPs (normalised 0–1 coordinates), and one Video Device In TOP per camera. It's a script because `.toe` files are binary and can't be written from code. After it runs, you save `superconductor.toe` and it is committed.
   - `README.md`: ports, what each operator holds, how to add a camera.

## Order of work

1. `playback/shared` + Standalone app → hear it on speakers, then through BlackHole.
2. Plugin (VST3/AU) → load in a DAW, check clock and monitor switching.
3. Load the plugin in TD's Audio VST CHOP.
4. Features + tempo + OSC → OSC In CHOP in TD.
5. `/ws/tracking` and `/ws/hands/monitor` → WebSocket DATs in TD.
6. Webcams in TD, then save the `.toe`.

## Verification

All tests run on the real ports through the make targets (`make server`, `make client`, `make conduct`), and I stop the servers afterwards.
- **Standalone:** start the app as clock. The camera page goes quiet, the music keeps going, and the client's stats count no underruns. Close the app and the page plays again.
- **BlackHole:** set the Standalone output to BlackHole 2ch, record it in the DAW.
- **DAW plugin as monitor while the page is the clock:** it plays along with no gaps for 10+ minutes. This tests the drift correction.
- **DAW plugin as clock:** the server keeps generating, paced by the DAW.
- **TD:** Audio VST CHOP shows audio channels; OSC In shows loudness/bands/onset/bpm/beat changing with the music; the beat matches what is heard.
- **Data:** wave an object at the camera page and the WebSocket DAT tables update. Conduct on the conducting page and the hand points update. Both pages behave exactly as before.
- Existing tests: `make test` on client-web and on the conducting branch still pass.
