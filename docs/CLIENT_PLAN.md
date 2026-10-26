# Browser detection + tracking for the web client (superconductor_client_web)

## Context

Today, in web mode (`sc-tracking`), the browser page (`superconductor_client/tracking/app.html`) only captures the camera and plays audio. The client process does the rest:
- decodes each JPEG
- runs YOLOE + BoT-SORT (`ObjectTracker.__call__`, `object_tracking/tracker.py:91`)
- runs DINO and `IdentityRegistry`, then `ParameterMapper`
- sends `UpdateRecipe` / `UpdateControls` to the MRT2 server
- relays the server's audio, already tempo-stretched by the server's `tempo.py`, back to the page

Because detection runs in the process, the process needs a GPU, and that is what made the container slow (Docker on the laptop) or broken (lh2300's L40S when the process could not open it).

**Goal:** detection and tracking run in the browser by default.
- **Tracking** (`tracker_js`) is plain JS on the CPU, at about 3 µs per frame.
- **Detection** (the YOLOE network) uses the GPU through WebGPU, at about 25 ms per frame on the M3 Pro.

WebGPU is on by default in Chrome/Edge (Windows, macOS), Safari 26 and Firefox 141+ (Windows) / Apple Silicon.

The browser sends the client process the objects it found, as `TrackedObject`s. Everything after detection stays in Python almost unchanged.

**Compatibility mode:** when the browser has no WebGPU, or when it's forced, the page sends plain frames. The client process then runs YOLOE + BoT-SORT in Python (`ObjectTracker.__call__`). The browser's CPU path (WASM, about 182 ms per frame) is not used.

**Visualization is always in the browser.** The client process no longer draws on frames or sends pictures back; that round trip made the view unresponsive. It sends what to show as JSON (boxes, names, colors, the prompt panel, …), and the page draws it over its own live video.

**One client, three ways to run it.** Native (conda env on disk), Docker and Singularity run the same `sc-tracking` in web mode.
- The camera and the view are always in the browser, so the OpenCV window mode is removed.
- Sound plays in the browser by default. The sounddevice playback becomes a separate small receiver program (`sc-audio`) that the audio can be routed to instead.

The parts talk only through documented interfaces, so other people can replace one part (another detector, another audio receiver) without touching the rest. Client and server keep talking over websocket/HTTP even when both run on lh2300, so they can be moved apart later.

**Already checked (feasibility):**
- `tracker_js/` reproduces Ultralytics' BoT-SORT, with the same IDs on every frame in all 8 scenarios.
- YOLOE-11s without the seg head gives the same boxes as `yoloe-11s-seg`.
- The ONNX export matches PyTorch.
- `vision_js/yoloe.js` gives the same detections as Ultralytics: boxes within 1.4 px.
- On the M3 Pro in Chrome, WebGPU takes about 34 ms per frame for everything, versus about 185 ms on the single-threaded CPU fallback.

## Parts and the interfaces between them

```
browser (app.html + vision_js + tracker_js)          detector (swappable) + the only display
   │  WS /ws/camera  up:   "frame" = JSON header (objects as TrackedObject fields, or none in compatibility mode) + JPEG
   │                 down: "overlay" JSON (what to draw), never pictures
   ▼
client process (sc-tracking)                     unchanged pipeline after detection
   │  identity (DINO, histograms) → mapper → recipe/controls
   │  other sources, as today: prompts page (/prompts), client_midi POST /pattern (Atlas, MIDI player), describe server
   │  WS ws://<server>/stream  UpdateRecipe / UpdateControls / Pattern / ReceivedChunk   (unchanged)
   ▼
MRT2 server (server branch, unchanged)               generates, tempo-stretches, streams audio
   │  audio blocks back over the same /stream
   ▼
client process → WS /audio (client_midi port 8470)   audio interface: one "clock" sink acks, others listen
   ▼
browser player (AudioWorklet → output bus → speakers)   later: other receivers on the same /audio protocol
```

## Changes

### 1. Detector in the browser (new code in `vision_js/`; `tracker_js/` is finished)
- **`vision_js/yoloe.js`** (exists): letterbox, NMS and box scaling, already checked against Ultralytics.
- **`vision_js/worker.js`** (new): a Web Worker.
  - It loads onnxruntime-web with the WebGPU backend only. If `navigator.gpu` is missing or the session fails to start, it reports "no WebGPU" and the page switches to compatibility mode (section 2).
  - It loads the model from `/web/models/yoloe.onnx` and the class names from `/api/web`.
  - Per frame: the main thread transfers an `ImageBitmap` → OffscreenCanvas → `preprocess` → model → `postprocess` (conf/iou from config) → `BOTSORT.update`. It posts the tracks back.
  - Only one frame is in flight at a time; newer frames replace a waiting one, as `latest_frame()` does in the process today.
- **`vision_js/index.js`** (new): small API for the page, `startDetector(config)` → `detect(bitmap)` → `{objects, ms}`.
- **`superconductor_client/tracking/app.html`**: `sendFrames()` captures one bitmap per frame and sends it to the worker. It also encodes the same frame as JPEG (`toBlob`, as now) and sends one message when both are ready. The status line shows the mode (browser/container) and the detector's ms per frame.

### 2. The frame message (the detector → client interface)
This is one binary WS message per frame on `/ws/camera`:

```
[uint32 LE header length][header JSON, UTF-8][JPEG]
```

```json
{"type":"frame","seq":123,"width":1280,"height":720,
 "objects":[{"track_id":7,"class_name":"toy","confidence":0.71,
             "box":[x1,y1,x2,y2],"center":[cx,cy],"box_normalized":[x1,y1,x2,y2]}]}
```

- Field names and meanings are exactly those of `TrackedObject` (`tracker.py:24`): pixels for `box`, 0–1 for the others.
- Hands are included as objects with `class_name` = the hand class. The process splits them off as it does now.
- The fields that need pixels (`hist`, `color`, `embedding`, `crop`, `occluded`) are still computed in the process from the JPEG. DINO and identities stay in Python.
- **Two modes, chosen per frame by what the page sends:**
  - **Browser (default):** a header with objects + the JPEG. The process skips YOLOE.
  - **Compatibility:** a plain JPEG with no header, as today. The process runs YOLOE + BoT-SORT in Python (`ObjectTracker.__call__`, unchanged).
- **How the page picks the mode:**
  - It uses compatibility mode automatically when WebGPU is unavailable.
  - `[web] detection = "auto" | "browser" | "container"` in the config can force either mode (default `"auto"`).
  - The camera page has a switch to force compatibility mode, for testing.
  - The status line shows which mode is active.
- The protocol is documented in a new `CAMERA_PROTOCOL.md`, next to the existing `MIDI_MSG_PROTOCOL.md`.

### 3. Client process: minimal Python changes
- **`object_tracking/tracker.py`**: split `__call__` into two steps.
  - `_detect(frame)`: `model.track` → raw `(track_id, cls, conf, box)` list.
  - `track_detections(frame, detections, now)`: everything from line 98 to line 139 (building `TrackedObject`s, hands, occlusion, DINO crops, `registry.update`).
  - `__call__` calls both, so the native client and compatibility mode are unchanged.
  - The YOLO model is loaded on the first plain frame (compatibility mode), not at startup. A browser-only session therefore never loads it. DINO is always loaded.
  - BoT-SORT state: browser and container track IDs are separate counters. When a session switches modes, the registry treats the new IDs as new tracks, which it already handles when tracks are lost.
- **`tracking_view.py`**:
  - `/ws/camera` parses the frame message. `latest_frame()` returns `(seq, jpeg, header|None)`.
  - Static routes: `/web/vision_js/*`, `/web/tracker_js/*`, `/web/ort/*` (the bundled runtime) and `/web/models/yoloe.onnx`.
  - `/api/web` adds the class names, conf/iou and the detection mode.
- **`collab.py`**:
  - `run_web()` passes the header to `step()`, which calls `tracker.track_detections` when there is a header and `tracker(frame)` otherwise.
  - `step()` no longer draws. It returns the overlay (section 3b), which `run_web()` sends to the page.
  - Web mode becomes the only mode. These are removed:
    - `run()` (local webcam + `cv2.imshow`), `find_camera`, `handle_key` via `cv2.waitKey` (the page already forwards `[`, `]`, `c`) and `_on_mouse` (the page already forwards clicks)
    - `draw()`, `draw_hud()`, `_dashed_rect`
    - the `--web`, `--camera` and `--iphone` options
  - `superconductor_client/client.py` no longer opens a sound device itself. All playback goes through `/audio` sinks: the browser, or `sc-audio` (section 5).
  - Makefile targets (`client`, `client-remote`, `vision`) all start web mode.

### 3b. Visualization: the overlay message (client → browser)
After each frame it handles, the process sends one JSON text message on `/ws/camera`. The data is what `draw()`/`draw_hud()` draw today, as data. Label text and colors are computed in Python, so the page only draws.

```json
{"type":"overlay","seq":123,"width":1280,"height":720,
 "reference":{"x":0.5,"y":0.6,"radius":0.6},
 "objects":[{"identity":3,"class_name":"toy","box":[x1,y1,x2,y2],"center":[cx,cy],
             "color":"#50c850","coasted":false,"combined":false,"progress":null,
             "label":"#3 toy -> flute 0.72"}],
 "pending":[{"box":[...],"label":"? new"}],
 "hands":[[x1,y1,x2,y2]],
 "combos":[{"active":true,"box":[...],"centers":[[cx,cy],...],"color":"#...","label":"..."}],
 "message":"touch the bulbasaur",
 "mrt2":{"rows":[{"prompt","source","value","share","color"}],"controls":{}},
 "log":["..."],
 "status":{"vision_fps":28,"track_ms":12,"music":"mrt2_small (mlx) x1.4 realtime"}}
```

- **Coordinates:** boxes are in the frame's unmirrored pixels, as `TrackedObject` has them. The page draws on a canvas laid over the video, scaled to the video's on-screen rectangle, and mirrors both together. Text is drawn unmirrored.
- **Two overlay builders in `collab.py`:**
  - `overlay(objects)` replaces `draw()`.
  - `hud()` replaces `draw_hud()`, reusing `mrt2_input()` as is.
- **Delay is shown:**
  - `seq` says which frame the overlay belongs to. The page remembers when it sent each frame and shows the processing delay (send → overlay back).
  - Boxes come from the process, so they visibly trail moving objects by that delay. That's intended, for debugging.
- **Stale data:** if no overlay has arrived for more than about 1 s, the page dims it and shows "no data".
- **`superconductor_client/tracking/app.html`:**
  - The `<img id="view">` that showed the returned JPEG is replaced by the live `<video>` plus a `<canvas>` on top.
  - The drawing code lives in `superconductor_client/tracking/overlay.js`, one function per element: boxes (dashed when coasted), reference circle, combo frames and lines, touch progress bar, prompt panel with bars, banners, log, status line.
- **Documentation:** the overlay is documented in `CAMERA_PROTOCOL.md` next to the frame message, so someone else's page can draw it too.

### 4. The model and the runtime, bundled and reproducible
- **`scripts/export_web_model.py`** + `make web-model`: build the detection-only YOLOE-11s from `yoloe-11s-seg.pt` (as checked), `set_classes` from `[tracking] classes`, and export ONNX at 480×288 to `superconductor_client/tracking/var/web/yoloe.onnx` (gitignored). The class names are stored in the ONNX metadata.
- **At client startup** (web + browser detection): if the model is missing or its stored classes differ from the config, run the export. In the container, it goes to `/data` (writable).
- **`superconductor_client/tracking/fetch_ort.sh`** + `make web-deps`: get `onnxruntime-web@1.30.0`, check its sha256, and copy its 3 files into `superconductor_client/tracking/var/web/ort/`. The runtime is served by the client, not a CDN, and it isn't committed to git because it's 27 MB.
- **`client_container/Dockerfile`**: run both steps at build time. The text encoder is already downloaded there by the `ObjectTracker` warm-up.
- **Torch with CUDA 12.6** (`client_container/requirements.txt`, from PyTorch's cu126 index). The lh2300 GPU is an L40S (Ada, sm_89). DINO, and YOLOE in compatibility mode, can use it when the card is in Default compute mode. Exclusive_Process leaves them on CPU.
- **Entrypoint GPU check** (`client_container/entrypoint.sh`): run a real CUDA operation instead of `torch.cuda.is_available()`, so an unsupported GPU falls back to CPU instead of crashing.

### 5. Audio set up for routing later
- **`superconductor_client/client.py` `_BrowserAudio` → `AudioSinks`**: `/audio?role=clock|monitor`.
  - The clock sink acks blocks, which keeps credits and the server's beat grid tied to what is actually heard. Only one clock exists at a time: a new clock connection takes over, and the previous clock becomes a monitor.
  - Monitor sinks get the same blocks and never ack.
  - The browser connects as the clock by default. When it is demoted to monitor, it mutes itself and its status shows where the sound plays.
  - When the clock disconnects, the most recent connection that asked to be clock takes over again (e.g. stopping sc-audio hands playback back to the browser). Blocks not yet acked are re-sent to it, as `_BrowserAudio` already does on reconnect.
- **`audio_receiver.py`** (repo root, `make audio`): the sounddevice playback moved out of `superconductor_client/client.py`. Its only job is to play the audio. It is not part of the client web app.
  - It connects to `/audio` on the client process (`--url`, default `ws://localhost:8470/audio`; through the gateway, `ws://localhost:8080/audio`), with `--role clock|monitor` and `--device`.
  - As clock it acks each block after the sounddevice callback has played it, as today.
  - It runs on the machine with the speakers. For lh2300 that means the laptop, through the SSH tunnel.
  - `make audio` starts it.
  - This is the template for later receivers (TouchDesigner, DAW, stem separation): the same `/audio` protocol, with no change to the server or the client process.
- **`app.html`**: the worklet → a `GainNode` output bus → `destination`, so a later routing step taps the bus.
- The protocol (binary `[uint32 seq][float32 stereo 48 kHz]`, JSON `ack`/`underrun`/`pause`) is documented in `AUDIO_PROTOCOL.md`.

### 6. lh2300 setup
- Nothing changes in the server. The client container reaches the server at `ws://localhost:9100/stream` (`make client-singularity`, as now).
- The browser on the laptop reaches the gateway through the SSH tunnel to port 8080. That counts as `localhost`, so WebGPU and the camera work.
- `README.md` and `CLIENT_CONTAINER.md` get a short "browser detection" section.

## Files
- New:
  - `vision_js/worker.js`
  - `vision_js/index.js`
  - `vision_js/test/` (fixtures + parity test, as in `tracker_js/test/`)
  - `superconductor_client/tracking/overlay.js` (draws the overlay message)
  - `audio_receiver.py` (`make audio`)
  - `scripts/export_web_model.py`
  - `superconductor_client/tracking/fetch_ort.sh`
  - `CAMERA_PROTOCOL.md`
  - `AUDIO_PROTOCOL.md`
- Modified:
  - `superconductor_client/tracking/app.html`
  - `superconductor_client/tracking/tracking_view.py`
  - `superconductor_client/tracking/collab.py` (`run_web`, `step`, overlay builders replacing `draw`/`draw_hud`, window mode removed, config)
  - `superconductor_client/tracking/object_tracking/tracker.py` (the split)
  - `superconductor_client/client.py` (audio sinks)
  - `collab.container.toml` / `collab.toml` (`[web] detection`)
  - `Makefile`
  - `client_container/Dockerfile`
  - `.gitignore` (`superconductor_client/tracking/var/`)
- Reused as-is:
  - `tracker_js/`
  - `vision_js/yoloe.js`
  - `IdentityRegistry`
  - `ParameterMapper`
  - `MagentaClient`
  - the server

Note: the working tree has uncommitted edits in `collab.py`, `superconductor_client/client.py`, `tracker.py`, `tracking_view.py` and `Makefile` that predate this work. The plan builds on them as they are.

## Verification
- **Unit tests:**
  - `tests/test_object_tracking.py`: a new test feeds identical detections through `__call__` (a mocked `_detect`) and through `track_detections`, and checks that the identities and objects are the same.
  - A test for frame-message parsing.
  - An overlay test: synthetic objects through `step()` give an overlay with the same boxes, labels and colors that `draw()` used to compute (JSON-serializable, coordinates in frame pixels).
  - A test for monitor sinks never acking.
- **JS:**
  - `node --test tracker_js/test/*.test.js` (exists).
  - `vision_js` parity against Ultralytics on recorded frames: boxes within 2 px, the same classes.
- **End to end, on the real ports** (`make server`, then `make client`):
  - Open `http://localhost:8467/` in Chrome. Check that the status shows `webgpu` and a detector time.
  - Check that objects get identities and the recipe changes.
  - Check that the overlay draws over the live video: mirrored correctly, the delay shown, and "no data" when the client process is stopped.
  - Check that audio plays in the browser.
  - Start `make audio` (sc-audio as clock): sound moves to the sound device, the browser mutes and says so, and the server keeps streaming.
  - Stop sc-audio: the browser takes over as clock again.
  - Check that a `--role monitor` receiver plays along without affecting credits.
  - Force compatibility mode with the page switch: objects are still detected (by the process) and get identities, and the status shows "container".
  - Stop the servers afterwards.
- **You:** the final test with the real camera and toys, plus `make client-singularity` on lh2300.
