# Camera protocol (`WS /ws/camera`)

The camera page (`superconductor/app.html`) and the client process (`sc-collab`, `collab.py` + `web_panel.py`) talk over one websocket, `ws://<client>:8467/ws/camera` (through the container's gateway: `ws://localhost:8080/ws/camera`). The page sends frames up. The client answers each frame it handles with an overlay: what to draw over the video. No pictures ever come back down.

Anything that speaks this protocol can replace the page's detector or the page itself.

## Page → client

### Frame (binary)

```
[uint32 little-endian: header length n][n bytes: header JSON, UTF-8][JPEG]
```

```json
{"type": "frame", "seq": 123, "session": "6f1c…", "width": 960, "height": 540,
 "objects": [{"track_id": 7, "class_name": "toy", "confidence": 0.71,
              "box": [x1, y1, x2, y2], "center": [cx, cy], "box_normalized": [x1, y1, x2, y2]}]}
```

| Field | Meaning |
|---|---|
| `seq` | Frame number. The overlay for this frame echoes it, so the sender can measure the delay. |
| `session` | Identifies the detector (one per page load). Each detector numbers its tracks from 1, so the client keeps each session's `track_id`s apart. |
| `width`, `height` | Size of the JPEG, in pixels. |
| `objects` | What the page detected and tracked in this frame. Fields are those of `TrackedObject` (`object_tracking/tracker.py`): `box` in pixels, `center` and `box_normalized` in 0–1. Hands are objects whose `class_name` is the hand class. **If `objects` is missing, the client detects** with YOLOE + BoT-SORT itself (compatibility mode). An empty list means "nothing detected". |

The JPEG is still needed when the page detects. The client cuts object crops from it for recognition (DINOv2), color histograms, the object library's pictures and the describe server.

A plain JPEG with no header is also accepted. It means the same as a header without `objects`. A header must be shorter than 55 KB, since its length must not look like a JPEG start marker.

Send one frame at a time: send the next after the overlay for the last came back, or after about a second. The client only ever keeps the newest frame.

### Click and key (text)

```json
{"type": "click", "x": 0.42, "y": 0.6}
{"type": "key", "key": "["}
```

The click is a position on the view, as displayed (mirrored), from 0 to 1. Keys: `[` and `]` change the combo distance; `c` calibrates (calibration mode).

## Client → page

### Overlay (text), one per frame handled

```json
{"type": "overlay", "seq": 123, "width": 960, "height": 540,
 "reference": {"x": 0.5, "y": 0.6, "radius": 0.6},
 "objects": [{"identity": 3, "track_id": 12, "class_name": "toy", "box": [x1, y1, x2, y2], "center": [cx, cy],
              "color": "#50c850", "coasted": false, "combined": false, "progress": null,
              "label": "#3 toy -> flute 0.72"}],
 "pending": [{"box": [x1, y1, x2, y2], "label": "? new"}],
 "hands": [[x1, y1, x2, y2]],
 "combos": [{"active": true, "box": [x1, y1, x2, y2], "centers": [[cx, cy], [cx, cy]],
             "color": "#dcdc50", "label": "bulbasaur + chimchar = friendly jungle beat"}],
 "message": "touch the bulbasaur",
 "mrt2": {"rows": [{"prompt": "ambient electronic", "source": "base", "value": 0.3, "share": 0.25,
                    "color": "#c8c8c8"}],
          "controls": {"temperature": 1.1}},
 "banners": [{"text": "bulbasaur + chimchar combined: friendly jungle beat", "color": "#dcdc50"}],
 "log": ["#3 toy: plays flute"],
 "status": {"vision_fps": 14.8, "track_ms": 21.0, "detector": "browser",
            "music": "mrt2_small (mlx) x1.42 realtime  underruns 0"}}
```

- **Coordinates** are the frame's pixels, **unmirrored** (as `TrackedObject` has them). The page shows the video mirrored, so it flips x when drawing (`superconductor/overlay.js`).
- `reference`: the point objects are measured from, in 0–1 of the frame, unmirrored. `radius` is the distance (a fraction of the frame width) at which an object's influence reaches 0.
- `objects`: the objects the client identified, with the client's view of them:
  - `identity`: the persistent object id.
  - `coasted`: the object isn't detected right now, so the box is its last known position (drawn dashed).
  - `combined`: it's part of an active combo.
  - `progress`: 0–1 while it is being touched in calibration mode.
  - `label`: ready to show, colored `color`.
- `pending`: tracks still being identified. `hands`: hand boxes. `combos`: an active combo's frame and links, or two members' distance while apart.
- `message`: the calibration instruction. `mrt2`: the prompts and sampling controls sent to MRT2, as client_midi's `GET /input` serves them. `banners`: active combos. `log`: recent messages.
- `status`:
  - `detector`: `"browser"` or `"container"`, i.e. who detected this frame.
  - `track_ms`: the client's time for the frame, from identities to the mapping.

Boxes are where the objects were when the frame was taken, so on screen they trail moving objects by the processing delay. The page also dims the overlay when none has arrived for a second.

### Error (text)

```json
{"type": "error", "error": "frame: frame header is not a frame"}
```

The client sends this when it can't parse a frame message. The frame is skipped.

## Listening in (`WS /ws/tracking`)

`ws://<client>:8467/ws/tracking` (through the gateway: `ws://localhost:8080/ws/tracking`) gets every overlay the camera page gets, as the same JSON text, while the page is sending frames. It's listen-only: anything sent to it is ignored, and the page is unaffected. TouchDesigner reads it this way.

Coordinates are the camera page's frame pixels, unmirrored. Divide by the overlay's `width` and `height` for 0–1.
