# tracker_js

Browser port of the multi-object tracker that `model.track()` runs inside the client (`tracking/object_tracking/tracker.py`). It is Ultralytics' BoT-SORT (`ultralytics/trackers/bot_sort.py`, 8.4.162) as configured by `botsort_superconductor.yaml`, which turns off camera-motion compensation and ReID. That leaves ByteTrack's association with BoT-SORT's `KalmanFilterXYWH`. It gives each detection the same `track_id` that `IdentityRegistry` gets now.

Plain ES modules with no dependencies or build step.

```js
import { BOTSORT } from "./tracker_js/index.js";

const tracker = new BOTSORT();          // botsort_superconductor.yaml values; pass overrides by YAML key
const tracks = tracker.update(boxes, scores, classes);
// boxes: flat (x1, y1, x2, y2) per detection in pixels; scores, classes: per detection
for (const t of tracks) t.trackId, t.box, t.score, t.cls, t.idx;   // idx: the detection it matched this frame
```

`update` returns the confirmed tracks in Ultralytics' order. `t.box` is the Kalman-filtered box, which is what `model.track()` reports, not the raw detection. The array and the `Track` objects are reused on the next call.

## Files

- `botsort.js`: `BYTETracker.update` / `BOTSORT` step by step. This includes the order of the track lists (it decides ties, and therefore ids) and Ultralytics' handling of removed tracks.
- `kalman.js`: `KalmanFilterXYWH` as four independent 2×2 filters (see below).
- `assignment.js`: `lap.lapjv(cost, extend_cost=True, cost_limit=thresh)`, the globally optimal matching that Ultralytics' `linear_assignment` uses.
- `test/make_fixtures.py`: runs Ultralytics' own `BOTSORT` and `lap` on synthetic scenes and writes `test/fixtures/`. Run it in sc_env; rerun after an Ultralytics upgrade.
- `test/parity.test.js`: checks that every frame of every scene gives the same track ids, detection indices, classes and scores. Boxes must agree within 0.01 px (Ultralytics works in float32).
- `test/bench.js`: time per frame.

```
node --test tracker_js/test/*.test.js
node tracker_js/test/bench.js
```

## Performance

Measured with `bench.js` on an M3 Pro (Node 23), with the same fixtures run through Ultralytics' Python tracker for comparison:

| scene | detections/frame | this port | Ultralytics (Python) |
|---|---|---|---|
| steady | 3.9 | 2.1 µs | 209 µs |
| low_scores | 6.2 | 2.8 µs | 341 µs |
| crowded | 21.5 | 80 µs | 1196 µs |

Where the speed comes from:

- **The Kalman filter decouples.** Ultralytics uses 8×8 matrices, but every matrix it combines is diagonal or only links a coordinate with its own velocity. The covariance therefore stays block-diagonal, so each step costs a few multiplications per coordinate instead of 8×8 products and a 4×4 solve. This is exact, not an approximation. It holds only without GMC, which mixes coordinates; turning GMC on would need the full matrices.
- **No allocation per frame.** Detections, cost matrices and solver state live in typed scratch buffers that grow on demand. A `Track` is allocated only when a new track starts.
- **Assignment pruning.** Rows and columns with no cost below the threshold are dropped before solving. They can never be matched, so the result is exactly the same.

## The reference ports

Four existing JS trackers were read for ideas. No code was copied.

- [nomi30701/YOLO-ByteTrack-ONNX-Web](https://github.com/nomi30701/YOLO-ByteTrack-ONNX-Web) `src/utils/tracker.js`, and [NyiNyiMyo/Makers.WebTrackYOLOv11onnx](https://github.com/NyiNyiMyo/Makers.WebTrackYOLOv11onnx), which has the same file.
  - Taken: reusing buffers instead of allocating each frame, and skipping pairs that can't match before the expensive step. That became exact pruning here, not a distance heuristic.
  - Not taken: its Kalman filter runs on generic mathjs matrices and uses different noise constants from Ultralytics. It also thresholds after a plain Hungarian assignment instead of solving with a cost limit, and it has no unconfirmed-track stage. Its ids would therefore differ from `model.track()`.
- [louischuang/V7RC-Webyolo-Bytetrack](https://github.com/louischuang/V7RC-Webyolo-Bytetrack) `app/lib/bytetrack.ts`.
  - Not taken: it uses greedy matching sorted by score, and a smoothed velocity instead of a Kalman filter. That's fast, but not optimal, and not what Ultralytics does.
- [opendatacam/node-moving-things-tracker](https://github.com/opendatacam/node-moving-things-tracker) `tracker.js`.
  - Not taken: a kd-tree for nearest-detection search. It pays off with hundreds of objects; at webcam scale the full cost matrix takes microseconds.
