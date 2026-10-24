// The JS tracker against Ultralytics' BOTSORT on the same detections (fixtures from
// make_fixtures.py): same track ids, scores, classes and detection indices every frame, boxes
// equal up to float32 rounding (Ultralytics keeps boxes in float32, this port in float64).
//
//   node --test tracker_js/test/

import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync, readdirSync } from "node:fs";
import { BOTSORT } from "../index.js";
import { linearAssignment } from "../assignment.js";

const FIXTURES = new URL("./fixtures/", import.meta.url);
const BOX_TOL = 1e-2;  // pixels
const SCORE_TOL = 1e-3;  // outputs are stored to 3 decimals

const load = (name) => JSON.parse(readFileSync(new URL(name, FIXTURES), "utf8"));

for (const file of readdirSync(FIXTURES).filter((f) => f.endsWith(".json") && f !== "assignment.json")) {
  test(`tracker parity: ${file}`, () => {
    const { config, frames, outputs } = load(file);
    const tracker = new BOTSORT(config);
    let maxBoxErr = 0;
    frames.forEach((dets, f) => {
      const boxes = new Float64Array(4 * dets.length), scores = new Float64Array(dets.length);
      const classes = new Float64Array(dets.length);
      dets.forEach((d, i) => {  // the float32 values Ultralytics got
        for (let j = 0; j < 4; j++) boxes[4 * i + j] = Math.fround(d[j]);
        scores[i] = Math.fround(d[4]);
        classes[i] = d[5];
      });
      const got = tracker.update(boxes, scores, classes);
      const want = outputs[f];
      const ids = (rows) => rows.map((r) => r[4]);
      assert.deepEqual(got.map((t) => t.trackId), ids(want), `frame ${f + 1}: track ids`);
      got.forEach((t, k) => {
        const [x1, y1, x2, y2, , score, cls, idx] = want[k];
        assert.equal(t.idx, idx, `frame ${f + 1}, track ${t.trackId}: detection index`);
        assert.equal(t.cls, cls, `frame ${f + 1}, track ${t.trackId}: class`);
        assert.ok(Math.abs(t.score - score) < SCORE_TOL, `frame ${f + 1}, track ${t.trackId}: score`);
        [x1, y1, x2, y2].forEach((v, j) => { maxBoxErr = Math.max(maxBoxErr, Math.abs(t.box[j] - v)); });
      });
    });
    assert.ok(maxBoxErr < BOX_TOL, `largest box difference ${maxBoxErr} px`);
  });
}

test("assignment parity: lap.lapjv with cost_limit", () => {
  const { cases } = load("assignment.json");
  const rowToCol = new Int32Array(64), colToRow = new Int32Array(64);
  for (const { n, m, thresh, cost, matches } of cases) {
    linearAssignment(Float64Array.from(cost), n, m, thresh, rowToCol, colToRow);
    const got = [];
    for (let r = 0; r < n; r++) if (rowToCol[r] >= 0) got.push([r, rowToCol[r]]);
    assert.deepEqual(got, matches, `${n}x${m}, thresh ${thresh}`);
  }
});
