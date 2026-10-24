// Time per frame of the tracker on the fixture scenes (detections prepared up front, as typed
// arrays, the way a detector's output would arrive).
//
//   node tracker_js/test/bench.js

import { readFileSync } from "node:fs";
import { BOTSORT } from "../index.js";

const ROUNDS = 50;

for (const name of ["steady", "occlusions", "low_scores", "crowded"]) {
  const { config, frames } = JSON.parse(readFileSync(new URL(`./fixtures/${name}.json`, import.meta.url), "utf8"));
  const input = frames.map((dets) => ({
    boxes: Float32Array.from(dets.flatMap((d) => d.slice(0, 4))),
    scores: Float32Array.from(dets, (d) => d[4]),
    classes: Int32Array.from(dets, (d) => d[5]),
  }));
  const run = () => {
    const tracker = new BOTSORT(config);
    for (const { boxes, scores, classes } of input) tracker.update(boxes, scores, classes);
  };
  for (let i = 0; i < 5; i++) run();  // warm up the JIT
  const t0 = performance.now();
  for (let i = 0; i < ROUNDS; i++) run();
  const us = ((performance.now() - t0) * 1000) / (ROUNDS * frames.length);
  const dets = frames.reduce((s, f) => s + f.length, 0) / frames.length;
  console.log(`${name.padEnd(11)} ${dets.toFixed(1).padStart(5)} detections/frame  ${us.toFixed(1).padStart(6)} µs/frame`);
}
