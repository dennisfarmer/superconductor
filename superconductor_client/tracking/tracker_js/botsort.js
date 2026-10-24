// Port of the tracker `model.track()` runs in superconductor (object_tracking/tracker.py):
// Ultralytics' BOTSORT (ultralytics/trackers/bot_sort.py, 8.4.162) as configured by
// object_tracking/botsort_superconductor.yaml, i.e. with camera-motion compensation (GMC) and
// ReID off. What remains is ByteTrack's association (byte_tracker.py) with BOTSORT's Kalman
// filter (KalmanFilterXYWH, see kalman.js). Step for step the same as Ultralytics, including the
// order of track lists, which decides ties and the output order, so track ids match.

import { initiate, predict, update as kalmanUpdate } from "./kalman.js";
import { linearAssignment } from "./assignment.js";

/** defaults: botsort_superconductor.yaml (the keys used without GMC / ReID); the camera page
 * passes the client's own copy of that file, so the two can't drift apart */
export const SUPERCONDUCTOR_CONFIG = Object.freeze({
  track_high_thresh: 0.15,
  track_low_thresh: 0.1,
  new_track_thresh: 0.15,
  track_buffer: 300,
  match_thresh: 0.8,
  fuse_score: true,
});

export const TrackState = Object.freeze({ New: 0, Tracked: 1, Lost: 2, Removed: 3 });

const REMOVED_BUFFER = 1000;  // stracks.merge_track_pools: removed tracks remembered
const DUPLICATE_THRESH = 0.15;  // stracks.remove_duplicate_stracks: IoU distance
const SECOND_MATCH_THRESH = 0.5;  // second association (low-score detections), IoU only
const UNCONFIRMED_MATCH_THRESH = 0.7;
const IOU_EPS = 1e-7;  // ultralytics.utils.metrics.bbox_ioa

export class Track {
  constructor() {
    this.trackId = 0;
    this.state = TrackState.New;
    this.isActivated = false;
    this.score = 0;
    this.cls = 0;
    this.idx = 0;  // index of the detection it was last updated with, in this frame's input
    this.frameId = 0;  // last frame it was matched
    this.startFrame = 0;
    this.trackletLen = 0;
    this.mean = new Float64Array(8);
    this.cov = new Float64Array(12);
    /** (x1, y1, x2, y2) of the Kalman state, set for the tracks update() returns */
    this.box = new Float64Array(4);
    this._mark = 0;  // list-membership stamp (joint / sub without allocating sets)
    this._removedRefs = 0;  // entries in the tracker's removed list
  }

  /** (x1, y1, x2, y2) of the Kalman state into out[o..o+3], as STrack.xyxy */
  xyxy(out, o = 0) {
    const m = this.mean;
    const x1 = m[0] - m[2] / 2, y1 = m[1] - m[3] / 2;
    out[o] = x1; out[o + 1] = y1; out[o + 2] = x1 + m[2]; out[o + 3] = y1 + m[3];
  }
}

export class BOTSORT {
  constructor(config = SUPERCONDUCTOR_CONFIG) {
    this.args = { ...SUPERCONDUCTOR_CONFIG, ...config };
    this.reset();
  }

  reset() {
    this.trackedTracks = [];
    this.lostTracks = [];
    this.removedTracks = [];  // last REMOVED_BUFFER removals (may repeat a track)
    this._removedHead = 0;
    this.frameId = 0;
    this._nextId = 0;
    this._stamp = 0;
    this._output = [];
    // per-frame scratch, grown as needed
    this._capDet = 0;
    this._capCost = 0;
    this._capTrk = 0;
    this._growDet(32);
    this._growCost(1024);
    this._growTrk(64);
  }

  /**
   * One frame of detections, as BYTETracker.update.
   * `boxes`: (x1, y1, x2, y2) per detection, flat (length >= 4n), in pixels.
   * `scores`, `classes`: per detection. `n`: number of detections (default scores.length).
   * Returns the confirmed tracks (state Tracked, activated) with `box` set, in Ultralytics'
   * order. The array and the Track objects are reused by the next call.
   */
  update(boxes, scores, classes, n = scores.length) {
    const args = this.args;
    this.frameId++;
    const frameId = this.frameId;
    this._growDet(n);
    // every association has at most this many rows, so rowToCol isn't reallocated midway
    this._growTrk(this.trackedTracks.length + this.lostTracks.length);

    // detections as STrack: xywh -> tlwh -> measurement, the same float steps as Ultralytics
    const dBox = this._dBox, dMeas = this._dMeas, dScore = this._dScore, dCls = this._dCls;
    const high = this._high, low = this._low;
    let nHigh = 0, nLow = 0;
    for (let i = 0; i < n; i++) {
      const x1 = boxes[4 * i], y1 = boxes[4 * i + 1], x2 = boxes[4 * i + 2], y2 = boxes[4 * i + 3];
      const cx = (x1 + x2) / 2, cy = (y1 + y2) / 2, w = x2 - x1, h = y2 - y1;
      const tx = cx - w / 2, ty = cy - h / 2;
      dBox[4 * i] = tx; dBox[4 * i + 1] = ty; dBox[4 * i + 2] = tx + w; dBox[4 * i + 3] = ty + h;
      dMeas[4 * i] = tx + w / 2; dMeas[4 * i + 1] = ty + h / 2; dMeas[4 * i + 2] = w; dMeas[4 * i + 3] = h;
      dScore[i] = scores[i];
      dCls[i] = classes[i];
      if (!(w > 0 && h > 0)) continue;  // zero-size boxes would divide by zero in the filter
      const s = scores[i];
      if (s >= args.track_high_thresh) high[nHigh++] = i;
      else if (s > args.track_low_thresh) low[nLow++] = i;
    }

    const activated = [], refind = [], lost = [], removed = [];

    // confirmed tracks and lost ones are predicted and matched first; unconfirmed (new last
    // frame) only get the high-score detections left over, without prediction
    const unconfirmed = [], pool = [];
    for (const t of this.trackedTracks) (t.isActivated ? pool : unconfirmed).push(t);
    this._joint(pool, this.lostTracks);
    for (const t of pool) {
      if (t.state !== TrackState.Tracked) { t.mean[6] = 0; t.mean[7] = 0; }  // BOTrack: no size velocity
      predict(t.mean, t.cov);
    }

    // first association: all of them with the high-score detections
    const rowToCol = this._rowToCol, colToRow = this._colToRow;
    this._dists(pool, high, nHigh, args.fuse_score);
    linearAssignment(this._cost, pool.length, nHigh, args.match_thresh, rowToCol, colToRow);
    const uTrack = [];
    for (let r = 0; r < pool.length; r++) {
      const c = rowToCol[r];
      if (c < 0) uTrack.push(r);
      else this._applyMatch(pool[r], high[c], activated, refind);
    }
    const uDet = this._uDet;
    let nUDet = 0;
    for (let c = 0; c < nHigh; c++) if (colToRow[c] < 0) uDet[nUDet++] = high[c];

    // second association: still-unmatched Tracked ones with the low-score detections, IoU only
    const rTracked = [];
    for (const r of uTrack) if (pool[r].state === TrackState.Tracked) rTracked.push(pool[r]);
    if (rTracked.length && nLow) {
      this._dists(rTracked, low, nLow, false);
      linearAssignment(this._cost, rTracked.length, nLow, SECOND_MATCH_THRESH, rowToCol, colToRow);
      for (let r = 0; r < rTracked.length; r++) {
        const c = rowToCol[r];
        if (c >= 0) this._applyMatch(rTracked[r], low[c], activated, refind);
        else if (rTracked[r].state !== TrackState.Lost) { rTracked[r].state = TrackState.Lost; lost.push(rTracked[r]); }
      }
    } else {
      for (const t of rTracked) if (t.state !== TrackState.Lost) { t.state = TrackState.Lost; lost.push(t); }
    }

    // unconfirmed tracks with the high-score detections nobody took; the rest are dropped
    let newDets = uDet, nNew = nUDet;
    if (unconfirmed.length) {
      this._dists(unconfirmed, uDet, nUDet, args.fuse_score);
      linearAssignment(this._cost, unconfirmed.length, nUDet, UNCONFIRMED_MATCH_THRESH, rowToCol, colToRow);
      for (let r = 0; r < unconfirmed.length; r++) {
        const t = unconfirmed[r], c = rowToCol[r];
        if (c >= 0) { this._updateTrack(t, uDet[c]); activated.push(t); }
        else { t.state = TrackState.Removed; removed.push(t); }
      }
      newDets = this._uDet2;
      nNew = 0;
      for (let c = 0; c < nUDet; c++) if (colToRow[c] < 0) newDets[nNew++] = uDet[c];
    }

    // new tracks; unconfirmed until matched next frame (except on the first frame)
    for (let k = 0; k < nNew; k++) {
      const d = newDets[k];
      if (dScore[d] < args.new_track_thresh) continue;
      const t = new Track();
      this._setDetection(t, d);
      t.trackId = ++this._nextId;
      initiate(t.mean, t.cov, dMeas[4 * d], dMeas[4 * d + 1], dMeas[4 * d + 2], dMeas[4 * d + 3]);
      t.trackletLen = 0;
      t.state = TrackState.Tracked;
      t.isActivated = frameId === 1;
      t.frameId = frameId;
      t.startFrame = frameId;
      activated.push(t);
    }

    for (const t of this.lostTracks) {
      if (frameId - t.frameId > args.track_buffer) { t.state = TrackState.Removed; removed.push(t); }
    }

    this._merge(activated, refind, lost, removed);

    const out = this._output;
    out.length = 0;
    for (const t of this.trackedTracks) {
      if (!t.isActivated) continue;
      t.xyxy(t.box);
      out.push(t);
    }
    return out;
  }

  // --- matching ---

  /** IoU cost (1 - IoU) of tracks x detections into this._cost, optionally fused with scores */
  _dists(tracks, dets, nDets, fuse) {
    const nT = tracks.length;
    this._growCost(nT * nDets);
    this._growTrk(nT);
    const cost = this._cost, dBox = this._dBox, dScore = this._dScore, tBox = this._tBox;
    for (let r = 0; r < nT; r++) tracks[r].xyxy(tBox, 4 * r);
    for (let r = 0; r < nT; r++) {
      const ax1 = tBox[4 * r], ay1 = tBox[4 * r + 1], ax2 = tBox[4 * r + 2], ay2 = tBox[4 * r + 3];
      const areaA = (ax2 - ax1) * (ay2 - ay1);
      const off = r * nDets;
      for (let k = 0; k < nDets; k++) {
        const d = dets[k];
        const bx1 = dBox[4 * d], by1 = dBox[4 * d + 1], bx2 = dBox[4 * d + 2], by2 = dBox[4 * d + 3];
        const iw = Math.min(ax2, bx2) - Math.max(ax1, bx1);
        const ih = Math.min(ay2, by2) - Math.max(ay1, by1);
        let iou = 0;
        if (iw > 0 && ih > 0) {
          const inter = iw * ih;
          iou = inter / (areaA + (bx2 - bx1) * (by2 - by1) - inter + IOU_EPS);
        }
        cost[off + k] = fuse ? 1 - iou * dScore[d] : 1 - iou;
      }
    }
  }

  _applyMatch(t, d, activated, refind) {
    if (t.state === TrackState.Tracked) {
      this._updateTrack(t, d);
      activated.push(t);
    } else {  // re_activate (same id)
      this._kalman(t, d);
      t.trackletLen = 0;
      t.state = TrackState.Tracked;
      t.isActivated = true;
      t.frameId = this.frameId;
      this._setDetection(t, d);
      refind.push(t);
    }
  }

  _updateTrack(t, d) {
    t.frameId = this.frameId;
    t.trackletLen++;
    this._kalman(t, d);
    t.state = TrackState.Tracked;
    t.isActivated = true;
    this._setDetection(t, d);
  }

  _kalman(t, d) {
    const m = this._dMeas;
    kalmanUpdate(t.mean, t.cov, m[4 * d], m[4 * d + 1], m[4 * d + 2], m[4 * d + 3]);
  }

  _setDetection(t, d) {
    t.score = this._dScore[d];
    t.cls = this._dCls[d];
    t.idx = d;
  }

  // --- track lists (stracks.py) ---

  /** joint_stracks in place: append the tracks of b whose id isn't in a */
  _joint(a, b) {
    const s = ++this._stamp;
    for (const t of a) t._mark = s;
    for (const t of b) if (t._mark !== s) { t._mark = s; a.push(t); }
    return a;
  }

  /** sub_stracks: the tracks of a whose id isn't in b */
  _sub(a, b) {
    const s = ++this._stamp;
    for (const t of b) t._mark = s;
    return a.filter((t) => t._mark !== s);
  }

  _merge(activated, refind, lost, removed) {
    let tracked = this.trackedTracks.filter((t) => t.state === TrackState.Tracked);
    tracked = this._joint(tracked, activated);
    tracked = this._joint(tracked, refind);
    let lostTracks = this._sub(this.lostTracks, tracked);
    for (const t of lost) lostTracks.push(t);
    // removed: the tracks in the removed list before this frame's removals are added
    lostTracks = lostTracks.filter((t) => t._removedRefs === 0);
    [this.trackedTracks, this.lostTracks] = this._removeDuplicates(tracked, lostTracks);
    for (const t of removed) { t._removedRefs++; this.removedTracks.push(t); }
    const excess = this.removedTracks.length - this._removedHead - REMOVED_BUFFER;
    for (let k = 0; k < excess; k++) this.removedTracks[this._removedHead++]._removedRefs--;
    if (this._removedHead > REMOVED_BUFFER) {  // compact now and then
      this.removedTracks = this.removedTracks.slice(this._removedHead);
      this._removedHead = 0;
    }
  }

  /** remove_duplicate_stracks: of a tracked and a lost track on the same spot, keep the older one */
  _removeDuplicates(a, b) {
    if (!a.length || !b.length) return [a, b];
    this._growTrk(a.length + b.length);
    const box = this._tBox;
    for (let i = 0; i < a.length; i++) a[i].xyxy(box, 4 * i);
    for (let j = 0; j < b.length; j++) b[j].xyxy(box, 4 * (a.length + j));
    const dropA = new Uint8Array(a.length), dropB = new Uint8Array(b.length);
    let any = false;
    for (let i = 0; i < a.length; i++) {
      const ax1 = box[4 * i], ay1 = box[4 * i + 1], ax2 = box[4 * i + 2], ay2 = box[4 * i + 3];
      const areaA = (ax2 - ax1) * (ay2 - ay1);
      for (let j = 0; j < b.length; j++) {
        const o = 4 * (a.length + j);
        const iw = Math.min(ax2, box[o + 2]) - Math.max(ax1, box[o]);
        const ih = Math.min(ay2, box[o + 3]) - Math.max(ay1, box[o + 1]);
        if (!(iw > 0 && ih > 0)) continue;
        const inter = iw * ih;
        const iou = inter / (areaA + (box[o + 2] - box[o]) * (box[o + 3] - box[o + 1]) - inter + IOU_EPS);
        if (1 - iou >= DUPLICATE_THRESH) continue;
        const timeP = a[i].frameId - a[i].startFrame, timeQ = b[j].frameId - b[j].startFrame;
        if (timeP > timeQ) dropB[j] = 1; else dropA[i] = 1;
        any = true;
      }
    }
    if (!any) return [a, b];
    return [a.filter((_, i) => !dropA[i]), b.filter((_, j) => !dropB[j])];
  }

  // --- scratch buffers ---

  _growDet(n) {
    if (n <= this._capDet) return;
    const c = Math.max(n, 2 * this._capDet);
    this._capDet = c;
    this._dBox = new Float64Array(4 * c);
    this._dMeas = new Float64Array(4 * c);
    this._dScore = new Float64Array(c);
    this._dCls = new Float64Array(c);
    this._high = new Int32Array(c);
    this._low = new Int32Array(c);
    this._uDet = new Int32Array(c);
    this._uDet2 = new Int32Array(c);
    this._colToRow = new Int32Array(c);
  }

  _growCost(n) {
    if (n <= this._capCost) return;
    this._capCost = Math.max(n, 2 * this._capCost);
    this._cost = new Float64Array(this._capCost);
  }

  _growTrk(n) {
    if (n <= this._capTrk) return;
    this._capTrk = Math.max(n, 2 * this._capTrk);
    this._tBox = new Float64Array(4 * this._capTrk);
    this._rowToCol = new Int32Array(this._capTrk);
  }
}
