// MediaPipe HandLandmarker off the main thread (hands.js starts it). It reads the camera
// frames, finds the hands, sends their landmarks up /ws/hands to the conducting process
// (conducting_web/server.py) and passes both the landmarks and the process's answers to
// the page. Nothing here waits on the page, so a hidden tab keeps conducting.
//
// Options match hand_tracking.py:create_hand_landmarker (VIDEO mode, up to two hands).

let landmarker = null, delegate = null, ws = null, wsUrl = null;
let reader = null, lastT = 0, seq = 0;
const sentAt = new Map();  // seq -> performance.now() when sent
const stats = { delay: 0, detectMs: 0, sent: 0, answered: 0 };
const smooth = (old, now) => 0.7 * old + 0.3 * now;

async function init(config) {
  const { FilesetResolver, HandLandmarker } = await import(config.bundle_url);
  const fileset = await FilesetResolver.forVisionTasks(new URL(config.wasm_url, self.location.href).href, true);
  const options = (d) => ({
    baseOptions: { modelAssetPath: config.model_url, delegate: d },
    runningMode: "VIDEO",
    numHands: 2,
    ...(d === "GPU" ? { canvas: new OffscreenCanvas(1, 1) } : {}),
  });
  try {
    landmarker = await HandLandmarker.createFromOptions(fileset, options("GPU"));
    delegate = "GPU";
  } catch (e) {
    console.warn("hands: GPU delegate unavailable, using the CPU:", e);
    landmarker = await HandLandmarker.createFromOptions(fileset, options("CPU"));
    delegate = "CPU";
  }
}

function connect() {
  ws = new WebSocket(wsUrl);
  ws.onopen = () => { sentAt.clear(); self.postMessage({ type: "socket", open: true }); };
  ws.onmessage = (e) => {
    let m;
    try { m = JSON.parse(e.data); } catch { return; }
    if (m.seq != null && sentAt.has(m.seq)) {
      stats.delay = smooth(stats.delay, performance.now() - sentAt.get(m.seq));
      for (const s of sentAt.keys()) if (s <= m.seq) sentAt.delete(s);
    }
    if (m.type === "conduct") stats.answered++;
    else if (m.type === "error") console.warn("conductor:", m.error);
    self.postMessage({ type: "reply", reply: m, stats: { ...stats } });
  };
  ws.onclose = () => { self.postMessage({ type: "socket", open: false }); setTimeout(connect, 1000); };
}

// one frame (VideoFrame or ImageBitmap): detect, send, tell the page
function handle(frame, width, height, tMs, fromBitmap) {
  // the landmarker (and the tempo estimator) need increasing timestamps
  const t = Math.max(tMs, lastT + 0.001);
  lastT = t;
  const start = performance.now();
  const result = landmarker.detectForVideo(frame, t);
  stats.detectMs = smooth(stats.detectMs, performance.now() - start);
  const hands = result.landmarks.map((hand) => hand.map((p) => ({ x: p.x, y: p.y, z: p.z })));
  const handedness = result.handedness.map((h) => h[0]?.categoryName ?? null);
  const id = ++seq;
  if (ws && ws.readyState === 1) {
    sentAt.set(id, performance.now());
    ws.send(JSON.stringify({ type: "hands", seq: id, t_ms: t, width, height, hands, handedness,
                             detect_ms: stats.detectMs }));
    stats.sent++;
  }
  self.postMessage({ type: "result", seq: id, t_ms: t, width, height, hands, handedness, fromBitmap,
                     delegate, stats: { ...stats } });
}

// frames straight from the camera track; with maxBufferSize 1 a frame that arrives while
// detecting replaces the waiting one, so the newest frame is always next
async function pump(readable) {
  if (reader) await reader.cancel().catch(() => {});
  const mine = readable.getReader();
  reader = mine;
  while (reader === mine) {
    const { value: frame, done } = await mine.read();
    if (done) break;
    try {
      // performance.now() when read: one clock across camera switches
      handle(frame, frame.displayWidth, frame.displayHeight, performance.now(), false);
    } catch (e) {
      console.warn("hands:", e);
    } finally {
      frame.close();
    }
  }
}

self.onmessage = async (e) => {
  const m = e.data;
  if (m.type === "init") {
    wsUrl = m.wsUrl;
    try {
      await init(m.config);
    } catch (err) {
      self.postMessage({ type: "unavailable", reason: err.message || String(err) });
      return;
    }
    connect();
    self.postMessage({ type: "ready", delegate });
  } else if (m.type === "frames") {
    pump(m.readable);
  } else if (m.type === "bitmap") {
    try {
      handle(m.bitmap, m.bitmap.width, m.bitmap.height, m.t_ms, true);
    } catch (err) {
      console.warn("hands:", err);
      self.postMessage({ type: "result", fromBitmap: true, hands: [], handedness: [], stats: { ...stats } });
    } finally {
      m.bitmap.close();
    }
  } else if (m.type === "key") {
    if (ws && ws.readyState === 1) ws.send(JSON.stringify({ type: "key", key: m.key }));
  }
};
