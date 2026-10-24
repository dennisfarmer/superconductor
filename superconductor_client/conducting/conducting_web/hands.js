// The conducting page's hand tracker (MediaPipe HandLandmarker in a Web Worker, see
// hands_worker.js), which also sends the landmarks to the conducting process itself.
//
//   const hands = startHands(config, wsUrl);   // config: /api/web
//   const { ok, reason } = await hands.ready;
//   hands.setTrack(track);    // the camera's video track; again after switching cameras
//   hands.onResult = r => …;  // every frame: {seq, hands, handedness, width, height, stats}
//   hands.onReply = r => …;   // the conducting process's answer: {reply, stats}
//   hands.key("f");
//
// Frames come straight from the camera track (MediaStreamTrackProcessor) and the worker
// detects and sends, so tracking goes on while this tab is hidden (requestAnimationFrame
// and the <video> stop there). Without MediaStreamTrackProcessor the page sends the
// worker ImageBitmaps of the video itself (`detectBitmap`), which stops while hidden.

export const backgroundCapable = typeof MediaStreamTrackProcessor !== "undefined";

export function startHands(config, wsUrl) {
  const worker = new Worker(new URL("./hands_worker.js", import.meta.url), { type: "module" });
  let settle, waiting = null;
  const ready = new Promise((resolve) => { settle = resolve; });
  const api = {
    ready,
    onResult: null,
    onReply: null,
    onSocket: null,
    setTrack(track) {
      if (!backgroundCapable) return;
      const { readable } = new MediaStreamTrackProcessor({ track, maxBufferSize: 1 });
      worker.postMessage({ type: "frames", readable }, [readable]);
    },
    // fallback: one frame at a time from the page; resolves once the worker handled it
    detectBitmap(bitmap) {
      return new Promise((resolve) => {
        waiting = resolve;
        worker.postMessage({ type: "bitmap", bitmap, t_ms: performance.now() }, [bitmap]);
      });
    },
    key(key) { worker.postMessage({ type: "key", key }); },
    stop() { worker.terminate(); },
  };
  worker.onmessage = (e) => {
    const m = e.data;
    if (m.type === "ready") settle({ ok: true, delegate: m.delegate });
    else if (m.type === "unavailable") settle({ ok: false, reason: m.reason });
    else if (m.type === "socket") api.onSocket?.(m.open);
    else if (m.type === "reply") api.onReply?.(m);
    else if (m.type === "result") {
      api.onResult?.(m);
      if (m.fromBitmap && waiting) { waiting(); waiting = null; }
    }
  };
  worker.onerror = (e) => settle({ ok: false, reason: e.message || "the hand tracker failed to start" });
  worker.postMessage({ type: "init", config, wsUrl });
  return api;
}
