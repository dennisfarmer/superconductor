// The camera page's detector (YOLOE + BoT-SORT in a Web Worker, see worker.js).
//
//   const detector = startDetector(config);       // config: /api/web's "detector"
//   const { ok, reason } = await detector.ready;  // ok false: no WebGPU (send plain frames instead)
//   const { objects, ms } = await detector.detect(bitmap);   // bitmap: ImageBitmap, transferred
//   detector.stop();

export function startDetector(config) {
  const worker = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
  const waiting = new Map();
  let nextId = 0, settle;
  const ready = new Promise((resolve) => { settle = resolve; });
  worker.onmessage = (e) => {
    const m = e.data;
    if (m.type === "ready") settle({ ok: true });
    else if (m.type === "unavailable") settle({ ok: false, reason: m.reason });
    else if (m.type === "result") {
      const resolve = waiting.get(m.id);
      waiting.delete(m.id);
      resolve?.(m);
    }
  };
  worker.onerror = (e) => settle({ ok: false, reason: e.message || "the detector failed to start" });
  worker.postMessage({ type: "init", config });
  return {
    ready,
    detect(bitmap) {
      const id = nextId++;
      return new Promise((resolve) => {
        waiting.set(id, resolve);
        worker.postMessage({ type: "frame", id, bitmap }, [bitmap]);
      });
    },
    stop() { worker.terminate(); },
  };
}
