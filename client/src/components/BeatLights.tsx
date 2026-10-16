// The beat: one light per beat of the chord, flashing on each quarter note as it is
// heard. The engine reports which beat its output is on and how far into it (plus the
// audio device's latency); `beat_sync` turns each report into a time anchor, and an
// animation-frame loop lights the dots from that anchor between reports. The dots are
// updated directly, so the page does not re-render 60 times a second.

import { useEffect, useRef } from "react";

// when beat `beat` was (or will be) heard, in performance.now() ms
const _sync = { anchor: 0.0, beat: 0, period: 500.0, beats: 4, running: false };
let _lights_el: HTMLDivElement | null = null;
let _raf: number | null = null;

/** Called on every engine status poll. `frac` is how far through `beat` the frame
 * being handed to the device is; it reaches the speaker `latency_ms` later. */
export function beat_sync(beat: number, frac: number, bpm: number, beats: number, latency_ms: number, running: boolean): void {
  const period = 60000.0 / Math.max(bpm, 1.0);
  let anchor = performance.now() + latency_ms - frac * period;
  const old = _sync;
  if (running && old.running && Math.abs(old.period - period) < 1.0 && old.beats === beats) {
    // same tempo: nudge towards the new reading instead of jumping, so poll jitter
    // (about one 40 ms frame) does not make the flashes stutter
    const k = Math.round((anchor - old.anchor) / period);
    const expected = old.anchor + k * period;
    const diff = anchor - expected;
    if (Math.abs(diff) < period * 0.25 && (((old.beat + k) % beats) + beats) % beats === beat) {
      anchor = expected + diff * 0.3;
    }
  }
  _sync.anchor = anchor;
  _sync.beat = beat;
  _sync.period = period;
  _sync.beats = Math.max(beats, 1);
  _sync.running = running;
}

function light_frame(): void {
  const el = _lights_el;
  if (el) {
    const dots = el.children;
    const n = dots.length;
    let idx = -1;
    let phase = 1.0;
    if (_sync.running) {
      const elapsed = performance.now() - _sync.anchor;
      const k = Math.floor(elapsed / _sync.period);
      idx = (((_sync.beat + k) % n) + n) % n;
      phase = elapsed / _sync.period - k;
    }
    for (let i = 0; i < n; i++) {
      const d = dots[i] as HTMLElement;
      const lit = i === idx;
      // a sharp flash at the start of the beat that fades over the first half of it
      d.style.opacity = lit ? String(Math.max(0.0, 1.0 - phase * 1.6) * 0.75 + 0.25) : "";
      d.classList.toggle("lit", lit);
      d.classList.toggle("flash", lit && phase < 0.14);
    }
  }
  _raf = requestAnimationFrame(light_frame);
}

export function BeatLights({ beats, running }: { beats: number; running: boolean }) {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    _lights_el = ref.current;
    if (!_raf) _raf = requestAnimationFrame(light_frame);
    return () => {
      if (_raf) cancelAnimationFrame(_raf);
      _raf = null;
      _lights_el = null;
    };
  }, []);

  return (
    <div
      ref={ref}
      className={"beat-lights" + (running ? "" : " idle")}
      title={
        running
          ? "The beat, as you hear it: " + String(beats) + " quarter notes per chord; the first light is the chord change"
          : "The beat (the audio is paused)"
      }
    >
      {Array.from({ length: Math.max(beats, 1) }, (_, i) => (
        <span key={"b" + String(i)} className={"beat-dot" + (i === 0 ? " down" : "")}></span>
      ))}
    </div>
  );
}
