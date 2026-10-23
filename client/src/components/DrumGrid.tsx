// The Drums page (/#drums, under the atlas's top bar): MRT2's DRUMS
// input, one value per 40 ms frame. -1: the model chooses, 0: no hit, 1: a hit now.
// The model picks the drum sound; this lane only says when hits happen.
//
// Modes: Free (every frame -1, the model drums as it likes), Silent (every frame 0),
// or Grid: a 4/4 step grid on the engine's beat clock. A hit cell is 1 on its first
// frame and 0 for the rest of the cell (a finer subdivision gives shorter cells, so
// more room for hits). Free and rest cells hold their value for the whole cell.
//
// The grid is saved in the session (server.atlas.set_drums) and sent to the engine,
// the local one or the SuperConductor sink, as settings.drum_grid. How strongly MRT2
// follows the grid is cfg_scales.drums, a knob in the controls row under the top bar.
//
// Drum control (a proof of concept): three levers on the drums at once. The DRUMS
// input above, cfg_scales.drums (how strictly MRT2 follows it), and drum_prompt, the
// weight 0..1 of the style prompt "drums" that the engine POSTs to the SuperConductor
// client's /prompts, which adds it to the recipe its tracking sends (sink only). With
// "link" on, the master knob sets cfg_scales.drums (-1..7) and drum_prompt (0..1)
// together; the knobs it moves can still be turned on their own, which doesn't move it.

import { useEffect, useRef, useState } from "react";
import { getState, setDrums } from "../api";
import { ControlKnob, spec_of, value_of } from "./ControlsBar";
import { Knob } from "./Knob";

const FRAME_RATE = 25.0;
const PER_BEAT: [number, string][] = [
  [1, "quarters"],
  [2, "eighths"],
  [4, "sixteenths"],
  [3, "triplets"],
];
const BARS = [1, 2, 4];
// the engine's status, read by the poll timer (outside React state)
let _status_url = "";
const CFG_KEYS = ["cfg_scales.musiccoca", "cfg_scales.notes", "cfg_scales.drums"];
const SEND_EVERY_MS = 60; // while the master knob turns, at most one /settings per this

/** The master knob (0..1) -> cfg_scales.drums (-1..7, whole steps) and drum_prompt (0..1). */
function master_body(m: number): Record<string, unknown> {
  return { cfg_scales: { drums: Math.round(-1.0 + 8.0 * m) }, drum_prompt: Math.round(m * 100.0) / 100.0 };
}

interface Grid {
  mode: string;
  bars: number;
  per_beat: number;
  cells: number[];
  after_hit: number;
}

function default_grid(): Grid {
  return { mode: "silent", bars: 1, per_beat: 2, cells: [-1, -1, -1, -1, -1, -1, -1, -1], after_hit: 0 };
}

/** The session's grid with every field present (older sessions have none). */
function grid_of(raw: string): Grid {
  const g = default_grid() as unknown as Record<string, any>;
  if (raw) {
    try {
      const saved = JSON.parse(raw);
      for (const k of ["mode", "bars", "per_beat", "cells"]) {
        if (saved[k] || saved[k] === 0) g[k] = saved[k];
      }
    } catch {
      // keep the defaults
    }
  }
  const n = g.bars * 4 * g.per_beat;
  const cells: number[] = [];
  for (let i = 0; i < n; i++) cells.push(i < g.cells.length ? g.cells[i] : -1);
  g.cells = cells;
  return g as Grid;
}

function with_field<K extends keyof Grid>(g: Grid, key: K, value: Grid[K]): Grid {
  const out: Grid = { mode: g.mode, bars: g.bars, per_beat: g.per_beat, cells: g.cells, after_hit: g.after_hit };
  out[key] = value;
  return out;
}

/** New bars / subdivision: cells that still start on the same spot keep their value
 * (a shorter grid is cut, a longer one repeats); new in-between cells are free. */
function regrid(g: Grid, bars: number, per_beat: number): Grid {
  const old = g.cells;
  const old_pb = g.per_beat;
  const cells: number[] = [];
  for (let i = 0; i < bars * 4 * per_beat; i++) {
    if ((i * old_pb) % per_beat === 0) cells.push(old[Math.trunc((i * old_pb) / per_beat) % old.length]);
    else cells.push(-1);
  }
  const out = with_field(g, "cells", cells);
  out.bars = bars;
  out.per_beat = per_beat;
  return out;
}

/** Quick fills; `beat` is the beat number in the bar (0..3), `sub` the cell within it. */
function fill(g: Grid, kind: string): Grid {
  const pb = g.per_beat;
  const cells: number[] = [];
  for (let i = 0; i < g.cells.length; i++) {
    const sub = i % pb;
    const beat = Math.trunc(i / pb) % 4;
    let v = -1;
    if (kind === "rest") v = 0;
    else if (kind === "pulse") v = sub === 0 ? 1 : 0;
    else if (kind === "one-three") v = sub === 0 && beat % 2 === 0 ? 1 : 0;
    else if (kind === "backbeat") v = sub === 0 && beat % 2 === 1 ? 1 : 0;
    else if (kind === "offbeats") v = sub === pb - 1 - Math.trunc((pb - 1) / 2) && pb > 1 ? 1 : 0;
    else if (kind === "random") v = Math.floor(Math.random() * 3) - 1;
    cells.push(v);
  }
  const out = with_field(g, "cells", cells);
  out.mode = "grid";
  return out;
}

/** Cell labels: 1 & 2 & ..., 1 e & a ..., 1 trip let ... */
function cell_label(i: number, pb: number): string {
  const sub = i % pb;
  if (sub === 0) return String((Math.trunc(i / pb) % 4) + 1);
  if (pb === 2) return "&";
  if (pb === 3) return sub === 1 ? "trip" : "let";
  return ["", "e", "&", "a"][sub];
}

/** The values the engine sends for one bar, frame by frame (engine.Conductor._drum_token). */
function bar_frames(g: Grid, bpm: number): number[] {
  const per_cell = (FRAME_RATE * 60.0) / Math.max(bpm, 1.0) / g.per_beat;
  const n_frames = Math.trunc(Math.ceil(((FRAME_RATE * 60.0) / Math.max(bpm, 1.0)) * 4.0));
  const out: number[] = [];
  for (let f = 0; f < n_frames; f++) {
    if (g.mode !== "grid") {
      out.push(g.mode === "silent" ? 0 : -1);
      continue;
    }
    const k = Math.floor(f / per_cell);
    let v = g.cells[k % g.cells.length];
    if (v === 1 && !(f === 0 || Math.floor((f - 1) / per_cell) !== k)) v = g.after_hit;
    out.push(v);
  }
  return out;
}

function value_class(v: number): string {
  return v === 1 ? "hit" : v === 0 ? "rest" : "free";
}

export interface DrumsPageProps {
  settings: any; // the engine's /status settings, or null while it is offline
  sink: boolean;
  onSend: (body: Record<string, unknown>) => void;
}

/** The drum-control levers: cfg_scales knobs, the "drums" prompt weight, the master knob. */
function DrumControl({ settings, sink, onSend }: DrumsPageProps) {
  const [linked, setLinked] = useState(false);
  const [master, setMaster] = useState(0.0);
  const [prompt_drag, setPromptDrag] = useState<number | null>(null);
  const last_sent = useRef(0);

  function throttled(body: Record<string, unknown>) {
    const now = Date.now();
    if (now - last_sent.current >= SEND_EVERY_MS) {
      last_sent.current = now;
      onSend(body);
    }
  }

  const off = settings == null;
  const engine_prompt = value_of(settings, "drum_prompt");
  const prompt_v = prompt_drag != null ? prompt_drag : engine_prompt == null ? 0.0 : engine_prompt;
  const cfg_drums = value_of(settings, "cfg_scales.drums");
  return (
    <section className="drums-control">
      <div className="drums-frames-head">
        <span className="drums-label">Drum control</span>
        <span className="drums-num">
          {"cfg_scales.drums " + (cfg_drums == null ? "–" : String(cfg_drums)) + " · prompt \"drums\" " + prompt_v.toFixed(2)}
        </span>
      </div>
      <div className="drums-control-row">
        <div className={"drums-master" + (linked && !off ? "" : " off")}>
          <label className="drums-link" title="On: the master knob sets cfg_scales.drums and the drums prompt together">
            <input type="checkbox" checked={linked} onChange={(e) => setLinked(e.target.checked)} />
            link
          </label>
          <div
            className="ctl"
            title={"Master: cfg_scales.drums -1 → 7 and prompt \"drums\" 0 → 1 together. Turning the others doesn't move it."}
          >
            <span className="ctl-name">master</span>
            <Knob
              value={master}
              size={44}
              color="#ff7a59"
              onChange={(k) => {
                if (!linked || off) return;
                setMaster(k);
                throttled(master_body(k));
              }}
              onCommit={() => {
                if (linked && !off) onSend(master_body(master));
              }}
            />
            <span className="drums-num">{master.toFixed(2)}</span>
          </div>
        </div>
        <div className="drums-control-knobs">
          {CFG_KEYS.map((key) => (
            <ControlKnob key={key} spec={spec_of(key)} settings={settings} onSend={onSend} size={34} />
          ))}
          <div
            className={"ctl drums-prompt" + (off || !sink ? " off" : "")}
            title={
              sink
                ? "drum_prompt: weight of the style prompt \"drums\" in the SuperConductor client's recipe (client_midi /prompts), next to its other prompts. 0 removes it."
                : "drum_prompt: SuperConductor only (make run); the local engine doesn't use it"
            }
          >
            <span className="ctl-name">prompt "drums"</span>
            <input
              type="range"
              min="0"
              max="1"
              step="0.01"
              disabled={off || !sink}
              value={prompt_v}
              onChange={(e) => {
                const v = parseFloat(e.target.value);
                setPromptDrag(v);
                throttled({ drum_prompt: v });
              }}
              onPointerUp={() => {
                if (prompt_drag != null) onSend({ drum_prompt: prompt_drag });
                setPromptDrag(null);
              }}
              onKeyUp={() => {
                if (prompt_drag != null) onSend({ drum_prompt: prompt_drag });
                setPromptDrag(null);
              }}
            />
            <span className="drums-num">{prompt_v.toFixed(2)}</span>
          </div>
        </div>
      </div>
    </section>
  );
}

export function DrumsPage({ settings, sink, onSend }: DrumsPageProps) {
  const [grid, setGrid] = useState<Grid>(default_grid());
  const [loaded, setLoaded] = useState(false);
  const [bpm, setBpm] = useState(84.0);
  const [now_cell, setNowCell] = useState(-1);

  async function load() {
    const s = await getState();
    setGrid(grid_of(s.drum_grid));
    setBpm(s.bpm);
    _status_url = s.engine_url + "/status";
    setLoaded(true);
  }

  async function poll() {
    if (!_status_url) return;
    try {
      const resp = await fetch(_status_url);
      const data = await resp.json();
      if (data.settings && data.settings.bpm) setBpm(parseFloat(data.settings.bpm));
      setNowCell(data.playing ? Math.trunc(Number(data.playing.drum_cell)) : -1);
    } catch {
      setNowCell(-1);
    }
  }

  useEffect(() => {
    load();
    const timer = setInterval(() => poll(), 120);
    return () => clearInterval(timer);
  }, []);

  async function save(g: Grid) {
    setGrid(g);
    await setDrums(JSON.stringify(g));
  }

  function click_cell(i: number) {
    // free -> hit -> rest -> free; editing the grid switches it on
    const cells = [...grid.cells];
    const v = cells[i];
    cells[i] = v === -1 ? 1 : v === 1 ? 0 : -1;
    const g = with_field(grid, "cells", cells);
    g.mode = "grid";
    save(g);
  }

  if (!loaded) {
    return (
      <div className="drums-page">
        <p className="empty">Loading the drums…</p>
      </div>
    );
  }

  const pb = grid.per_beat;
  const on_grid = grid.mode === "grid";
  const frames = bar_frames(grid, bpm);
  const frames_per_cell = (FRAME_RATE * 60.0) / Math.max(bpm, 1.0) / pb;
  const bars: number[][] = [];
  for (let b = 0; b < grid.bars; b++) bars.push(Array.from({ length: 4 * pb }, (_, i) => b * 4 * pb + i));

  return (
    <div className="drums-page">
      <header className="drums-head">
        <div>
          <h1>Drums</h1>
          <p className="drums-sub">When hits happen. The model picks the drum sounds and plays them over the harmony from the main page.</p>
        </div>
      </header>

      <section className="drums-controls">
        <div className="drums-field">
          <span className="drums-label">Mode</span>
          <div className="seg">
            <button
              className={grid.mode === "free" ? "on" : ""}
              title="Every frame -1: the model decides whether and when to drum"
              onClick={() => save(with_field(grid, "mode", "free"))}
            >
              Free
            </button>
            <button className={grid.mode === "silent" ? "on" : ""} title="Every frame 0: no drum hits" onClick={() => save(with_field(grid, "mode", "silent"))}>
              Silent
            </button>
            <button className={on_grid ? "on" : ""} title="The grid below, on the beat" onClick={() => save(with_field(grid, "mode", "grid"))}>
              Grid
            </button>
          </div>
        </div>
        <div className="drums-field">
          <span className="drums-label">Subdivision</span>
          <div className="seg">
            {PER_BEAT.map((opt) => (
              <button key={"pb" + String(opt[0])} className={pb === opt[0] ? "on" : ""} onClick={() => save(regrid(grid, grid.bars, opt[0]))}>
                {opt[1]}
              </button>
            ))}
          </div>
        </div>
        <div className="drums-field">
          <span className="drums-label">Bars</span>
          <div className="seg">
            {BARS.map((n) => (
              <button key={"bars" + String(n)} className={grid.bars === n ? "on" : ""} onClick={() => save(regrid(grid, n, pb))}>
                {String(n)}
              </button>
            ))}
          </div>
        </div>
      </section>

      <section className={"drums-grid" + (on_grid ? "" : " off")}>
        {bars.map((row, b) => (
          <div key={"bar" + String(b)} className="drums-bar">
            <span className="drums-bar-n">{"bar " + String(b + 1)}</span>
            <div className="drums-cells" style={{ gridTemplateColumns: "repeat(" + String(4 * pb) + ", 1fr)" }}>
              {row.map((i) => (
                <button
                  key={"c" + String(i)}
                  className={"drum-cell " + value_class(grid.cells[i]) + (i % pb === 0 ? " beat" : "") + (on_grid && i === now_cell ? " now" : "")}
                  title={cell_label(i, pb) + ": " + String(grid.cells[i]) + " (click: free → hit → rest)"}
                  onClick={() => click_cell(i)}
                >
                  <span className="drum-cell-label">{cell_label(i, pb)}</span>
                  <span className="drum-cell-mark">{grid.cells[i] === 1 ? "●" : grid.cells[i] === 0 ? "×" : "○"}</span>
                </button>
              ))}
            </div>
          </div>
        ))}
      </section>

      <section className="drums-fills">
        <span className="drums-label">Fill</span>
        <button className="btn ghost" title="Every cell -1" onClick={() => save(fill(grid, "free"))}>
          all free
        </button>
        <button className="btn ghost" title="Every cell 0" onClick={() => save(fill(grid, "rest"))}>
          all rest
        </button>
        <button className="btn ghost" title="A hit on every beat" onClick={() => save(fill(grid, "pulse"))}>
          pulse
        </button>
        <button className="btn ghost" title="Hits on beats 1 and 3" onClick={() => save(fill(grid, "one-three"))}>
          1 + 3
        </button>
        <button className="btn ghost" title="Hits on beats 2 and 4" onClick={() => save(fill(grid, "backbeat"))}>
          backbeat
        </button>
        <button className="btn ghost" disabled={pb === 1} title="Hits between the beats" onClick={() => save(fill(grid, "offbeats"))}>
          offbeats
        </button>
        <button className="btn ghost" title="Each cell -1, 0 or 1 at random" onClick={() => save(fill(grid, "random"))}>
          random
        </button>
      </section>

      <DrumControl settings={settings} sink={sink} onSend={onSend} />

      <section className="drums-frames">
        <div className="drums-frames-head">
          <span className="drums-label">What the model gets, frame by frame (bar 1)</span>
          <span className="drums-num">
            {String(frames.length) +
              " frames of 40 ms at " +
              String(Math.trunc(bpm)) +
              " BPM · " +
              String(Math.round(frames_per_cell * 10.0) / 10.0) +
              " frames per cell"}
          </span>
        </div>
        <div className="drums-strip">
          {frames.map((v, f) => (
            <span key={"f" + String(f)} className={"drum-frame " + value_class(v)} title={"frame " + String(f) + ": " + String(v)}>
              {String(v)}
            </span>
          ))}
        </div>
      </section>

      <section className="drums-legend">
        <p>
          <span className="drum-key free">○ −1</span> no instruction: the model decides whether to hit.
        </p>
        <p>
          <span className="drum-key hit">● 1</span> a hit now (on the cell's first frame).
        </p>
        <p>
          <span className="drum-key rest">× 0</span> no hit for the whole cell.
        </p>
        <p className="hint">
          The grid runs on the engine's beat clock: set the tempo in the top bar. With SuperConductor, changes are heard about 0.6 s later
          (its buffer).
        </p>
      </section>
    </div>
  );
}
