// The controls row (under the top bar, on every page): MRT2's sampling arguments,
// named as magenta_rt's generate() names them: cfg_scales (musiccoca, notes,
// drums), temperature and top_k. Turning a knob POSTs /settings to the engine. The
// local engine passes them to generate() on the next frame; the SuperConductor sink
// POSTs them to client_midi's /controls, which relays them to the server as
// UpdateControls (heard after the client's lead, ~0.6 s).

import { useRef, useState } from "react";
import { Knob } from "./Knob";

export interface Spec {
  key: string; // "temperature" | "top_k" | "cfg_scales.<name>"
  min: number;
  max: number;
  step: number;
  log: boolean; // knob position on a log scale (top_k)
  model_default: number;
  color: string;
  hint: string;
}

const CFG_HINT =
  " Range -1 to 7 (magenta_rt/config.py). The value is sent to MRT2 as a conditioning token on every 40 ms frame, so it is rounded to MRT2's step.";

const SPECS: Spec[] = [
  {
    key: "cfg_scales.musiccoca",
    min: -1.0,
    max: 7.0,
    step: 0.2,
    log: false,
    model_default: 3.0,
    color: "#c792ea",
    hint:
      "cfg_scales.musiccoca: classifier-free guidance toward the style (the MusicCoCa embedding of the prompts). Higher: closer to the prompts; 0 or below: the style matters little." +
      CFG_HINT +
      " Steps of 0.2. MRT2's default 3.0.",
  },
  {
    key: "cfg_scales.notes",
    min: -1.0,
    max: 7.0,
    step: 0.2,
    log: false,
    model_default: 1.0,
    color: "#7bd88f",
    hint:
      "cfg_scales.notes: classifier-free guidance toward the NOTES input (the chords Atlas sends). Higher: the pitches and onsets are followed more strictly." +
      CFG_HINT +
      " Steps of 0.2. MRT2's default 1.0; Atlas starts at 3.0 so the chords are followed.",
  },
  {
    key: "cfg_scales.drums",
    min: -1.0,
    max: 7.0,
    step: 1.0,
    log: false,
    model_default: 1.0,
    color: "#ffb74d",
    hint:
      "cfg_scales.drums: how strongly MRT2 follows the DRUMS input (the Drums page). Higher: hits land on the grid and rest cells stay quiet; 0 or below: the style decides." +
      CFG_HINT +
      " Steps of 1 (9 tokens). Unlike musiccoca and notes, MRT2 runs no extra masked pass for drums. MRT2's default 1.0.",
  },
  {
    key: "temperature",
    min: 0.1,
    max: 3.0,
    step: 0.05,
    log: false,
    model_default: 1.3,
    color: "#5aa9ff",
    hint: "temperature: sampling temperature for every audio token. Higher: more varied and noisier (more 8-20 kHz hiss, see docs/VOICE.md); lower: more repetitive. MRT2's default 1.3.",
  },
  {
    key: "top_k",
    min: 1.0,
    max: 1024.0,
    step: 1.0,
    log: true,
    model_default: 40.0,
    color: "#56d4c9",
    hint: "top_k: each audio token is sampled from only the k likeliest of its codebook's 1024. Lower: safer, less varied; 1 always takes the likeliest. MRT2's default 40.",
  },
];

function clamp(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}

function snap(sp: Spec, v: number): number {
  const s = Math.round(clamp(v, sp.min, sp.max) / sp.step) * sp.step;
  return Math.round(s * 100.0) / 100.0;
}

function to_knob(sp: Spec, v: number): number {
  if (sp.log) return Math.log(v / sp.min) / Math.log(sp.max / sp.min);
  return (v - sp.min) / (sp.max - sp.min);
}

function from_knob(sp: Spec, k: number): number {
  if (sp.log) return snap(sp, sp.min * Math.pow(sp.max / sp.min, k));
  return snap(sp, sp.min + k * (sp.max - sp.min));
}

function digits(sp: Spec): number {
  return sp.step >= 1.0 ? 0 : sp.step >= 0.1 ? 1 : 2;
}

/** The value of `key` in the engine's settings ({temperature, top_k, cfg_scales: {...}}). */
export function value_of(settings: any, key: string): number | null {
  if (settings == null) return null;
  const v = key.startsWith("cfg_scales.") ? (settings.cfg_scales || {})[key.slice(11)] : settings[key];
  return v == null ? null : Number(v);
}

/** {"cfg_scales.notes": 2} -> {cfg_scales: {notes: 2}}, the engine's /settings body. */
function body_of(key: string, v: number): Record<string, unknown> {
  if (key.startsWith("cfg_scales.")) return { cfg_scales: { [key.slice(11)]: v } };
  return { [key]: v };
}

const SEND_EVERY_MS = 60; // while a knob turns, at most one /settings per this

/** The spec of a controls-row knob ("cfg_scales.drums", "temperature", ...). */
export function spec_of(key: string): Spec {
  return SPECS.find((sp) => sp.key === key) as Spec;
}

export interface ControlKnobProps {
  spec: Spec;
  settings: any; // the engine's /status settings, or null while it is offline
  onSend: (body: Record<string, unknown>) => void;
  size?: number;
  held?: boolean; // released: shown, but not sent to the client
}

/** One knob of the controls row: knob, number field and reset to MRT2's default. */
export function ControlKnob({ spec: sp, settings, onSend, size = 26, held = false }: ControlKnobProps) {
  // the value while the knob is being turned (it wins over the engine's until committed)
  const [drag, setDrag] = useState<number | null>(null);
  const last_sent = useRef(0);

  function turn(v: number) {
    if (held) return;
    setDrag(v);
    const now = Date.now();
    if (now - last_sent.current >= SEND_EVERY_MS) {
      last_sent.current = now;
      onSend(body_of(sp.key, v));
    }
  }

  function commit() {
    if (held || drag == null) return;
    onSend(body_of(sp.key, drag));
    setDrag(null);
  }

  const off = settings == null || held;
  const engine_v = value_of(settings, sp.key);
  const v = drag != null ? drag : engine_v;
  const shown = v == null ? sp.model_default : v;
  const held_note = held ? " Released: Atlas is not sending this to the client." : "";
  return (
    <div className={"ctl" + (off ? " off" : "")} title={sp.hint + held_note}>
      <span className="ctl-name">{sp.key}</span>
      <Knob
        value={to_knob(sp, shown)}
        size={size}
        color={sp.color}
        onChange={(k) => {
          if (!off) turn(from_knob(sp, k));
        }}
        onCommit={commit}
      />
      <input
        className="ctl-in"
        type="number"
        disabled={off}
        min={String(sp.min)}
        max={String(sp.max)}
        step={String(sp.step)}
        key={sp.key + String(shown)}
        defaultValue={shown.toFixed(digits(sp))}
        onBlur={(e) => {
          if (held || e.target.value === "") return;
          const nv = snap(sp, parseFloat(e.target.value));
          if (nv !== engine_v) onSend(body_of(sp.key, nv));
        }}
        onKeyDown={(e) => {
          if (e.key === "Enter") e.currentTarget.blur();
        }}
      />
      <button
        className="ctl-reset"
        disabled={off || held || shown === sp.model_default}
        title={"Back to MRT2's default, " + sp.model_default.toFixed(digits(sp))}
        onClick={() => onSend(body_of(sp.key, sp.model_default))}
      >
        ↺
      </button>
    </div>
  );
}

export interface OutputSend {
  harmony: boolean;
  drums: boolean;
}

export interface ControlsBarProps {
  settings: any; // the engine's /status settings, or null while it is offline
  sink: boolean;
  output: OutputSend; // which pages are sending to the SuperConductor client
  onSend: (body: Record<string, unknown>) => void;
  onOutput: (body: { harmony?: boolean; drums?: boolean }) => void;
}

const HARMONY_TITLE =
  "Harmony control. On: Atlas sends the NOTES lane (the chords on this page) and cfg_scales.notes to the SuperConductor client. Off: Atlas resets both to the model default — the NOTES lane cleared, cfg_scales.notes back to MRT2's default — and then stops sending them, so another program can take harmony, or it stays at that default.";

const DRUMS_TITLE =
  "Drum control. On: the Drums page sends the DRUMS lane, cfg_scales.drums and the style prompt \"drums\" to the SuperConductor client. Off: Atlas resets those to the defaults — the DRUMS lane cleared, cfg_scales.drums back to MRT2's default, the drums prompt removed — and then stops sending them.";

/** One page's send toggle. Checked: Atlas controls that part of the client. */
function SendToggle({
  name,
  on,
  disabled,
  title,
  onChange,
}: {
  name: string;
  on: boolean;
  disabled: boolean;
  title: string;
  onChange: (on: boolean) => void;
}) {
  return (
    <label className={"send-toggle" + (on ? "" : " off")} title={title}>
      <input type="checkbox" checked={on} disabled={disabled} onChange={(e) => onChange(e.target.checked)} />
      {on ? name : name + " released"}
    </label>
  );
}

/** Which of Atlas's pages are allowed to send to the SuperConductor client. */
export function ClientToggles({ output, disabled, onOutput }: { output: OutputSend; disabled: boolean; onOutput: ControlsBarProps["onOutput"] }) {
  return (
    <div className="client-send">
      <span className="controls-label">to client</span>
      <SendToggle name="harmony" on={output.harmony} disabled={disabled} title={HARMONY_TITLE} onChange={(v) => onOutput({ harmony: v })} />
      <SendToggle name="drums" on={output.drums} disabled={disabled} title={DRUMS_TITLE} onChange={(v) => onOutput({ drums: v })} />
    </div>
  );
}

function held_key(key: string, output: OutputSend): boolean {
  if (key === "cfg_scales.notes") return !output.harmony;
  if (key === "cfg_scales.drums") return !output.drums;
  return false;
}

export function ControlsBar({ settings, sink, output, onSend, onOutput }: ControlsBarProps) {
  return (
    <div
      className="controls-row"
      title={
        sink
          ? "MRT2's sampling controls, sent to SuperConductor's server as UpdateControls (via client_midi /controls); heard after the client's lead. The switches at the left release harmony or drums."
          : "MRT2's sampling controls, passed to the local engine's generate() from the next frame"
      }
    >
      {sink && <ClientToggles output={output} disabled={settings == null} onOutput={onOutput} />}
      <span className="controls-label">generate()</span>
      {SPECS.map((sp) => (
        <ControlKnob key={sp.key} spec={sp} settings={settings} onSend={onSend} held={sink && held_key(sp.key, output)} />
      ))}
    </div>
  );
}
