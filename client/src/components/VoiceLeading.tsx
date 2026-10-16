// The voice-leading diagram for one edge of the progression: the chord before and
// the chord after on a grand staff, as the walkers voiced them (soprano and alto on
// the treble staff, tenor and bass on the bass staff), with an arrow for every voice
// a rule bears on. Hovering an arrow shows the rule below the staff; hovering a note
// head shows its name and scale degree there, so a head's line or space is easy to read
// (heads are drawn open, so the staff line shows through).
//
// The rules come from the server's fixed rule bank (server/theory/leading.py), carried
// on the step as `leads`; voices no rule speaks to are drawn without an arrow.
// Notes are spelled as the server spells them, with every accidental written out
// (no key signature).

import { useState } from "react";
import type { LeadView, StepView } from "../api";

const W = 300;
const H = 152;
const SP = 3.5; // half a staff space: one diatonic step
const TREBLE_TOP = 22.0; // y of the top line (F5)
const BASS_TOP = 92.0; // y of the top line (A3)
const XL = 118.0; // x of the chord before
const XR = 228.0; // x of the chord after
const STAFF_L = 6.0;
const STAFF_R = 294.0;

const LETTER_IDX: Record<string, number> = { C: 0, D: 1, E: 2, F: 3, G: 4, A: 5, B: 6 };
const VOICE_NAMES = ["bass", "tenor", "alto", "soprano"];
const RANK: Record<string, number> = { hold: 0, good: 1, warn: 2, bad: 3 };

export const KIND_COLOR: Record<string, string> = { hold: "#8fb3ff", good: "#7bd88f", warn: "#ffcf66", bad: "#ff7a85" };

/** Semitones the spelled name is altered by (♯ +1, ♭ -1, 𝄪 +2, 𝄫 -2). */
function alteration(name: string): number {
  if (name.includes("𝄪")) return 2;
  if (name.includes("𝄫")) return -2;
  if (name.includes("♯")) return 1;
  if (name.includes("♭")) return -1;
  return 0;
}

function accidental_glyph(alt: number): string {
  if (alt === 2) return "𝄪";
  if (alt === -2) return "𝄫";
  if (alt === 1) return "♯";
  if (alt === -1) return "♭";
  return "";
}

/** Diatonic staff index of a spelled note (C4 = 28): letter plus octave, where the
 * octave comes from the MIDI number less the alteration (so B♯3 is index 27, not 28). */
function diatonic(name: string, midi: number): number {
  const octave = Math.floor((midi - alteration(name)) / 12) - 1;
  return octave * 7 + LETTER_IDX[name[0]];
}

/** Voices 2-3 (alto, soprano) sit on the treble staff, 0-1 on the bass staff. */
function staff_y(voice: number, d: number): number {
  if (voice >= 2) return TREBLE_TOP + (38 - d) * SP;
  return BASS_TOP + (26 - d) * SP;
}

/** Ledger-line staff indices a note at `d` needs on its staff. */
function ledgers(voice: number, d: number): number[] {
  const out: number[] = [];
  const lo = voice >= 2 ? 30 : 18;
  const hi = voice >= 2 ? 38 : 26;
  for (let k = hi + 2; k <= d; k += 2) out.push(k);
  for (let k = lo - 2; k >= d; k -= 2) out.push(k);
  return out;
}

interface Head {
  voice: number;
  x: number;
  y: number;
  d: number;
  alt: number;
}

/** Note heads of one chord at x; the upper note of a second (or unison) on the same
 * staff steps to the right, as engraved. */
function heads(x: number, notes: number[], names: string[]): Head[] {
  const out: Head[] = [];
  for (let v = 0; v < 4; v++) {
    const d = diatonic(names[v], notes[v]);
    out.push({ voice: v, x, y: staff_y(v, d), d, alt: alteration(names[v]) });
  }
  for (const pr of [
    [0, 1],
    [2, 3],
  ]) {
    const gap = out[pr[1]].d - out[pr[0]].d;
    if (gap === 0 || gap === 1) out[pr[1]].x = x + 9.0;
  }
  return out;
}

export function worst(ls: LeadView[]): string {
  let k = "hold";
  for (const l of ls) if (RANK[l.kind] > RANK[k]) k = l.kind;
  return k;
}

export interface VoiceLeadingProps {
  prev: StepView;
  step: StepView;
  onClose: () => void;
}

export function VoiceLeading({ prev, step, onClose }: VoiceLeadingProps) {
  const [hover, setHover] = useState(-1);
  // the note head under the mouse: [0 before | 1 after, voice], or []
  const [hover_note, setHoverNote] = useState<number[]>([]);

  const left = heads(XL, prev.notes, prev.names);
  const right = heads(XR, step.notes, step.names);
  const by_voice: LeadView[][] = [[], [], [], []];
  for (const l of step.leads) by_voice[l.voice].push(l);
  const shown = hover >= 0 ? by_voice[hover] : [];
  // a rule between two voices (parallels, overlap, cross relation) lights both arrows
  const related: number[] = [];
  for (let v = 0; v < 4; v++) {
    for (const l of by_voice[v]) {
      if (shown.some((s) => l.title === s.title && l.detail === s.detail) && !related.includes(v)) related.push(v);
    }
  }

  function staff_lines(top: number, key: string) {
    return [0, 1, 2, 3, 4].map((k) => (
      <line key={key + String(k)} x1={STAFF_L} x2={STAFF_R} y1={top + k * 2 * SP} y2={top + k * 2 * SP} className="vl-staff" />
    ));
  }

  function head_g(h: Head, key: string) {
    return (
      <g key={key}>
        {ledgers(h.voice, h.d).map((d) => (
          <line key={key + "l" + String(d)} x1={h.x - 8.0} x2={h.x + 8.0} y1={staff_y(h.voice, d)} y2={staff_y(h.voice, d)} className="vl-staff" />
        ))}
        {h.alt !== 0 && (
          <text x={h.x - 10.0} y={h.y + (h.alt === -1 ? 2.5 : 4.0)} className="vl-acc" textAnchor="end">
            {accidental_glyph(h.alt)}
          </text>
        )}
        <ellipse cx={h.x} cy={h.y} rx={5.0} ry={3.4} transform={`rotate(-18 ${h.x} ${h.y})`} className="vl-head" />
      </g>
    );
  }

  /** An invisible target over a head (drawn above the arrows, so it wins near them). */
  function head_hit(h: Head, side: number) {
    const on = hover_note.length === 2 && hover_note[0] === side && hover_note[1] === h.voice;
    return (
      <ellipse
        key={"h" + String(side) + String(h.voice)}
        cx={h.x}
        cy={h.y}
        rx={6.5}
        ry={4.5}
        className={"vl-head-hit" + (on ? " on" : "")}
        onMouseEnter={() => {
          setHoverNote([side, h.voice]);
          setHover(-1);
        }}
        onMouseLeave={() => setHoverNote([])}
      />
    );
  }

  /** A note as the rule details write it: "soprano: D5 (2̂)". */
  function note_label(st: StepView, v: number): string {
    const deg = v < st.degrees.length ? st.degrees[v] : "";
    return VOICE_NAMES[v] + ": " + st.names[v] + (deg ? " (" + deg + ")" : "");
  }

  function arrow(v: number) {
    const a = left[v];
    const b = right[v];
    const kind = worst(by_voice[v]);
    const on = hover === v || related.includes(v);
    const x1 = a.x + 7.0;
    const x2 = b.x - (b.alt !== 0 ? 19.0 : 8.0);
    // bow away from the other voice on the same staff: upper voices up, lower ones down
    const bow = v === 1 || v === 3 ? -6.0 : 6.0;
    const mx = (x1 + x2) / 2;
    const my = (a.y + b.y) / 2 + bow;
    const d = `M ${x1} ${a.y} Q ${mx} ${my} ${x2} ${b.y}`;
    return (
      <g
        key={"a" + String(v)}
        onMouseEnter={() => {
          setHover(v);
          setHoverNote([]);
        }}
        onMouseLeave={() => setHover(-1)}
      >
        <path
          d={d}
          fill="none"
          stroke={KIND_COLOR[kind]}
          strokeWidth={on ? 2.4 : 1.4}
          opacity={on || hover < 0 ? 1.0 : 0.35}
          markerEnd={"url(#vl-m-" + kind + ")"}
        />
        <path d={d} fill="none" stroke="transparent" strokeWidth={12} className="vl-hit" />
      </g>
    );
  }

  return (
    <div className="vl-card" onPointerDown={(e) => e.stopPropagation()} onWheel={(e) => e.stopPropagation()}>
      <div className="vl-head-row">
        <span className="vl-title">
          {prev.rn} → {step.rn}
        </span>
        <span className="vl-sub">
          {prev.symbol} → {step.symbol}
        </span>
        <button
          className="vl-close"
          title="Close"
          onClick={(e) => {
            e.stopPropagation();
            onClose();
          }}
        >
          ×
        </button>
      </div>
      <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} className="vl-svg">
        <defs>
          {["hold", "good", "warn", "bad"].map((k) => (
            <marker key={k} id={"vl-m-" + k} viewBox="0 0 8 8" refX={7} refY={4} markerWidth={5} markerHeight={5} orient="auto-start-reverse">
              <path d="M 0 0 L 8 4 L 0 8 z" fill={KIND_COLOR[k]} />
            </marker>
          ))}
        </defs>
        {staff_lines(TREBLE_TOP, "t")}
        {staff_lines(BASS_TOP, "b")}
        <line x1={STAFF_L} x2={STAFF_L} y1={TREBLE_TOP} y2={BASS_TOP + 8 * SP} className="vl-staff" />
        <text x={STAFF_L + 3} y={TREBLE_TOP + 6 * SP + 9} className="vl-clef">
          𝄞
        </text>
        <text x={STAFF_L + 3} y={BASS_TOP + 2 * SP + 8} className="vl-clef bass">
          𝄢
        </text>
        {left.map((h) => head_g(h, "L" + String(h.voice)))}
        {right.map((h) => head_g(h, "R" + String(h.voice)))}
        {[0, 1, 2, 3].filter((v) => by_voice[v].length > 0).map((v) => arrow(v))}
        {left.map((h) => head_hit(h, 0))}
        {right.map((h) => head_hit(h, 1))}
        <text x={XL} y={H - 4} className="vl-rn" textAnchor="middle">
          {prev.rn}
        </text>
        <text x={XR} y={H - 4} className="vl-rn" textAnchor="middle">
          {step.rn}
        </text>
      </svg>
      <div className="vl-why">
        {hover_note.length === 2 ? (
          <div className="vl-rule">
            <div className="vl-detail">{note_label(hover_note[0] === 0 ? prev : step, hover_note[1])}</div>
            <div className="vl-text">{"in " + (hover_note[0] === 0 ? prev.rn + " (" + prev.symbol + ")" : step.rn + " (" + step.symbol + ")")}</div>
          </div>
        ) : hover < 0 ? (
          <p className="vl-hint">
            {step.leads.length > 0
              ? "Hover an arrow for the rule behind it, or a note for its name."
              : "No rule in the bank applies to this change: no tendency tones, held common tones or faults. Hover a note for its name."}
          </p>
        ) : (
          <div>
            {shown.map((l, k) => (
              <div key={String(k)} className="vl-rule">
                <div className={"vl-rule-title " + l.kind}>{l.title}</div>
                <div className="vl-detail">{l.detail}</div>
                <div className="vl-text">{l.text}</div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
