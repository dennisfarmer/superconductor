// The sound as a blend of MusicCoCa prompts. Each layer has a knob for its share;
// the engine mixes the layers' style embeddings by these weights.

import { Knob } from "./Knob";

// Instrument presets. The first is the saved voice (VOICE.md).
const INSTRUMENTS: string[][] = [
  ["Grand piano (saved voice)", "warm grand piano, intimate, slow"],
  ["String quartet", "string quartet, warm legato, chamber music"],
  ["Rhodes", "lofi rhodes electric piano, mellow"],
  ["Nylon guitar", "nylon string classical guitar, fingerpicked"],
  ["Harp", "concert harp, shimmering, gentle"],
  ["Choir", "soft choir, ooh vowels, cathedral reverb"],
  ["Vibraphone", "vibraphone, soft mallets, jazz"],
  ["Analog pads", "warm analog synth pads, slow attack"],
  ["Organ", "church pipe organ, soft stops"],
  ["Solo cello", "solo cello, expressive, close mic"],
  ["Flute", "solo flute, breathy, airy"],
  ["Upright bass + brushes", "jazz trio, upright bass, brushed drums"],
];

const LAYER_COLORS = ["#8fb3ff", "#c792ea", "#7bd88f", "#ffcf66", "#f78fb3", "#56d4c9"];

export interface SoundMixerProps {
  texts: string[];
  weights: number[];
  engineStyle: string;
  onText: (i: number, t: string) => void;
  onWeight: (i: number, w: number) => void;
  onCommit: () => void;
  onAdd: (t: string) => void;
  onRemove: (i: number) => void;
  onReset: () => void;
}

export function SoundMixer({ texts, weights, engineStyle, onText, onWeight, onCommit, onAdd, onRemove, onReset }: SoundMixerProps) {
  let total = 0.0;
  for (const w of weights) total = total + w;
  return (
    <div className="mixer">
      <span className="prog-label" title="The sound is a blend of text prompts, mixed by the knobs">
        Sound
      </span>
      <div className="layers">
        {texts.map((t, i) => (
          <div key={"layer" + String(i)} className="layer">
            <div title="This sound's share of the blend: drag up or down">
              <Knob
                value={weights[i]}
                size={42}
                color={LAYER_COLORS[i % LAYER_COLORS.length]}
                onChange={(v) => onWeight(i, v)}
                onCommit={onCommit}
              />
            </div>
            <div className="layer-body">
              <input
                className="layer-text"
                value={t}
                placeholder="describe a sound…"
                title="A text prompt for the sound (MusicCoCa style); Enter or click away to apply"
                onChange={(e) => onText(i, e.target.value)}
                onBlur={() => onCommit()}
                onKeyDown={(e) => {
                  if (e.key === "Enter") onCommit();
                }}
              />
              <span className="layer-share">{total > 0.0 ? String(Math.round((100.0 * weights[i]) / total)) + "%" : "0%"}</span>
            </div>
            {texts.length > 1 && (
              <button className="layer-x" title="remove" onClick={() => onRemove(i)}>
                ✕
              </button>
            )}
          </div>
        ))}
        <select
          className="layer-add"
          value=""
          title="add another sound"
          onChange={(e) => {
            if (e.target.value) onAdd(e.target.value);
          }}
        >
          <option value="">＋ add sound</option>
          <option value="custom">custom prompt…</option>
          {INSTRUMENTS.map((inst) => (
            <option key={inst[1]} value={inst[1]}>
              {inst[0]}
            </option>
          ))}
        </select>
        <button className="layer-reset" title="back to the saved voice: warm grand piano, intimate, slow" onClick={() => onReset()}>
          ↺ reset sound
        </button>
      </div>
      <span className="mixer-now" title="what the engine is using right now">
        {"engine: " + engineStyle}
      </span>
    </div>
  );
}
