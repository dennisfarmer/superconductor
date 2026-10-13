"""The sound as a blend of MusicCoCa prompts. Each layer has a knob for its share;
the engine mixes the layers' style embeddings by these weights."""

import from .Knob { Knob }

# Instrument presets. The first is the saved voice (VOICE.md).
glob INSTRUMENTS: list[list[str]] = [
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
    ["Upright bass + brushes", "jazz trio, upright bass, brushed drums"]
];

glob LAYER_COLORS: list[str] = ["#8fb3ff", "#c792ea", "#7bd88f", "#ffcf66", "#f78fb3", "#56d4c9"];


def:pub SoundMixer(
    texts: list[str],
    weights: list[float],
    engineStyle: str,
    onText: Callable[[int, str], None],
    onWeight: Callable[[int, float], None],
    onCommit: Callable[[], None],
    onAdd: Callable[[str], None],
    onRemove: Callable[[int], None],
    onReset: Callable[[], None]
) -> JsxElement {
    total = 0.0;
    for w in weights {
        total = total + w;
    }
    return <div className="mixer">
        <span className="prog-label" title="The sound is a blend of text prompts, mixed by the knobs">Sound</span>
        <div className="layers">
            {for (i, t) in enumerate(texts) {
                <div key={"layer" + str(i)} className="layer">
                    <div title="This sound's share of the blend: drag up or down">
                    <Knob
                        value={weights[i]}
                        size={42}
                        color={LAYER_COLORS[i % len(LAYER_COLORS)]}
                        onChange={lambda v: float { onWeight(i, v); }}
                        onCommit={onCommit}
                    />
                    </div>
                    <div className="layer-body">
                        <input
                            className="layer-text"
                            value={t}
                            placeholder="describe a sound…"
                            title="A text prompt for the sound (MusicCoCa style); Enter or click away to apply"
                            onChange={lambda e: ChangeEvent { onText(i, e.target.value); }}
                            onBlur={lambda e: FocusEvent { onCommit(); }}
                            onKeyDown={lambda e: KeyboardEvent { if e.key == "Enter" { onCommit(); } }}
                        />
                        <span className="layer-share">
                            {(str(round(100.0 * weights[i] / total)) + "%") if total > 0.0 else "0%"}
                        </span>
                    </div>
                    {if len(texts) > 1 {
                        <button className="layer-x" title="remove" onClick={lambda e: MouseEvent { onRemove(i); }}>✕</button>
                    }}
                </div>
            }}
            <select
                className="layer-add"
                value=""
                title="add another sound"
                onChange={lambda e: ChangeEvent { if e.target.value { onAdd(e.target.value); } }}
            >
                <option value="">＋ add sound</option>
                <option value="custom">custom prompt…</option>
                {for inst in INSTRUMENTS { <option key={inst[1]} value={inst[1]}>{inst[0]}</option> }}
            </select>
            <button
                className="layer-reset"
                title="back to the saved voice: warm grand piano, intimate, slow"
                onClick={lambda e: MouseEvent { onReset(); }}
            >↺ reset sound</button>
        </div>
        <span className="mixer-now" title="what the engine is using right now">{"engine: " + engineStyle}</span>
    </div>;
}
