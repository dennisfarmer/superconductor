"""The rules the BeamSearch walker scores with, written as Markdown and rendered
below the interface. Numbers are the costs in theory/syntax.jac, theory/voicing.jac
and theory/phrase.jac (lower = more idiomatic)."""

glob RULES_MD: str = """## How the walkers choose the next chord

Every candidate move is scored, and the beam keeps the cheapest paths:

`step = 1.5 × syntax + voice leading + phrase`

A move whose syntax cost exceeds 3.0 is never followed. These numbers, and the colour tax, cadence bonus and voice-leading weights below, are the defaults: all of them can be changed under **⚙ Walker settings**.

### Harmonic function flow

[[flow]]

Obligated chords resolve one way only: **Cad⁶₄ → V**; **V/x, V⁷/x, vii°⁷/V → x**; **It⁺⁶, Fr⁺⁶, Ger⁺⁶ → V** (or Cad⁶₄); **N⁶ → V** (or vii°⁷/V, Cad⁶₄). T may also go straight to D (0.6). The retrogression D → PD (4.0) is over the limit and never taken, except the passing V–IV⁶–I⁶ bass descent (2.4).

### Voicing (SATB)

- **Ranges** S C4–G5 · A G3–D5 · T C3–G4 · B E2–D4; no voice crossing.
- **Spacing** S–A ≤ P8, A–T ≤ P8, T–B ≤ P12.
- **Completeness** every chord tone present; only the 5th of a root-position chord may be omitted.
- **Doubling** prefer the root; in vii° and N⁶ double the 3rd; in Cad⁶₄ double the bass (5̂). Never double 7̂, a chordal 7th, ♭2̂, or an augmented-sixth tone.

### Voice leading, chord to chord

- **Forbidden** parallel P5/P8 in any pair of voices (+20); contrary P5/P8 (+5); direct 5ths/8ves in the outer voices with a soprano leap (+3); a leap beyond P8 in S, A or T (+8); A2 ♭6̂–♯7̂ (+6); melodic tritone (+3); cross relation (+2.5).
- **Discouraged** d5 → P5 against the bass (+1.5); voice overlap (+2); all four voices in similar motion (+1.5); upper-voice leaps beyond P5 (scaled).
- **Preferred** the nearest voicing: 0.3 per semitone in S, A, T, 0.08 in B; −0.2 per held common tone.

### Tendency tones

Enforced whenever the next chord contains the resolution; an unresolved tendency tone costs +4 to +8.

- **7̂ → 1̂** up by step. In an inner voice, 7̂ may fall to 5̂ into I (frustrated leading tone, +0.8).
- **Chordal 7th** down by step (held as a common tone: +1.5).
- **Cad⁶₄** 6–5 and 4–3 over the bass.
- **Augmented sixth** ♭6̂ and ♯4̂ expand outward by semitone to the octave on 5̂.
- **N⁶** ♭2̂ falls to 7̂.
- **d5 of vii°** and **borrowed ♭6̂** fall by step (+1 if not).

### Phrase, across the whole path

- **Colour tax** applied +0.35, mixture +0.45, N⁶ / ⁺⁶ +0.55.
- **Circling** X–Y–X oscillation +1.6; revisiting a harmony +0.8; three chords in one functional area +1.2; cancelling a borrowed chord (iv → IV) +1.6.
- **Cadence bonus** PAC −0.9 · IAC −0.5 · PHC −0.5 · HC −0.45 · PC −0.45 · DC −0.35.
- **Cadences** PAC: V(⁷) → I, both in root position, soprano on 1̂. IAC: any other D → I. HC: ends on root-position V. PHC: iv⁶ → V in minor. PC: IV/iv → I or ♭VII → I. DC: V → vi.""";


"""Inline Markdown: **bold** spans."""
def inline(text: str) -> list[any] {
    out: list[any] = [];
    for (i, part) in enumerate(text.split("**")) {
        if i % 2 == 1 {
            out.append(<b key={"b" + str(i)}>{part}</b>);
        } elif part {
            out.append(<span key={"t" + str(i)}>{part}</span>);
        }
    }
    return out;
}


"""Roman numeral in a box, outlined in its function's colour."""
def rn_box(x: int, y: int, f: str, rn: str, w: int = 68) -> JsxElement {
    return <g key={"b" + rn + str(x)}>
        <rect className="fn-box" x={x - w // 2} y={y - 18} width={w} height={36} rx={9} style={{"stroke": "var(--f-" + f + ")"}}/>
        <text className="fn-rn" x={x} y={y + 6} textAnchor="middle">{rn}</text>
    </g>;
}


"""A function heading (underlined, in its colour) with the rest of its chords below."""
def heading(x: int, f: str, name: str, also: str) -> JsxElement {
    return <g key={"h" + name + str(x)}>
        <text className="fn-name" x={x} y={0} textAnchor="middle" style={{"fill": "var(--f-" + f + ")"}}>{name}</text>
        <line className="fn-under" x1={x - 52} y1={6} x2={x + 52} y2={6} style={{"stroke": "var(--f-" + f + ")"}}/>
        {if also { <text className="fn-also" x={x} y={22} textAnchor="middle">{also}</text> }}
    </g>;
}


def arrow(d: str, cls: str = "") -> JsxElement {
    return <path key={d} className={"fn-edge " + cls} d={d} markerEnd="url(#fn-ah)"/>;
}


def label(x: int, y: int, t: str, cls: str = "") -> JsxElement {
    return <text key={t + str(x)} className={"fn-label " + cls} x={x} y={y} textAnchor="middle">{t}</text>;
}


"""Dashed arrow down the right side of a two-chord column: movement within the area."""
def within(x: int, cost: str) -> JsxElement {
    return <g key={"w" + str(x)}>
        {arrow(f"M {x + 36} {UP + 4} C {x + 62} {UP + 16}, {x + 62} {LO - 16}, {x + 36} {LO - 4}", "dashed")}
        {label(x, MID + 4, cost)}
    </g>;
}


"""Short arrow between two columns at mid-height (from the right of a to the left of b)."""
def step(xa: int, xb: int, y: int) -> JsxElement {
    return arrow(f"M {xa + 42} {y} L {xb - 42} {y}");
}


glob UP: int = 80,
     MID: int = 110,
     LO: int = 140;


"""Tonic → tonic prolongation → pre-dominant → dominant → tonic, drawn after the two
standard flow charts (major and minor). Numbers are the syntax costs in
theory/syntax.jac; dashed arrows are possible moves within a functional area."""
def FunctionFlow(mode: str) -> JsxElement {
    major = mode != "minor";
    T = 70;
    TP = 270 if major else 330;
    PD = 470 if major else 520;
    D = 670 if major else 700;
    T2 = 860;
    VII = 190;
    return <svg className="fn-flow" viewBox="0 -80 930 430" role="img" aria-label={"Harmonic function flow chart, " + ("major" if major else "minor")}>
        <defs>
            <marker id="fn-ah" viewBox="0 0 10 10" refX={9} refY={5} markerWidth={7} markerHeight={7} orient="auto-start-reverse">
                <path d="M0,0 L10,5 L0,10 z" className="fn-head"/>
            </marker>
        </defs>

        {heading(T, "T", "Tonic", "also I⁶" if major else "")}
        {heading(TP, "TX", "Tonic prolongation", "")}
        {heading(PD, "PD", "Pre-dominant", "also ii⁷ · iv · ♭VI · iiø⁶₅" if major else "also iiø⁷")}
        {heading(D, "D", "Dominant", "also V⁷ · viiø⁷" if major else "also V⁷ · vii°⁷")}
        {heading(T2, "T", "Tonic", "")}

        {arrow(f"M {T + 40} -4 L {TP - 82} -4")}
        {arrow(f"M {TP + 82} -4 L {PD - 62} -4")}
        {arrow(f"M {PD + 62} -4 L {D - 50} -4")}
        {arrow(f"M {D + 50} -4 L {T2 - 40} -4")}
        {label((T + TP) // 2 + 8, -12, "0.6")}
        {label((TP + PD) // 2, -12, "0.4")}
        {label((PD + D) // 2, -12, "0.4")}
        {label((D + T2) // 2, -12, "0.2")}
        {arrow(f"M {T} -16 C {T} -52, {PD} -52, {PD} -20")}
        {arrow(f"M {T} -16 C {T} -76, {D} -76, {D} -20")}
        {label((T + PD) // 2, -46, "T → PD 0.45")}
        {label((T + D) // 2 + 60, -66, "T → D 0.6")}

        {if major {
            <g>
                {rn_box(T, MID, "T", "I")}
                {step(T, TP, MID)}
                {rn_box(TP, UP, "TX", "iii")}
                {rn_box(TP, LO, "TX", "vi")}
                {within(TP, "0.45")}
                {step(TP + 30, PD, MID)}
                {rn_box(PD, UP, "PD", "IV")}
                {rn_box(PD, LO, "PD", "ii")}
                {within(PD, "0.5")}
                {step(PD + 30, D, MID)}
                {rn_box(D, UP, "D", "vii°")}
                {rn_box(D, LO, "D", "V")}
                {within(D, "1.6")}
                {step(D + 30, T2, MID)}
                {rn_box(T2, MID, "T", "I")}
            </g>
        } else {
            <g>
                {rn_box(T, MID, "T", "i")}
                {step(T, VII, MID)}
                {rn_box(VII, MID, "MIX", "VII")}
                {arrow(f"M {VII + 36} {MID - 8} L {TP - 38} {UP + 6}")}
                {label((VII + TP) // 2, UP - 2, "0.5")}
                {rn_box(TP, UP, "TX", "III")}
                {rn_box(TP, LO, "TX", "VI")}
                {within(TP, "0.45")}
                {arrow(f"M {T} {MID + 20} C {T} {LO + 52}, {TP - 40} {LO + 44}, {TP - 38} {LO + 8}")}
                {step(TP + 30, PD, MID)}
                {rn_box(PD, UP, "PD", "iv")}
                {rn_box(PD, LO, "PD", "ii°")}
                {within(PD, "0.5")}
                {step(PD + 30, D, MID)}
                {rn_box(D, UP, "D", "vii°")}
                {rn_box(D, LO, "D", "V")}
                {within(D, "1.6")}
                {step(D + 30, T2, MID)}
                {rn_box(T2, MID, "T", "i")}
            </g>
        }}

        {arrow(f"M {D - 8} {LO + 20} C {D - 8} {LO + 70}, {TP + 8} {LO + 70}, {TP + 8} {LO + 22}", "dashed deceptive")}
        {label((TP + D) // 2, LO + 66, "deceptive V → " + ("vi" if major else "VI") + " 1.1")}

        <line className="fn-rule" x1={10} y1={238} x2={920} y2={238}/>
        <text className="fn-also" x={920} y={256} textAnchor="end">colour and chromatic chords</text>
        {rn_box(TP, 300, "MIX", "♭VII · ♭III" if major else "Picardy I", 112)}
        {if major {
            <g>
                {arrow(f"M {TP - 58} 300 C {T + 40} 300, {T} 280, {T} {MID + 22}")}
                {label(T + 96, 322, "♭VII → I 0.7")}
            </g>
        } else {
            <g>{label(TP, 336, "change of mode 1.5")}</g>
        }}
        {rn_box(PD, 300, "NEAP", "N⁶", 76)}
        {arrow(f"M {PD + 40} 300 L {D - 40} 300")}
        {rn_box(D, 300, "CAD64", "Cad⁶₄", 76)}
        {arrow(f"M {D} 282 L {D} {LO + 22}")}
        {label((PD + D) // 2, 292, "0.45")}
        {label(D + 30, 266, "0.1")}
        {label(PD, 336, "→ V 0.35")}
        {rn_box(T2 - 30, 300, "APP", "V/x · vii°⁷/x", 128)}
        {label(T2 - 30, 336, "→ x 0.25")}
    </svg>;
}


"""Block Markdown: ##/### headings, - lists, `code` lines, paragraphs, [[flow]].
The flow chart follows the session's mode until the selector picks one."""
def:pub Rules(mode: str) -> JsxElement {
    has shown: str = "";
    chart_mode = shown if shown else mode;
    blocks: list[any] = [];
    items: list[any] = [];
    for (i, raw) in enumerate(RULES_MD.split("\n")) {
        line = raw.strip();
        if line.startswith("- ") {
            items.append(<li key={"li" + str(i)}>{inline(line[2:])}</li>);
            continue;
        }
        if items {
            blocks.append(<ul key={"ul" + str(i)}>{items}</ul>);
            items = [];
        }
        if not line {
            continue;
        }
        if line.startswith("### ") {
            blocks.append(<h3 key={"h" + str(i)}>{line[4:]}</h3>);
        } elif line.startswith("## ") {
            blocks.append(<h2 key={"h" + str(i)}>{line[3:]}</h2>);
        } elif line == "[[flow]]" {
            blocks.append(
                <div key="flow" className="fn-wrap">
                    <div className="seg">
                        <button className={"on" if chart_mode != "minor" else ""} title="Show the flow chart for major keys" onClick={lambda e: MouseEvent { shown = "major"; }}>Major</button>
                        <button className={"on" if chart_mode == "minor" else ""} title="Show the flow chart for minor keys" onClick={lambda e: MouseEvent { shown = "minor"; }}>Minor</button>
                    </div>
                    {FunctionFlow(chart_mode)}
                    <p className="fn-note">Dashed arrows: possible movement within a functional area. Numbers: syntax cost (lower = more idiomatic).</p>
                </div>
            );
        } elif line.startswith("`") and line.endswith("`") {
            blocks.append(<pre key={"c" + str(i)}><code>{line[1:-1]}</code></pre>);
        } else {
            blocks.append(<p key={"p" + str(i)}>{inline(line)}</p>);
        }
    }
    if items {
        blocks.append(<ul key="ul-end">{items}</ul>);
    }
    return <div className="card rules"><div className="card-b md">{blocks}</div></div>;
}
