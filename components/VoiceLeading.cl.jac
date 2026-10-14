"""The voice-leading diagram for one edge of the progression: the chord before and
the chord after on a grand staff, as the walkers voiced them (soprano and alto on
the treble staff, tenor and bass on the bass staff), with an arrow for every voice
a rule bears on. Hovering an arrow shows the rule below the staff; hovering a note
head shows its name and scale degree there, so a head's line or space is easy to read
(heads are drawn open, so the staff line shows through).

The rules come from the server's fixed rule bank (theory/leading.jac), carried on
the step as `leads`; voices no rule speaks to are drawn without an arrow.
Notes are spelled as the server spells them, with every accidental written out
(no key signature).
"""

sv import from ..services.atlas { StepView, LeadView }

glob W: int = 300,
     H: int = 152,
     SP: float = 3.5,         # half a staff space: one diatonic step
     TREBLE_TOP: float = 22.0,  # y of the top line (F5)
     BASS_TOP: float = 92.0,    # y of the top line (A3)
     XL: float = 118.0,       # x of the chord before
     XR: float = 228.0,       # x of the chord after
     STAFF_L: float = 6.0,
     STAFF_R: float = 294.0;

glob LETTER_IDX: dict[str, int] = {"C": 0, "D": 1, "E": 2, "F": 3, "G": 4, "A": 5, "B": 6},
     VOICE_NAMES: list[str] = ["bass", "tenor", "alto", "soprano"],
     RANK: dict[str, int] = {"hold": 0, "good": 1, "warn": 2, "bad": 3};

glob:pub KIND_COLOR: dict[str, str] = {"hold": "#8fb3ff", "good": "#7bd88f", "warn": "#ffcf66", "bad": "#ff7a85"};


"""Semitones the spelled name is altered by (♯ +1, ♭ -1, 𝄪 +2, 𝄫 -2)."""
def alteration(name: str) -> int {
    if "𝄪" in name {
        return 2;
    }
    if "𝄫" in name {
        return -2;
    }
    if "♯" in name {
        return 1;
    }
    if "♭" in name {
        return -1;
    }
    return 0;
}


def accidental_glyph(alt: int) -> str {
    if alt == 2 {
        return "𝄪";
    }
    if alt == -2 {
        return "𝄫";
    }
    if alt == 1 {
        return "♯";
    }
    if alt == -1 {
        return "♭";
    }
    return "";
}


"""Diatonic staff index of a spelled note (C4 = 28): letter plus octave, where the
octave comes from the MIDI number less the alteration (so B♯3 is index 27, not 28)."""
def diatonic(name: str, midi: int) -> int {
    octave = int(Math.floor((midi - alteration(name)) / 12)) - 1;
    return octave * 7 + LETTER_IDX[name[0]];
}


"""Voices 2-3 (alto, soprano) sit on the treble staff, 0-1 on the bass staff."""
def staff_y(voice: int, d: int) -> float {
    if voice >= 2 {
        return TREBLE_TOP + (38 - d) * SP;
    }
    return BASS_TOP + (26 - d) * SP;
}


"""Ledger-line staff indices a note at `d` needs on its staff."""
def ledgers(voice: int, d: int) -> list[int] {
    out: list[int] = [];
    lo = 30 if voice >= 2 else 18;
    hi = 38 if voice >= 2 else 26;
    k = hi + 2;
    while k <= d {
        out.append(k);
        k += 2;
    }
    k = lo - 2;
    while k >= d {
        out.append(k);
        k -= 2;
    }
    return out;
}


obj Head {
    has voice: int,
        x: float,
        y: float,
        d: int,
        alt: int;
}


"""Note heads of one chord at x; the upper note of a second (or unison) on the same
staff steps to the right, as engraved."""
def heads(x: float, notes: list[int], names: list[str]) -> list[Head] {
    out: list[Head] = [];
    for v in range(4) {
        d = diatonic(names[v], notes[v]);
        out.append(Head(voice=v, x=x, y=staff_y(v, d), d=d, alt=alteration(names[v])));
    }
    for pr in [[0, 1], [2, 3]] {
        gap = out[pr[1]].d - out[pr[0]].d;
        if gap == 0 or gap == 1 {
            out[pr[1]].x = x + 9.0;
        }
    }
    return out;
}


def:pub worst(ls: list[LeadView]) -> str {
    k = "hold";
    for l in ls {
        if RANK[l.kind] > RANK[k] {
            k = l.kind;
        }
    }
    return k;
}


def:pub VoiceLeading(prev: StepView, step: StepView, onClose: Callable[[], None]) -> JsxElement {
    has hover: int = -1,
        # the note head under the mouse: [0 before | 1 after, voice], or []
        hover_note: list[int] = [];

    left = heads(XL, prev.notes, prev.names);
    right = heads(XR, step.notes, step.names);
    by_voice: list[list[LeadView]] = [[], [], [], []];
    for l in step.leads {
        by_voice[l.voice].append(l);
    }
    shown: list[LeadView] = by_voice[hover] if hover >= 0 else [];
    # a rule between two voices (parallels, overlap, cross relation) lights both arrows
    related: list[int] = [];
    for v in range(4) {
        for l in by_voice[v] {
            if any([l.title == s.title and l.detail == s.detail for s in shown]) and v not in related {
                related.append(v);
            }
        }
    }

    def staff_lines(top: float, key: str) -> list[JsxElement] {
        return [<line key={key + str(k)} x1={STAFF_L} x2={STAFF_R} y1={top + k * 2 * SP} y2={top + k * 2 * SP} className="vl-staff"/> for k in range(5)];
    }

    def head_g(h: Head, key: str) -> JsxElement {
        return <g key={key}>
            {[<line key={key + "l" + str(d)} x1={h.x - 8.0} x2={h.x + 8.0} y1={staff_y(h.voice, d)} y2={staff_y(h.voice, d)} className="vl-staff"/> for d in ledgers(h.voice, h.d)]}
            {if h.alt != 0 {
                <text x={h.x - 10.0} y={h.y + (2.5 if h.alt == -1 else 4.0)} className="vl-acc" textAnchor="end">{accidental_glyph(h.alt)}</text>
            }}
            <ellipse cx={h.x} cy={h.y} rx={5.0} ry={3.4} transform={f"rotate(-18 {h.x} {h.y})"} className="vl-head"/>
        </g>;
    }

    """An invisible target over a head (drawn above the arrows, so it wins near them)."""
    def head_hit(h: Head, side: int) -> JsxElement {
        on = len(hover_note) == 2 and hover_note[0] == side and hover_note[1] == h.voice;
        return <ellipse
            key={"h" + str(side) + str(h.voice)}
            cx={h.x} cy={h.y} rx={6.5} ry={4.5}
            className={"vl-head-hit" + (" on" if on else "")}
            onMouseEnter={lambda e: any { hover_note = [side, h.voice]; hover = -1; }}
            onMouseLeave={lambda e: any { hover_note = []; }}
        />;
    }

    """A note as the rule details write it: "soprano: D5 (2̂)"."""
    def note_label(st: StepView, v: int) -> str {
        deg = st.degrees[v] if v < len(st.degrees) else "";
        return VOICE_NAMES[v] + ": " + st.names[v] + (" (" + deg + ")" if deg else "");
    }

    def arrow(v: int) -> JsxElement {
        a = left[v];
        b = right[v];
        kind = worst(by_voice[v]);
        on = hover == v or v in related;
        x1 = a.x + 7.0;
        x2 = b.x - (19.0 if b.alt != 0 else 8.0);
        # bow away from the other voice on the same staff: upper voices up, lower ones down
        bow = -6.0 if v in [1, 3] else 6.0;
        mx = (x1 + x2) / 2;
        my = (a.y + b.y) / 2 + bow;
        d = f"M {x1} {a.y} Q {mx} {my} {x2} {b.y}";
        return <g key={"a" + str(v)} onMouseEnter={lambda e: any { hover = v; hover_note = []; }} onMouseLeave={lambda e: any { hover = -1; }}>
            <path d={d} fill="none" stroke={KIND_COLOR[kind]} strokeWidth={2.4 if on else 1.4} opacity={1.0 if on or hover < 0 else 0.35} markerEnd={"url(#vl-m-" + kind + ")"}/>
            <path d={d} fill="none" stroke="transparent" strokeWidth={12} className="vl-hit"/>
        </g>;
    }

    return <div className="vl-card" onPointerDown={lambda e: any { e.stopPropagation(); }} onWheel={lambda e: any { e.stopPropagation(); }}>
        <div className="vl-head-row">
            <span className="vl-title">{prev.rn} → {step.rn}</span>
            <span className="vl-sub">{prev.symbol} → {step.symbol}</span>
            <button className="vl-close" title="Close" onClick={lambda e: MouseEvent { e.stopPropagation(); onClose(); }}>×</button>
        </div>
        <svg width={W} height={H} viewBox={f"0 0 {W} {H}"} className="vl-svg">
            <defs>
                {[
                    <marker key={k} id={"vl-m-" + k} viewBox="0 0 8 8" refX={7} refY={4} markerWidth={5} markerHeight={5} orient="auto-start-reverse">
                        <path d="M 0 0 L 8 4 L 0 8 z" fill={KIND_COLOR[k]}/>
                    </marker>
                    for k in ["hold", "good", "warn", "bad"]
                ]}
            </defs>
            {staff_lines(TREBLE_TOP, "t")}
            {staff_lines(BASS_TOP, "b")}
            <line x1={STAFF_L} x2={STAFF_L} y1={TREBLE_TOP} y2={BASS_TOP + 8 * SP} className="vl-staff"/>
            <text x={STAFF_L + 3} y={TREBLE_TOP + 6 * SP + 9} className="vl-clef">𝄞</text>
            <text x={STAFF_L + 3} y={BASS_TOP + 2 * SP + 8} className="vl-clef bass">𝄢</text>
            {[head_g(h, "L" + str(h.voice)) for h in left]}
            {[head_g(h, "R" + str(h.voice)) for h in right]}
            {[arrow(v) for v in range(4) if len(by_voice[v]) > 0]}
            {[head_hit(h, 0) for h in left]}
            {[head_hit(h, 1) for h in right]}
            <text x={XL} y={H - 4} className="vl-rn" textAnchor="middle">{prev.rn}</text>
            <text x={XR} y={H - 4} className="vl-rn" textAnchor="middle">{step.rn}</text>
        </svg>
        <div className="vl-why">
            {if len(hover_note) == 2 {
                <div className="vl-rule">
                    <div className="vl-detail">{note_label(prev if hover_note[0] == 0 else step, hover_note[1])}</div>
                    <div className="vl-text">{"in " + (prev.rn + " (" + prev.symbol + ")" if hover_note[0] == 0 else step.rn + " (" + step.symbol + ")")}</div>
                </div>
            } elif hover < 0 {
                <p className="vl-hint">
                    {("Hover an arrow for the rule behind it, or a note for its name." if len(step.leads) > 0 else "No rule in the bank applies to this change: no tendency tones, held common tones or faults. Hover a note for its name.")}
                </p>
            } else {
                <div>
                    {for (k, l) in enumerate(shown) {
                        <div key={str(k)} className="vl-rule">
                            <div className={"vl-rule-title " + l.kind}>{l.title}</div>
                            <div className="vl-detail">{l.detail}</div>
                            <div className="vl-text">{l.text}</div>
                        </div>
                    }}
                </div>
            }}
        </div>
    </div>;
}
