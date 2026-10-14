"""Why each voice moves where it does: the voice-leading diagram's rule bank.

`leads` looks at one chord change (voicing `va` of chord a to voicing `vb` of chord
b, as the walkers chose them) and names, per voice, the rules that bear on its
motion. Only voices a rule speaks to get an entry: tendency tones and how they
resolve, common tones kept in place, and the faults `theory.voicing.transition`
charges for. Free motion between chord tones gets none.

Every rule's wording is fixed text in `RULES`, the standard common-practice
four-part rules as taught in Aldwell & Schachter (Harmony and Voice Leading),
Kostka & Payne (Tonal Harmony) and Laitz (The Complete Musician). Nothing is
generated; only the notes, voices and scale degrees are filled in. The detection
mirrors `transition` so the diagram explains exactly what the walkers scored.
"""

import from theory.pitch { Key, iabs, signed }
import from theory.vocab { ChordSpec, key_for }
import from theory.voicing { VOICE_NAMES, VOICE_ABBR, note_name }


"""One rule of the bank. kind: good (a rule followed), hold (a common tone kept),
warn (a licensed exception or a mild fault), bad (a fault)."""
obj Rule {
    has kind: str,
        title: str,
        text: str;
}


glob RULES: dict[str, Rule] = {
    # --- tendency tones ---
    "lt": Rule(
        kind="good", title="Leading tone rises to the tonic",
        text="7̂ lies a half step below 1̂. When the next chord contains 1̂, the leading tone resolves up by step."
    ),
    "lt_app": Rule(
        kind="good", title="Applied leading tone rises",
        text="The third of an applied dominant is the leading tone of the tonicized chord and rises a half step to its root."
    ),
    "lt_frustrated": Rule(
        kind="warn", title="Frustrated leading tone",
        text="In an inner voice, 7̂ may fall a third to 5̂ in V(7)–I so the tonic triad is complete. Not acceptable in an outer voice."
    ),
    "lt_unresolved": Rule(
        kind="bad", title="Leading tone left unresolved",
        text="The leading tone is the most active tone of the dominant; when 1̂ is available it should rise to it, above all in an outer voice."
    ),
    "sev": Rule(
        kind="good", title="Chordal seventh resolves down by step",
        text="The seventh is a dissonance against the root and resolves down by step into the next chord."
    ),
    "sev_held": Rule(
        kind="warn", title="Chordal seventh held",
        text="The seventh stays as a common tone instead of falling by step, so its resolution is postponed to a later chord."
    ),
    "sev_unresolved": Rule(
        kind="bad", title="Chordal seventh left unresolved",
        text="A chordal seventh should resolve down by step; leaping away from it or rising leaves the dissonance unresolved."
    ),
    "d5": Rule(
        kind="good", title="Diminished fifth falls",
        text="The fifth of a diminished chord falls by step, so the diminished fifth above the root contracts to a third."
    ),
    "d5_up": Rule(
        kind="warn", title="Diminished fifth does not fall",
        text="The fifth of a diminished chord normally falls by step. Rising is acceptable where the tritone sounds as an augmented fourth, which expands to a sixth."
    ),
    "aug6": Rule(
        kind="good", title="Augmented sixth expands to the octave",
        text="♭6̂ (6̂ in minor) falls a half step and ♯4̂ rises a half step; the augmented sixth opens to an octave on 5̂."
    ),
    "aug6_unresolved": Rule(
        kind="bad", title="Augmented sixth not expanded",
        text="Both tones of the augmented sixth are directed: ♭6̂ (6̂ in minor) down to 5̂, ♯4̂ up to 5̂, each by half step."
    ),
    "cad6": Rule(
        kind="good", title="Cadential ⁶₄: the sixth falls to the fifth",
        text="Over the held bass 5̂, the sixth above the bass (3̂) falls by step to 2̂, the fifth of V (6–5)."
    ),
    "cad4": Rule(
        kind="good", title="Cadential ⁶₄: the fourth falls to the third",
        text="Over the held bass 5̂, the fourth above the bass (1̂) falls by step to 7̂, the third of V (4–3)."
    ),
    "cad_unresolved": Rule(
        kind="bad", title="Cadential ⁶₄ not resolved",
        text="The sixth and fourth above the bass of a cadential ⁶₄ are non-chord tones over V and resolve down by step (6–5, 4–3)."
    ),
    "neap": Rule(
        kind="good", title="♭2̂ of the Neapolitan descends",
        text="♭2̂ moves down: through 1̂ as a passing tone, or directly to 7̂ of V by a diminished third."
    ),
    "neap_unresolved": Rule(
        kind="warn", title="♭2̂ of the Neapolitan does not descend",
        text="♭2̂ tends downward, to 1̂ or to 7̂; moving it up or away weakens the chord's pull to the dominant."
    ),
    "b6": Rule(
        kind="good", title="Borrowed ♭6̂ falls to 5̂",
        text="Modal mixture's ♭6̂ is a half step above 5̂ and resolves down to it."
    ),
    "b6_unresolved": Rule(
        kind="warn", title="Borrowed ♭6̂ does not fall",
        text="♭6̂ borrowed from the minor mode tends down by half step to 5̂."
    ),
    # --- common tones ---
    "common": Rule(
        kind="hold", title="Common tone kept",
        text="A pitch shared by both chords stays in the same voice, which keeps the voice leading smooth."
    ),
    # --- faults ---
    "par5": Rule(
        kind="bad", title="Parallel fifths",
        text="Two voices move in the same direction from a perfect fifth to another perfect fifth and lose their independence."
    ),
    "par8": Rule(
        kind="bad", title="Parallel octaves",
        text="Two voices move in the same direction from an octave or unison to another and merge into one line."
    ),
    "con5": Rule(
        kind="bad", title="Fifths by contrary motion",
        text="Consecutive perfect fifths reached by contrary motion are still consecutive fifths and are avoided in strict four-part writing."
    ),
    "con8": Rule(
        kind="bad", title="Octaves by contrary motion",
        text="Consecutive octaves reached by contrary motion are still consecutive octaves and are avoided in strict four-part writing."
    ),
    "unequal5": Rule(
        kind="warn", title="Unequal fifths against the bass",
        text="A diminished fifth moving to a perfect fifth (d5–P5) is avoided when it involves the bass. P5–d5 is acceptable."
    ),
    "direct5": Rule(
        kind="bad", title="Direct fifths in the outer voices",
        text="Bass and soprano reach a perfect fifth in similar motion with the soprano leaping. Acceptable only when the soprano moves by step."
    ),
    "direct8": Rule(
        kind="bad", title="Direct octaves in the outer voices",
        text="Bass and soprano reach an octave in similar motion with the soprano leaping. Acceptable only when the soprano moves by step."
    ),
    "overlap": Rule(
        kind="warn", title="Voice overlap",
        text="A voice moves past the note its neighbouring voice just sounded, which blurs which line is which."
    ),
    "tritone": Rule(
        kind="bad", title="Melodic tritone",
        text="An augmented fourth or diminished fifth as a melodic interval is avoided in the upper voices."
    ),
    "aug2": Rule(
        kind="bad", title="Augmented second",
        text="♭6̂ to ♯7̂ in minor is an augmented second; it is avoided as a melodic interval."
    ),
    "leap8": Rule(
        kind="bad", title="Leap larger than an octave",
        text="Upper voices move by step or small leap; leaps beyond an octave are avoided."
    ),
    "leap": Rule(
        kind="warn", title="Large leap",
        text="Upper voices move mostly by step or small leap; leaps larger than a fifth are used sparingly."
    ),
    "cross": Rule(
        kind="bad", title="Cross relation",
        text="A note and its chromatic alteration sound in successive chords in different voices; the inflection should stay within one voice."
    ),
    "similar_all": Rule(
        kind="warn", title="All voices in similar motion",
        text="All four voices move in the same direction; at least one should move in contrary or oblique motion against the others."
    )
};


"""A rule applied to one voice of a chord change, with the notes it concerns."""
obj Lead {
    has voice: int,
        rule: str,
        detail: str;
}


"""Scale degree of `midi` in chord `c` (1̂…7̂, with ♭/♯ when it is altered)."""
def degree_label(key: Key, c: ChordSpec, midi: int) -> str {
    t = c.tone_for_pc(midi % 12);
    if t is None {
        return "";
    }
    acc = signed(midi % 12 - key.degree_pc(t.deg));
    return {-2: "𝄫", -1: "♭", 1: "♯", 2: "𝄪"}.get(acc, "") + str(t.deg + 1) + "̂";
}


def leads(key_id: str, a: ChordSpec, va: list[int], b: ChordSpec, vb: list[int]) -> list[Lead] {
    key = key_for(key_id);
    d = [vb[i] - va[i] for i in range(4)];
    bpcs = b.pcs();
    out: list[Lead] = [];
    tended: list[int] = [];

    def move(i: int) -> str {
        return f"{VOICE_NAMES[i]}: {note_name(key, a, va[i])} ({degree_label(key, a, va[i])}) → {note_name(key, b, vb[i])} ({degree_label(key, b, vb[i])})";
    }

    def pair(i: int, j: int) -> str {
        return f"{VOICE_ABBR[i]}–{VOICE_ABBR[j]}: {note_name(key, a, va[i])}/{note_name(key, a, va[j])} → {note_name(key, b, vb[i])}/{note_name(key, b, vb[j])}";
    }

    def add(i: int, rule: str, detail: str) {
        out.append(Lead(voice=i, rule=rule, detail=detail));
    }

    # Tendency tones, under the same conditions transition() scores them.
    for tend in a.tendencies {
        if not any([p in bpcs for p in tend.targets]) {
            continue;
        }
        for i in range(4) {
            if va[i] % 12 != tend.pc {
                continue;
            }
            step = d[i];
            dest = vb[i] % 12;
            rule = "";
            if tend.kind == "LT" {
                if step > 0 and step <= 2 and dest in tend.targets {
                    rule = "lt_app" if a.func == "APP" else "lt";
                } elif i in [1, 2] and b.func == "T" and dest == key.rel(7) and step < 0 {
                    rule = "lt_frustrated";
                } else {
                    rule = "lt_unresolved";
                }
            } elif tend.kind == "7th" {
                if step < 0 and step >= -2 and dest in tend.targets {
                    rule = "sev";
                } elif step == 0 and dest in bpcs {
                    rule = "sev_held";
                } else {
                    rule = "sev_unresolved";
                }
            } elif tend.kind == "aug6" {
                rule = "aug6" if iabs(step) == 1 and dest in tend.targets else "aug6_unresolved";
            } elif tend.kind in ["cad6", "cad4"] {
                rule = tend.kind if step < 0 and step >= -2 and dest in tend.targets else "cad_unresolved";
            } elif tend.kind == "N" {
                rule = "neap" if step < 0 and step >= -3 and dest in tend.targets else "neap_unresolved";
            } elif tend.kind == "d5" {
                rule = "d5" if step < 0 and step >= -2 and dest in tend.targets else "d5_up";
            } elif tend.kind == "b6" {
                rule = "b6" if step < 0 and step >= -2 and dest in tend.targets else "b6_unresolved";
            }
            if rule {
                add(i, rule, move(i));
                tended.append(i);
            }
        }
    }

    # Common tones held in place (a held seventh is already explained above).
    for i in range(4) {
        if d[i] == 0 and vb[i] % 12 in bpcs and i not in tended {
            add(i, "common", f"{VOICE_NAMES[i]}: {note_name(key, a, va[i])} ({degree_label(key, a, va[i])}) held");
        }
    }

    # Consecutive and unequal perfect intervals.
    for i in range(4) {
        for j in range(i + 1, 4) {
            if d[i] == 0 and d[j] == 0 {
                continue;
            }
            p1 = (va[j] - va[i]) % 12;
            p2 = (vb[j] - vb[i]) % 12;
            if p1 == p2 and p1 in [0, 7] {
                kind = "5" if p1 == 7 else "8";
                rule = ("par" if d[i] * d[j] > 0 else "con") + kind;
                if d[i] * d[j] != 0 {
                    add(i, rule, pair(i, j));
                    add(j, rule, pair(i, j));
                }
            }
            if i == 0 and p1 == 6 and p2 == 7 and d[i] * d[j] > 0 {
                add(i, "unequal5", pair(i, j));
                add(j, "unequal5", pair(i, j));
            }
        }
    }

    outer = (vb[3] - vb[0]) % 12;
    if d[0] * d[3] > 0 and outer in [0, 7] and iabs(d[3]) > 2 {
        rule = "direct8" if outer == 0 else "direct5";
        add(0, rule, pair(0, 3));
        add(3, rule, pair(0, 3));
    }

    for i in range(3) {
        if vb[i] > va[i + 1] or vb[i + 1] < va[i] {
            add(i, "overlap", pair(i, i + 1));
            add(i + 1, "overlap", pair(i, i + 1));
        }
    }

    # Melodic intervals.
    for i in range(4) {
        m: int = iabs(d[i]);
        if i > 0 and m > 12 {
            add(i, "leap8", move(i));
        } elif i > 0 and m > 7 {
            add(i, "leap", move(i));
        }
        if i > 0 and m == 6 {
            add(i, "tritone", move(i));
        }
        if m == 3 and sorted([va[i] % 12, vb[i] % 12]) == sorted([key.rel(8), key.rel(11)]) {
            add(i, "aug2", move(i));
        }
    }

    # Cross relations: a letter altered chromatically, the change not carried by one voice.
    for i in range(4) {
        ta = a.tone_for_pc(va[i]);
        if ta is None {
            continue;
        }
        ta_deg = ta.deg;
        ta_pc = ta.pc;
        for j in range(4) {
            tb = b.tone_for_pc(vb[j]);
            if tb is None {
                continue;
            }
            tb_pc = tb.pc;
            if j == i or tb.deg != ta_deg or tb_pc == ta_pc {
                continue;
            }
            carried = any([va[k] % 12 == ta_pc and vb[k] % 12 == tb_pc for k in range(4)]);
            if not carried {
                text = f"{note_name(key, a, va[i])} ({VOICE_ABBR[i]}) against {note_name(key, b, vb[j])} ({VOICE_ABBR[j]})";
                add(i, "cross", text);
                add(j, "cross", text);
            }
        }
    }

    # Shown on the outer voices only, so it does not draw an arrow on every voice.
    if all([x > 0 for x in d]) or all([x < 0 for x in d]) {
        text = "all four voices " + ("rise" if d[0] > 0 else "fall");
        add(0, "similar_all", text);
        add(3, "similar_all", text);
    }
    return out;
}
