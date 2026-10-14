"""Functional harmonic syntax: may chord `b` follow chord `a`, and how idiomatic is it?

These rules are the edge weights of the harmony graph. They encode common-practice
phrase grammar (T -> PD -> D -> T), with specific obligations for chords that
*must* resolve (cadential 6/4, applied chords, augmented sixths, the Neapolitan)
and graded costs for weaker motions (retrogressions, deceptive motion, mixture).
Lower cost = more idiomatic. `None` = not a legal progression.

Voice leading is deliberately NOT considered here - it depends on the actual
voicing a path arrives with, so the beam-search walker scores it per path.
"""

import from theory.pitch { Key }
import from theory.vocab { ChordSpec, ADDED_BASE }


obj Rule {
    has cost: float,
        label: str;
}


def is_root_position_dominant(c: ChordSpec) -> bool {
    return c.func == "D" and c.root_deg == 4 and c.inversion == 0;
}


def syntax(a: ChordSpec, b: ChordSpec, key: Key) -> Rule | None {
    if a.cid == b.cid {
        return None;
    }

    # Same harmony, new inversion or an added seventh: prolongation.
    if a.root_pc == b.root_pc and a.func == b.func and a.target_pc == b.target_pc and a.func != "CAD64" {
        if a.quality == b.quality {
            return Rule(cost=0.9, label="prolongation: same harmony, new inversion");
        }
        base_a = QUALITY_BASE.get(a.quality, a.quality);
        if b.is_seventh and not a.is_seventh and QUALITY_BASE.get(b.quality, "") == base_a {
            return Rule(cost=0.5, label="adds the chordal 7th over a held harmony");
        }
        if not a.is_seventh and not b.is_seventh and base_a == QUALITY_BASE.get(b.quality, b.quality) {
            if b.quality in ADDED_BASE {
                return Rule(cost=0.6, label="colours the held harmony with an added tone");
            }
            return Rule(cost=0.7, label="the added tone settles into the plain triad");
        }
        if not b.is_seventh and a.is_seventh {
            return None;
        }
    }

    # Cadential 6/4 is a double suspension over 5̂: it can only go to V.
    if a.func == "CAD64" {
        if is_root_position_dominant(b) {
            return Rule(cost=0.1, label="cadential ⁶₄ resolves (6–5, 4–3 over the bass)");
        }
        return None;
    }
    if b.func == "CAD64" {
        if a.func in ["PD", "NEAP", "AUG6"] or (a.func == "APP" and a.target_deg == 4) {
            return Rule(cost=0.45, label="pre-dominant → cadential ⁶₄");
        }
        return None;
    }

    # Applied chords are obligated to their target.
    if a.func == "APP" {
        if b.func != "APP" and b.root_pc == a.target_pc and b.root_deg == a.target_deg {
            return Rule(cost=0.25, label="applied chord resolves to the chord it tonicizes");
        }
        if b.func == "APP" and b.root_pc == a.target_pc {
            return Rule(cost=0.7, label="chain of applied dominants (descending fifths)");
        }
        if b.func != "APP" and b.root_pc == (a.target_pc + 9) % 12 {
            return Rule(cost=1.8, label="deceptive resolution of an applied chord");
        }
        return None;
    }

    if a.func == "AUG6" {
        if is_root_position_dominant(b) {
            return Rule(cost=0.3, label="augmented sixth expands outward to the octave on 5̂");
        }
        return None;
    }

    if a.func == "NEAP" {
        if b.func == "D" and b.root_deg == 4 {
            return Rule(cost=0.35, label="Neapolitan → V (♭2̂ falls to 7̂)");
        }
        if b.func == "APP" and b.target_deg == 4 and b.quality == "dim7" {
            return Rule(cost=0.6, label="Neapolitan → vii°⁷/V");
        }
        return None;
    }

    if a.func == "D" {
        if b.func == "T" {
            cost = 0.2 if a.root_deg == 4 else 0.35;
            if b.inversion > 0 {
                cost += 0.25;
            }
            return Rule(cost=cost, label="authentic resolution (D → T)");
        }
        if b.func == "TX" and b.root_deg == 5 and a.root_deg == 4 {
            return Rule(cost=1.1, label="deceptive motion (V → vi)");
        }
        if b.func == "D" {
            return Rule(cost=1.6, label="dominant expansion");
        }
        if b.func == "PD" and b.root_deg == 3 and b.inversion == 1 and a.root_deg == 4 and not a.is_seventh {
            return Rule(cost=2.4, label="passing IV⁶ (V–IV⁶–I⁶ bass descent)");
        }
        if b.func == "PD" {
            return Rule(cost=4.0, label="retrogression (D → PD)");
        }
        return None;
    }

    if a.func == "PD" {
        if b.func == "D" {
            return Rule(cost=0.4, label="pre-dominant → dominant");
        }
        if b.func == "APP" and b.target_deg == 4 {
            return Rule(cost=0.5, label="pre-dominant → applied V/V (intensified approach)");
        }
        if b.func == "AUG6" {
            # iv6 / ♭VI already hold ♭6̂ in the bass: the augmented sixth grows out of them
            cost = 0.55 if a.root_pc in [key.rel(5), key.rel(8)] else 1.2;
            return Rule(cost=cost, label="pre-dominant intensifies to an augmented sixth");
        }
        if b.func == "NEAP" {
            return Rule(cost=1.0, label="pre-dominant → Neapolitan");
        }
        if b.func == "PD" {
            fall = (a.root_pc - b.root_pc) % 12;
            if fall in [3, 4] {
                return Rule(cost=0.5, label="pre-dominant expansion (descending third, e.g. IV → ii)");
            }
            if a.root_pc == b.root_pc {
                return Rule(cost=0.6, label="modal mixture within the pre-dominant (IV → iv)");
            }
            return Rule(cost=1.8, label="weak pre-dominant succession");
        }
        if b.func == "T" and b.root_deg == 0 and a.root_deg == 3 {
            cost = 0.9 if b.inversion == 0 else 1.6;
            return Rule(cost=cost, label="plagal motion (IV → I)");
        }
        if b.func == "MIX" {
            if a.root_pc == key.rel(8) and b.root_pc == key.rel(10) {
                return Rule(cost=0.6, label="♭VI → ♭VII (Aeolian ascent to the tonic)");
            }
            return Rule(cost=1.3, label="pre-dominant → borrowed colour chord");
        }
        if b.func == "TX" {
            return Rule(cost=2.6, label="retrogression (PD → tonic extension)");
        }
        return None;
    }

    if a.func == "T" and a.family == "mixture" {
        # A Picardy third is an arrival; the only way on is to hear it as V of iv.
        if b.func == "PD" and b.root_deg == 3 {
            return Rule(cost=1.0, label="Picardy I heard as V/iv");
        }
        return None;
    }

    if a.func == "T" {
        if b.func == "T" {
            return Rule(cost=1.5, label="change of tonic mode (Picardy / mixture)");
        }
        return {
            "TX": Rule(cost=0.6, label="tonic → tonic extension"),
            "PD": Rule(cost=0.45, label="tonic → pre-dominant"),
            "D": Rule(cost=0.6, label="tonic → dominant"),
            "APP": Rule(cost=0.8, label="tonicization of a diatonic chord"),
            "MIX": Rule(cost=0.9, label="borrowed chord (modal mixture)"),
            "NEAP": Rule(cost=1.6, label="tonic → Neapolitan"),
            "AUG6": Rule(cost=2.2, label="tonic → augmented sixth")
        }.get(b.func, None);
    }

    if a.func == "TX" {
        fall = (a.root_pc - b.root_pc) % 12;
        if b.func == "PD" {
            cost = 0.4 if fall in [7, 3, 4] else 0.7;
            return Rule(cost=cost, label="tonic extension → pre-dominant");
        }
        if b.func == "TX" {
            cost = 0.45 if fall == 7 else 1.4;
            return Rule(cost=cost, label="tonic extension succession (e.g. iii → vi)");
        }
        if b.func == "D" {
            return Rule(cost=1.1, label="tonic extension → dominant");
        }
        if b.func == "APP" {
            return Rule(cost=0.8, label="tonic extension → applied chord");
        }
        if b.func == "AUG6" {
            cost = 0.7 if a.root_pc == key.rel(8) else 1.6;
            return Rule(cost=cost, label="VI → augmented sixth (♭6̂ held in the bass)");
        }
        if b.func == "MIX" {
            return Rule(cost=1.3, label="tonic extension → borrowed chord");
        }
        if b.func == "NEAP" {
            return Rule(cost=1.4, label="tonic extension → Neapolitan");
        }
        if b.func == "T" {
            return Rule(cost=2.4, label="retrogression (back to the tonic)");
        }
        return None;
    }

    if a.func == "MIX" {
        subtonic = a.root_pc == key.rel(10);
        if b.func == "T" and b.inversion == 0 {
            if subtonic {
                return Rule(cost=0.7, label="♭VII → I (backdoor / Mixolydian cadence)");
            }
            return Rule(cost=1.4, label="borrowed chord → tonic");
        }
        if b.func == "PD" {
            cost = 0.8 if subtonic and b.root_pc == key.rel(5) else 1.1;
            return Rule(cost=cost, label="♭VII → IV (double plagal)" if cost < 1.0 else "borrowed chord → pre-dominant");
        }
        if b.func == "TX" {
            fall = (a.root_pc - b.root_pc) % 12;
            cost = 0.5 if fall == 7 else 1.2;
            return Rule(cost=cost, label="descending fifth into a tonic extension (VII → III)" if fall == 7 else "borrowed chord → tonic extension");
        }
        if b.func == "MIX" {
            return Rule(cost=0.8, label="succession of borrowed chords");
        }
        if b.func == "D" {
            return Rule(cost=1.6, label="borrowed chord → dominant");
        }
        return None;
    }
    return None;
}


glob QUALITY_BASE: dict[str, str] = {
    "dom7": "maj", "min7": "min", "maj7": "maj", "hdim7": "dim", "dim7": "dim", "7sus4": "sus4", **ADDED_BASE
};
