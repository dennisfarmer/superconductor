"""Cadence classification for the last two chords of a path.

PAC  perfect authentic: root-position V(7) → root-position I, soprano on 1̂
IAC  imperfect authentic: any other D → T close
HC   half cadence: arrival on a root-position V triad
PHC  Phrygian half cadence (minor): iv⁶ → V
PC   plagal: IV/iv → I, or the backdoor ♭VII → I
DC   deceptive: V(7) → vi / VI
"""

import from theory.pitch { Key }
import from theory.vocab { ChordSpec }

glob CADENCE_NAMES: dict[str, str] = {
    "PAC": "Perfect authentic cadence",
    "IAC": "Imperfect authentic cadence",
    "HC": "Half cadence",
    "PHC": "Phrygian half cadence",
    "PC": "Plagal cadence",
    "DC": "Deceptive cadence"
};


def cadence(key: Key, a: ChordSpec, b: ChordSpec, vb: list[int]) -> str {
    if b.func == "T" and b.root_deg == 0 {
        if a.func == "D" {
            if a.root_deg == 4 and a.inversion == 0 and b.inversion == 0 and vb[3] % 12 == key.tonic_pc {
                return "PAC";
            }
            return "IAC";
        }
        if b.inversion == 0 and (
            (a.func == "PD" and a.root_deg == 3) or (a.func == "MIX" and a.root_pc == key.rel(10))
        ) {
            return "PC";
        }
        return "";
    }
    if b.func == "D" and b.root_deg == 4 and b.inversion == 0 and not b.is_seventh {
        if key.mode == "minor" and a.func == "PD" and a.root_deg == 3 and a.inversion == 1 {
            return "PHC";
        }
        return "HC";
    }
    if a.func == "D" and a.root_deg == 4 and b.func == "TX" and b.root_deg == 5 {
        return "DC";
    }
    return "";
}
