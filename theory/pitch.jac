"""Pitch-class arithmetic, keys, and correct enharmonic spelling.

Spelling is degree-based: a note is named from the letter of its scale degree
relative to the tonic letter, then given whatever accidental reaches its pitch
class. So C-sharp in V/ii (A-C#-E, key of C) is spelled C#, never D-flat.
"""

glob LETTERS: list[str] = ["C", "D", "E", "F", "G", "A", "B"],
     LETTER_PC: list[int] = [0, 2, 4, 5, 7, 9, 11],
     MAJOR_SCALE: list[int] = [0, 2, 4, 5, 7, 9, 11],
     MINOR_SCALE: list[int] = [0, 2, 3, 5, 7, 8, 10],
     NUMERALS: list[str] = ["I", "II", "III", "IV", "V", "VI", "VII"];

"""Tonic names offered in the UI (major keys, then minor keys)."""
glob MAJOR_TONICS: list[str] = ["C", "G", "D", "A", "E", "F", "B♭", "E♭", "A♭", "D♭"],
     MINOR_TONICS: list[str] = ["A", "E", "B", "F♯", "C♯", "D", "G", "C", "F"];


def accidental(diff: int) -> str {
    return {-2: "𝄫", -1: "♭", 0: "", 1: "♯", 2: "𝄪"}.get(diff, "?");
}


def iabs(x: int) -> int {
    return x if x >= 0 else -x;
}


def r2(x: float) -> float {
    return int(x * 100.0 + (0.5 if x >= 0.0 else -0.5)) / 100.0;
}


"""Signed semitone difference folded into -6..+5."""
def signed(diff: int) -> int {
    return ((diff + 6) % 12) - 6;
}


obj Key {
    has tonic_letter: int,
        tonic_pc: int,
        mode: str;

    """Parse a tonic like "E♭", "Eb", "F#", with mode "major" or "minor"."""
    static def parse(tonic: str, mode: str) -> Key {
        letter = LETTERS.index(tonic[0].upper());
        acc = 0;
        for ch in tonic[1:] {
            if ch in ["b", "♭"] { acc -= 1; }
            elif ch in ["#", "♯"] { acc += 1; }
        }
        return Key(
            tonic_letter=letter,
            tonic_pc=(LETTER_PC[letter] + acc) % 12,
            mode="minor" if mode == "minor" else "major"
        );
    }

    def tonic_name -> str {
        return self.spell(0, self.tonic_pc);
    }

    def name -> str {
        return f"{self.tonic_name()} {self.mode}";
    }

    def scale -> list[int] {
        return MAJOR_SCALE if self.mode == "major" else MINOR_SCALE;
    }

    """Diatonic pitch class of 0-based scale degree `deg`."""
    def degree_pc(deg: int) -> int {
        return (self.tonic_pc + self.scale()[deg % 7]) % 12;
    }

    """Pitch class of a chromatic offset above the tonic (0..11)."""
    def rel(semitones: int) -> int {
        return (self.tonic_pc + semitones) % 12;
    }

    """Spell pitch class `pc` on the letter of scale degree `deg`."""
    def spell(deg: int, pc: int) -> str {
        letter = (self.tonic_letter + deg) % 7;
        return LETTERS[letter] + accidental(signed(pc - LETTER_PC[letter]));
    }

    """Spell a MIDI note on the letter of degree `deg`, with octave (C4 = 60)."""
    def spell_midi(deg: int, midi: int) -> str {
        letter = (self.tonic_letter + deg) % 7;
        diff = signed((midi % 12) - LETTER_PC[letter]);
        octave = (midi - diff) // 12 - 1;
        return LETTERS[letter] + accidental(diff) + str(octave);
    }
}
