"""The chord vocabulary of a key.

Every chord a composer can reach from a given key: diatonic triads and
sevenths (with inversions), applied (secondary) dominants, modal mixture,
the Neapolitan, the three augmented-sixth chords, and the cadential 6/4.

Each chord carries its harmonic *function* (which drives the syntax rules in
`theory.syntax`) and its *tendency tones* (which drive the voice-leading
rules in `theory.voicing`).

Function codes:
    T      tonic (I, i, I6, Picardy I)
    TX     tonic extension / substitute (vi, iii, VI, III)
    PD     pre-dominant (ii, ii7, IV, iv, iiø7, ♭VI)
    D      dominant (V, V7, vii°, viiø7, vii°7)
    CAD64  cadential 6/4 (a dominant-function embellishment of V)
    APP    applied chord (V/x, V7/x, vii°7/x) - must resolve to its target
    NEAP   Neapolitan sixth
    AUG6   augmented sixth (It, Fr, Ger) - must expand to V
    MIX    borrowed colour chords that are neither PD nor D (♭VII, ♭III, VII)
"""


from __future__ import annotations
from dataclasses import dataclass, field

from server.theory.pitch import Key, NUMERALS, LETTERS, LETTER_PC, signed

QUALITY_INTERVALS: dict[str, list[int]] = {
    "maj": [0, 4, 7],
    "min": [0, 3, 7],
    "dim": [0, 3, 6],
    "aug": [0, 4, 8],
    "dom7": [0, 4, 7, 10],
    "maj7": [0, 4, 7, 11],
    "min7": [0, 3, 7, 10],
    "hdim7": [0, 3, 6, 10],
    "dim7": [0, 3, 6, 9],
    "sus4": [0, 5, 7],
    "sus2": [0, 2, 7],
    "7sus4": [0, 5, 7, 10],
    # added-note chords: a triad plus one tone that is not a seventh
    "add9": [0, 4, 7, 2],
    "madd9": [0, 3, 7, 2],
    "add11": [0, 4, 7, 5],
    "madd11": [0, 3, 7, 5],
    "add13": [0, 4, 7, 9],
    "madd13": [0, 3, 7, 9],
    "maj6": [0, 4, 7, 9],
    "min6": [0, 3, 7, 9]
}

# Added-note qualities: the added tone's name (its role in the chord), and the
# triad underneath.
ADDED_TONE: dict[str, str] = {
        "add9": "9", "madd9": "9", "add11": "11", "madd11": "11",
        "add13": "13", "madd13": "13", "maj6": "6", "min6": "6"
}
ADDED_BASE: dict[str, str] = {
        "add9": "maj", "madd9": "min", "add11": "maj", "madd11": "min",
        "add13": "maj", "madd13": "min", "maj6": "maj", "min6": "min"
}

# Scale steps of each chord tone above the root (stacked thirds unless suspended).
QUALITY_STEPS: dict[str, list[int]] = {
    "sus4": [0, 3, 4], "sus2": [0, 1, 4], "7sus4": [0, 3, 4, 6],
    "add9": [0, 2, 4, 1], "madd9": [0, 2, 4, 1], "add11": [0, 2, 4, 3], "madd11": [0, 2, 4, 3],
    "add13": [0, 2, 4, 5], "madd13": [0, 2, 4, 5], "maj6": [0, 2, 4, 5], "min6": [0, 2, 4, 5]
}

ROLES: list[str] = ["R", "3", "5", "7"]
TRIAD_FIGURES: list[str] = ["", "⁶", "⁶₄"]
SEVENTH_FIGURES: list[str] = ["⁷", "⁶₅", "⁴₃", "⁴₂"]
SYMBOL_SUFFIX: dict[str, str] = {
        "maj": "", "min": "m", "dim": "°", "aug": "+", "dom7": "7",
        "maj7": "maj7", "min7": "m7", "hdim7": "ø7", "dim7": "°7",
        "sus4": "sus4", "sus2": "sus2", "7sus4": "7sus4",
        "add9": "add9", "madd9": "madd9", "add11": "add11", "madd11": "madd11",
        "add13": "add13", "madd13": "madd13", "maj6": "6", "min6": "m6"
}


@dataclass(eq=False)
class Tone:
    pc: int
    deg: int
    role: str


@dataclass(eq=False)
class Tendency:
    """A tone that must move a particular way when its resolution is available.

    kind: LT (leading tone, up by step), 7th (chordal seventh, down by step),
    aug6 (the two tones of the augmented sixth expand outward by half step),
    cad6 / cad4 (the 6th and 4th above the bass of a cadential 6/4 fall by step),
    N (♭2̂ of the Neapolitan falls), d5 (diminished fifth falls), b6 (borrowed ♭6̂ falls)
    """
    pc: int
    kind: str
    targets: list[int]
    label: str


@dataclass(eq=False)
class ChordSpec:
    cid: str
    rn: str
    harmony: str
    symbol: str
    func: str
    family: str
    quality: str
    root_pc: int
    root_deg: int
    tones: list[Tone]
    bass_role: str
    inversion: int
    is_seventh: bool
    target_pc: int = -1
    target_deg: int = -1
    tendencies: list[Tendency] = field(default_factory=list)
    # the other Roman numeral these pitches go by ("VI" for V/ii), and why each
    # applies ("" if none)
    alt_rn: str = ""
    also: str = ""

    def pcs(self) -> list[int]:
        return [t.pc for t in self.tones]

    def bass_pc(self) -> int:
        for t in self.tones:
            if t.role == self.bass_role:
                return t.pc
        return self.root_pc

    def tone_for_pc(self, pc: int) -> Tone | None:
        for t in self.tones:
            if t.pc == pc % 12:
                return t
        return None


def is_minor_quality(quality: str) -> bool:
    return quality in ["min", "dim", "min7", "hdim7", "dim7"] or ADDED_BASE.get(quality, "") == "min"


def quality_mark(quality: str) -> str:
    return {"dim": "°", "hdim7": "ø", "dim7": "°", "aug": "⁺"}.get(quality, "")


def tertian(
    key: Key,
    root_deg: int,
    root_pc: int,
    quality: str,
    inversion: int,
    func: str,
    family: str,
    target_deg: int = -1,
    target_rn: str = "",
    numeral_override: str = ""
) -> ChordSpec:
    """Build a tertian chord (stacked thirds on `root_deg`) and everything derived from it."""
    intervals = QUALITY_INTERVALS[quality]
    added = ADDED_TONE.get(quality, "")
    is_seventh = len(intervals) == 4 and not added
    tones: list[Tone] = []
    steps = QUALITY_STEPS.get(quality, [0, 2, 4, 6])
    for (i, iv) in enumerate(intervals):
        tones.append(Tone(pc=(root_pc + iv) % 12, deg=(root_deg + steps[i]) % 7, role=added if i == 3 and added else ROLES[i]))
    bass_role = tones[inversion].role

    # Roman numeral: case from quality, accidental from chromatic root, figures from inversion.
    # `plain` is the numeral read in the home key; an applied chord overrides it (V/ii).
    plain = NUMERALS[root_deg % 7]
    if is_minor_quality(quality):
        plain = plain.lower()
    diff = signed(root_pc - key.degree_pc(root_deg))
    # the raised leading tone in minor is conventionally unmarked (vii°7, not ♯vii°7)
    if not (key.mode == "minor" and root_deg % 7 == 6 and diff == 1):
        plain = {-1: "♭", 1: "♯"}.get(diff, "") + plain
    numeral: str = numeral_override
    if not numeral:
        numeral = plain if target_deg < 0 else (plain.lstrip("♭♯"))
    figures = SEVENTH_FIGURES if is_seventh else TRIAD_FIGURES
    mark = quality_mark(quality)
    rn: str = numeral + mark + figures[inversion]
    harmony: str = numeral + mark + ("⁷" if is_seventh else "")
    if added:
        # the added tone is named after the figure: IV(add6), I⁶(add9)
        rn = numeral + figures[inversion] + "(add" + added + ")"
        harmony = numeral + "(add" + added + ")"
    elif quality in QUALITY_STEPS:
        sus = "sus2" if quality == "sus2" else "sus4"
        rn = numeral + figures[inversion] + sus
        harmony = numeral + ("⁷" if is_seventh else "") + sus
    # the same chord named with another numeral in place of this one (same figures)
    def renamed(other: str) -> str:
        return other + rn[len(numeral):].split("/")[0]
    alt_rn = ""
    also = ""
    if target_deg >= 0 and numeral != plain:
        alt_rn = renamed(plain)
        also = f"{rn}/{target_rn} names what it does (leads to {target_rn}); {alt_rn} names what it is in the key"
    elif target_deg < 0 and func not in ["CAD64", "AUG6", "NEAP"] and (quality in ["maj", "dom7"] or ADDED_BASE.get(quality, "") == "maj"):
        # a major or dominant-7th chord a 5th above a diatonic degree, not itself
        # diatonic, could also be that degree's applied dominant
        tdeg = (root_deg + 3) % 7
        scale = [key.degree_pc(d) for d in range(7)]
        core = [t.pc for t in tones[:3]]
        if tdeg != 0 and (root_pc + 5) % 12 == key.degree_pc(tdeg) and not all([p in scale for p in core]):
            trn = diatonic_numeral(key, tdeg)
            alt_rn = renamed("V") + "/" + trn
            also = f"{rn} names what it is in the key; {alt_rn} names what it does when it leads to {trn}"
    target_pc = -1
    if target_deg >= 0:
        rn = rn + "/" + target_rn
        harmony = harmony + "/" + target_rn
        target_pc = key.degree_pc(target_deg)
        if target_rn == "V" and key.mode == "minor":
            target_pc = key.rel(7)
    if func == "CAD64":
        harmony = "cad⁶₄"

    root_name = key.spell(root_deg, root_pc)
    symbol = root_name + SYMBOL_SUFFIX[quality]
    if inversion > 0:
        bt = tones[inversion]
        symbol = symbol + "/" + key.spell(bt.deg, bt.pc)

    spec = ChordSpec(
        cid=rn,
        rn=rn,
        harmony=harmony,
        symbol=symbol,
        func=func,
        family=family,
        quality=quality,
        root_pc=root_pc,
        root_deg=root_deg % 7,
        tones=tones,
        bass_role=bass_role,
        inversion=inversion,
        is_seventh=is_seventh,
        target_pc=target_pc,
        target_deg=target_deg,
        alt_rn=alt_rn,
        also=also
    )
    spec.tendencies = tendencies_for(key, spec, target_rn)
    return spec


def tendencies_for(key: Key, c: ChordSpec, target_rn: str) -> list[Tendency]:
    """Tendency tones: which pitches of this chord are obliged to move, and where."""
    out: list[Tendency] = []
    pcs = c.pcs()
    lt = key.rel(11)
    if c.func == "D" and lt in pcs:
        out.append(Tendency(pc=lt, kind="LT", targets=[key.tonic_pc], label="leading tone"))
    if c.func == "APP" and c.target_pc >= 0:
        local_lt = (c.target_pc + 11) % 12
        if local_lt in pcs:
            out.append(
                Tendency(pc=local_lt, kind="LT", targets=[c.target_pc], label=f"leading tone of {target_rn}")
            )
    if c.is_seventh:
        sev = c.tones[3].pc
        out.append(
            Tendency(pc=sev, kind="7th", targets=[(sev + 11) % 12, (sev + 10) % 12], label="chordal 7th")
        )
    if c.quality in ["dim", "hdim7", "dim7"] and c.func in ["D", "APP"]:
        fifth = c.tones[2].pc
        out.append(
            Tendency(pc=fifth, kind="d5", targets=[(fifth + 11) % 12, (fifth + 10) % 12], label="diminished 5th")
        )
    if c.func == "CAD64":
        third = c.tones[1].pc
        out.append(Tendency(pc=third, kind="cad6", targets=[key.rel(2)], label="6th above the bass"))
        out.append(Tendency(pc=key.tonic_pc, kind="cad4", targets=[lt], label="4th above the bass"))
    if c.func == "NEAP":
        out.append(Tendency(pc=key.rel(1), kind="N", targets=[lt, key.tonic_pc], label="♭2̂"))
    if c.family == "mixture" and key.mode == "major" and key.rel(8) in pcs and c.func == "PD":
        out.append(Tendency(pc=key.rel(8), kind="b6", targets=[key.rel(7)], label="borrowed ♭6̂"))
    return out


def aug_sixth(key: Key, kind: str) -> ChordSpec:
    """Italian, French and German augmented sixths, spelled on ♭6̂ / 1̂ / (2̂ | ♭3̂) / ♯4̂."""
    tones = [
        Tone(pc=key.rel(8), deg=5, role="b6"),
        Tone(pc=key.tonic_pc, deg=0, role="1"),
        Tone(pc=key.rel(6), deg=3, role="#4")
    ]
    if kind == "Fr":
        tones.insert(2, Tone(pc=key.rel(2), deg=1, role="2"))
    elif kind == "Ger":
        tones.insert(2, Tone(pc=key.rel(3), deg=2, role="b3"))
    rn = kind + "⁺⁶"
    spelled = "–".join([key.spell(t.deg, t.pc) for t in tones])
    return ChordSpec(
        cid=rn,
        rn=rn,
        harmony=rn,
        symbol=f"{kind}+6 ({spelled})",
        func="AUG6",
        family="chromatic",
        quality=kind,
        root_pc=key.rel(8),
        root_deg=5,
        tones=tones,
        bass_role="b6",
        inversion=0,
        is_seventh=False,
        tendencies=[
            Tendency(pc=key.rel(8), kind="aug6", targets=[key.rel(7)], label="♭6̂ of the augmented sixth"),
            Tendency(pc=key.rel(6), kind="aug6", targets=[key.rel(7)], label="♯4̂ of the augmented sixth")
        ]
    )


def build_vocab(key: Key, extended: bool = False) -> list[ChordSpec]:
    v: list[ChordSpec] = []
    major = key.mode == "major"
    applied_targets: list[tuple[int, str]] = []

    if major:
        # (degree, quality, function, inversions)
        for (deg, q, f, invs) in [
            (0, "maj", "T", [0, 1]),
            (1, "min", "PD", [0, 1]),
            (2, "min", "TX", [0]),
            (3, "maj", "PD", [0, 1]),
            (4, "maj", "D", [0, 1]),
            (5, "min", "TX", [0]),
            (6, "dim", "D", [1]),
            (1, "min7", "PD", [0, 1]),
            (4, "dom7", "D", [0, 1, 2, 3]),
            (6, "hdim7", "D", [0, 1, 2])
        ]:
            for inv in invs:
                v.append(tertian(key, deg, key.degree_pc(deg), q, inv, f, "diatonic"))
        # modal mixture from the parallel minor
        for (deg, semis, q, f, invs) in [
            (3, 5, "min", "PD", [0, 1]),
            (5, 8, "maj", "PD", [0]),
            (6, 10, "maj", "MIX", [0]),
            (2, 3, "maj", "MIX", [0]),
            (1, 2, "hdim7", "PD", [1])
        ]:
            for inv in invs:
                v.append(tertian(key, deg, key.rel(semis), q, inv, f, "mixture"))
        applied_targets = [(1, "ii"), (2, "iii"), (3, "IV"), (4, "V"), (5, "vi")]
    else:
        for (deg, semis, q, f, invs) in [
            (0, 0, "min", "T", [0, 1]),
            (1, 2, "dim", "PD", [1]),
            (2, 3, "maj", "TX", [0]),
            (3, 5, "min", "PD", [0, 1]),
            (4, 7, "maj", "D", [0, 1]),
            (5, 8, "maj", "TX", [0]),
            (6, 10, "maj", "MIX", [0]),
            (6, 11, "dim", "D", [1]),
            (1, 2, "hdim7", "PD", [0, 1]),
            (4, 7, "dom7", "D", [0, 1, 2, 3]),
            (6, 11, "dim7", "D", [0, 1, 2])
        ]:
            for inv in invs:
                v.append(tertian(key, deg, key.rel(semis), q, inv, f, "diatonic"))
        # Picardy third
        v.append(tertian(key, 0, key.tonic_pc, "maj", 0, "T", "mixture"))
        applied_targets = [(2, "III"), (3, "iv"), (4, "V"), (5, "VI")]

    # Applied dominants V/x and V7/x, plus vii°7/V.
    for (tdeg, trn) in applied_targets:
        tpc = key.rel(7) if trn == "V" else key.degree_pc(tdeg)
        app_root = (tpc + 7) % 12
        root_deg = (tdeg + 4) % 7
        # V/IV (major) and V/iv, V/III, V/VI (minor) duplicate an existing triad, so only their V7 is used.
        if trn not in ["IV", "iv", "III", "VI"]:
            for inv in [0, 1]:
                v.append(tertian(key, root_deg, app_root, "maj", inv, "APP", "applied", tdeg, trn, "V"))
        for inv in [0, 1, 3]:
            v.append(tertian(key, root_deg, app_root, "dom7", inv, "APP", "applied", tdeg, trn, "V"))
    for inv in [0, 1]:
        v.append(tertian(key, 3, key.rel(6), "dim7", inv, "APP", "applied", 4, "V", "vii"))

    # Chromatic pre-dominants.
    v.append(tertian(key, 1, key.rel(1), "maj", 1, "NEAP", "chromatic", -1, "", "N"))
    for kind in ["It", "Ger", "Fr"]:
        v.append(aug_sixth(key, kind))

    # Cadential 6/4: tonic pitches over 5̂, functioning as an embellished V.
    v.append(tertian(key, 0, key.tonic_pc, "maj" if major else "min", 2, "CAD64", "cadential"))

    # Added-note chords: each diatonic major or minor triad with an added 9th, 11th
    # or 6th, where that tone is in the scale too; same function as the triad. (An
    # added 13th is the same pitches as an added 6th, so only the 6th is offered.)
    if extended:
        scale = [key.degree_pc(d) for d in range(7)]
        triads = [c for c in v if c.family == "diatonic" and c.quality in ["maj", "min"] and c.inversion == 0]
        for t in triads:
            minor = t.quality == "min"
            for q in (["madd9", "madd11", "min6"] if minor else ["add9", "add11", "maj6"]):
                if (t.root_pc + QUALITY_INTERVALS[q][3]) % 12 not in scale:
                    continue
                for inv in [0, 1]:
                    v.append(tertian(key, t.root_deg, t.root_pc, q, inv, t.func, "diatonic"))
    return v


_VOCAB: dict[str, list[ChordSpec]] = {}
_BY_CID: dict[str, dict[str, ChordSpec]] = {}
_KEYS: dict[str, Key] = {}


def key_id_of(tonic: str, mode: str, extended: bool = False) -> str:
    """A key's id: "C|major", or "C|major|ext" for its vocabulary with the added-note
    chords (add9, add11, 6) that the walkers may then use."""
    return tonic + "|" + mode + ("|ext" if extended else "")


def is_extended(key_id: str) -> bool:
    return key_id.endswith("|ext")


def key_for(key_id: str) -> Key:
    if key_id not in _KEYS:
        parts = key_id.split("|")
        _KEYS[key_id] = Key.parse(parts[0], parts[1])
    return _KEYS[key_id]


def vocab_for(key_id: str) -> list[ChordSpec]:
    if key_id not in _VOCAB:
        chords = build_vocab(key_for(key_id), is_extended(key_id))
        _VOCAB[key_id] = chords
        _BY_CID[key_id] = {c.cid: c for c in chords}
    return _VOCAB[key_id]


def has_spec(key_id: str, cid: str) -> bool:
    """Is `cid` a chord this key knows? (The saved harmony graph may still hold edges to
    chords since removed from the vocabulary, e.g. the French and German sixths.)"""
    vocab_for(key_id)
    return cid in _BY_CID[key_id]


def spec_of(key_id: str, cid: str) -> ChordSpec:
    vocab_for(key_id)
    return _BY_CID[key_id][cid]


def tonic_cid(key_id: str) -> str:
    """The tonic chord's cid for a key (I or i)."""
    return "I" if key_for(key_id).mode == "major" else "i"


# ---------------------------------------------------------------- typed chords

# Chord-symbol suffixes, longest first so "m7" wins over "m" and "maj7" over "maj".
SUFFIX_QUALITY: list[tuple[str, str]] = [
    ("7sus4", "7sus4"), ("7sus", "7sus4"), ("sus4", "sus4"), ("sus2", "sus2"), ("sus", "sus4"),
    ("min6", "m6"), ("m6", "m6"), ("6", "6"),
    ("add9", "add9"), ("add2", "add9"), ("madd9", "madd9"), ("m(add9)", "madd9"), ("madd2", "madd9"),
    ("add11", "add11"), ("add4", "add11"), ("madd11", "madd11"), ("m(add11)", "madd11"), ("madd4", "madd11"),
    ("add13", "add13"), ("add6", "maj6"), ("madd13", "madd13"), ("m(add13)", "madd13"), ("madd6", "min6"),
    ("(add9)", "add9"), ("(add11)", "add11"), ("(add13)", "add13"),
    ("m7b5", "hdim7"), ("m7♭5", "hdim7"), ("maj7", "maj7"), ("dim7", "dim7"), ("min7", "min7"),
    ("ø7", "hdim7"), ("°7", "dim7"), ("o7", "dim7"), ("Δ7", "maj7"), ("M7", "maj7"),
    ("m7", "min7"), ("-7", "min7"), ("dim", "dim"), ("aug", "aug"), ("min", "min"),
    ("maj", "maj"), ("mi", "min"), ("ø", "hdim7"), ("°", "dim"), ("o", "dim"), ("+", "aug"),
    ("7", "dom7"), ("m", "min"), ("-", "min"), ("M", "maj"), ("", "maj")
]

# Function of a diatonic chord by scale degree, when the vocabulary has no template.
DEGREE_FUNC: list[str] = ["T", "PD", "TX", "PD", "D", "TX", "D"]


def read_note(text: str) -> tuple[int, int, int]:
    """Read a note name ("E♭", "Bb", "F#") as (letter index, pitch class, characters used)."""
    if not text or text[0].upper() not in LETTERS:
        raise ValueError(f"'{text}' does not start with a note name A–G")
    letter = LETTERS.index(text[0].upper())
    pc = LETTER_PC[letter]
    used = 1
    while used < len(text) and text[used] in ["b", "♭", "#", "♯"]:
        pc += -1 if text[used] in ["b", "♭"] else 1
        used += 1
    return (letter, pc % 12, used)


def diatonic_numeral(key: Key, deg: int) -> str:
    """Numeral (case from quality) of the diatonic triad on `deg`, e.g. "ii" or "V"."""
    third = (key.degree_pc(deg + 2) - key.degree_pc(deg)) % 12
    return NUMERALS[deg % 7].lower() if third == 3 else NUMERALS[deg % 7]


def parse_chord(key_id: str, symbol: str) -> ChordSpec:
    """Interpret a typed chord in a key: a chord symbol ("Gm/D", "Cm/E♭", "B♭maj7") or a
    Roman numeral ("ii65", "V7/V", "bVI", "viio7", "N6", "Ger6", "Cad64").

    A symbol's root letter fixes its scale degree, so spelling is respected (D♯ and E♭
    are different chords to a theorist). The chord is then analysed against the key:
    1. if the vocabulary already has exactly this chord, that chord is used;
    2. if it has the same harmony in another inversion, its analysis is reused
    (Gm/D in F major is ii in second inversion: ii⁶₄, a pre-dominant);
    3. otherwise it is analysed from scratch: an applied dominant if it is a major
    or dominant-7th chord a fifth above a diatonic degree (D/A in F major → V⁶₄/ii),
    else diatonic by degree, else a borrowed colour chord (Cm/E♭ in F major → v⁶).
    A Roman numeral already is an analysis, so it is taken at its word.
    Raises ValueError with a readable message when the input cannot be read.
    """
    text = symbol.strip()
    if is_roman(text):
        return parse_roman(key_id, text)
    key = key_for(key_id)
    (letter, root_pc, used) = read_note(text)
    rest = text[used:]
    bass_text = ""
    if "/" in rest:
        parts = rest.split("/")
        rest = parts[0]
        bass_text = parts[1]
    quality = ""
    for (suffix, q) in SUFFIX_QUALITY:
        if rest == suffix:
            quality = q
            break
    if not quality:
        raise ValueError(f"unknown chord quality “{rest}” in {text} (try m, 7, maj7, m7, dim, ø7, °7, aug)")
    # With the added-note chords switched on, C6 and Cm6 are added-sixth chords.
    if quality in ["6", "m6"] and is_extended(key_id):
        quality = "maj6" if quality == "6" else "min6"
    # Otherwise C6 is Am7 over C and Cm6 is Aø7 over C: spell them from the sixth,
    # with the written root in the bass unless another bass is given.
    named_root = ""
    if quality in ["6", "m6"]:
        named_root = text[:used] + ("6" if quality == "6" else "m6")
        if not bass_text:
            bass_text = text[:used]
        letter = (letter + 5) % 7
        root_pc = (root_pc + 9) % 12
        quality = "min7" if quality == "6" else "hdim7"
    intervals = QUALITY_INTERVALS[quality]
    pcs = [(root_pc + iv) % 12 for iv in intervals]
    inversion = 0
    if bass_text:
        (_, bass_pc, bused) = read_note(bass_text)
        if bused != len(bass_text) or bass_pc not in pcs:
            raise ValueError(f"'{bass_text}' is not a note of {text.split('/')[0]}")
        inversion = pcs.index(bass_pc)
        if inversion == 3 and quality in ADDED_TONE:
            raise ValueError(f"put the added {ADDED_TONE[quality]} of {text.split('/')[0]} above the bass (with it in the bass it is a different chord)")
    made = analyse(key_id, (letter - key.tonic_letter) % 7, root_pc, quality, inversion, bool(named_root))
    if named_root:
        sym = named_root.replace("b", "♭").replace("#", "♯")
        made.symbol = sym + (made.symbol[made.symbol.index("/"):] if "/" in made.symbol and inversion != 1 else "")
    return register_chord(key_id, made)


def analyse(key_id: str, root_deg: int, root_pc: int, quality: str, inversion: int, fresh: bool = False) -> ChordSpec:
    """Analyse a chord given by root degree, root pitch class, quality and inversion
    (steps 1-3 of `parse_chord`). `fresh` skips reusing a vocabulary chord outright."""
    key = key_for(key_id)
    pcs = [(root_pc + iv) % 12 for iv in QUALITY_INTERVALS[quality]]

    # 1. the vocabulary already has it
    same_harmony: list[ChordSpec] = []
    for c in vocab_for(key_id):
        if c.root_pc != root_pc or c.root_deg != root_deg or c.quality != quality or c.func in ["CAD64", "AUG6"]:
            continue
        if c.inversion == inversion and not fresh:
            return c
        same_harmony.append(c)

    # 2. the same harmony in another inversion: reuse its analysis
    if same_harmony:
        t = same_harmony[0]
        target_rn = t.rn.split("/")[1] if "/" in t.rn else ""
        override = ""
        if t.func == "APP":
            override = "V" if quality in ["maj", "dom7"] else "vii"
        elif t.func == "NEAP":
            override = "N"
        return tertian(key, root_deg, root_pc, quality, inversion, t.func, t.family, t.target_deg, target_rn, override)

    # 3. analyse from scratch
    diatonic = all([p in [key.degree_pc(d) for d in range(7)] for p in pcs])
    tdeg = (root_deg + 3) % 7
    # an added-note major chord is read by its triad: C(add9) in E♭ is V(add9)/ii
    major_core = quality in ["maj", "dom7"] or ADDED_BASE.get(quality, "") == "maj"
    core_diatonic = all([p in [key.degree_pc(d) for d in range(7)] for p in pcs[:3]])
    if major_core and not core_diatonic and tdeg != 0 and (root_pc + 5) % 12 == key.degree_pc(tdeg):
        return tertian(key, root_deg, root_pc, quality, inversion, "APP", "applied", tdeg, diatonic_numeral(key, tdeg), "V")
    if diatonic:
        return tertian(key, root_deg, root_pc, quality, inversion, DEGREE_FUNC[root_deg], "diatonic")
    func = "PD" if root_deg in [1, 3] else "MIX"
    if root_deg == 0:
        func = "T"   # a borrowed tonic (i in major) is still the tonic
    elif root_deg == 6 and root_pc == key.rel(11) and quality in ["dim", "hdim7", "dim7"]:
        func = "D"
    elif root_deg == 4 and root_pc == key.rel(7) and quality == "aug":
        func = "D"
    return tertian(key, root_deg, root_pc, quality, inversion, func, "mixture")


# ---------------------------------------------------------------- Roman numerals

ROMAN: list[str] = ["VII", "III", "VI", "IV", "II", "V", "I"]   # longest first
FIGURE_DIGITS: dict[str, str] = {
        "⁶": "6", "⁴": "4", "₄": "4", "⁵": "5", "₅": "5", "³": "3", "₃": "3", "²": "2", "₂": "2", "⁷": "7"
}
TRIAD_INVERSION: dict[str, int] = {"": 0, "5": 0, "53": 0, "6": 1, "63": 1, "64": 2}
SEVENTH_INVERSION: dict[str, int] = {"7": 0, "65": 1, "43": 2, "42": 3, "2": 3}
AUG_SIXTHS: list[str] = ["It", "Fr", "Ger"]


def roman_head(text: str) -> str:
    """Strip the accidental and return the numeral at the front of `text` ("" if none)."""
    t = text
    while t and t[0] in ["b", "♭", "#", "♯"]:
        t = t[1:]
    for r in ROMAN:
        if t.upper().startswith(r):
            head = t[:len(r)]
            # case must be consistent: "IV" or "iv", never "Iv"
            if head == r or head == r.lower():
                return head
    return ""


def is_roman(text: str) -> bool:
    """Roman numeral (or N / It / Fr / Ger / Cad64) rather than a chord symbol?
    Chord roots are A-G, so a leading I/V/i/v (after an accidental) can't be a symbol;
    a lone "b"/"#" before a numeral is an accidental, not the note B."""
    # "Cad64" is the cadential 6/4; "Cadd9" is a C chord with an added 9th
    if text.startswith("N") or (text.lower().startswith("cad") and not text.lower().startswith("cadd")):
        return True
    for a6 in AUG_SIXTHS:
        if text.startswith(a6):
            return True
    return roman_head(text) != ""


def normalise_figures(text: str) -> str:
    out = ""
    for ch in text:
        out += FIGURE_DIGITS.get(ch, ch)
    return out


def seventh_quality(upper: bool, seventh: int) -> str:
    """Seventh chord for a bare "7": a major seventh above a major triad is maj7 (IV7 in
    major), otherwise dom7 over a major triad and min7 over a minor one."""
    if upper:
        return "maj7" if seventh == 11 else "dom7"
    return "min7"


def parse_roman(key_id: str, raw: str) -> ChordSpec:
    """A Roman numeral in this key, taken as the analysis it names."""
    key = key_for(key_id)
    text = normalise_figures(raw.replace(" ", ""))
    major = key.mode == "major"

    # specials: Neapolitan, augmented sixths, cadential 6/4 - all in the vocabulary
    low = text.lower()
    if low.startswith("cad") and not low.startswith("cadd"):
        for c in vocab_for(key_id):
            if c.func == "CAD64":
                return c
    if text.startswith("N") and text[1:] in ["", "6"]:
        return spec_of(key_id, "N⁶")
    for a6 in AUG_SIXTHS:
        if text.startswith(a6) and text[len(a6):] in ["", "6", "+6", "⁺6"]:
            return spec_of(key_id, a6 + "⁺⁶")

    # applied: "V7/V", "viio7/ii"
    target = ""
    if "/" in text:
        parts = text.split("/")
        text = parts[0]
        target = parts[1]

    # accidental, numeral, quality marks, figures
    shift = 0
    while text and text[0] in ["b", "♭", "#", "♯"]:
        shift += -1 if text[0] in ["b", "♭"] else 1
        text = text[1:]
    head = roman_head(text)
    if not head:
        raise ValueError(f"can't read the numeral in {raw}")
    upper = head.isupper()
    root_deg = ["I", "II", "III", "IV", "V", "VI", "VII"].index(head.upper())
    rest = text[len(head):]
    mark = ""
    if rest.startswith("maj") or rest.startswith("M"):
        mark = "maj"
        rest = rest[3:] if rest.startswith("maj") else rest[1:]
    elif rest and rest[0] in ["o", "°", "ø", "+", "⁺"]:
        mark = {"o": "dim", "°": "dim", "ø": "hdim", "+": "aug", "⁺": "aug"}[rest[0]]
        rest = rest[1:]
    figures = rest
    if figures in SEVENTH_INVERSION or mark == "maj":
        inversion = SEVENTH_INVERSION.get(figures if figures else "7", -1)
        if inversion < 0:
            raise ValueError(f"unknown figures “{figures}” in {raw} (sevenths: 7, 65, 43, 42)")
        if mark == "dim":
            quality = "dim7"
        elif mark == "hdim":
            quality = "hdim7"
        elif mark == "maj":
            quality = "maj7"
        else:
            quality = "diatonic7"   # the scale's own seventh: resolved once the root is known
    elif figures in TRIAD_INVERSION:
        inversion = TRIAD_INVERSION[figures]
        if mark == "hdim":
            raise ValueError(f"ø needs a seventh: try {head}ø7 or {head}ø65")
        quality = {"dim": "dim", "aug": "aug"}.get(mark, "maj" if upper else "min")
    else:
        raise ValueError(f"unknown figures “{figures}” in {raw} (triads: 6, 64; sevenths: 7, 65, 43, 42)")

    if target:
        # applied chord: the numeral is read in the key of the target
        thead = roman_head(target)
        if not thead or target != thead:
            raise ValueError(f"can't read the target {target} in {raw} (e.g. V/V, viio7/ii)")
        tdeg = ["I", "II", "III", "IV", "V", "VI", "VII"].index(thead.upper())
        tpc = key.rel(7) if thead == "V" else key.degree_pc(tdeg)
        local = [0, 2, 4, 5, 7, 9, 11] if thead.isupper() else [0, 2, 3, 5, 7, 8, 10]
        root_pc = (tpc + local[root_deg] + shift) % 12
        if root_deg == 6 and quality in ["dim", "hdim7", "dim7"]:
            root_pc = (tpc + 11 + shift) % 12   # leading-tone chords use the raised 7th
        if quality == "diatonic7":
            quality = seventh_quality(upper, (tpc + local[(root_deg + 6) % 7] - root_pc) % 12 if shift == 0 else 10)
        made = tertian(
            key, (tdeg + root_deg) % 7, root_pc, quality, inversion, "APP", "applied", tdeg, thead,
            "V" if root_deg == 4 else ("vii" if root_deg == 6 else NUMERALS[root_deg] if upper else NUMERALS[root_deg].lower())
        )
        return register_chord(key_id, made)

    root_pc = (key.degree_pc(root_deg) + shift) % 12
    if quality == "diatonic7":
        quality = seventh_quality(upper, (key.degree_pc(root_deg + 6) - root_pc) % 12 if shift == 0 else 10)
    # in minor, vii° and V name the raised leading tone, as in the vocabulary
    if not major and root_deg == 6 and shift == 0 and quality in ["dim", "hdim7", "dim7"]:
        root_pc = key.rel(11)
    return register_chord(key_id, analyse(key_id, root_deg, root_pc, quality, inversion))


def alias_chord(key_id: str, cid: str, c: ChordSpec):
    """Let `spec_of` find chord `c` by an older id as well (a stored step whose chord
    has since been re-analysed under a new name)."""
    vocab_for(key_id)
    _BY_CID[key_id][cid] = c


def register_chord(key_id: str, c: ChordSpec) -> ChordSpec:
    """Make a typed chord known to `spec_of` (the search vocabulary itself is unchanged,
    so walkers can start from or arrive at it, but never invent it). Returns the spec to
    use: an existing one when it is the same chord with the same analysis."""
    vocab_for(key_id)
    known = _BY_CID[key_id]
    while c.cid in known:
        k = known[c.cid]
        if k.pcs() == c.pcs() and k.bass_role == c.bass_role and k.func == c.func and k.target_pc == c.target_pc:
            return k
        c.cid = c.cid + "*"
    known[c.cid] = c
    return c
