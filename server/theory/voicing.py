"""Four-part (SATB) voicing and voice-leading evaluation.

`voicings_for` enumerates every legal SATB voicing of a chord (ranges, spacing,
no crossing, complete or properly-incomplete, no doubled tendency tones) and
gives each a static cost (doubling preferences, register).

`best_motion` finds the best next voicing given where the voices are *now*,
scoring the transition against common-practice voice-leading rules:
parallel/contrary perfect intervals, direct fifths/octaves in the outer voices,
overlap, awkward melodic intervals (tritones, augmented seconds, big leaps),
and resolution of every tendency tone (leading tones, chordal sevenths,
augmented sixths, cadential 6/4 suspensions, ♭2̂, ♭6̂).
"""


from __future__ import annotations
from dataclasses import dataclass

from server.theory.pitch import Key, iabs
from server.theory.vocab import ChordSpec, key_for, ADDED_TONE

# Bass, tenor, alto, soprano ranges (MIDI, inclusive).
RANGES: list[list[int]] = [[40, 62], [48, 67], [55, 74], [60, 79]]
VOICE_NAMES: list[str] = ["bass", "tenor", "alto", "soprano"]
VOICE_ABBR: list[str] = ["B", "T", "A", "S"]

# How many cheapest-by-motion voicings get the full rule evaluation.
SHORTLIST: int = 18


@dataclass(eq=False)
class Voicing:
    notes: list[int]
    static_cost: float
    doubling: str


@dataclass(eq=False)
class Mark:
    """One rule outcome shown to the user. kind: good | warn | bad."""
    kind: str
    text: str
    cost: float = 0.0


@dataclass(eq=False)
class Motion:
    notes: list[int]
    cost: float
    marks: list[Mark]


_VOICINGS: dict[str, list[Voicing]] = {}


def note_name(key: Key, c: ChordSpec, midi: int) -> str:
    t = c.tone_for_pc(midi % 12)
    deg = t.deg if t is not None else 0
    return key.spell_midi(deg, midi)


def voicing_names(key_id: str, c: ChordSpec, notes: list[int]) -> list[str]:
    key = key_for(key_id)
    return [note_name(key, c, n) for n in notes]


def undoublable(c: ChordSpec) -> list[int]:
    """Tendency-tone pitch classes that must never be doubled in this chord."""
    return [t.pc for t in c.tendencies if t.kind in ["LT", "7th", "aug6", "N"]]


def doubling_cost(c: ChordSpec, role: str) -> float:
    if c.func == "CAD64":
        return 0.0 if role == c.bass_role else 1.5
    if c.func == "NEAP":
        return {"3": 0.0, "5": 0.9, "R": 2.0}.get(role, 1.0)
    if c.quality == "dim":
        return {"3": 0.0, "5": 1.2, "R": 3.0}.get(role, 1.0)
    if c.func == "AUG6":
        return 0.0 if role == "1" else 3.0
    if c.inversion == 0:
        return {"R": 0.0, "5": 0.7, "3": 1.4}.get(role, 1.0)
    return {"R": 0.0, "5": 0.2, "3": 0.9}.get(role, 1.0)


def voicings_for(key_id: str, c: ChordSpec) -> list[Voicing]:
    ck = key_id + "#" + c.cid
    if ck in _VOICINGS:
        return _VOICINGS[ck]
    pcs = c.pcs()
    pcset = set(pcs)
    role_of = {t.pc: t.role for t in c.tones}
    bass_pc = c.bass_pc()
    no_double = undoublable(c)
    fifth_pc = -1
    if c.inversion == 0 and c.quality in ["maj", "min", "dom7", "min7", "maj7"] + list(ADDED_TONE.keys()):
        fifth_pc = c.tones[2].pc
    out: list[Voicing] = []
    for b in range(RANGES[0][0], RANGES[0][1] + 1):
        if b % 12 != bass_pc:
            continue
        for t in range(max(RANGES[1][0], b + 1), RANGES[1][1] + 1):
            if t % 12 not in pcset or t - b > 19:
                continue
            for a in range(max(RANGES[2][0], t + 1), RANGES[2][1] + 1):
                if a % 12 not in pcset or a - t > 12:
                    continue
                for s in range(max(RANGES[3][0], a + 1), RANGES[3][1] + 1):
                    if s % 12 not in pcset or s - a > 12:
                        continue
                    notes = [b, t, a, s]
                    counts: dict[int, int] = {}
                    for n in notes:
                        counts[n % 12] = counts.get(n % 12, 0) + 1
                    missing = [p for p in pcs if p not in counts]
                    cost = 0.0
                    doubling = ""
                    if missing:
                        # only the fifth of a root-position chord may be left out
                        if missing != [fifth_pc]:
                            continue
                        cost += 0.9
                        doubling = "omits 5th"
                    bad = False
                    for (p, n) in counts.items():
                        if n > 1:
                            if p in no_double:
                                bad = True
                            cost += doubling_cost(c, role_of[p]) * (n - 1)
                            if not doubling:
                                doubling = "doubles " + role_of[p]
                    if bad:
                        continue
                    if s < 64:
                        cost += (64 - s) * 0.08
                    elif s > 76:
                        cost += (s - 76) * 0.08
                    if t - b > 16:
                        cost += 0.3
                    out.append(Voicing(notes=notes, static_cost=cost, doubling=doubling))
    out.sort(key=lambda v: v.static_cost)
    _VOICINGS[ck] = out
    return out


def home_voicing(key_id: str, c: ChordSpec) -> list[int]:
    """A comfortable default voicing for the first chord of a session (soprano near A4)."""
    vs = voicings_for(key_id, c)
    best = vs[0]
    best_score = 1e9
    for v in vs:
        score = v.static_cost + iabs(v.notes[3] - 69) * 0.15 + iabs(v.notes[0] - 48) * 0.05
        if score < best_score:
            best = v
            best_score = score
    return best.notes


def transition(key: Key, a: ChordSpec, va: list[int], b: ChordSpec, vb: list[int], explain: bool, smooth: float = 1.0) -> Motion:
    """Full voice-leading cost of moving voicing `va` (chord a) to `vb` (chord b).
    `smooth` scales only the nearness-of-voicing term (semitones moved, held common
    tones); the voice-leading rules (parallels, leaps, tendency tones) are unaffected."""
    cost = 0.0
    marks: list[Mark] = []
    d = [vb[i] - va[i] for i in range(4)]

    # Smoothness: upper voices should move as little as possible; the bass may leap.
    cost += smooth * (0.3 * (iabs(d[1]) + iabs(d[2]) + iabs(d[3])) + 0.08 * iabs(d[0]))
    bpcs = b.pcs()
    held = len([i for i in [1, 2, 3] if d[i] == 0 and vb[i] % 12 in bpcs])
    cost -= 0.2 * smooth * held
    if explain and held > 0:
        marks.append(Mark(kind="good", text=f"{held} common tone{'s' if held > 1 else ''} held"))

    # Parallel and contrary perfect fifths / octaves between every voice pair.
    for i in range(4):
        for j in range(i + 1, 4):
            if d[i] == 0 and d[j] == 0:
                continue
            p1 = (va[j] - va[i]) % 12
            p2 = (vb[j] - vb[i]) % 12
            if p1 == p2 and p1 in [0, 7]:
                name = "5ths" if p1 == 7 else "octaves"
                if d[i] * d[j] > 0:
                    cost += 20.0
                    marks.append(Mark(kind="bad", text=f"parallel {name} ({VOICE_ABBR[i]}–{VOICE_ABBR[j]})", cost=20.0))
                elif d[i] * d[j] < 0:
                    cost += 5.0
                    marks.append(Mark(kind="bad", text=f"contrary {name} ({VOICE_ABBR[i]}–{VOICE_ABBR[j]})", cost=5.0))
            if i == 0 and p1 == 6 and p2 == 7 and d[i] * d[j] > 0:
                cost += 1.5
                marks.append(Mark(kind="warn", text=f"unequal fifths d5→P5 against the bass ({VOICE_ABBR[j]})", cost=1.5))

    # Direct (hidden) fifths / octaves: outer voices in similar motion into a perfect
    # interval, with the soprano arriving by leap.
    outer = (vb[3] - vb[0]) % 12
    if d[0] * d[3] > 0 and outer in [0, 7] and iabs(d[3]) > 2:
        cost += 3.0
        marks.append(Mark(kind="bad", text="direct " + ("octaves" if outer == 0 else "5ths") + " in the outer voices", cost=3.0))

    # Overlap: a voice moves past where its neighbour just was.
    for i in range(3):
        if vb[i] > va[i + 1] or vb[i + 1] < va[i]:
            cost += 2.0
            marks.append(Mark(kind="warn", text=f"voice overlap ({VOICE_ABBR[i]}/{VOICE_ABBR[i + 1]})", cost=2.0))

    # Melodic intervals.
    for i in range(4):
        m: int = iabs(d[i])
        if i > 0 and m > 12:
            cost += 8.0
            marks.append(Mark(kind="bad", text=f"{VOICE_NAMES[i]} leaps more than an octave", cost=8.0))
        elif i == 3 and m > 7:
            leap_cost = (m - 7) * 0.6
            cost += leap_cost
            marks.append(Mark(kind="warn", text=f"large soprano leap ({m} semitones)", cost=leap_cost))
        elif i in [1, 2] and m > 7:
            leap_cost = (m - 7) * 0.5
            cost += leap_cost
            marks.append(Mark(kind="warn", text=f"large {VOICE_NAMES[i]} leap ({m} semitones)", cost=leap_cost))
        elif i == 0 and m > 12:
            cost += 2.0
        if i > 0 and m == 6:
            cost += 3.0
            marks.append(Mark(kind="bad", text=f"melodic tritone in the {VOICE_NAMES[i]}", cost=3.0))
        if m == 3 and sorted([va[i] % 12, vb[i] % 12]) == sorted([key.rel(8), key.rel(11)]):
            cost += 6.0
            marks.append(Mark(kind="bad", text=f"augmented 2nd in the {VOICE_NAMES[i]} (♭6̂–♯7̂)", cost=6.0))
    # Cross relation: a letter is chromatically altered (e.g. F → F♯) but the change
    # appears in a different voice instead of being carried by one voice.
    for i in range(4):
        ta = a.tone_for_pc(va[i])
        if ta is None:
            continue
        ta_deg = ta.deg
        ta_pc = ta.pc
        for j in range(4):
            tb = b.tone_for_pc(vb[j])
            if tb is None:
                continue
            tb_pc = tb.pc
            if tb.deg != ta_deg or tb_pc == ta_pc or j == i:
                continue
            carried = any([va[k] % 12 == ta_pc and vb[k] % 12 == tb_pc for k in range(4)])
            if not carried:
                cost += 2.5
                marks.append(
                    Mark(
                        kind="bad",
                        text=f"cross relation {note_name(key, a, va[i])} ({VOICE_ABBR[i]}) against {note_name(key, b, vb[j])} ({VOICE_ABBR[j]})",
                        cost=2.5
                    )
                )
    if all([x > 0 for x in d]) or all([x < 0 for x in d]):
        cost += 1.5
        marks.append(Mark(kind="warn", text="all four voices move in the same direction", cost=1.5))

    # Tendency tones: when the chord of resolution offers the target, the voice must go there.
    for tend in a.tendencies:
        if not any([p in bpcs for p in tend.targets]):
            continue
        for i in range(4):
            if va[i] % 12 != tend.pc:
                continue
            step = d[i]
            dest = vb[i] % 12
            if tend.kind == "LT":
                ok = step > 0 and step <= 2 and dest in tend.targets
            elif tend.kind == "aug6":
                ok = iabs(step) == 1 and dest in tend.targets
            elif tend.kind == "N":
                ok = step < 0 and step >= -3 and dest in tend.targets
            else:
                ok = step < 0 and step >= -2 and dest in tend.targets
            what = f"{tend.label} in the {VOICE_NAMES[i]} ({note_name(key, a, va[i])}→{note_name(key, b, vb[i])})"
            if ok:
                if explain:
                    marks.append(Mark(kind="good", text=what + " resolves"))
                continue
            penalty = 4.0
            text = what + " does not resolve"
            if tend.kind == "LT":
                if i in [0, 3]:
                    penalty = 8.0
                elif b.func == "T" and dest == key.rel(7) and step < 0:
                    penalty = 0.8
                    text = what + ": frustrated leading tone (inner voice, allowed)"
            elif tend.kind == "7th":
                if step == 0 and dest in bpcs:
                    penalty = 1.5
                    text = what + " is held rather than resolved"
                else:
                    penalty = 8.0
            elif tend.kind == "aug6":
                penalty = 8.0
            elif tend.kind in ["cad6", "cad4"]:
                penalty = 5.0
            elif tend.kind == "N":
                penalty = 3.0
            elif tend.kind in ["d5", "b6"]:
                penalty = 1.0
            cost += penalty
            marks.append(Mark(kind="warn" if penalty < 2.0 else "bad", text=text, cost=penalty))
    return Motion(notes=vb, cost=cost, marks=marks)


def best_motion(key_id: str, a: ChordSpec, va: list[int], b: ChordSpec, smooth: float = 1.0) -> Motion | None:
    """Best voicing of chord `b` reachable from voicing `va` of chord `a`, or None if none exist."""
    key = key_for(key_id)
    vs = voicings_for(key_id, b)
    if not vs:
        return None
    scored: list[tuple[float, int]] = []
    for (idx, v) in enumerate(vs):
        n = v.notes
        quick = v.static_cost + smooth * (0.3 * (iabs(n[1] - va[1]) + iabs(n[2] - va[2]) + iabs(n[3] - va[3])) + 0.08 * iabs(n[0] - va[0]))
        scored.append((quick, idx))
    scored.sort()
    best_idx = -1
    best_cost = 1e9
    for (_, idx) in scored[:SHORTLIST]:
        v = vs[idx]
        m = transition(key, a, va, b, v.notes, False, smooth)
        total = m.cost + v.static_cost
        if total < best_cost:
            best_cost = total
            best_idx = idx
    if best_idx < 0:
        return None
    # Re-run the winner with explanations for the UI.
    win = vs[best_idx]
    final = transition(key, a, va, b, win.notes, True, smooth)
    if win.doubling:
        final.marks.append(Mark(kind="warn" if win.static_cost > 0.8 else "good", text=f"voicing {win.doubling}", cost=win.static_cost))
    final.cost = final.cost + win.static_cost
    return final


def best_motion_into(key_id: str, a: ChordSpec, b: ChordSpec, vb: list[int], smooth: float = 1.0) -> Motion | None:
    """Best voicing of chord `a` that leads into the fixed voicing `vb` of chord `b`:
    the mirror of `best_motion`, used when walkers search backwards in time."""
    key = key_for(key_id)
    vs = voicings_for(key_id, a)
    if not vs:
        return None
    scored: list[tuple[float, int]] = []
    for (idx, v) in enumerate(vs):
        n = v.notes
        quick = v.static_cost + smooth * (0.3 * (iabs(n[1] - vb[1]) + iabs(n[2] - vb[2]) + iabs(n[3] - vb[3])) + 0.08 * iabs(n[0] - vb[0]))
        scored.append((quick, idx))
    scored.sort()
    best_idx = -1
    best_cost = 1e9
    for (_, idx) in scored[:SHORTLIST]:
        v = vs[idx]
        m = transition(key, a, v.notes, b, vb, False, smooth)
        total = m.cost + v.static_cost
        if total < best_cost:
            best_cost = total
            best_idx = idx
    if best_idx < 0:
        return None
    win = vs[best_idx]
    final = transition(key, a, win.notes, b, vb, True, smooth)
    if win.doubling:
        final.marks.append(Mark(kind="warn" if win.static_cost > 0.8 else "good", text=f"voicing {win.doubling}", cost=win.static_cost))
    # transition() reports the arriving voicing; here the chosen voicing is the departing one.
    final.notes = win.notes
    final.cost = final.cost + win.static_cost
    return final
