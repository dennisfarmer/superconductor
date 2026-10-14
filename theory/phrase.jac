"""Phrase-level rules: costs that depend on the whole path, not just the last move.

These are the reason the search is a *stateful walker*: a progression edge
cannot know that the path already visited this harmony two chords ago, or that
it has been circling in the pre-dominant area for three chords, or that it is
about to cancel the borrowed colour it just introduced. Only something that
carries the path with it can judge that.
"""

import from theory.vocab { ChordSpec, spec_of, ADDED_TONE }
import from theory.voicing { Mark }

# Extra cost for leaving the diatonic palette: chromatic chords are marked choices.
glob COLOUR_TAX: dict[str, float] = {"applied": 0.35, "mixture": 0.45, "chromatic": 0.55};

# Added-note chords are colour too: a small tax keeps them from crowding out plain triads.
glob ADDED_TAX: float = 0.3;

glob CLOSURE_BONUS: dict[str, float] = {"PAC": 0.9, "IAC": 0.5, "HC": 0.45, "PHC": 0.5, "PC": 0.45, "DC": 0.35};


"""Coarse functional area, for detecting a phrase that stalls in one place."""
def area(c: ChordSpec) -> str {
    if c.func in ["D", "CAD64"] {
        return "D";
    }
    if c.func in ["PD", "NEAP", "AUG6"] {
        return "PD";
    }
    return c.func;
}


"""Cost of extending `path` with `b`. `colour` scales the chromatic tax and `pull`
the cadence bonus (the composer's walker settings; 1.0 = the defaults). Searching backwards, `path` runs back in time
(reference chord first) and `b` comes *before* the path's last chord."""
def phrase_cost(
    key_id: str, path: list[str], b: ChordSpec, cad: str, backward: bool = False, colour: float = 1.0, pull: float = 1.0
) -> tuple[float, list[Mark]] {
    specs = [spec_of(key_id, cid) for cid in path];
    last = specs[-1];
    harmonies = [s.harmony for s in specs];
    # colour scales the tax on chromatic families: 0 = indifferent, negative = sought out
    cost = (COLOUR_TAX.get(b.family, 0.0) + (ADDED_TAX if b.quality in ADDED_TONE else 0.0)) * colour;
    marks: list[Mark] = [];

    if len(specs) >= 2 and harmonies[-2] == b.harmony and harmonies[-1] != b.harmony and b.func != "T" {
        cost += 1.6;
        marks.append(Mark(kind="bad", text=f"oscillates back to {b.harmony}", cost=1.6));
    } elif b.harmony in harmonies[:-1] and b.func != "T" {
        cost += 0.8;
        marks.append(Mark(kind="warn", text=f"revisits {b.harmony} (already used in this phrase)", cost=0.8));
    }

    run = [area(s) for s in specs[-2:]] + [area(b)];
    if len(run) == 3 and run[0] == run[1] and run[1] == run[2] and run[2] in ["PD", "TX", "APP", "MIX"] {
        cost += 1.2;
        marks.append(Mark(kind="warn", text=f"stalls in the {run[2]} area for three chords", cost=1.2));
    }

    (earlier, later) = (b, last) if backward else (last, b);
    if earlier.family == "mixture" and later.family == "diatonic" and earlier.root_pc == later.root_pc {
        cost += 1.6;
        marks.append(Mark(kind="bad", text="cancels the borrowed colour it just introduced", cost=1.6));
    }

    if cad {
        bonus = CLOSURE_BONUS.get(cad, 0.0) * pull;
        cost -= bonus;
        marks.append(Mark(kind="good", text=f"closes the phrase ({cad})", cost=-bonus));
    }
    return (cost, marks);
}
