"""SuperConductor Atlas server API.

The composer's session is one `Session`, kept in memory and saved as JSON by
`server.store` (server/app.py saves it after each call). Every exploration runs a
`BeamSearch` walker over the key's harmony graph; its results are cached so the UI
can audition, inspect, and commit any node of the search tree.
"""


from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

from server.theory.pitch import MAJOR_TONICS, MINOR_TONICS, Key, r2, iabs, signed
from server.theory.vocab import (
    ChordSpec,
    spec_of,
    has_spec,
    alias_chord,
    key_for,
    key_id_of,
    tonic_cid,
    parse_chord,
    vocab_for,
    is_roman,
    read_note,
    analyse,
    QUALITY_INTERVALS,
)
from server.theory.phrase import phrase_cost
from server.theory.voicing import (
    home_voicing,
    voicing_names,
    voicings_for,
    best_motion,
    best_motion_into,
    Mark,
)
from server.theory.leading import leads, RULES, degree_label
from server.theory.cadence import CADENCE_NAMES, cadence
from server.theory.syntax import syntax
from server.graph.harmony import space_for, chord_in, graph_stats
from server.graph.beam import BeamSearch, Hyp, LayerStat
from server.engine_client import (
    ENGINE_PUBLIC_URL,
    engine_online,
    send_style_layers,
    send_settings,
    restart_engine as relaunch_engine,
)
from server.models import (
    MarkView,
    ChordView,
    LayerView,
    GraphView,
    SearchView,
    SavedView,
    AltView,
    LeadView,
    StepView,
    SessionView,
    OptionResult,
    PathStory,
)
from server.narrate import narrate_path
from server import store
import json
import tomllib


@dataclass(eq=False)
class Alt:
    """Another chord that was on offer where a kept chord now stands."""
    cid: str
    symbol: str = ""   # set for a chord typed into the graph


@dataclass(eq=False)
class Step:
    cid: str
    notes: list[int]
    rule: str = ""
    cadence: str = ""
    symbol: str = ""   # set when the composer typed the chord in
    alts: list[Alt] = field(default_factory=list)   # the other options it was kept from: swappable later


def alts_of(st: Step) -> list[Alt]:
    """A step's alternatives (steps stored before alternatives existed have none)."""
    if hasattr(st, "alts") and st.alts:
        return st.alts
    return []


def restep(st: Step, rule: str, cad: str) -> Step:
    return Step(cid=st.cid, notes=st.notes, rule=rule, cadence=cad, symbol=st.symbol, alts=alts_of(st))


@dataclass(eq=False)
class Saved:
    """A committed progression: kept, in its own key, while a new one is written."""
    name: str
    tonic: str
    mode: str
    steps: list[Step]
    extended: bool = False   # written with the added-note chords switched on


def saved_kid(kept: Saved) -> str:
    """A saved progression's key id (progressions saved before `extended` existed have none)."""
    return key_id_of(kept.tonic, kept.mode, hasattr(kept, "extended") and kept.extended)


@dataclass(eq=False)
class Session:
    tonic: str = "C"
    mode: str = "major"
    style_texts: list[str] = field(default_factory=lambda: ["warm grand piano, intimate, slow"])
    style_weights: list[float] = field(default_factory=lambda: [1.0])
    texture: str = "block"
    drum_grid: str = ""   # the Drums page's grid, as JSON ("" = engine default: no drums)
    llm_enabled: bool = False
    extended: bool = False   # the walkers may use added-note chords (add9, add11, 6)
    bpm: float = 84.0
    beats_per_chord: int = 4
    # walker settings, read from walker.toml (1.0 multipliers = the defaults built into the rules)
    width: int = 12
    depth: int = 4
    options: int = 8
    max_syntax: float = 3.0     # the costliest rule a walker may follow at all
    syntax_weight: float = 1.5  # how much rule cost counts against voice leading
    colour: float = 1.0         # × the tax on applied / borrowed / chromatic chords
    cadence_pull: float = 1.0   # × the bonus for reaching a cadence
    smooth: float = 1.0         # × the cost of voices moving
    steps: list[Step] = field(default_factory=list)
    history: list[list[Step]] = field(default_factory=list)
    saved: list[Saved] = field(default_factory=list)
    loop_from: int = -1   # loop range start (-1: just the last chord)
    loop_to: int = -1     # loop range end (-1: to the last chord)
    auditions: int = 0
    commits: int = 0


# ---------------------------------------------------------------- session helpers

# Node ids of the backward tree start here, so one id names a node in either tree.
BACK_BASE: int = 100000

# How the engine plays inside each chord (audio_engine/engine.py TEXTURES).
TEXTURES: list[str] = ["block", "arpeggio", "improvise"]

# typed map options get ids from 90000 (forward) / BACK_BASE + 90000 (backward)
_NEXT_TYPED: int = 90000
_LIVE: list[int] = []   # ids of search nodes the page shows

_SEARCH: dict[int, Hyp] = {}
_SEARCH_ID: int = 0
_SEARCH_KEY: str = ""
_TYPED: list[str] = []


def is_back(node_id: int) -> bool:
    return node_id >= BACK_BASE


def know_typed(kid: str, steps: list[Step]):
    """Typed chords live outside the vocabulary, so after a restart they must be
    re-analysed before anything asks for them by cid. A chord stored under the id of an
    older analysis (C(add9) in E♭ was VI(add9), now V(add9)/ii) is found by that id too."""
    for st in steps:
        typed = [[st.cid, st.symbol]] + [[a.cid, a.symbol] for a in alts_of(st)]
        for pair in typed:
            (cid, sym) = (pair[0], pair[1])
            if sym and (kid + cid + "|" + sym) not in _TYPED:
                c = parse_chord(kid, sym)
                if cid != c.cid and not has_spec(kid, cid):
                    alias_chord(kid, cid, c)
                _TYPED.append(kid + cid + "|" + sym)


# Local mode (`python -m server.app --local`, make run-local): the engine runs Magenta
# itself. Otherwise SuperConductor mode: the engine is a sink, SuperConductor makes the
# sound, and the engine restart and the LLM descriptions are not offered.
LOCAL: bool = False


def require_local(what: str) -> None:
    if not LOCAL:
        raise RuntimeError(f"{what}: only in local mode (make run-local)")


# walker.toml: [key, min, max] for each setting; values are clamped to their range.
WALKER_FILE: str = "walker.toml"
WALKER_KEYS: list[list[Any]] = [
        ["width", 2, 40], ["depth", 1, 6], ["options", 2, 14],
        ["max_syntax", 0.5, 6.0], ["syntax_weight", 0.0, 4.0], ["colour", -2.0, 3.0],
        ["cadence_pull", 0.0, 4.0], ["smooth", 0.1, 4.0]
]


def walker_config(s: Session):
    """Copy walker.toml onto the session. A missing or broken file, or a missing key,
    leaves that setting as it was."""
    try:
        with open(WALKER_FILE, "rb") as f:
            cfg = tomllib.load(f)
    except Exception as e:
        print(f"walker.toml not read ({e}); keeping the walker settings as they are")
        return
    for (k, lo, hi) in WALKER_KEYS:
        if k in cfg:
            v = max(lo, min(hi, cfg[k]))
            setattr(s, k, int(v) if isinstance(lo, int) else float(v))


_SESSION: Session | None = None


def session() -> Session:
    global _SESSION
    if _SESSION is None:
        _SESSION = store.load(Session)
    s = _SESSION
    kid = key_id(s)
    if not s.steps:
        cid = tonic_cid(kid)
        s.steps = [Step(cid=cid, notes=home_voicing(kid, spec_of(kid, cid)))]
    if s.texture not in TEXTURES:
        s.texture = "block"
    walker_config(s)
    know_typed(kid, s.steps)
    for kept in s.saved:
        know_typed(saved_kid(kept), kept.steps)
    for past in s.history:
        know_typed(kid, past)
    return s


def save_session() -> None:
    """Write the session to disk (server/app.py calls this after each request)."""
    if _SESSION is not None:
        store.save(_SESSION)


def remember(s: Session):
    """Remember the progression before a change, so undo can restore it."""
    s.history = (s.history + [list(s.steps)])[-40:]


def key_id(s: Session) -> str:
    return key_id_of(s.tonic, s.mode, hasattr(s, "extended") and s.extended)


def loop_start(s: Session) -> int:
    """The loop range the composer set in the Progression bar (-1 where unset). The page
    turns it into what the engine plays; the server only remembers it."""
    return s.loop_from if 0 <= s.loop_from and s.loop_from < len(s.steps) else -1


def loop_end(s: Session) -> int:
    start = loop_start(s)
    return s.loop_to if start >= 0 and start <= s.loop_to and s.loop_to < len(s.steps) else -1


def saved_view(i: int, kept: Saved) -> SavedView:
    kid = saved_kid(kept)
    return SavedView(
        index=i,
        name=kept.name,
        key_name=key_for(kid).name(),
        steps=[step_view(kid, j, st, [], kept.steps[j - 1] if j > 0 else None) for (j, st) in enumerate(kept.steps)]
    )


def voice_at(kid: str, steps: list[Step], i: int, c: ChordSpec, smooth: float = 1.0) -> list[int] | None:
    """Voice chord `c` to stand at `i` in `steps` (led from the chord before, or into
    the one after when it is first). None if no voicing works."""
    if i > 0:
        prev = steps[i - 1]
        m = best_motion(kid, spec_of(kid, prev.cid), prev.notes, c, smooth)
        return m.notes if m is not None else None
    if len(steps) > 1:
        nxt = steps[1]
        m = best_motion_into(kid, c, spec_of(kid, nxt.cid), nxt.notes, smooth)
        return m.notes if m is not None else None
    return home_voicing(kid, c) if voicings_for(kid, c) else None


def alt_views(kid: str, steps: list[Step], i: int) -> list[AltView]:
    out: list[AltView] = []
    for a in alts_of(steps[i]):
        if not has_spec(kid, a.cid):
            continue
        c = spec_of(kid, a.cid)
        notes = voice_at(kid, steps, i, c)
        if notes is not None:
            out.append(AltView(cid=a.cid, rn=c.rn, symbol=c.symbol, func=c.func, notes=notes, alt_rn=c.alt_rn))
    return out


def lead_views(kid: str, prev: Step, st: Step) -> list[LeadView]:
    out: list[LeadView] = []
    for l in leads(kid, spec_of(kid, prev.cid), prev.notes, spec_of(kid, st.cid), st.notes):
        r = RULES[l.rule]
        out.append(LeadView(voice=l.voice, kind=r.kind, title=r.title, text=r.text, detail=l.detail))
    return out


def step_view(kid: str, i: int, st: Step, alts: list[AltView] = [], prev: Step | None = None) -> StepView:
    c = spec_of(kid, st.cid)
    return StepView(
        index=i,
        cid=st.cid,
        rn=c.rn,
        symbol=c.symbol,
        func=c.func,
        notes=st.notes,
        names=voicing_names(kid, c, st.notes),
        rule=st.rule,
        cadence=st.cadence,
        degrees=[degree_label(key_for(kid), c, n) for n in st.notes],
        alt_rn=c.alt_rn,
        also=c.also,
        alts=alts,
        leads=lead_views(kid, prev, st) if prev is not None else [],
        typed=bool(st.symbol)
    )


def session_view(s: Session, notice: str = "") -> SessionView:
    kid = key_id(s)
    return SessionView(
        tonic=s.tonic,
        mode=s.mode,
        key_name=key_for(kid).name(),
        style_texts=s.style_texts,
        style_weights=s.style_weights,
        texture=s.texture,
        drum_grid=s.drum_grid,
        llm_enabled=s.llm_enabled,
        extended=key_id(s).endswith("|ext"),
        bpm=s.bpm,
        beats_per_chord=s.beats_per_chord,
        steps=[step_view(kid, i, st, alt_views(kid, s.steps, i), s.steps[i - 1] if i > 0 else None) for (i, st) in enumerate(s.steps)],
        loop_from=loop_start(s),
        loop_to=loop_end(s),
        can_undo=len(s.history) > 0 or len(s.steps) > 1,
        saved=[saved_view(i, kept) for (i, kept) in enumerate(s.saved)],
        notice=notice,
        engine_url=ENGINE_PUBLIC_URL,
        engine_online=engine_online(),
        local=LOCAL,
        major_tonics=MAJOR_TONICS,
        minor_tonics=MINOR_TONICS
    )


def mark_views(marks: list[Mark]) -> list[MarkView]:
    return [MarkView(kind=m.kind, text=m.text, cost=r2(m.cost)) for m in marks]


def chord_view(kid: str, h: Hyp, status: str) -> ChordView:
    c = spec_of(kid, h.cid)
    return ChordView(
        id=h.id,
        parent=h.parent,
        layer=h.layer,
        first=h.first,
        cid=h.cid,
        rn=c.rn,
        symbol=c.symbol,
        func=c.func,
        family=c.family,
        notes=h.notes,
        names=voicing_names(kid, c, h.notes),
        syntax_cost=r2(h.syntax_cost),
        vl_cost=r2(h.vl_cost),
        phrase_cost=r2(h.phrase_cost),
        step_cost=r2(h.step_cost),
        total=r2(h.total),
        rule=h.rule,
        marks=mark_views(h.marks),
        cadence=h.cadence,
        cadence_name=CADENCE_NAMES.get(h.cadence, ""),
        status=status,
        alt_rn=c.alt_rn,
        also=c.also
    )


def chain_to(node_id: int) -> list[Hyp]:
    """The chain of hypotheses from the search start (exclusive) to `node_id` (inclusive)."""
    chain: list[Hyp] = []
    cur = _SEARCH.get(node_id, None)
    while cur is not None and cur.id != 0 and cur.id != BACK_BASE:
        chain.insert(0, cur)
        cur = _SEARCH.get(cur.parent, None)
    return chain


# ---------------------------------------------------------------- endpoints

def get_state() -> SessionView:
    s = session()
    send_style_layers(s.style_texts, s.style_weights)
    settings: dict[str, Any] = {"bpm": s.bpm, "beats_per_chord": s.beats_per_chord, "texture": s.texture}
    if s.drum_grid:
        settings["drum_grid"] = json.loads(s.drum_grid)
    send_settings(settings)
    return session_view(s)


# ---------------------------------------------------------------- transposing

def transpose_note(text: str, a: Key, b: Key) -> str:
    """A note name moved from key `a` to key `b`, keeping its scale degree (and any text
    after it), so the spelling follows the new key."""
    (letter, pc, used) = read_note(text)
    name = b.spell((letter - a.tonic_letter) % 7, (pc + b.tonic_pc - a.tonic_pc) % 12)
    if name[1:] not in ["", "♭", "♯"]:
        raise ValueError(f"{text} has no plain spelling in {b.name()}")
    return name + text[used:]


def transpose_symbol(text: str, a: Key, b: Key) -> str:
    """A typed chord symbol ("F♯m7/A", "Bb") moved from key `a` to key `b`."""
    return "/".join([transpose_note(part, a, b) for part in text.split("/")])


def counterpart(kid: str, c: ChordSpec) -> ChordSpec | None:
    """The chord of the new key's vocabulary that plays the same part as `c` did in a key
    of the other mode (I → i, ♭VI → VI, V⁷/vi → V⁷/VI), or None if it has none."""
    best: ChordSpec | None = None
    best_score = -1
    for n in vocab_for(kid):
        if n.root_deg != c.root_deg or n.is_seventh != c.is_seventh or n.target_deg != c.target_deg:
            continue
        if (n.func == "CAD64") != (c.func == "CAD64"):
            continue
        score = (
            8 * int(n.inversion == c.inversion)
            + 4 * int(n.family == c.family or (c.family == "mixture" and n.family == "diatonic"))
            + 2 * int(n.func == c.func)
            + int(n.quality == c.quality)
        )
        if score > best_score:
            best = n
            best_score = score
    return best


def transpose_chord(a_kid: str, b_kid: str, cid: str, symbol: str) -> tuple[str, str]:
    """A chord (cid, typed symbol) of key `a` as it stands in key `b`: the same Roman
    numeral in the same mode, its counterpart across modes, else the same chord moved
    literally with the tonic."""
    (a, b) = (key_for(a_kid), key_for(b_kid))
    if not has_spec(a_kid, cid):
        return (tonic_cid(b_kid), "")
    old = spec_of(a_kid, cid)
    tries: list[str] = []
    if symbol and is_roman(symbol):
        tries.append(symbol)
    elif not symbol and a.mode == b.mode and has_spec(b_kid, cid):
        return (cid, "")
    elif not symbol and a.mode != b.mode:
        twin = counterpart(b_kid, old)
        if twin is not None:
            return (twin.cid, "")
    for text in [symbol, old.symbol]:
        if text and not is_roman(text):
            try:
                tries.append(transpose_symbol(text, a, b))
            except ValueError:
                continue
    tries.append(old.rn)
    for text in tries:
        try:
            return (parse_chord(b_kid, text).cid, text)
        except ValueError:
            continue
    return (tonic_cid(b_kid), "")


def transpose_step(a_kid: str, b_kid: str, st: Step) -> Step:
    """A step moved to key `b`: its chord and alternatives transposed, voiced as near as
    the new chord allows to the old voicing moved by the tonic's shift."""
    (cid, symbol) = transpose_chord(a_kid, b_kid, st.cid, st.symbol)
    alts: list[Alt] = []
    for alt in alts_of(st):
        (acid, asym) = transpose_chord(a_kid, b_kid, alt.cid, alt.symbol)
        alts.append(Alt(cid=acid, symbol=asym))
    spec = spec_of(b_kid, cid)
    shift = signed(key_for(b_kid).tonic_pc - key_for(a_kid).tonic_pc)
    target = [n + shift for n in st.notes]
    notes = home_voicing(b_kid, spec)
    best = 1e9
    for v in voicings_for(b_kid, spec):
        d = sum([iabs(v.notes[i] - target[i]) for i in range(4)]) + 0.1 * v.static_cost
        if d < best:
            best = d
            notes = v.notes
    return Step(cid=cid, notes=notes, rule=st.rule, cadence=st.cadence, symbol=symbol, alts=alts)


def set_extended(on: bool) -> SessionView:
    """Let the walkers use the added-note chords (add9, add11, 6) or not. The progression
    is carried over: a chord the other vocabulary lacks becomes the same chord typed in."""
    s = session()
    a_kid = key_id(s)
    b_kid = key_id_of(s.tonic, s.mode, on)
    if a_kid != b_kid:
        s.steps = [transpose_step(a_kid, b_kid, st) for st in s.steps]
        s.history = [[transpose_step(a_kid, b_kid, st) for st in past] for past in s.history]
    s.extended = on
    s = session()
    return session_view(s)


def set_key(tonic: str, mode: str) -> SessionView:
    """Change key: the progression (and its undo history) is transposed, not cleared."""
    s = session()
    a_kid = key_id(s)
    b_kid = key_id_of(tonic, "minor" if mode == "minor" else "major", s.extended)
    if a_kid != b_kid:
        s.steps = [transpose_step(a_kid, b_kid, st) for st in s.steps]
        s.history = [[transpose_step(a_kid, b_kid, st) for st in past] for past in s.history]
    s.tonic = tonic
    s.mode = "minor" if mode == "minor" else "major"
    s = session()
    return session_view(s)


def set_tempo(bpm: float, beats_per_chord: int) -> SessionView:
    s = session()
    s.bpm = max(40.0, min(180.0, bpm))
    s.beats_per_chord = max(1, min(8, beats_per_chord))
    send_settings({"bpm": s.bpm, "beats_per_chord": s.beats_per_chord})
    return session_view(s)


def set_texture(texture: str) -> SessionView:
    """How the engine plays inside each chord: block | arpeggio | improvise."""
    s = session()
    if texture in TEXTURES:
        s.texture = texture
        send_settings({"texture": texture})
    return session_view(s)


def set_drums(grid: str) -> SessionView:
    """The Drums page's grid (JSON: mode, bars, per_beat, cells, after_hit; see
    audio_engine/engine.py), sent to the engine as its drum_pianoroll input. Kept only
    if the engine accepts it (or is offline: it gets it with the state on reconnect)."""
    s = session()
    parsed = json.loads(grid)
    if send_settings({"drum_grid": parsed}) or not engine_online():
        s.drum_grid = json.dumps(parsed)
    return session_view(s)


def set_llm(enabled: bool) -> SessionView:
    """Allow or forbid LLM calls. Ollama shares the GPU with Magenta, so turning this
    off guarantees the LLM never competes with real-time generation. Local mode only."""
    require_local("LLM path descriptions")
    s = session()
    s.llm_enabled = enabled
    return session_view(s)


def set_style_layers(texts: list[str], weights: list[float]) -> SessionView:
    """Set the sound as a blend of MusicCoCa prompts, each with a weight (0..1)."""
    s = session()
    keep: list[str] = []
    amounts: list[float] = []
    for (t, w) in zip(texts, weights):
        if t.strip():
            keep.append(t.strip())
            amounts.append(max(0.0, min(1.0, float(w))))
    if keep:
        s.style_texts = keep
        s.style_weights = amounts
        send_style_layers(s.style_texts, s.style_weights)
    return session_view(s)


def layer_views(searcher: BeamSearch) -> list[LayerView]:
    return [
        LayerView(
            layer=L.layer,
            frontier=L.frontier,
            frontier_rns=L.frontier_rns,
            nodes_visited=L.nodes_visited,
            edges_followed=L.edges_followed,
            voicings_scored=L.voicings_scored,
            newly_expanded=L.newly_expanded,
            candidates=L.candidates,
            kept=L.kept,
            pruned=L.pruned,
            cadences=L.cadences,
            ms=r2(L.ms)
        )
        for L in searcher.layers
    ]


def tree_nodes(kid: str, searcher: BeamSearch) -> list[ChordView]:
    kept_ids = [h.id for h in searcher.kept]
    nodes = [chord_view(kid, h, "kept") for h in searcher.kept]
    return nodes + [chord_view(kid, h, "dest") for h in searcher.destinations if h.id not in kept_ids]


def ranked_options(s: Session, kid: str, i: int) -> list[Alt]:
    """The options beside kept step `i`: every chord of the vocabulary that can follow
    the chord before it and lead into the chord after it (either side may be missing at
    the ends), ranked like a walker's move - syntax cost × weight plus voice leading on
    each side - with the best inversion of each harmony, best `s.options` of them."""
    key = key_for(kid)
    steps = s.steps
    here = steps[i]
    before = spec_of(kid, steps[i - 1].cid) if i > 0 else None
    after = spec_of(kid, steps[i + 1].cid) if i + 1 < len(steps) else None
    scored: list[tuple[float, ChordSpec]] = []
    for b in vocab_for(kid):
        if b.cid == here.cid:
            continue
        cost = 0.0
        if before is not None:
            r = syntax(before, b, key)
            if r is None or r.cost > s.max_syntax:
                continue
            cost += s.syntax_weight * r.cost
        if after is not None:
            r = syntax(b, after, key)
            if r is None or r.cost > s.max_syntax:
                continue
            cost += s.syntax_weight * r.cost
        scored.append((cost, b))
    # voice leading only for chords the syntax allows (the costly part)
    ranked: list[tuple[float, ChordSpec]] = []
    for (cost, b) in scored:
        notes: list[int] = []
        if before is not None:
            m = best_motion(kid, before, steps[i - 1].notes, b, s.smooth)
            if m is None:
                continue
            cost += m.cost
            notes = m.notes
        if after is not None:
            if notes:
                m2 = best_motion(kid, b, notes, after, s.smooth)
            else:
                m2 = best_motion_into(kid, b, after, steps[i + 1].notes, s.smooth)
            if m2 is None:
                continue
            cost += m2.cost
        ranked.append((cost, b))
    ranked.sort(key=lambda t: t[0])
    out: list[Alt] = []
    seen: list[str] = []
    for (_, b) in ranked:
        if b.harmony in seen:
            continue
        seen.append(b.harmony)
        out.append(Alt(cid=b.cid))
        if len(out) >= s.options:
            break
    return out


def refresh_options() -> SessionView:
    """Re-rank the options beside every kept chord against its neighbours as they are
    now (Send walkers). Chords the composer typed in as options stay."""
    s = session()
    kid = key_id(s)
    if len(s.steps) < 2:
        return session_view(s)
    steps: list[Step] = []
    for (i, st) in enumerate(s.steps):
        typed = [a for a in alts_of(st) if a.symbol]
        fresh = [a for a in ranked_options(s, kid, i) if a.cid not in [t.cid for t in typed]]
        steps.append(Step(cid=st.cid, notes=st.notes, rule=st.rule, cadence=st.cadence, symbol=st.symbol, alts=typed + fresh))
    s.steps = steps
    return session_view(s)


def explore() -> SearchView:
    """Send two BeamSearch walkers out at once: one forwards from the last chord of the
    progression, one backwards in time from the first."""
    global _SEARCH, _SEARCH_ID, _SEARCH_KEY
    s = session()
    kid = key_id(s)
    space = space_for(kid)
    last = s.steps[-1]
    first = s.steps[0]
    fwd_start = Hyp(
        id=0, parent=-1, layer=0, first=-1, cid=last.cid, notes=last.notes,
        path=[st.cid for st in s.steps][-3:], at=chord_in(space, last.cid)
    )
    # backwards, the phrase context is read back in time from the first chord
    back_start = Hyp(
        id=BACK_BASE, parent=-1, layer=0, first=-1, cid=first.cid, notes=first.notes,
        path=[st.cid for st in s.steps[:3]][::-1], at=chord_in(space, first.cid)
    )
    fwd = BeamSearch(
        key_id=kid, start=fwd_start, width=s.width, depth=s.depth, options=s.options,
        max_syntax=s.max_syntax, syntax_weight=s.syntax_weight, colour=s.colour, cadence_pull=s.cadence_pull, smooth=s.smooth
    ).run()
    back = BeamSearch(
        key_id=kid, start=back_start, width=s.width, depth=s.depth, options=s.options, backward=True, next_id=BACK_BASE + 1,
        max_syntax=s.max_syntax, syntax_weight=s.syntax_weight, colour=s.colour, cadence_pull=s.cadence_pull, smooth=s.smooth
    ).run()
    _SEARCH_ID += 1
    _SEARCH_KEY = kid
    _SEARCH = {0: fwd_start, BACK_BASE: back_start}
    _LIVE.clear()
    for h in fwd.kept + fwd.destinations + back.kept + back.destinations:
        _LIVE.append(h.id)
    for h in fwd.kept + fwd.destinations + back.kept + back.destinations:
        _SEARCH[h.id] = h
    st = graph_stats(space)
    return SearchView(
        search_id=_SEARCH_ID,
        key_name=key_for(kid).name(),
        start=chord_view(kid, fwd_start, "start"),
        nodes=tree_nodes(kid, fwd),
        layers=layer_views(fwd),
        graph=GraphView(
            key_name=key_for(kid).name(),
            chord_nodes=st.chord_nodes,
            progression_edges=st.progression_edges,
            expanded_nodes=st.expanded_nodes,
            vocabulary=st.vocabulary
        ),
        elapsed_ms=r2(fwd.elapsed_ms),
        width=s.width,
        depth=s.depth,
        options=s.options,
        back_start=chord_view(kid, back_start, "start"),
        back_nodes=tree_nodes(kid, back),
        back_layers=layer_views(back),
        back_elapsed_ms=r2(back.elapsed_ms)
    )


def siblings(h: Hyp) -> list[Alt]:
    """The other chords the graph offered beside `h` (same parent): kept with the step so
    it can be swapped later."""
    out: list[Alt] = []
    seen: list[str] = [h.cid]
    for nid in _LIVE:
        g = _SEARCH.get(nid, None)
        if g is not None and g.parent == h.parent and g.cid not in seen:
            seen.append(g.cid)
            out.append(Alt(cid=g.cid, symbol=g.symbol))
    return out[:12]


def add_path(search_id: int, node_id: int, whole: bool) -> SessionView:
    """Keep the path to `node_id` in the progression (whole=True) or only its first chord."""
    s = session()
    if search_id == _SEARCH_ID and node_id in _SEARCH and chain_to(node_id):
        chain = chain_to(node_id)
        if not whole:
            chain = chain[:1]
        remember(s)
        if is_back(node_id):
            # chain runs back in time from the first chord; each hypothesis carries the
            # move *out of* it, which belongs to the step after it.
            ahead = [Step(cid=h.cid, notes=h.notes, symbol=h.symbol, alts=siblings(h)) for h in chain[::-1]]
            later = ahead[1:] + [s.steps[0]]
            moved = [restep(st, h.rule, h.cadence) for (st, h) in zip(later, chain[::-1])]
            s.steps = ahead[:1] + moved + s.steps[1:]
            if s.loop_from >= 0:
                s.loop_from += len(chain)
                s.loop_to = s.loop_to + len(chain) if s.loop_to >= 0 else -1
        else:
            s.steps = s.steps + [
                Step(cid=h.cid, notes=h.notes, rule=h.rule, cadence=h.cadence, symbol=h.symbol, alts=siblings(h))
                for h in chain
            ]
        s.commits += 1
    return session_view(s)


def undo() -> SessionView:
    s = session()
    s.loop_from = -1
    s.loop_to = -1
    if s.history:
        s.steps = s.history[-1]
        s.history = s.history[:-1]
    elif len(s.steps) > 1:
        s.steps = s.steps[:-1]
    return session_view(s)


def clear_progression() -> SessionView:
    s = session()
    remember(s)
    s.steps = []
    s.loop_from = -1
    s.loop_to = -1
    s = session()
    return session_view(s)


def describe(search_id: int, node_id: int) -> PathStory:
    """Ask the LLM (server.narrate) to characterise the path to a node of the current search.
    Local mode only."""
    require_local("LLM path descriptions")
    s = session()
    if not s.llm_enabled:
        return PathStory(title="", mood="", explanation="LLM calls are switched off (so Magenta keeps the GPU).")
    if search_id != _SEARCH_ID or node_id not in _SEARCH:
        return PathStory(title="", mood="", explanation="That search is no longer current.")
    kid = _SEARCH_KEY
    chain = chain_to(node_id)
    anchor = s.steps[0] if is_back(node_id) else s.steps[-1]
    cids = [h.cid for h in chain]
    cids = cids[::-1] + [anchor.cid] if is_back(node_id) else [anchor.cid] + cids
    progression = " → ".join([f"{spec_of(kid, c).rn} ({spec_of(kid, c).symbol})" for c in cids])
    moves = [h.rule for h in chain][::-1] if is_back(node_id) else [h.rule for h in chain]
    notes: list[str] = []
    for h in chain:
        notes = notes + [m.text for m in h.marks if m.kind != "good" or "resolves" in m.text]
    cad = (chain[0].cadence if is_back(node_id) else chain[-1].cadence) if chain else ""
    try:
        return narrate_path(key_for(kid).name(), progression, CADENCE_NAMES.get(cad, "open"), moves, notes[:12])
    except Exception as e:
        return PathStory(title="", mood="", explanation=f"The language model is unavailable ({str(e)[:160]}).")


def split_symbols(text: str) -> list[str]:
    """Split typed input like "Gm/D -> Cm/Eb, Gm/Bb  D/A" into chord symbols."""
    flat = text.replace("→", " ").replace("->", " ").replace(",", " ").replace("|", " ").replace(";", " ")
    return [t for t in flat.split() if t and t != "-"]


def describe_move(key_id: str, a: ChordSpec, b: ChordSpec) -> str:
    rule = syntax(a, b, key_for(key_id))
    return rule.label if rule is not None else f"typed: {a.rn} → {b.rn} is outside the walkers' rules"


def voiced_after(kid: str, prev: Step, c: ChordSpec) -> list[int]:
    m = best_motion(kid, spec_of(kid, prev.cid), prev.notes, c)
    if m is None:
        raise ValueError(f"{c.symbol} has no playable four-voice voicing in this inversion")
    return m.notes


def add_chords(text: str, where: str) -> SessionView:
    """Add typed chords to the progression: "after" the last chord, "before" the first,
    "first" in place of the first chord, or "replace" the whole progression. Each chord
    is analysed in the key and voice-led from (or, before, into) its neighbour, so the
    walkers can carry on from it."""
    s = session()
    kid = key_id(s)
    key = key_for(kid)
    symbols = split_symbols(text)
    if not symbols:
        return session_view(s, "type one or more chord symbols, e.g. Gm/D Cm/Eb D/A")
    specs: list[ChordSpec] = []
    try:
        for sym in symbols:
            specs.append(parse_chord(kid, sym))
    except ValueError as e:
        return session_view(s, str(e))
    for sym in symbols:
        if (kid + sym) not in _TYPED:
            _TYPED.append(kid + sym)
    first = s.steps[0]
    starting = where == "first"
    try:
        steps: list[Step] = list(s.steps)
        if starting:
            # the typed chords open the progression in place of its first chord
            steps = steps[1:]
            where = "before" if steps else "replace"
        if where == "before":
            # insert last-typed first, so each chord is voiced into the one after it
            for i in range(len(specs) - 1, -1, -1):
                c = specs[i]
                nxt = steps[0]
                nspec = spec_of(kid, nxt.cid)
                m = best_motion_into(kid, c, nspec, nxt.notes)
                if m is None:
                    raise ValueError(f"{c.symbol} has no playable four-voice voicing in this inversion")
                steps[0] = restep(nxt, describe_move(kid, c, nspec), cadence(key, c, nspec, nxt.notes))
                steps.insert(0, Step(cid=c.cid, notes=m.notes, symbol=symbols[i]))
        else:
            start = 0
            if where == "replace":
                c0 = specs[0]
                if not voicings_for(kid, c0):
                    raise ValueError(f"{c0.symbol} has no playable four-voice voicing in this inversion")
                steps = [Step(cid=c0.cid, notes=home_voicing(kid, c0), symbol=symbols[0])]
                start = 1
            for i in range(start, len(specs)):
                c = specs[i]
                prev = steps[-1]
                pspec = spec_of(kid, prev.cid)
                notes = voiced_after(kid, prev, c)
                steps.append(
                    Step(cid=c.cid, notes=notes, rule=describe_move(kid, pspec, c), cadence=cadence(key, pspec, c, notes), symbol=symbols[i])
                )
    except ValueError as e:
        return session_view(s, str(e))
    if starting and steps[0].cid != first.cid:
        # a new opening chord: the old one stays on offer beside it, to swap back
        keep = [a for a in alts_of(first) if a.cid != steps[0].cid]
        steps[0] = Step(
            cid=steps[0].cid, notes=steps[0].notes, rule=steps[0].rule, cadence=steps[0].cadence,
            symbol=steps[0].symbol, alts=[Alt(cid=first.cid, symbol=first.symbol)] + keep
        )
    remember(s)
    if where == "replace":
        s.loop_from = -1
        s.loop_to = -1
    elif where == "before" and s.loop_from >= 0:
        s.loop_from += len(steps) - len(s.steps)
        s.loop_to = s.loop_to + len(steps) - len(s.steps) if s.loop_to >= 0 else -1
    s.steps = steps
    return session_view(s)


def restart_engine() -> bool:
    """Restart the audio engine process (for when the audio glitches). The client calls
    get_state once the engine is back, which re-sends the sound, settings and chord.
    Local mode only: it starts the local Magenta engine."""
    require_local("Restarting the engine")
    return relaunch_engine()


def set_loop(start: int, end: int) -> SessionView:
    """Set the loop range: steps `start`..`end` (end -1: to the last chord; start -1:
    the default, just the last chord). The Progression bar decides what clicks mean."""
    s = session()
    n = len(s.steps)
    s.loop_from = start if 0 <= start and start < n else -1
    s.loop_to = end if s.loop_from >= 0 and s.loop_from <= end and end < n else -1
    return session_view(s)


def add_option(search_id: int, parent_ids: list[int], text: str, force: bool = False) -> OptionResult:
    """Type a chord into a column of the map. It joins the tree once for every chord in
    the previous column (`parent_ids`) that may legally move to it - forwards, parent →
    chord; backwards, chord → parent - voice-led and scored like the walker's own.
    Its other inversions and seventh forms join it wherever the rules allow them, and
    stand in for it if it can go nowhere. `force` links the typed chord even where the
    rules forbid the move (it still needs a playable voicing).

    Then a mini walker searches on from the typed chord through the remaining columns
    (status "grown"). The client draws a found chord that is already in its column as
    one more arrow into the existing box, and anything else as a new entry."""
    if search_id != _SEARCH_ID:
        return OptionResult(nodes=[], notice="that search is no longer current")
    kid = _SEARCH_KEY
    symbol = text.strip()
    try:
        c = parse_chord(kid, symbol)
    except ValueError as e:
        return OptionResult(nodes=[], notice=str(e))
    if (kid + symbol) not in _TYPED:
        _TYPED.append(kid + symbol)
    made = link_option(kid, c, symbol, parent_ids, force)
    # its other inversions and seventh forms join it wherever the rules allow them
    extra: list[ChordView] = []
    used: list[str] = []
    for (v, vsym) in chord_variants(kid, c):
        vm = link_option(kid, v, vsym, parent_ids)
        if vm:
            extra = extra + vm
            used.append(f"{v.rn}")
            if (kid + vsym) not in _TYPED:
                _TYPED.append(kid + vsym)
    if not made and not extra:
        return OptionResult(nodes=[], notice=f"no chord in the previous column can move to {c.symbol} ({c.rn}) under the walkers' rules, in any inversion or as a seventh")
    info = ""
    if not made:
        info = f"{c.symbol} ({c.rn}) can't follow here under the walkers' rules; added " + ", ".join(used) + " instead"
    # the mini walker grows on from the typed chord itself when it could go in
    grow = made if made else extra
    best = _SEARCH[min(grow, key=lambda v: v.total).id]
    return OptionResult(nodes=made + extra + grow_from(best), notice="", info=info)


def link_option(kid: str, c: ChordSpec, symbol: str, parent_ids: list[int], force: bool = False) -> list[ChordView]:
    """Link chord `c` (typed as `symbol`) from every given node it may legally follow
    (or, in the backward tree, precede); returns the new nodes."""
    global _NEXT_TYPED
    key = key_for(kid)
    tune = session()
    made: list[ChordView] = []
    for pid in parent_ids:
        p = _SEARCH.get(pid, None)
        if p is None:
            continue
        ps = spec_of(kid, p.cid)
        back = is_back(pid)
        rule = syntax(c, ps, key) if back else syntax(ps, c, key)
        if rule is None and not force:
            continue
        # a forced move costs as much as the costliest rule a walker may follow
        rule_cost = rule.cost if rule is not None else tune.max_syntax
        first_rn = c.rn if back else ps.rn
        then_rn = ps.rn if back else c.rn
        rule_label = rule.label if rule is not None else f"forced: {first_rn} → {then_rn} is outside the walkers' rules"
        m = best_motion_into(kid, c, ps, p.notes, tune.smooth) if back else best_motion(kid, ps, p.notes, c, tune.smooth)
        if m is None:
            continue
        cad = cadence(key, c, ps, p.notes) if back else cadence(key, ps, c, m.notes)
        (pc, pmarks) = phrase_cost(kid, p.path, c, cad, back, tune.colour, tune.cadence_pull)
        step = max(0.05, tune.syntax_weight * rule_cost + m.cost + pc)
        _NEXT_TYPED += 1
        nid = _NEXT_TYPED + (BACK_BASE if back else 0)
        h = Hyp(
            id=nid, parent=pid, layer=p.layer + 1, first=p.first if p.layer > 0 else nid,
            cid=c.cid, notes=m.notes, path=p.path + [c.cid], syntax_cost=rule_cost, vl_cost=m.cost,
            phrase_cost=pc, step_cost=step, total=p.total + step, rule=rule_label,
            marks=m.marks + pmarks, cadence=cad, symbol=symbol
        )
        _SEARCH[nid] = h
        _LIVE.append(nid)
        made.append(chord_view(kid, h, "custom"))
    return made


def chord_variants(kid: str, c: ChordSpec) -> list[tuple[ChordSpec, str]]:
    """Other forms of a typed chord, nearest first: its other inversions, then (for a
    triad) the seventh chord on the same root, typed inversion first. Each comes with a
    chord symbol that reads back as it."""
    if c.func in ["CAD64", "AUG6"]:
        return []
    key = key_for(kid)
    forms: list[tuple[str, int]] = [(c.quality, i) for i in range(len(QUALITY_INTERVALS[c.quality])) if i != c.inversion]
    seventh = (key.degree_pc(c.root_deg + 6) - c.root_pc) % 12
    q7 = ""
    if c.quality == "maj":
        q7 = "dom7" if c.func in ["D", "APP"] or seventh != 11 else "maj7"
    elif c.quality == "min":
        q7 = "min7"
    elif c.quality == "dim":
        q7 = "dim7" if seventh == 9 or c.func == "APP" else "hdim7"
    elif c.quality == "sus4":
        q7 = "7sus4"
    if q7:
        first = min(c.inversion, 3)
        forms = forms + [(q7, first)] + [(q7, i) for i in range(4) if i != first]
    out: list[tuple[ChordSpec, str]] = []
    for (q, inv) in forms:
        try:
            sym = analyse(kid, c.root_deg, c.root_pc, q, inv).symbol
            out.append((parse_chord(kid, sym), sym))
        except Exception:
            continue
    return out


def grow_from(h: Hyp) -> list[ChordView]:
    """Mini walker: a BeamSearch from one typed chord over the columns left after it."""
    global _NEXT_TYPED
    s = session()
    kid = _SEARCH_KEY
    remaining = s.depth - h.layer
    if remaining < 1:
        return []
    back = is_back(h.id)
    space = space_for(kid)
    root_hyp = Hyp(
        id=h.id, parent=h.parent, layer=0, first=-1, cid=h.cid, notes=h.notes, path=h.path,
        total=h.total, at=chord_in(space, h.cid)
    )
    base_id = _NEXT_TYPED + 1 + (BACK_BASE if back else 0)
    w = BeamSearch(
        key_id=kid, start=root_hyp, width=s.width, depth=remaining, options=s.options,
        backward=back, next_id=base_id,
        max_syntax=s.max_syntax, syntax_weight=s.syntax_weight, colour=s.colour, cadence_pull=s.cadence_pull, smooth=s.smooth
    ).run()
    kept_ids = [g.id for g in w.kept]
    found = w.kept + [g for g in w.destinations if g.id not in kept_ids]
    out: list[ChordView] = []
    for g in found:
        g.layer += h.layer   # the walker counts layers from the typed chord
        _SEARCH[g.id] = g
        _LIVE.append(g.id)
        out.append(chord_view(kid, g, "grown"))
    _NEXT_TYPED = w.next_id - (BACK_BASE if back else 0)
    return out


def commit_progression(name: str) -> SessionView:
    """Commit the progression: keep it (named, in its key) and start a fresh one."""
    s = session()
    label = name.strip() if name.strip() else f"Progression {len(s.saved) + 1}"
    s.saved = s.saved + [Saved(name=label, tonic=s.tonic, mode=s.mode, steps=list(s.steps), extended=s.extended)]
    s.steps = []
    s.history = []
    s.loop_from = -1
    s.loop_to = -1
    s = session()
    return session_view(s)


def rename_saved(index: int, name: str) -> SessionView:
    """Rename a committed progression (a blank name leaves it as it was)."""
    s = session()
    if 0 <= index and index < len(s.saved) and name.strip():
        s.saved = [
            Saved(name=name.strip(), tonic=k.tonic, mode=k.mode, steps=k.steps, extended=hasattr(k, "extended") and k.extended) if i == index else k
            for (i, k) in enumerate(s.saved)
        ]
    return session_view(s)


def delete_saved(index: int) -> SessionView:
    """Forget a committed progression."""
    s = session()
    if 0 <= index and index < len(s.saved):
        s.saved = [kept for (i, kept) in enumerate(s.saved) if i != index]
    return session_view(s)


def resume_saved(index: int) -> SessionView:
    """Make a committed progression current again. The current one is committed in its
    place first, unless it is still just the opening tonic."""
    s = session()
    if index < 0 or index >= len(s.saved):
        return session_view(s)
    kept = s.saved[index]
    rest = [k for (i, k) in enumerate(s.saved) if i != index]
    if len(s.steps) > 1:
        rest = rest + [Saved(name=f"Progression {len(s.saved) + 1}", tonic=s.tonic, mode=s.mode, steps=list(s.steps), extended=s.extended)]
    s.saved = rest
    s.tonic = kept.tonic
    s.mode = kept.mode
    s.extended = hasattr(kept, "extended") and kept.extended
    s.steps = list(kept.steps)
    s.history = []
    s.loop_from = -1
    s.loop_to = -1
    return session_view(s)


def remove_step(index: int) -> SessionView:
    """Remove a typed chord from the progression. A chord typed in place of another gives
    that chord back; any other is taken out. The chords after it are voice-led again."""
    s = session()
    kid = key_id(s)
    if index < 0 or index >= len(s.steps) or not s.steps[index].symbol:
        return session_view(s)
    steps = list(s.steps)
    back = [a for a in alts_of(steps[index]) if has_spec(kid, a.cid)]
    if back:
        c = spec_of(kid, back[0].cid)
        notes = voice_at(kid, steps, index, c)
        if notes is None:
            return session_view(s, f"{c.symbol} has no playable voicing here")
        steps[index] = Step(cid=c.cid, notes=notes, symbol=back[0].symbol, alts=back[1:])
        steps = relink(kid, steps, index)
    else:
        steps = steps[:index] + steps[index + 1:]
        if steps and index == 0:
            steps[0] = restep(steps[0], "", "")
        steps = relink(kid, steps, index, False)
    remember(s)
    s.steps = steps
    s.loop_from = -1
    s.loop_to = -1
    s = session()
    return session_view(s)


def replace_step(index: int, text: str) -> SessionView:
    """Replace the kept chord at `index` (after the first) with a typed one, voice-led from
    the chord before. The chord it replaces stays on offer beside it (× gives it back)."""
    s = session()
    kid = key_id(s)
    symbol = text.strip()
    if index <= 0 or index >= len(s.steps) or not symbol:
        return session_view(s)
    try:
        c = parse_chord(kid, symbol)
    except ValueError as e:
        return session_view(s, str(e))
    if (kid + symbol) not in _TYPED:
        _TYPED.append(kid + symbol)
    steps = list(s.steps)
    old = steps[index]
    notes = voice_at(kid, steps, index, c)
    if notes is None:
        return session_view(s, f"{c.symbol} has no playable four-voice voicing here")
    alts = [Alt(cid=old.cid, symbol=old.symbol)] + [a for a in alts_of(old) if a.cid != c.cid and a.cid != old.cid]
    steps[index] = Step(cid=c.cid, notes=notes, symbol=symbol, alts=alts)
    remember(s)
    s.steps = relink(kid, steps, index)
    return session_view(s)


def relink(kid: str, steps: list[Step], changed: int, keep: bool = True) -> list[Step]:
    """Recompute the move into step `changed` (from the one before) and voice-lead every
    chord after it again; `keep` keeps the changed step's own voicing. A changed first
    chord keeps the second's voicing (it was voiced into it); only the move into the
    second is recomputed."""
    key = key_for(kid)
    if changed <= 0:
        if len(steps) > 1:
            a = spec_of(kid, steps[0].cid)
            b = spec_of(kid, steps[1].cid)
            steps[1] = restep(steps[1], describe_move(kid, a, b), cadence(key, a, b, steps[1].notes))
        return steps
    for j in range(changed, len(steps)):
        led = steps[j - 1]
        ps = spec_of(kid, led.cid)
        st = steps[j]
        cs = spec_of(kid, st.cid)
        vb = st.notes
        if j > changed or not keep:
            m = best_motion(kid, ps, led.notes, cs)
            vb = m.notes if m is not None else st.notes
        steps[j] = Step(cid=st.cid, notes=vb, rule=describe_move(kid, ps, cs), cadence=cadence(key, ps, cs, vb), symbol=st.symbol, alts=alts_of(st))
    return steps


def swap_step(index: int, cid: str) -> SessionView:
    """Swap the kept chord at `index` for one of its alternatives. The chords after it are
    re-voiced so the voice leading stays smooth; the old chord becomes an alternative."""
    s = session()
    kid = key_id(s)
    key = key_for(kid)
    if index < 0 or index >= len(s.steps) or not has_spec(kid, cid):
        return session_view(s)
    old = s.steps[index]
    picked = [a for a in alts_of(old) if a.cid == cid]
    if not picked:
        return session_view(s, f"{cid} was not one of the options at step {index + 1}")
    c = spec_of(kid, cid)
    steps = list(s.steps)
    notes = voice_at(kid, steps, index, c)
    if notes is None:
        return session_view(s, f"{c.symbol} has no playable voicing here")
    # the old chord stays on offer, unless it was typed in: swapped out, it is gone (like ×)
    alts = [a for a in alts_of(old) if a.cid != cid] + ([] if old.symbol else [Alt(cid=old.cid, symbol=old.symbol)])
    rule = ""
    cad = ""
    if index > 0:
        before = spec_of(kid, steps[index - 1].cid)
        rule = describe_move(kid, before, c)
        cad = cadence(key, before, c, notes)
    steps[index] = Step(cid=cid, notes=notes, rule=rule, cadence=cad, symbol=picked[0].symbol, alts=alts)
    for j in range(index + 1, len(steps)):
        led = steps[j - 1]
        ps = spec_of(kid, led.cid)
        st = steps[j]
        cs = spec_of(kid, st.cid)
        m = best_motion(kid, ps, led.notes, cs)
        vb = m.notes if m is not None else st.notes
        steps[j] = Step(cid=st.cid, notes=vb, rule=describe_move(kid, ps, cs), cadence=cadence(key, ps, cs, vb), symbol=st.symbol, alts=alts_of(st))
    remember(s)
    s.steps = steps
    return session_view(s)
