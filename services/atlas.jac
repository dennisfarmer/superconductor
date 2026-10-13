"""SuperConductor Atlas server API.

The composer's session lives on the graph (a `Session` node under root). Every
exploration spawns a `BeamSearch` walker on the key's `BeamGate`; its results are
cached so the UI can audition, inspect, and commit any node of the search tree.
"""

import from theory.pitch { MAJOR_TONICS, MINOR_TONICS, r2 }
import from theory.vocab { ChordSpec, spec_of, has_spec, key_for, key_id_of, tonic_cid, parse_chord }
import from theory.phrase { phrase_cost }
import from theory.voicing { home_voicing, voicing_names, voicings_for, best_motion, best_motion_into, Mark }
import from theory.cadence { CADENCE_NAMES, cadence }
import from theory.syntax { syntax }
import from graph.harmony { space_for, chord_in, gate_of, graph_stats }
import from graph.beam { BeamSearch, Hyp, LayerStat }
import from services.engine_client {
    ENGINE_URL, engine_online, send_style_layers, send_settings, stop_engine,
    restart_engine as relaunch_engine
}
import gc;
import json;
import signal;
import tomllib;
import threading;
import uvicorn;


"""Another chord that was on offer where a kept chord now stands."""
obj Alt {
    has cid: str,
        symbol: str = "";   # set for a chord typed into the graph
}


obj Step {
    has cid: str,
        notes: list[int],
        rule: str = "",
        cadence: str = "",
        symbol: str = "",   # set when the composer typed the chord in
        alts: list[Alt] = [];   # the other options it was kept from: swappable later
}


"""A step's alternatives (steps stored before alternatives existed have none)."""
def alts_of(st: Step) -> list[Alt] {
    if hasattr(st, "alts") and st.alts {
        return st.alts;
    }
    return [];
}


def restep(st: Step, rule: str, cad: str) -> Step {
    return Step(cid=st.cid, notes=st.notes, rule=rule, cadence=cad, symbol=st.symbol, alts=alts_of(st));
}


"""A committed progression: kept, in its own key, while a new one is written."""
obj Saved {
    has name: str,
        tonic: str,
        mode: str,
        steps: list[Step];
}


node Session {
    has tonic: str = "C",
        mode: str = "major",
        style_texts: list[str] = ["warm grand piano, intimate, slow"],
        style_weights: list[float] = [1.0],
        texture: str = "block",
        drum_grid: str = "",   # the Drums page's grid, as JSON ("" = engine default: no drums)
        llm_enabled: bool = False,
        bpm: float = 84.0,
        beats_per_chord: int = 4,
        # walker settings, read from walker.toml (1.0 multipliers = the defaults built into the rules)
        width: int = 12,
        depth: int = 4,
        options: int = 8,
        max_syntax: float = 3.0,     # the costliest rule a walker may follow at all
        syntax_weight: float = 1.5,  # how much rule cost counts against voice leading
        colour: float = 1.0,         # × the tax on applied / borrowed / chromatic chords
        cadence_pull: float = 1.0,   # × the bonus for reaching a cadence
        smooth: float = 1.0,         # × the cost of voices moving
        steps: list[Step] = [],
        history: list[list[Step]] = [],
        saved: list[Saved] = [],
        loop_from: int = -1,   # loop range start (-1: just the last chord)
        loop_to: int = -1,     # loop range end (-1: to the last chord)
        auditions: int = 0,
        commits: int = 0;
}


# ---------------------------------------------------------------- wire types

obj MarkView {
    has kind: str,
        text: str,
        cost: float;
}


obj ChordView {
    has id: int,
        parent: int,
        layer: int,
        first: int,
        cid: str,
        rn: str,
        symbol: str,
        func: str,
        family: str,
        notes: list[int],
        names: list[str],
        syntax_cost: float,
        vl_cost: float,
        phrase_cost: float,
        step_cost: float,
        total: float,
        rule: str,
        marks: list[MarkView],
        cadence: str,
        cadence_name: str,
        status: str;
}


obj LayerView {
    has layer: int,
        frontier: list[int],
        frontier_rns: list[str],
        nodes_visited: int,
        edges_followed: int,
        voicings_scored: int,
        newly_expanded: int,
        candidates: int,
        kept: int,
        pruned: int,
        cadences: int,
        ms: float;
}


obj GraphView {
    has key_name: str,
        chord_nodes: int,
        progression_edges: int,
        expanded_nodes: int,
        vocabulary: int;
}


obj SearchView {
    has search_id: int,
        key_name: str,
        start: ChordView,
        nodes: list[ChordView],
        layers: list[LayerView],
        graph: GraphView,
        elapsed_ms: float,
        width: int,
        depth: int,
        options: int,
        # the backward tree: what could come before the first chord (ids from BACK_BASE)
        back_start: ChordView,
        back_nodes: list[ChordView],
        back_layers: list[LayerView],
        back_elapsed_ms: float;
}


obj SavedView {
    has index: int,
        name: str,
        key_name: str,
        steps: list[StepView];
}


obj AltView {
    has cid: str,
        rn: str,
        symbol: str,
        func: str,
        notes: list[int];   # voiced from the chord before (or into the one after)
}


obj StepView {
    has index: int,
        cid: str,
        rn: str,
        symbol: str,
        func: str,
        notes: list[int],
        names: list[str],
        rule: str,
        cadence: str,
        alts: list[AltView] = [];
}


obj SessionView {
    has tonic: str,
        mode: str,
        key_name: str,
        style_texts: list[str],
        style_weights: list[float],
        texture: str,
        drum_grid: str,
        llm_enabled: bool,
        bpm: float,
        beats_per_chord: int,
        steps: list[StepView],
        loop_from: int,
        loop_to: int,
        can_undo: bool,
        saved: list[SavedView],
        notice: str,
        engine_url: str,
        engine_online: bool,
        major_tonics: list[str],
        minor_tonics: list[str];
}


obj OptionResult {
    has nodes: list[ChordView],
        notice: str;
}


obj PathStory {
    has title: str,
        mood: str,
        explanation: str;
}

sem PathStory.title = "An evocative 2-5 word name for the progression, like a chapter title.";
sem PathStory.mood = "One sentence on the emotional character of the progression.";
sem PathStory.explanation = "Two sentences a composer would find useful: why the progression works in terms of harmonic function, voice leading, and how it closes.";


def narrate_path(
    key: str, progression: str, cadence: str, moves: list[str], voice_leading: list[str]
) -> PathStory by llm(temperature=0.6);

sem narrate_path = "You are a warm, precise composition teacher. A composer is auditioning a chord progression; characterise it for them. Only use the facts given.";
sem narrate_path.key = "The key, e.g. 'C major'.";
sem narrate_path.progression = "The progression as Roman numerals with chord symbols.";
sem narrate_path.cadence = "The cadence the path ends with, or 'open' if none.";
sem narrate_path.moves = "The harmonic-syntax rule behind each move.";
sem narrate_path.voice_leading = "Voice-leading observations the analysis engine made.";


# ---------------------------------------------------------------- session helpers

"""Node ids of the backward tree start here, so one id names a node in either tree."""
glob BACK_BASE: int = 100000;

"""How the engine plays inside each chord (audio_engine/engine.py TEXTURES)."""
glob TEXTURES: list[str] = ["block", "arpeggio", "improvise"];

# typed map options get ids from 90000 (forward) / BACK_BASE + 90000 (backward)
glob _NEXT_TYPED: int = 90000,
     _LIVE: list[int] = [];   # ids of search nodes the page shows

glob _SEARCH: dict[int, Hyp] = {},
     _SEARCH_ID: int = 0,
     _SEARCH_KEY: str = "",
     _TYPED: list[str] = [];


def is_back(node_id: int) -> bool {
    return node_id >= BACK_BASE;
}


"""Typed chords live outside the vocabulary, so after a restart they must be
re-analysed before anything asks for them by cid."""
def know_typed(kid: str, steps: list[Step]) {
    for st in steps {
        symbols = [st.symbol] + [a.symbol for a in alts_of(st)];
        for sym in symbols {
            if sym and (kid + sym) not in _TYPED {
                parse_chord(kid, sym);
                _TYPED.append(kid + sym);
            }
        }
    }
}


"""walker.toml: [key, min, max] for each setting; values are clamped to their range."""
glob WALKER_FILE: str = "walker.toml",
     WALKER_KEYS: list[list[any]] = [
        ["width", 2, 40], ["depth", 1, 6], ["options", 2, 14],
        ["max_syntax", 0.5, 6.0], ["syntax_weight", 0.0, 4.0], ["colour", -2.0, 3.0],
        ["cadence_pull", 0.0, 4.0], ["smooth", 0.1, 4.0]
    ];


"""Copy walker.toml onto the session. A missing or broken file, or a missing key,
leaves that setting as it was."""
def walker_config(s: Session) {
    try {
        with open(WALKER_FILE, "rb") as f {
            cfg = tomllib.load(f);
        }
    } except Exception as e {
        print(f"walker.toml not read ({e}); keeping the walker settings as they are");
        return;
    }
    for (k, lo, hi) in WALKER_KEYS {
        if k in cfg {
            v = max(lo, min(hi, cfg[k]));
            setattr(s, k, int(v) if isinstance(lo, int) else float(v));
        }
    }
}


def session -> Session {
    found = [root -->[?:Session]];
    if found {
        s = found[0];
    } else {
        s = (root ++> Session())[0];
    }
    kid = key_id_of(s.tonic, s.mode);
    if not s.steps {
        cid = tonic_cid(kid);
        s.steps = [Step(cid=cid, notes=home_voicing(kid, spec_of(kid, cid)))];
    }
    if s.texture not in TEXTURES {
        s.texture = "block";
    }
    walker_config(s);
    know_typed(kid, s.steps);
    for kept in s.saved {
        know_typed(key_id_of(kept.tonic, kept.mode), kept.steps);
    }
    for past in s.history {
        know_typed(kid, past);
    }
    return s;
}


"""Remember the progression before a change, so undo can restore it."""
def remember(s: Session) {
    s.history = (s.history + [list(s.steps)])[-40:];
}


def key_id(s: Session) -> str {
    return key_id_of(s.tonic, s.mode);
}


"""The loop range the composer set in the Progression bar (-1 where unset). The page
turns it into what the engine plays; the server only remembers it."""
def loop_start(s: Session) -> int {
    return s.loop_from if 0 <= s.loop_from and s.loop_from < len(s.steps) else -1;
}


def loop_end(s: Session) -> int {
    start = loop_start(s);
    return s.loop_to if start >= 0 and start <= s.loop_to and s.loop_to < len(s.steps) else -1;
}


def saved_view(i: int, kept: Saved) -> SavedView {
    kid = key_id_of(kept.tonic, kept.mode);
    return SavedView(
        index=i,
        name=kept.name,
        key_name=key_for(kid).name(),
        steps=[step_view(kid, j, st) for (j, st) in enumerate(kept.steps)]
    );
}


"""Voice chord `c` to stand at `i` in `steps` (led from the chord before, or into
the one after when it is first). None if no voicing works."""
def voice_at(kid: str, steps: list[Step], i: int, c: ChordSpec, smooth: float = 1.0) -> list[int] | None {
    if i > 0 {
        prev = steps[i - 1];
        m = best_motion(kid, spec_of(kid, prev.cid), prev.notes, c, smooth);
        return m.notes if m is not None else None;
    }
    if len(steps) > 1 {
        nxt = steps[1];
        m = best_motion_into(kid, c, spec_of(kid, nxt.cid), nxt.notes, smooth);
        return m.notes if m is not None else None;
    }
    return home_voicing(kid, c) if voicings_for(kid, c) else None;
}


def alt_views(kid: str, steps: list[Step], i: int) -> list[AltView] {
    out: list[AltView] = [];
    for a in alts_of(steps[i]) {
        if not has_spec(kid, a.cid) {
            continue;
        }
        c = spec_of(kid, a.cid);
        notes = voice_at(kid, steps, i, c);
        if notes is not None {
            out.append(AltView(cid=a.cid, rn=c.rn, symbol=c.symbol, func=c.func, notes=notes));
        }
    }
    return out;
}


def step_view(kid: str, i: int, st: Step, alts: list[AltView] = []) -> StepView {
    c = spec_of(kid, st.cid);
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
        alts=alts
    );
}


def session_view(s: Session, notice: str = "") -> SessionView {
    kid = key_id(s);
    return SessionView(
        tonic=s.tonic,
        mode=s.mode,
        key_name=key_for(kid).name(),
        style_texts=s.style_texts,
        style_weights=s.style_weights,
        texture=s.texture,
        drum_grid=s.drum_grid,
        llm_enabled=s.llm_enabled,
        bpm=s.bpm,
        beats_per_chord=s.beats_per_chord,
        steps=[step_view(kid, i, st, alt_views(kid, s.steps, i)) for (i, st) in enumerate(s.steps)],
        loop_from=loop_start(s),
        loop_to=loop_end(s),
        can_undo=len(s.history) > 0 or len(s.steps) > 1,
        saved=[saved_view(i, kept) for (i, kept) in enumerate(s.saved)],
        notice=notice,
        engine_url=ENGINE_URL,
        engine_online=engine_online(),
        major_tonics=MAJOR_TONICS,
        minor_tonics=MINOR_TONICS
    );
}


def mark_views(marks: list[Mark]) -> list[MarkView] {
    return [MarkView(kind=m.kind, text=m.text, cost=r2(m.cost)) for m in marks];
}


def chord_view(kid: str, h: Hyp, status: str) -> ChordView {
    c = spec_of(kid, h.cid);
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
        status=status
    );
}


"""The chain of hypotheses from the search start (exclusive) to `node_id` (inclusive)."""
def chain_to(node_id: int) -> list[Hyp] {
    chain: list[Hyp] = [];
    cur = _SEARCH.get(node_id, None);
    while cur is not None and cur.id != 0 and cur.id != BACK_BASE {
        chain.insert(0, cur);
        cur = _SEARCH.get(cur.parent, None);
    }
    return chain;
}


# ---------------------------------------------------------------- endpoints

def:pub get_state -> SessionView {
    s = session();
    send_style_layers(s.style_texts, s.style_weights);
    settings: dict[str, any] = {"bpm": s.bpm, "beats_per_chord": s.beats_per_chord, "texture": s.texture};
    if s.drum_grid {
        settings["drum_grid"] = json.loads(s.drum_grid);
    }
    send_settings(settings);
    return session_view(s);
}


def:pub set_key(tonic: str, mode: str) -> SessionView {
    s = session();
    s.tonic = tonic;
    s.mode = "minor" if mode == "minor" else "major";
    s.steps = [];
    s.history = [];
    s.loop_from = -1;
    s.loop_to = -1;
    s = session();
    return session_view(s);
}


def:pub set_tempo(bpm: float, beats_per_chord: int) -> SessionView {
    s = session();
    s.bpm = max(40.0, min(180.0, bpm));
    s.beats_per_chord = max(1, min(8, beats_per_chord));
    send_settings({"bpm": s.bpm, "beats_per_chord": s.beats_per_chord});
    return session_view(s);
}


"""How the engine plays inside each chord: block | arpeggio | improvise."""
def:pub set_texture(texture: str) -> SessionView {
    s = session();
    if texture in TEXTURES {
        s.texture = texture;
        send_settings({"texture": texture});
    }
    return session_view(s);
}


"""The Drums page's grid (JSON: mode, bars, per_beat, cells, after_hit, cfg; see
audio_engine/engine.py), sent to the engine as its drum_pianoroll input. Kept only
if the engine accepts it (or is offline: it gets it with the state on reconnect)."""
def:pub set_drums(grid: str) -> SessionView {
    s = session();
    parsed = json.loads(grid);
    if send_settings({"drum_grid": parsed}) or not engine_online() {
        s.drum_grid = json.dumps(parsed);
    }
    return session_view(s);
}


"""Allow or forbid LLM calls. Ollama shares the GPU with Magenta, so turning this
off guarantees the LLM never competes with real-time generation."""
def:pub set_llm(enabled: bool) -> SessionView {
    s = session();
    s.llm_enabled = enabled;
    return session_view(s);
}


"""Set the sound as a blend of MusicCoCa prompts, each with a weight (0..1)."""
def:pub set_style_layers(texts: list[str], weights: list[float]) -> SessionView {
    s = session();
    keep: list[str] = [];
    amounts: list[float] = [];
    for (t, w) in zip(texts, weights) {
        if t.strip() {
            keep.append(t.strip());
            amounts.append(max(0.0, min(1.0, float(w))));
        }
    }
    if keep {
        s.style_texts = keep;
        s.style_weights = amounts;
        send_style_layers(s.style_texts, s.style_weights);
    }
    return session_view(s);
}


def layer_views(searcher: BeamSearch) -> list[LayerView] {
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
    ];
}


def tree_nodes(kid: str, searcher: BeamSearch) -> list[ChordView] {
    kept_ids = [h.id for h in searcher.kept];
    nodes = [chord_view(kid, h, "kept") for h in searcher.kept];
    return nodes + [chord_view(kid, h, "dest") for h in searcher.destinations if h.id not in kept_ids];
}


"""Send two BeamSearch walkers out at once: one forwards from the last chord of the
progression, one backwards in time from the first."""
def:pub explore -> SearchView {
    global _SEARCH, _SEARCH_ID, _SEARCH_KEY;
    s = session();
    kid = key_id(s);
    space = space_for(kid);
    last = s.steps[-1];
    first = s.steps[0];
    fwd_start = Hyp(
        id=0, parent=-1, layer=0, first=-1, cid=last.cid, notes=last.notes,
        path=[st.cid for st in s.steps][-3:], at=chord_in(space, last.cid)
    );
    # backwards, the phrase context is read back in time from the first chord
    back_start = Hyp(
        id=BACK_BASE, parent=-1, layer=0, first=-1, cid=first.cid, notes=first.notes,
        path=[st.cid for st in s.steps[:3]][::-1], at=chord_in(space, first.cid)
    );
    gate = gate_of(space);
    fwd = gate spawn BeamSearch(
        key_id=kid, start=fwd_start, width=s.width, depth=s.depth, options=s.options,
        max_syntax=s.max_syntax, syntax_weight=s.syntax_weight, colour=s.colour, cadence_pull=s.cadence_pull, smooth=s.smooth
    );
    back = gate spawn BeamSearch(
        key_id=kid, start=back_start, width=s.width, depth=s.depth, options=s.options, backward=True, next_id=BACK_BASE + 1,
        max_syntax=s.max_syntax, syntax_weight=s.syntax_weight, colour=s.colour, cadence_pull=s.cadence_pull, smooth=s.smooth
    );
    _SEARCH_ID += 1;
    _SEARCH_KEY = kid;
    _SEARCH = {0: fwd_start, BACK_BASE: back_start};
    _LIVE.clear();
    for h in fwd.kept + fwd.destinations + back.kept + back.destinations {
        _LIVE.append(h.id);
    }
    for h in fwd.kept + fwd.destinations + back.kept + back.destinations {
        _SEARCH[h.id] = h;
    }
    st = graph_stats(space);
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
    );
}


"""The other chords the graph offered beside `h` (same parent): kept with the step so
it can be swapped later."""
def siblings(h: Hyp) -> list[Alt] {
    out: list[Alt] = [];
    seen: list[str] = [h.cid];
    for nid in _LIVE {
        g = _SEARCH.get(nid, None);
        if g is not None and g.parent == h.parent and g.cid not in seen {
            seen.append(g.cid);
            out.append(Alt(cid=g.cid, symbol=g.symbol));
        }
    }
    return out[:12];
}


"""Keep the path to `node_id` in the progression (whole=True) or only its first chord."""
def:pub add_path(search_id: int, node_id: int, whole: bool) -> SessionView {
    s = session();
    if search_id == _SEARCH_ID and node_id in _SEARCH and chain_to(node_id) {
        chain = chain_to(node_id);
        if not whole {
            chain = chain[:1];
        }
        remember(s);
        if is_back(node_id) {
            # chain runs back in time from the first chord; each hypothesis carries the
            # move *out of* it, which belongs to the step after it.
            ahead = [Step(cid=h.cid, notes=h.notes, symbol=h.symbol, alts=siblings(h)) for h in chain[::-1]];
            later = ahead[1:] + [s.steps[0]];
            moved = [restep(st, h.rule, h.cadence) for (st, h) in zip(later, chain[::-1])];
            s.steps = ahead[:1] + moved + s.steps[1:];
            if s.loop_from >= 0 {
                s.loop_from += len(chain);
                s.loop_to = s.loop_to + len(chain) if s.loop_to >= 0 else -1;
            }
        } else {
            s.steps = s.steps + [
                Step(cid=h.cid, notes=h.notes, rule=h.rule, cadence=h.cadence, symbol=h.symbol, alts=siblings(h))
                for h in chain
            ];
        }
        s.commits += 1;
    }
    return session_view(s);
}


def:pub undo -> SessionView {
    s = session();
    s.loop_from = -1;
    s.loop_to = -1;
    if s.history {
        s.steps = s.history[-1];
        s.history = s.history[:-1];
    } elif len(s.steps) > 1 {
        s.steps = s.steps[:-1];
    }
    return session_view(s);
}


def:pub clear_progression -> SessionView {
    s = session();
    remember(s);
    s.steps = [];
    s.loop_from = -1;
    s.loop_to = -1;
    s = session();
    return session_view(s);
}


"""Ask the LLM (byllm) to characterise the path to a node of the current search."""
def:pub describe(search_id: int, node_id: int) -> PathStory {
    s = session();
    if not s.llm_enabled {
        return PathStory(title="", mood="", explanation="LLM calls are switched off (so Magenta keeps the GPU).");
    }
    if search_id != _SEARCH_ID or node_id not in _SEARCH {
        return PathStory(title="", mood="", explanation="That search is no longer current.");
    }
    kid = _SEARCH_KEY;
    chain = chain_to(node_id);
    anchor = s.steps[0] if is_back(node_id) else s.steps[-1];
    cids = [h.cid for h in chain];
    cids = cids[::-1] + [anchor.cid] if is_back(node_id) else [anchor.cid] + cids;
    progression = " → ".join([f"{spec_of(kid, c).rn} ({spec_of(kid, c).symbol})" for c in cids]);
    moves = [h.rule for h in chain][::-1] if is_back(node_id) else [h.rule for h in chain];
    notes: list[str] = [];
    for h in chain {
        notes = notes + [m.text for m in h.marks if m.kind != "good" or "resolves" in m.text];
    }
    cad = (chain[0].cadence if is_back(node_id) else chain[-1].cadence) if chain else "";
    try {
        return narrate_path(key_for(kid).name(), progression, CADENCE_NAMES.get(cad, "open"), moves, notes[:12]);
    } except Exception as e {
        return PathStory(title="", mood="", explanation=f"The language model is unavailable ({str(e)[:160]}).");
    }
}


"""Split typed input like "Gm/D -> Cm/Eb, Gm/Bb  D/A" into chord symbols."""
def split_symbols(text: str) -> list[str] {
    flat = text.replace("→", " ").replace("->", " ").replace(",", " ").replace("|", " ").replace(";", " ");
    return [t for t in flat.split() if t and t != "-"];
}


def describe_move(key_id: str, a: ChordSpec, b: ChordSpec) -> str {
    rule = syntax(a, b, key_for(key_id));
    return rule.label if rule is not None else f"typed: {a.rn} → {b.rn} is outside the walkers' rules";
}


def voiced_after(kid: str, prev: Step, c: ChordSpec) -> list[int] {
    m = best_motion(kid, spec_of(kid, prev.cid), prev.notes, c);
    if m is None {
        raise ValueError(f"{c.symbol} has no playable four-voice voicing in this inversion");
    }
    return m.notes;
}


"""Add typed chords to the progression: "after" the last chord, "before" the first,
or "replace" the whole progression. Each chord is analysed in the key and voice-led
from (or, before, into) its neighbour, so the walkers can carry on from it."""
def:pub add_chords(text: str, where: str) -> SessionView {
    s = session();
    kid = key_id(s);
    key = key_for(kid);
    symbols = split_symbols(text);
    if not symbols {
        return session_view(s, "type one or more chord symbols, e.g. Gm/D Cm/Eb D/A");
    }
    specs: list[ChordSpec] = [];
    try {
        for sym in symbols {
            specs.append(parse_chord(kid, sym));
        }
    } except ValueError as e {
        return session_view(s, str(e));
    }
    for sym in symbols {
        if (kid + sym) not in _TYPED {
            _TYPED.append(kid + sym);
        }
    }
    try {
        steps: list[Step] = list(s.steps);
        if where == "before" {
            # insert last-typed first, so each chord is voiced into the one after it
            for i in range(len(specs) - 1, -1, -1) {
                c = specs[i];
                nxt = steps[0];
                nspec = spec_of(kid, nxt.cid);
                m = best_motion_into(kid, c, nspec, nxt.notes);
                if m is None {
                    raise ValueError(f"{c.symbol} has no playable four-voice voicing in this inversion");
                }
                steps[0] = restep(nxt, describe_move(kid, c, nspec), cadence(key, c, nspec, nxt.notes));
                steps.insert(0, Step(cid=c.cid, notes=m.notes, symbol=symbols[i]));
            }
        } else {
            start = 0;
            if where == "replace" {
                c0 = specs[0];
                if not voicings_for(kid, c0) {
                    raise ValueError(f"{c0.symbol} has no playable four-voice voicing in this inversion");
                }
                steps = [Step(cid=c0.cid, notes=home_voicing(kid, c0), symbol=symbols[0])];
                start = 1;
            }
            for i in range(start, len(specs)) {
                c = specs[i];
                prev = steps[-1];
                pspec = spec_of(kid, prev.cid);
                notes = voiced_after(kid, prev, c);
                steps.append(
                    Step(cid=c.cid, notes=notes, rule=describe_move(kid, pspec, c), cadence=cadence(key, pspec, c, notes), symbol=symbols[i])
                );
            }
        }
    } except ValueError as e {
        return session_view(s, str(e));
    }
    remember(s);
    if where == "replace" {
        s.loop_from = -1;
        s.loop_to = -1;
    } elif where == "before" and s.loop_from >= 0 {
        s.loop_from += len(steps) - len(s.steps);
        s.loop_to = s.loop_to + len(steps) - len(s.steps) if s.loop_to >= 0 else -1;
    }
    s.steps = steps;
    return session_view(s);
}


"""Restart the audio engine process (for when the audio glitches). The client calls
get_state once the engine is back, which re-sends the sound, settings and chord."""
def:pub restart_engine -> bool {
    return relaunch_engine();
}


"""Quit SuperConductor Atlas from the page: ask the audio engine to exit and wait for it,
then ask this app server to exit. Nothing is killed: the engine winds down through
its own /shutdown, and the server through uvicorn's own exit path (`handle_exit`,
which jac-scale wraps to drain in-flight requests first). The server stops a moment
after this reply, so the page receives it."""
def:pub shutdown_app -> dict[str, any] {
    engine_stopped = stop_engine();
    threading.Timer(0.4, _stop_server).start();
    return {"engine_stopped": engine_stopped};
}


"""Tell this process's uvicorn server to shut down gracefully, as Ctrl-C would."""
def _stop_server -> None {
    servers = [o for o in gc.get_objects() if isinstance(o, uvicorn.Server)];
    for srv in servers {
        srv.handle_exit(signal.SIGINT, None);
    }
    if not servers {
        # no server object to ask (not running under uvicorn): deliver the same
        # request to this process only, which runs its own SIGINT handler
        signal.raise_signal(signal.SIGINT);
    }
}


"""Set the loop range: steps `start`..`end` (end -1: to the last chord; start -1:
the default, just the last chord). The Progression bar decides what clicks mean."""
def:pub set_loop(start: int, end: int) -> SessionView {
    s = session();
    n = len(s.steps);
    s.loop_from = start if 0 <= start and start < n else -1;
    s.loop_to = end if s.loop_from >= 0 and s.loop_from <= end and end < n else -1;
    return session_view(s);
}


"""Type a chord into a column of the map. It joins the tree once for every chord in
the previous column (`parent_ids`) that may legally move to it - forwards, parent →
chord; backwards, chord → parent - voice-led and scored like the walker's own.

Then a mini walker searches on from the typed chord through the remaining columns
(status "grown"). The client draws a found chord that is already in its column as
one more arrow into the existing box, and anything else as a new entry."""
def:pub add_option(search_id: int, parent_ids: list[int], text: str) -> OptionResult {
    global _NEXT_TYPED;
    if search_id != _SEARCH_ID {
        return OptionResult(nodes=[], notice="that search is no longer current");
    }
    kid = _SEARCH_KEY;
    key = key_for(kid);
    tune = session();
    symbol = text.strip();
    try {
        c = parse_chord(kid, symbol);
    } except ValueError as e {
        return OptionResult(nodes=[], notice=str(e));
    }
    if (kid + symbol) not in _TYPED {
        _TYPED.append(kid + symbol);
    }
    made: list[ChordView] = [];
    for pid in parent_ids {
        p = _SEARCH.get(pid, None);
        if p is None {
            continue;
        }
        ps = spec_of(kid, p.cid);
        back = is_back(pid);
        rule = syntax(c, ps, key) if back else syntax(ps, c, key);
        if rule is None {
            continue;
        }
        m = best_motion_into(kid, c, ps, p.notes, tune.smooth) if back else best_motion(kid, ps, p.notes, c, tune.smooth);
        if m is None {
            continue;
        }
        cad = cadence(key, c, ps, p.notes) if back else cadence(key, ps, c, m.notes);
        (pc, pmarks) = phrase_cost(kid, p.path, c, cad, back, tune.colour, tune.cadence_pull);
        step = max(0.05, tune.syntax_weight * rule.cost + m.cost + pc);
        _NEXT_TYPED += 1;
        nid = _NEXT_TYPED + (BACK_BASE if back else 0);
        h = Hyp(
            id=nid, parent=pid, layer=p.layer + 1, first=p.first if p.layer > 0 else nid,
            cid=c.cid, notes=m.notes, path=p.path + [c.cid], syntax_cost=rule.cost, vl_cost=m.cost,
            phrase_cost=pc, step_cost=step, total=p.total + step, rule=rule.label,
            marks=m.marks + pmarks, cadence=cad, symbol=symbol
        );
        _SEARCH[nid] = h;
        _LIVE.append(nid);
        made.append(chord_view(kid, h, "custom"));
    }
    if not made {
        return OptionResult(nodes=[], notice=f"no chord in the previous column can move to {c.symbol} ({c.rn}) under the walkers' rules");
    }
    return OptionResult(nodes=made + grow_from(_SEARCH[min(made, key=lambda v: ChordView : v.total).id]), notice="");
}


"""Mini walker: a BeamSearch from one typed chord over the columns left after it."""
def grow_from(h: Hyp) -> list[ChordView] {
    global _NEXT_TYPED;
    s = session();
    kid = _SEARCH_KEY;
    remaining = s.depth - h.layer;
    if remaining < 1 {
        return [];
    }
    back = is_back(h.id);
    space = space_for(kid);
    root_hyp = Hyp(
        id=h.id, parent=h.parent, layer=0, first=-1, cid=h.cid, notes=h.notes, path=h.path,
        total=h.total, at=chord_in(space, h.cid)
    );
    base_id = _NEXT_TYPED + 1 + (BACK_BASE if back else 0);
    w = gate_of(space) spawn BeamSearch(
        key_id=kid, start=root_hyp, width=s.width, depth=remaining, options=s.options,
        backward=back, next_id=base_id,
        max_syntax=s.max_syntax, syntax_weight=s.syntax_weight, colour=s.colour, cadence_pull=s.cadence_pull, smooth=s.smooth
    );
    kept_ids = [g.id for g in w.kept];
    found = w.kept + [g for g in w.destinations if g.id not in kept_ids];
    out: list[ChordView] = [];
    for g in found {
        g.layer += h.layer;   # the walker counts layers from the typed chord
        _SEARCH[g.id] = g;
        _LIVE.append(g.id);
        out.append(chord_view(kid, g, "grown"));
    }
    _NEXT_TYPED = w.next_id - (BACK_BASE if back else 0);
    return out;
}


"""Commit the progression: keep it (named, in its key) and start a fresh one."""
def:pub commit_progression(name: str) -> SessionView {
    s = session();
    label = name.strip() if name.strip() else f"Progression {len(s.saved) + 1}";
    s.saved = s.saved + [Saved(name=label, tonic=s.tonic, mode=s.mode, steps=list(s.steps))];
    s.steps = [];
    s.history = [];
    s.loop_from = -1;
    s.loop_to = -1;
    s = session();
    return session_view(s);
}


"""Forget a committed progression."""
def:pub delete_saved(index: int) -> SessionView {
    s = session();
    if 0 <= index and index < len(s.saved) {
        s.saved = [kept for (i, kept) in enumerate(s.saved) if i != index];
    }
    return session_view(s);
}


"""Make a committed progression current again. The current one is committed in its
place first, unless it is still just the opening tonic."""
def:pub resume_saved(index: int) -> SessionView {
    s = session();
    if index < 0 or index >= len(s.saved) {
        return session_view(s);
    }
    kept = s.saved[index];
    rest = [k for (i, k) in enumerate(s.saved) if i != index];
    if len(s.steps) > 1 {
        rest = rest + [Saved(name=f"Progression {len(s.saved) + 1}", tonic=s.tonic, mode=s.mode, steps=list(s.steps))];
    }
    s.saved = rest;
    s.tonic = kept.tonic;
    s.mode = kept.mode;
    s.steps = list(kept.steps);
    s.history = [];
    s.loop_from = -1;
    s.loop_to = -1;
    return session_view(s);
}


"""Swap the kept chord at `index` for one of its alternatives. The chords after it are
re-voiced so the voice leading stays smooth; the old chord becomes an alternative."""
def:pub swap_step(index: int, cid: str) -> SessionView {
    s = session();
    kid = key_id(s);
    key = key_for(kid);
    if index < 0 or index >= len(s.steps) or not has_spec(kid, cid) {
        return session_view(s);
    }
    old = s.steps[index];
    picked = [a for a in alts_of(old) if a.cid == cid];
    if not picked {
        return session_view(s, f"{cid} was not one of the options at step {index + 1}");
    }
    c = spec_of(kid, cid);
    steps = list(s.steps);
    notes = voice_at(kid, steps, index, c);
    if notes is None {
        return session_view(s, f"{c.symbol} has no playable voicing here");
    }
    alts = [a for a in alts_of(old) if a.cid != cid] + [Alt(cid=old.cid, symbol=old.symbol)];
    rule = "";
    cad = "";
    if index > 0 {
        before = spec_of(kid, steps[index - 1].cid);
        rule = describe_move(kid, before, c);
        cad = cadence(key, before, c, notes);
    }
    steps[index] = Step(cid=cid, notes=notes, rule=rule, cadence=cad, symbol=picked[0].symbol, alts=alts);
    for j in range(index + 1, len(steps)) {
        led = steps[j - 1];
        ps = spec_of(kid, led.cid);
        st = steps[j];
        cs = spec_of(kid, st.cid);
        m = best_motion(kid, ps, led.notes, cs);
        vb = m.notes if m is not None else st.notes;
        steps[j] = Step(cid=st.cid, notes=vb, rule=describe_move(kid, ps, cs), cadence=cadence(key, ps, cs, vb), symbol=st.symbol, alts=alts_of(st));
    }
    remember(s);
    s.steps = steps;
    return session_view(s);
}
