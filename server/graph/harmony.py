"""The harmony graph: one `HarmonySpace` per key, holding `Chord` nodes joined by
`Progression` edges whose weights come from the functional-syntax rules.
The graph is grown lazily: a Chord's outgoing progressions are only materialised
when a walker first arrives there (`expand`), and its incoming ones only when a
walker searching backwards in time arrives (`expand_back`). Nobody ever enumerates
the full space up front - it grows exactly as far as the composers explore.

The graph lives in memory: it follows from the rules, so each run regrows it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from server.theory.vocab import ChordSpec, vocab_for, spec_of, key_for, tonic_cid
from server.theory.syntax import syntax


@dataclass(eq=False)
class HarmonySpace:
    key_id: str
    expansions: int = 0
    # the chords it contains, by cid, in the order they joined
    chords: dict[str, Chord] = field(default_factory=dict)

    def add(self, chord: Chord) -> Chord:
        chord.space = self
        self.chords[chord.cid] = chord
        return chord


@dataclass(eq=False)
class Chord:
    cid: str
    rn: str
    symbol: str
    func: str
    family: str
    expanded: bool = False
    back_expanded: bool = False
    visits: int = 0
    space: HarmonySpace | None = None
    # Progression edges out of and into this chord, in the order they were laid
    out: list[Progression] = field(default_factory=list)
    inc: list[Progression] = field(default_factory=list)


@dataclass(eq=False)
class Progression:
    src: Chord
    dst: Chord
    to_cid: str
    cost: float
    rule: str


def connect(a: Chord, b: Chord, cost: float, rule: str) -> Progression:
    e = Progression(src=a, dst=b, to_cid=b.cid, cost=cost, rule=rule)
    a.out.append(e)
    b.inc.append(e)
    return e


def chord_node_for(spec: ChordSpec) -> Chord:
    return Chord(cid=spec.cid, rn=spec.rn, symbol=spec.symbol, func=spec.func, family=spec.family)


def expand(here: Chord) -> int:
    """Materialise a Chord's legal successors as Progression edges, on first arrival.
    Returns the number of edges laid."""
    if here.expanded:
        return 0
    created = 0
    space = here.space
    key = key_for(space.key_id)
    a = spec_of(space.key_id, here.cid)
    existing = dict(space.chords)
    # expand_back may already have laid some of these edges.
    have = [e.to_cid for e in here.out]
    for b in vocab_for(space.key_id):
        if b.cid in have:
            continue
        rule = syntax(a, b, key)
        if rule is None:
            continue
        if b.cid not in existing:
            existing[b.cid] = space.add(chord_node_for(b))
        connect(here, existing[b.cid], rule.cost, rule.label)
        created += 1
    here.expanded = True
    space.expansions += 1
    return created


def expand_back(here: Chord) -> int:
    """Materialise a Chord's legal predecessors as Progression edges *into* it, on the
    first arrival of a backward search. Same edges, same syntax rules: the graph is one
    graph, it is only read in the other direction. Returns the number of edges laid."""
    if here.back_expanded:
        return 0
    created = 0
    space = here.space
    key = key_for(space.key_id)
    b = spec_of(space.key_id, here.cid)
    existing = dict(space.chords)
    have = [e.src.cid for e in here.inc]
    for a in vocab_for(space.key_id):
        if a.cid in have:
            continue
        rule = syntax(a, b, key)
        if rule is None:
            continue
        if a.cid not in existing:
            existing[a.cid] = space.add(chord_node_for(a))
        connect(existing[a.cid], here, rule.cost, rule.label)
        created += 1
    here.back_expanded = True
    space.expansions += 1
    return created


_SPACES: dict[str, HarmonySpace] = {}


def space_for(key_id: str) -> HarmonySpace:
    if key_id not in _SPACES:
        space = HarmonySpace(key_id=key_id)
        space.add(chord_node_for(spec_of(key_id, tonic_cid(key_id))))
        _SPACES[key_id] = space
    return _SPACES[key_id]


def chord_in(space: HarmonySpace, cid: str) -> Chord:
    if cid in space.chords:
        return space.chords[cid]
    return space.add(chord_node_for(spec_of(space.key_id, cid)))


@dataclass(eq=False)
class GraphStats:
    key_id: str
    chord_nodes: int
    progression_edges: int
    expanded_nodes: int
    vocabulary: int


def graph_stats(space: HarmonySpace) -> GraphStats:
    chords = list(space.chords.values())
    edges = 0
    expanded = 0
    for c in chords:
        if c.expanded:
            expanded += 1
            edges += len(c.out)
    return GraphStats(
        key_id=space.key_id,
        chord_nodes=len(chords),
        progression_edges=edges,
        expanded_nodes=expanded,
        vocabulary=len(vocab_for(space.key_id)),
    )
