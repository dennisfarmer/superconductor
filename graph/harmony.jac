"""The harmony graph: one `HarmonySpace` per key, holding `Chord` nodes joined by
`Progression` edges whose weights come from the functional-syntax rules.

The graph is grown lazily: a Chord's outgoing progressions are only materialised
when a walker first arrives there (the `Expand` walker), and its incoming ones only
when a walker searching backwards in time arrives (`ExpandBack`). Nobody ever
enumerates the full space up front - it grows exactly as far as the composers explore.
"""

import from theory.vocab { ChordSpec, vocab_for, spec_of, key_for, tonic_cid }
import from theory.syntax { syntax }


node HarmonySpace {
    has key_id: str,
        expansions: int = 0;
}


node Chord {
    has cid: str,
        rn: str,
        symbol: str,
        func: str,
        family: str,
        expanded: bool = False,
        back_expanded: bool = False,
        visits: int = 0;
}


edge Contains: HarmonySpace --> Chord {}


edge Progression: Chord --> Chord {
    has to_cid: str,
        cost: float,
        rule: str;
}


"""Hub a BeamSearch walker passes through between layers (see graph.beam)."""
node BeamGate {}


edge HasGate: HarmonySpace --> BeamGate {}


def chord_node_for(spec: ChordSpec) -> Chord {
    return Chord(cid=spec.cid, rn=spec.rn, symbol=spec.symbol, func=spec.func, family=spec.family);
}


"""Materialise a Chord's legal successors as Progression edges, on first arrival."""
walker Expand {
    has created_nodes: int = 0,
        created_edges: int = 0;

    can grow with Chord entry {
        if here.expanded {
            disengage;
        }
        spaces = [here <-:Contains:<-];
        space = spaces[0];
        key = key_for(space.key_id);
        a = spec_of(space.key_id, here.cid);
        existing = {c.cid: c for c in [space ->:Contains:->]};
        # ExpandBack may already have laid some of these edges.
        have = [e.to_cid for e in [edge here ->:Progression:->]];
        for b in vocab_for(space.key_id) {
            if b.cid in have {
                continue;
            }
            rule = syntax(a, b, key);
            if rule is None {
                continue;
            }
            if b.cid not in existing {
                fresh = chord_node_for(b);
                space +>:Contains:+> fresh;
                existing[b.cid] = fresh;
                self.created_nodes += 1;
            }
            here +>:Progression(to_cid=b.cid, cost=rule.cost, rule=rule.label):+> existing[b.cid];
            self.created_edges += 1;
        }
        here.expanded = True;
        space.expansions += 1;
    }
}


"""Materialise a Chord's legal predecessors as Progression edges *into* it, on the
first arrival of a backward search. Same edges, same syntax rules: the graph is one
graph, it is only read in the other direction."""
walker ExpandBack {
    has created_nodes: int = 0,
        created_edges: int = 0;

    can grow with Chord entry {
        if here.back_expanded {
            disengage;
        }
        spaces = [here <-:Contains:<-];
        space = spaces[0];
        key = key_for(space.key_id);
        b = spec_of(space.key_id, here.cid);
        existing = {c.cid: c for c in [space ->:Contains:->]};
        have = [p.cid for p in [here <-:Progression:<-]];
        for a in vocab_for(space.key_id) {
            if a.cid in have {
                continue;
            }
            rule = syntax(a, b, key);
            if rule is None {
                continue;
            }
            if a.cid not in existing {
                fresh = chord_node_for(a);
                space +>:Contains:+> fresh;
                existing[a.cid] = fresh;
                self.created_nodes += 1;
            }
            existing[a.cid] +>:Progression(to_cid=here.cid, cost=rule.cost, rule=rule.label):+> here;
            self.created_edges += 1;
        }
        here.back_expanded = True;
        space.expansions += 1;
    }
}


def space_for(key_id: str) -> HarmonySpace {
    for s in [root -->[?:HarmonySpace]] {
        if s.key_id == key_id {
            return s;
        }
    }
    space = HarmonySpace(key_id=key_id);
    root ++> space;
    space +>:Contains:+> chord_node_for(spec_of(key_id, tonic_cid(key_id)));
    space +>:HasGate:+> BeamGate();
    return space;
}


def chord_in(space: HarmonySpace, cid: str) -> Chord {
    for c in [space ->:Contains:->] {
        if c.cid == cid {
            return c;
        }
    }
    fresh = chord_node_for(spec_of(space.key_id, cid));
    space +>:Contains:+> fresh;
    return fresh;
}


def gate_of(space: HarmonySpace) -> BeamGate {
    return [space ->:HasGate:->][0];
}


obj GraphStats {
    has key_id: str,
        chord_nodes: int,
        progression_edges: int,
        expanded_nodes: int,
        vocabulary: int;
}


def graph_stats(space: HarmonySpace) -> GraphStats {
    chords = [space ->:Contains:->];
    edges = 0;
    expanded = 0;
    for c in chords {
        if c.expanded {
            expanded += 1;
            edges += len([edge c ->:Progression:->]);
        }
    }
    return GraphStats(
        key_id=space.key_id,
        chord_nodes=len(chords),
        progression_edges=edges,
        expanded_nodes=expanded,
        vocabulary=len(vocab_for(space.key_id))
    );
}
