"""Beam search over the harmony graph, performed by a walker.

Why a walker and not a shortest-path call: the cost of a progression edge is
not static. The syntax part is (it lives on the edge), but the voice-leading
part depends on the exact SATB voicing the path *arrives with*, which depends
on every choice before it. So each hypothesis carries its own voicing, and the
search has to physically stand on each chord to extend the paths that end there.

The walker moves in lock-step layers:

    gate ──visit──▶ every chord on the frontier   (extend each path along every
      ▲                     │                      outgoing Progression edge,
      └─────────visit───────┘                      voice-leading it from the
                                                   path's current voicing)
    back at the gate: prune candidates to the beam, record the layer, and
    queue the next frontier + the gate again.

With `backward` set, the same walker searches back in time from the first chord
of the progression: it follows Progression edges against their direction (grown
by `ExpandBack`), and voice-leads each predecessor *into* the voicing the path
already has. Layer 1 is then the menu of chords that could come just before.

Layer 1 is special: it is the composer's menu of immediate options - the best
inversion of each distinct harmony, ranked. Deeper layers use a *diverse* beam:
the best continuation of every option always survives (so every option shows
a future), then remaining slots go to the globally best paths.
"""

import time;
import from theory.vocab { spec_of, key_for, has_spec }
import from theory.voicing { best_motion, best_motion_into, Mark }
import from theory.cadence { cadence }
import from theory.phrase { phrase_cost }
import from graph.harmony { Chord, Progression, BeamGate, Expand, ExpandBack }


obj Hyp {
    has id: int,
        parent: int,
        layer: int,
        first: int,
        cid: str,
        notes: list[int],
        path: list[str] = [],
        syntax_cost: float = 0.0,
        vl_cost: float = 0.0,
        phrase_cost: float = 0.0,
        step_cost: float = 0.0,
        total: float = 0.0,
        rule: str = "",
        marks: list[Mark] = [],
        cadence: str = "",
        symbol: str = "",   # set when the composer typed this chord into the map
        at: Chord | None = None;
}


obj LayerStat {
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


glob CATEGORY: dict[str, str] = {
    "T": "tonic", "TX": "tonic", "PD": "pre-dominant", "NEAP": "pre-dominant",
    "D": "dominant", "CAD64": "dominant", "APP": "applied", "MIX": "colour", "AUG6": "colour"
};


walker BeamSearch {
    has key_id: str,
        start: Hyp,
        width: int = 12,
        depth: int = 4,
        options: int = 8,
        backward: bool = False,
        syntax_weight: float = 1.5,
        max_syntax: float = 3.0,
        colour: float = 1.0,
        cadence_pull: float = 1.0,
        smooth: float = 1.0,
        frontier: list[Hyp] = [],
        candidates: list[Hyp] = [],
        kept: list[Hyp] = [],
        destinations: list[Hyp] = [],
        layers: list[LayerStat] = [],
        layer: int = 0,
        next_id: int = 1,
        passes: int = 0,
        elapsed_ms: float = 0.0,
        _visited: int = 0,
        _edges: int = 0,
        _scored: int = 0,
        _expanded: int = 0,
        _t_start: float = 0.0,
        _t_layer: float = 0.0;

    can extend with Chord entry {
        grower = (here spawn ExpandBack()) if self.backward else (here spawn Expand());
        if grower.created_edges > 0 {
            self._expanded += 1;
        }
        here.visits += 1;
        self._visited += 1;
        key = key_for(self.key_id);
        here_spec = spec_of(self.key_id, here.cid);
        # (edge, the chord at its other end) for every move out of here, in search direction
        moves: list[tuple[Progression, Chord]] = [];
        if self.backward {
            for p in [here <-:Progression:<-] {
                for e in [edge p ->:Progression:->] {
                    if e.to_cid == here.cid {
                        moves.append((e, p));
                    }
                }
            }
        } else {
            succ = {n.cid: n for n in [here ->:Progression:->]};
            for e in [edge here ->:Progression:->] {
                moves.append((e, succ[e.to_cid]));
            }
        }
        for h in self.frontier {
            if h.cid != here.cid {
                continue;
            }
            for (e, there) in moves {
                if e.cost > self.max_syntax or not has_spec(self.key_id, there.cid) {
                    continue;
                }
                self._edges += 1;
                other = spec_of(self.key_id, there.cid);
                if self.backward {
                    m = best_motion_into(self.key_id, other, here_spec, h.notes, self.smooth);
                } else {
                    m = best_motion(self.key_id, here_spec, h.notes, other, self.smooth);
                }
                self._scored += 1;
                if m is None {
                    continue;
                }
                # cadences are judged in time order: earlier chord -> later chord (later voicing)
                if self.backward {
                    cad = cadence(key, other, here_spec, h.notes);
                } else {
                    cad = cadence(key, here_spec, other, m.notes);
                }
                (pc, pmarks) = phrase_cost(self.key_id, h.path, other, cad, self.backward, self.colour, self.cadence_pull);
                step = max(0.05, self.syntax_weight * e.cost + m.cost + pc);
                self.candidates.append(
                    Hyp(
                        id=self.next_id,
                        parent=h.id,
                        layer=self.layer + 1,
                        first=h.first,
                        cid=other.cid,
                        notes=m.notes,
                        path=h.path + [other.cid],
                        syntax_cost=e.cost,
                        vl_cost=m.cost,
                        phrase_cost=pc,
                        step_cost=step,
                        total=h.total + step,
                        rule=e.rule,
                        marks=m.marks + pmarks,
                        cadence=cad,
                        at=there
                    )
                );
                self.next_id += 1;
            }
        }
    }

    can checkpoint with BeamGate entry {
        now = time.perf_counter();
        if self.passes == 0 {
            self._t_start = now;
            self.frontier = [self.start];
        } else {
            self.prune((now - self._t_layer) * 1000.0);
        }
        self.passes += 1;
        if self.layer >= self.depth or not self.frontier {
            self.elapsed_ms = (time.perf_counter() - self._t_start) * 1000.0;
            disengage;
        }
        self._visited = 0;
        self._edges = 0;
        self._scored = 0;
        self._expanded = 0;
        self._t_layer = time.perf_counter();
        stops: list[Chord] = [];
        seen: list[str] = [];
        for h in self.frontier {
            if h.cid not in seen and h.at is not None {
                seen.append(h.cid);
                stops.append(h.at);
            }
        }
        visit stops;
        visit here;
    }

    def prune(ms: float) {
        ranked = sorted(self.candidates, key=lambda h: Hyp : h.total);
        keep: list[Hyp] = [];
        if self.layer == 0 {
            # The composer's menu: best inversion of each distinct harmony, with the
            # best option from every functional category guaranteed a seat.
            by_harmony: list[Hyp] = [];
            seen_harmony: list[str] = [];
            for h in ranked {
                harmony = spec_of(self.key_id, h.cid).harmony;
                if harmony not in seen_harmony {
                    seen_harmony.append(harmony);
                    by_harmony.append(h);
                }
            }
            seen_cat: list[str] = [];
            for h in by_harmony {
                cat = CATEGORY.get(spec_of(self.key_id, h.cid).func, "other");
                if cat not in seen_cat and len(keep) < self.options {
                    seen_cat.append(cat);
                    keep.append(h);
                }
            }
            chosen = [h.id for h in keep];
            for h in by_harmony {
                if len(keep) >= self.options {
                    break;
                }
                if h.id not in chosen {
                    keep.append(h);
                    chosen.append(h.id);
                }
            }
            keep = sorted(keep, key=lambda h: Hyp : h.total);
            for h in keep {
                h.first = h.id;
            }
        } else {
            has_future: list[int] = [];
            for h in ranked {
                if h.first not in has_future {
                    has_future.append(h.first);
                    keep.append(h);
                }
            }
            kept_ids = [h.id for h in keep];
            for h in ranked {
                if len(keep) >= self.width {
                    break;
                }
                if h.id not in kept_ids {
                    keep.append(h);
                    kept_ids.append(h.id);
                }
            }
        }
        kept_ids = [h.id for h in keep];
        # Destinations: past the first layer, the best path reaching each kind of
        # cadence at this depth is recorded even if the beam prunes it.
        new_dests: list[Hyp] = [];
        if self.layer > 0 {
            found: list[str] = [];
            for h in ranked {
                if h.cadence and h.cadence not in found {
                    found.append(h.cadence);
                    new_dests.append(h);
                }
            }
        }
        self.destinations.extend(new_dests);
        dest_ids = [h.id for h in new_dests];
        pruned = [h for h in ranked if h.id not in kept_ids and h.id not in dest_ids];
        self.layers.append(
            LayerStat(
                layer=self.layer + 1,
                frontier=[h.id for h in self.frontier],
                frontier_rns=[spec_of(self.key_id, h.cid).rn for h in self.frontier],
                nodes_visited=self._visited,
                edges_followed=self._edges,
                voicings_scored=self._scored,
                newly_expanded=self._expanded,
                candidates=len(ranked),
                kept=len(keep),
                pruned=len(pruned),
                cadences=len(new_dests),
                ms=ms
            )
        );
        self.kept.extend(keep);
        self.frontier = keep;
        self.candidates = [];
        self.layer += 1;
    }
}
