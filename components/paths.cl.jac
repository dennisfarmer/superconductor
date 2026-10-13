"""Path helpers shared by the shell and the map: walking a node's chain back to its
tree's start, and the half-commit locks.

A lock pins one chord in one column of one tree. A node is *reachable* when, for
every locked column before it in the same tree, its path passes through the
locked chord. Unreachable nodes are dimmed: hovering them would play a path the
composer has already ruled out.
"""

sv import from ..services.atlas { SearchView, ChordView }

# Ids of the backward search tree start here (services.atlas.BACK_BASE).
glob BACK_BASE: int = 100000;


"""Every node of both trees, both starts, and the typed options, by id."""
def:pub node_map(search: SearchView, typed: list[ChordView]) -> dict[int, ChordView] {
    by_id: dict[int, ChordView] = {0: search.start, BACK_BASE: search.back_start};
    for n in search.nodes {
        by_id[n.id] = n;
    }
    for n in search.back_nodes {
        by_id[n.id] = n;
    }
    for n in typed {
        by_id[n.id] = n;
    }
    return by_id;
}


def:pub is_back_id(node_id: int) -> bool {
    return node_id >= BACK_BASE;
}


"""Nodes from the tree's start (exclusive) to `node_id` (inclusive). For the backward
tree this runs back in time from the first chord."""
def:pub chain_in(by_id: dict[int, ChordView], node_id: int) -> list[ChordView] {
    chain: list[ChordView] = [];
    cur = node_id;
    while cur in by_id and by_id[cur].layer > 0 {
        chain.insert(0, by_id[cur]);
        cur = by_id[cur].parent;
    }
    return chain;
}


"""The id of `node_id`'s ancestor on `layer` (itself if it is on that layer), or -1."""
def:pub ancestor_at(by_id: dict[int, ChordView], node_id: int, layer: int) -> int {
    cur = node_id;
    while cur in by_id {
        if by_id[cur].layer == layer {
            return cur;
        }
        if by_id[cur].layer < layer {
            return -1;
        }
        cur = by_id[cur].parent;
    }
    return -1;
}


"""The lock on (tree, layer), or -1."""
def:pub lock_at(by_id: dict[int, ChordView], locks: list[int], back: bool, layer: int) -> int {
    for l in locks {
        if l in by_id and is_back_id(l) == back and by_id[l].layer == layer {
            return l;
        }
    }
    return -1;
}


"""Does `node_id`'s path pass through every lock on an earlier column of its tree?"""
def:pub reachable(by_id: dict[int, ChordView], locks: list[int], node_id: int) -> bool {
    if node_id not in by_id {
        return False;
    }
    n = by_id[node_id];
    for l in locks {
        if l in by_id and is_back_id(l) == is_back_id(node_id) and by_id[l].layer < n.layer {
            if ancestor_at(by_id, node_id, by_id[l].layer) != l {
                return False;
            }
        }
    }
    return True;
}


"""The deepest lock in one tree, or -1."""
def:pub deepest_lock(by_id: dict[int, ChordView], locks: list[int], back: bool) -> int {
    best = -1;
    for l in locks {
        if l in by_id and is_back_id(l) == back and (best < 0 or by_id[l].layer > by_id[best].layer) {
            best = l;
        }
    }
    return best;
}
