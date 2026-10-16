// Path helpers shared by the shell and the map: walking a node's chain back to its
// tree's start, and the half-commit locks.
//
// A lock pins one chord in one column of one tree. A node is *reachable* when, for
// every locked column before it in the same tree, its path passes through the
// locked chord. Unreachable nodes are dimmed: hovering them would play a path the
// composer has already ruled out.

import type { ChordView, SearchView } from "../api";

// Ids of the backward search tree start here (server.atlas.BACK_BASE).
export const BACK_BASE = 100000;

export type NodeMap = Record<number, ChordView>;

/** Every node of both trees, both starts, and the typed options, by id. */
export function node_map(search: SearchView, typed: ChordView[]): NodeMap {
  const by_id: NodeMap = { 0: search.start, [BACK_BASE]: search.back_start };
  for (const n of search.nodes) by_id[n.id] = n;
  for (const n of search.back_nodes) by_id[n.id] = n;
  for (const n of typed) by_id[n.id] = n;
  return by_id;
}

export function is_back_id(node_id: number): boolean {
  return node_id >= BACK_BASE;
}

/** Nodes from the tree's start (exclusive) to `node_id` (inclusive). For the backward
 * tree this runs back in time from the first chord. */
export function chain_in(by_id: NodeMap, node_id: number): ChordView[] {
  const chain: ChordView[] = [];
  let cur = node_id;
  while (cur in by_id && by_id[cur].layer > 0) {
    chain.unshift(by_id[cur]);
    cur = by_id[cur].parent;
  }
  return chain;
}

/** The id of `node_id`'s ancestor on `layer` (itself if it is on that layer), or -1. */
export function ancestor_at(by_id: NodeMap, node_id: number, layer: number): number {
  let cur = node_id;
  while (cur in by_id) {
    if (by_id[cur].layer === layer) return cur;
    if (by_id[cur].layer < layer) return -1;
    cur = by_id[cur].parent;
  }
  return -1;
}

/** The lock on (tree, layer), or -1. */
export function lock_at(by_id: NodeMap, locks: number[], back: boolean, layer: number): number {
  for (const l of locks) {
    if (l in by_id && is_back_id(l) === back && by_id[l].layer === layer) return l;
  }
  return -1;
}

/** Does `node_id`'s path pass through every lock on an earlier column of its tree? */
export function reachable(by_id: NodeMap, locks: number[], node_id: number): boolean {
  if (!(node_id in by_id)) return false;
  const n = by_id[node_id];
  for (const l of locks) {
    if (l in by_id && is_back_id(l) === is_back_id(node_id) && by_id[l].layer < n.layer) {
      if (ancestor_at(by_id, node_id, by_id[l].layer) !== l) return false;
    }
  }
  return true;
}

/** The deepest lock in one tree, or -1. */
export function deepest_lock(by_id: NodeMap, locks: number[], back: boolean): number {
  let best = -1;
  for (const l of locks) {
    if (l in by_id && is_back_id(l) === back && (best < 0 || by_id[l].layer > by_id[best].layer)) best = l;
  }
  return best;
}
