// Path helpers shared by the shell and the map: the search's nodes by id, and walking
// a node's chain back to its tree's start.

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
