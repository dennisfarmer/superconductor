// Stateless helpers for the shell (App.tsx): engine timelines, per-browser
// preferences, and the saved map locks.

import type { SearchView, SessionView } from "./api";
import { chain_in, is_back_id, node_map } from "./components/paths";
import type { NodeMap } from "./components/paths";

export interface EngineChord {
  label: string;
  symbol: string;
  notes: number[];
}

export interface Timeline {
  id: string;
  role: string;
  chords: EngineChord[];
}

/** A timeline whose id is its content, so any change of chords or voicing is a new id. */
export function make_timeline(role: string, chords: EngineChord[], slots: string[]): Timeline {
  let tid = role;
  chords.forEach((c, i) => {
    tid = tid + "|" + slots[i] + ":" + String(c.notes);
  });
  return { id: tid, role, chords };
}

export function engine_chord(rn: string, symbol: string, notes: number[]): EngineChord {
  return { label: rn, symbol, notes };
}

/** Per-browser layout preferences (side panel tab, open, hidden tabs). Storage can
 * be missing or throw (private windows, previews): the page works without it. */
export function pref_get<T>(key: string, fallback: T): T {
  try {
    const raw = localStorage.getItem("atlas." + key);
    return raw ? JSON.parse(raw) : fallback;
  } catch {
    return fallback;
  }
}

export function pref_set(key: string, value: unknown): void {
  try {
    localStorage.setItem("atlas." + key, JSON.stringify(value));
  } catch {
    // no storage: nothing to keep
  }
}

/** The progression the map's locks were made on: locks saved for it are put back
 * when the app is reopened on the same progression. */
export function lock_sig(s: SessionView): string {
  return JSON.stringify([s.tonic, s.mode, s.steps.map((st) => st.cid)]);
}

interface StoredLocks {
  sig: string;
  locks: { back: boolean; path: string[] }[];
}

/** Save the locks as chord paths from their tree's start (node ids change with
 * every search). */
export function save_locks(by_id: NodeMap, lock_ids: number[], sig: string): void {
  const paths: StoredLocks["locks"] = [];
  for (const l of lock_ids) {
    if (l in by_id) paths.push({ back: is_back_id(l), path: chain_in(by_id, l).map((c) => c.cid) });
  }
  pref_set("locks", { sig, locks: paths });
}

/** The nodes of a fresh search standing where the saved locks were (a lock whose
 * path the search no longer has, e.g. through a typed chord, is dropped). */
export function restore_locks(res: SearchView, sig: string): number[] {
  const stored = pref_get<StoredLocks | null>("locks", null);
  if (!stored || stored.sig !== sig) return [];
  const by_id = node_map(res, []);
  const found: number[] = [];
  for (const want of stored.locks) {
    const path = JSON.stringify(want.path);
    const column = want.back ? res.back_nodes : res.nodes;
    for (const n of column) {
      if (n.layer === want.path.length && JSON.stringify(chain_in(by_id, n.id).map((c) => c.cid)) === path) {
        found.push(n.id);
        break;
      }
    }
  }
  return found;
}

export function sleep_ms(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/** Pitch class of a tonic name like "E♭" or "F♯" (as server/theory/pitch.py Key.parse). */
export function tonic_pc(name: string): number {
  let pc = [0, 2, 4, 5, 7, 9, 11]["CDEFGAB".indexOf(name[0])];
  for (const ch of name.slice(1)) pc += ch === "♯" ? 1 : ch === "♭" ? -1 : 0;
  return ((pc % 12) + 12) % 12;
}

/** The page under the top bar: /#drums, /#bpm, or the Harmony page. */
export function page_of(hash: string): string {
  return hash === "#drums" ? "drums" : hash === "#bpm" ? "bpm" : "harmony";
}
