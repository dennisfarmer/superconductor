// Stateless helpers for the shell (App.tsx): engine timelines and per-browser
// preferences.

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
