// SuperConductor Atlas - client shell. Owns all page state; sections are stateless.
// The handlers are defined inside App, so each render's handlers see that render's
// state (the status poll keeps the first render's, so it reads only module state).

import { useEffect, useState } from "react";
import type { CSSProperties, ReactNode } from "react";
import { ErrorBoundary } from "react-error-boundary";
import * as api from "./api";
import type { ChordView, PathStory, SearchView, SessionView, StepView } from "./api";
import { engine_chord, lock_sig, make_timeline, page_of, pref_get, pref_set, restore_locks, save_locks, sleep_ms, tonic_pc } from "./appLogic";
import type { EngineChord, Timeline } from "./appLogic";
import { beat_sync } from "./components/BeatLights";
import { DrumsPage } from "./components/DrumGrid";
import { Graph3D } from "./components/Graph3D";
import { Header } from "./components/Header";
import { ControlsBar } from "./components/ControlsBar";
import { Inspector } from "./components/Inspector";
import { Progression } from "./components/Progression";
import { Rules } from "./components/Rules";
import { SidePanel } from "./components/SidePanel";
import { SoundMixer } from "./components/SoundMixer";
import { SuperConductorInput } from "./components/SuperConductorInput";
import { TempoPage } from "./components/TempoPage";
import { Tooltip } from "./components/Tooltip";
import { TypingGuide } from "./components/TypingGuide";
import { BACK_BASE, ancestor_at, chain_in, deepest_lock, is_back_id, lock_at, node_map, reachable } from "./components/paths";

// Non-reactive module state: timers and values read from long-lived callbacks.
let _search_id = -1;
let _engine_url = "";
// the session loaded at startup: its saved locks are put back on the first map
let _restore_sig = "";
let _engine_was_ok = false;
let _restarting = false;
let _bpm = 84.0;
// set once the app has been quit: no more polling or timelines
let _stopped = false;
// when the volume was last changed here (the poll leaves it alone for a moment)
let _vol_at = 0.0;
// the engine's bpm last copied into the session while tempo sync is on
let _synced_bpm = 0.0;

// ---- what the engine should be playing -------------------------------------------
// The page is the single authority: every render works out the one timeline that
// should be sounding (`_want`), sends it when it changes, and the status poll keeps
// re-sending it until the engine reports playing it. Nothing else talks timelines to
// the engine, so playback cannot drift from what the page shows.
let _want: Timeline | null = null;
let _sent_id = "";
let _sent_at = 0.0;
let _push_timer: ReturnType<typeof setTimeout> | null = null;
let _play_seen = false;
let _scheduled_id = "";
// timeline id -> what each of its chords is on the page: "s3" (step 3) or "n120" (map node 120)
let _slots: Record<string, string[]> = {};
let _slot_count = 0;

function engine_post(path: string, body: Record<string, unknown>): Promise<Response> {
  return fetch(_engine_url + path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
}

/** Send the wanted timeline. An idle loop also ends any preview, so it takes over. */
async function push_want(): Promise<void> {
  if (_stopped || !_engine_url || _want == null) return;
  const want = _want;
  _sent_id = String(want.id);
  _sent_at = Date.now();
  try {
    await engine_post("/timeline", { role: want.role, id: want.id, chords: want.chords });
    if (want.role === "idle") await engine_post("/release", {});
  } catch {
    _sent_at = 0.0; // the poll retries
  }
}

function schedule_push(delay: number): void {
  if (_push_timer) clearTimeout(_push_timer);
  _push_timer = setTimeout(() => push_want(), delay);
}

// The saved voice (VOICE.md).
const DEFAULT_SOUND = "warm grand piano, intimate, slow";

export function App() {
  const [session, setSession] = useState<SessionView | null>(null);
  const [search, setSearch] = useState<SearchView | null>(null);
  const [anim, setAnim] = useState(-1);
  const [walking, setWalking] = useState(false);
  const [hover_id, setHoverId] = useState(-1);
  const [selected_id, setSelectedId] = useState(-1);
  const [mix_texts, setMixTexts] = useState<string[]>([]);
  const [mix_weights, setMixWeights] = useState<number[]>([]);
  const [engine_style, setEngineStyle] = useState("");
  // make run (SuperConductor sink): the client's client_midi URL, and its GET /input
  const [sc_url, setScUrl] = useState("");
  const [sc_input, setScInput] = useState<any>(null);
  const [engine_ok, setEngineOk] = useState(false);
  const [engine_info, setEngineInfo] = useState("engine offline");
  const [paused, setPaused] = useState(false);
  const [, setPlayingRole] = useState("");
  const [playing_tid, setPlayingTid] = useState("");
  const [playing_idx, setPlayingIdx] = useState(-1);
  // the 3D view: every saved progression behind the current one, or only the current one
  const [show_all, setShowAll] = useState(false);
  // chords typed into the map's columns skip the walkers' rules
  const [force_add, setForceAdd] = useState(false);
  const [show_legend, setShowLegend] = useState(true);
  // the side panel: open tab, collapsed or not, tabs hidden from its bar
  const [side_tab, setSideTab] = useState<string>(pref_get("side_tab", "inspect"));
  const [side_open, setSideOpen] = useState<boolean>(pref_get("side_open", true));
  const [side_hidden, setSideHidden] = useState<string[]>(pref_get("side_hidden", []));
  const [side_w, setSideW] = useState<number>(pref_get("side_w", 400)); // the side panel's width, dragged at its border
  const [story, setStory] = useState<PathStory | null>(null);
  const [story_loading, setStoryLoading] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [play_all, setPlayAll] = useState(false);
  // a saved progression being played (-1: the current one), and its loop range
  const [viewing, setViewing] = useState(-1);
  const [view_from, setViewFrom] = useState(-1);
  const [view_to, setViewTo] = useState(-1);
  // pins: one pinned node per column per tree (see components/paths)
  const [locks, setLocks] = useState<number[]>([]);
  // chords typed into the map's columns, as tree nodes (status "custom")
  const [typed, setTyped] = useState<ChordView[]>([]);
  const [restarting, setRestarting] = useState(false);
  const [volume, setVolume] = useState(1.0);
  const [muted, setMuted] = useState(false);
  const [quitting, setQuitting] = useState(false);
  const [stopped, setStopped] = useState(false);
  // the alternative of a kept chord under the mouse: [step index, alternative index]
  const [alt_hover, setAltHover] = useState<number[]>([]);
  // the page under the top bar: "harmony" or "drums" (/#drums)
  const [page, setPage] = useState(page_of(window.location.hash));
  // the engine's last /status (the BPM page reads it)
  const [engine_status, setEngineStatus] = useState<any>(null);

  useEffect(() => {
    load();
  }, []);

  useEffect(() => {
    const timer = setInterval(() => poll_engine(), 300);
    return () => clearInterval(timer);
  }, []);

  useEffect(() => {
    const on_hash = () => setPage(page_of(window.location.hash));
    window.addEventListener("hashchange", on_hash);
    return () => window.removeEventListener("hashchange", on_hash);
  }, []);

  function open_tab(t: string) {
    setSideTab(t);
    pref_set("side_tab", t);
  }

  function set_side_open(v: boolean) {
    setSideOpen(v);
    pref_set("side_open", v);
  }

  function set_side_w(w: number) {
    setSideW(w);
    pref_set("side_w", w);
  }

  function set_side_hidden(h: string[]) {
    setSideHidden(h);
    pref_set("side_hidden", h);
  }

  // the stage changes width with the panel: make the 3D canvas re-measure at once
  useEffect(() => {
    setTimeout(() => window.dispatchEvent(new Event("resize")), 30);
  }, [side_open]);

  // pinning a chord brings up its details
  useEffect(() => {
    if (selected_id >= 0 && side_open && side_tab !== "inspect") open_tab("inspect");
  }, [selected_id]);

  // ---- handlers --------------------------------------------------------------------

  async function load() {
    try {
      const s = await api.getState();
      setSession(s);
      setMixTexts(s.style_texts);
      setMixWeights(s.style_weights);
      _engine_url = s.engine_url;
      _restore_sig = lock_sig(s);
    } catch (e) {
      setError("could not load session: " + String(e));
      return;
    }
    await send_walkers();
  }

  /** Spawn the BeamSearch walker server-side, then replay its layers on the map. */
  async function send_walkers() {
    setWalking(true);
    setError("");
    setHoverId(-1);
    setSelectedId(-1);
    setLocks([]);
    setTyped([]);
    setStory(null);
    if (!_restore_sig) pref_set("locks", null);
    try {
      // fresh options beside every kept chord, ranked against its neighbours as they are now
      setSession(await api.refreshOptions());
      const res = await api.explore();
      _search_id = res.search_id;
      setAnim(0);
      setSearch(res);
      if (_restore_sig) {
        // reopened on the progression the saved locks were made on: put them back
        const held = restore_locks(res, _restore_sig);
        _restore_sig = "";
        if (held.length) {
          const by_id = node_map(res, []);
          const deepest = deepest_lock(by_id, held, false);
          setLocks(held);
          setSelectedId(deepest >= 0 ? deepest : deepest_lock(by_id, held, true));
        }
      }
      for (let k = 0; k < res.depth + 1; k++) {
        setAnim(k);
        await sleep_ms(620);
      }
    } catch (e) {
      setError("walkers failed: " + String(e));
    }
    setAnim(-1);
    setWalking(false);
  }

  async function poll_engine() {
    if (_stopped || !_engine_url) return;
    try {
      const resp = await fetch(_engine_url + "/status");
      const data = await resp.json();
      setEngineOk(true);
      setEngineStatus(data);
      if (!_engine_was_ok) {
        // the engine (re)started: give it the sound and settings, and the timeline again
        _engine_was_ok = true;
        _restarting = false;
        setRestarting(false);
        _sent_id = "";
        setSession(await api.getState());
      }
      const ms = data.ms_per_frame;
      if (data.sink) {
        setEngineInfo(String(data.model)); // make run: the sound comes from SuperConductor
        // the Sound tab shows what the SuperConductor client sends to MRT2
        // (a local: sc_url read here would be this timer's stale copy)
        const client_midi = String(data.client_midi);
        setScUrl(client_midi);
        try {
          const sc_resp = await fetch(client_midi + "/input");
          setScInput(await sc_resp.json());
        } catch {
          setScInput(null);
        }
      } else {
        setScUrl("");
        setEngineInfo(String(data.model) + " · " + (ms ? String(ms) + " ms/frame" : "warming up"));
      }
      setEngineStyle(String(data.style));
      setPaused(Boolean(data.paused));
      if (Date.now() - _vol_at > 1500 && data.volume != null) {
        setVolume(parseFloat(data.volume));
        setMuted(Boolean(data.muted));
      }
      if (data.settings && data.settings.bpm) {
        _bpm = parseFloat(data.settings.bpm);
        if (data.sync && data.sync.on && Math.abs(_bpm - _synced_bpm) > 0.05) {
          // tempo sync moved the engine's tempo: keep the session (and the top bar) with it
          _synced_bpm = _bpm;
          setSession(await api.setTempo(_bpm, Math.trunc(Number(data.settings.beats_per_chord))));
        }
      }
      const playing = data.playing;
      if (playing && data.settings) {
        beat_sync(
          Math.trunc(Number(playing.beat)),
          parseFloat(playing.beat_frac || 0),
          parseFloat(data.settings.bpm),
          Math.trunc(Number(data.settings.beats_per_chord)),
          parseFloat(data.latency_ms || 0),
          !data.paused
        );
      } else {
        beat_sync(0, 0.0, _bpm, 4, 0.0, false);
      }
      let now_tid = "";
      if (playing) {
        now_tid = String(playing.timeline_id);
        setPlayingRole(String(playing.role));
        setPlayingTid(now_tid);
        setPlayingIdx(Math.trunc(Number(playing.index)));
      }
      // keep the engine on the wanted timeline: resend until it reports playing it
      if (_want != null) {
        const want = _want;
        if (want.role === "play") {
          if (now_tid === want.id) {
            _play_seen = true;
          } else if (_play_seen) {
            // "Play all" ran to the end: back to the loop
            _play_seen = false;
            setPlayAll(false);
            return;
          }
        }
        const grace = 60000.0 / _bpm + 1500.0;
        if (now_tid !== want.id && (_sent_id !== want.id || Date.now() - _sent_at > grace)) push_want();
      }
    } catch {
      _engine_was_ok = false;
      setEngineOk(false);
      setEngineInfo(_restarting ? "engine restarting…" : "engine offline");
      setPlayingRole("");
      beat_sync(0, 0.0, _bpm, 4, 0.0, false);
    }
  }

  /** Pause/resume the engine's audio directly (like the status poll). */
  async function toggle_pause() {
    if (!_engine_url) return;
    const want = !paused;
    setPaused(want);
    await fetch(_engine_url + "/pause", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ paused: want }) });
  }

  /** Hover a map node: the page plays the path to it (see `_want` above). */
  function on_hover(node_id: number) {
    setHoverId(node_id);
    setPlayAll(false);
  }

  /** Leave: playback falls back to the half-committed path, else the loop. */
  function on_leave() {
    setHoverId(-1);
  }

  /** Click a map node to half-commit its column: hover there is ignored, and later
   * columns only offer paths through it. Click it again to release the column (and any
   * later locks that hung off it); click another chord in the column to move the lock. */
  function on_select(node_id: number) {
    setStory(null);
    if (search == null) return;
    const by_id = node_map(search, typed);
    if (!(node_id in by_id)) return;
    const back = is_back_id(node_id);
    const layer = by_id[node_id].layer;
    const kept: number[] = [];
    if (locks.includes(node_id)) {
      for (const l of locks) if (is_back_id(l) !== back || by_id[l].layer < layer) kept.push(l);
    } else {
      for (const l of locks) {
        if (is_back_id(l) !== back || by_id[l].layer < layer) kept.push(l);
        else if (by_id[l].layer > layer && ancestor_at(by_id, l, layer) === node_id) kept.push(l); // a later lock on a path through the new one still holds
      }
      kept.push(node_id);
    }
    setLocks(kept);
    save_locks(by_id, kept, lock_sig(session!));
    pin_deepest(back, kept, typed);
    setHoverId(-1);
  }

  /** Make the deepest lock of a tree the pinned path (Inspector, chips, idle loop). */
  function pin_deepest(back: boolean, lock_ids: number[], typed_now: ChordView[]) {
    if (search == null) return;
    const by_id = node_map(search, typed_now);
    let deepest = deepest_lock(by_id, lock_ids, back);
    if (deepest < 0) deepest = deepest_lock(by_id, lock_ids, !back);
    setSelectedId(deepest);
  }

  /** Type a chord into column `layer` of one tree. It links from every chord in the
   * previous column you can still reach (only the locked one, if that column is locked)
   * that may legally move to it. */
  async function add_typed(back: boolean, layer: number, text: string) {
    if (search == null || !text.trim()) return;
    const cur = search;
    const by_id = node_map(cur, typed);
    const start_id = back ? BACK_BASE : 0;
    let parents: number[] = [];
    if (layer === 1) {
      parents = [start_id];
    } else {
      const held = lock_at(by_id, locks, back, layer - 1);
      if (held >= 0) {
        parents = [held];
      } else {
        const column = [...(back ? cur.back_nodes : cur.nodes), ...typed];
        for (const n of column) {
          if (n.layer === layer - 1 && is_back_id(n.id) === back && reachable(by_id, locks, n.id)) parents.push(n.id);
        }
      }
    }
    const res = await api.addOption(_search_id, parents, text, force_add);
    setError(res.notice);
    if (res.notice) return;
    setNotice(res.info);
    // the typed chord first, then the mini walker's finds one column at a time
    let shown = [...typed, ...res.nodes.filter((n) => n.status === "custom")];
    setTyped(shown);
    const grown = res.nodes.filter((n) => n.status === "grown");
    const later = [...new Set(grown.map((n) => n.layer))].sort((a, b) => a - b);
    for (const L of later) {
      await sleep_ms(450);
      shown = [...shown, ...grown.filter((n) => n.layer === L)];
      setTyped(shown);
    }
  }

  /** Remove a typed option (all its links), anything typed after it, and their locks. */
  function remove_typed(ids: number[]) {
    const gone = [...ids];
    let changed = true;
    while (changed) {
      changed = false;
      for (const n of typed) {
        if (gone.includes(n.parent) && !gone.includes(n.id)) {
          gone.push(n.id);
          changed = true;
        }
      }
    }
    const remaining = typed.filter((n) => !gone.includes(n.id));
    setTyped(remaining);
    if (locks.some((l) => gone.includes(l))) {
      const held = locks.filter((l) => !gone.includes(l));
      setLocks(held);
      save_locks(node_map(search!, remaining), held, lock_sig(session!));
      pin_deepest(selected_id >= BACK_BASE, held, remaining);
    }
    if (gone.includes(hover_id)) setHoverId(-1);
  }

  /** Keep: put the hovered/pinned chord (or its whole path) into the progression. The
   * server keeps the other options beside each kept chord, so it can be swapped later. */
  async function do_commit(whole: boolean) {
    const target = hover_id >= 0 ? hover_id : selected_id;
    if (target <= 0 || _search_id < 0) return;
    setSession(await api.addPath(_search_id, target, whole));
    await send_walkers();
  }

  /** Type a chord in place of the kept chord at step `index` (after the first). */
  async function do_replace_step(index: number, text: string) {
    const s = await api.replaceStep(index, text);
    setSession(s);
    setNotice(s.notice);
    if (s.notice) return;
    await send_walkers();
  }

  /** Switch the added-note chords (add9, add11, 6) in or out of the walkers' vocabulary. */
  async function do_set_extended(on: boolean) {
    setSession(await api.setExtended(on));
    await send_walkers();
  }

  /** Remove a typed chord from the progression (an opening chord typed in place of
   * another gives that one back). */
  async function do_remove_step(index: number) {
    const s = await api.removeStep(index);
    setSession(s);
    setNotice(s.notice);
    await send_walkers();
  }

  /** Swap the kept chord at step `index` for one of its alternatives. */
  async function do_swap(index: number, cid: string) {
    setAltHover([]);
    const s = await api.swapStep(index, cid);
    setSession(s);
    setNotice(s.notice);
    await send_walkers();
  }

  async function do_undo() {
    setSession(await api.undo());
    await send_walkers();
  }

  async function do_clear() {
    setSession(await api.clearProgression());
    await send_walkers();
  }

  /** Play the progression once from the top (its last 16 chords), then back to the loop. */
  function do_play() {
    _play_seen = false;
    setHoverId(-1);
    setPlayAll(true);
  }

  async function change_key(tonic: string, mode: string) {
    if (session == null) return;
    const tonics = mode === "major" ? session.major_tonics : session.minor_tonics;
    // a spelling the mode lacks (D♭ minor, D♯ major) becomes its enharmonic twin
    const t = tonics.includes(tonic) ? tonic : (tonics.find((x) => tonic_pc(x) === tonic_pc(tonic)) ?? tonics[0]);
    setSession(await api.setKey(t, mode));
    await send_walkers();
  }

  async function change_tempo(bpm: number, beats: number) {
    if (!bpm || !beats) return;
    if (bpm !== session!.bpm && engine_status && engine_status.sync && engine_status.sync.on) {
      // a tempo set by hand ends the sync, or the sync would undo it
      await set_sync(false);
    }
    setSession(await api.setTempo(bpm, beats));
  }

  /** Tempo sync with the SuperConductor server (the BPM page; engine POST /sync). */
  async function set_sync(on: boolean) {
    if (!_engine_url) return;
    _synced_bpm = session!.bpm;
    try {
      const resp = await engine_post("/sync", { on });
      setEngineStatus(await resp.json());
    } catch (e) {
      setError("tempo sync: " + String(e));
    }
  }

  async function do_describe() {
    const target = hover_id >= 0 ? hover_id : selected_id;
    if (target <= 0) return;
    setStoryLoading(true);
    try {
      setStory(await api.describe(_search_id, target));
    } catch (e) {
      setError("describe failed: " + String(e));
    }
    setStoryLoading(false);
  }

  async function change_texture(texture: string) {
    setSession(await api.setTexture(texture));
  }

  async function toggle_llm(enabled: boolean) {
    setSession(await api.setLlm(enabled));
  }

  /** Send the current layers + knob weights; the engine re-blends the style embedding. */
  async function commit_mix() {
    if (session == null || mix_texts.length === 0) return;
    setSession(await api.setStyleLayers(mix_texts, mix_weights));
  }

  function mix_add(prompt: string) {
    setMixTexts([...mix_texts, prompt === "custom" ? "" : prompt]);
    setMixWeights([...mix_weights, 0.5]);
    if (prompt !== "custom") setTimeout(() => commit_mix(), 50);
  }

  function mix_remove(i: number) {
    const texts = mix_texts.filter((_, j) => j !== i);
    const weights = mix_weights.filter((_, j) => j !== i);
    setMixTexts(texts);
    setMixWeights(weights);
    setTimeout(() => api.setStyleLayers(texts, weights), 50);
  }

  async function mix_reset() {
    setMixTexts([DEFAULT_SOUND]);
    setMixWeights([1.0]);
    setSession(await api.setStyleLayers([DEFAULT_SOUND], [1.0]));
  }

  /** Typed chords: "before" the first chord, "after" the last, "first" in place of the
   * first chord, or "replace" everything. */
  async function do_add(text: string, where: string) {
    const s = await api.addChords(text, where);
    setSession(s);
    setNotice(s.notice);
    if (s.notice) return;
    await send_walkers();
  }

  /** Relaunch the audio engine process; poll_engine re-syncs it once it is back. */
  async function do_restart_engine() {
    _restarting = true;
    setRestarting(true);
    setEngineOk(false);
    _engine_was_ok = false;
    setEngineInfo("engine restarting…");
    try {
      if (!(await api.restartEngine())) {
        _restarting = false;
        setRestarting(false);
        setError("The audio engine did not stop when asked, so it was not restarted.");
      }
    } catch (e) {
      _restarting = false;
      setRestarting(false);
      setError("engine restart failed: " + String(e));
    }
  }

  /** Progression clicks set the loop range. Click: loop from this chord to the end
   * (click the start again for the default, just the last chord). Shift-click: end the
   * loop at this chord, so any stretch can be looped. */
  async function do_loop(index: number, extend: boolean) {
    const cur = session!.loop_from;
    let start = index;
    let end = -1;
    if (extend && cur >= 0 && index !== cur) {
      start = Math.min(cur, index);
      end = Math.max(cur, index);
    } else if (!extend && index === cur) {
      start = -1;
    }
    setSession(await api.setLoop(start, end));
  }

  /** Loop range inside the committed progression being played (same clicks as for the
   * current one; kept on the page only). */
  function view_loop(index: number, extend: boolean) {
    if (extend && view_from >= 0 && index !== view_from) {
      setViewTo(Math.max(view_from, index));
      setViewFrom(Math.min(view_from, index));
    } else if (!extend && index === view_from) {
      setViewFrom(-1);
      setViewTo(-1);
    } else {
      setViewFrom(index);
      setViewTo(-1);
    }
  }

  /** Play a committed progression (its loop takes over the engine); -1: back to the current one. */
  function pick_saved(index: number) {
    setViewing(index);
    setViewFrom(-1);
    setViewTo(-1);
    setPlayAll(false);
  }

  /** Save: shelve this progression under a name (blank: "Progression N"; it moves
   * behind in the graph view) and start a new one. */
  async function do_commit_progression(name: string) {
    setSession(await api.commitProgression(name));
    setViewing(-1);
    await send_walkers();
  }

  async function do_rename_saved(index: number, name: string) {
    setSession(await api.renameSaved(index, name));
  }

  async function do_delete_saved() {
    if (viewing < 0) return;
    setSession(await api.deleteSaved(viewing));
    setViewing(-1);
  }

  /** Make the committed progression being played current again (the current one is
   * committed in its place unless it is only the opening tonic). */
  async function do_resume_saved() {
    if (viewing < 0) return;
    setSession(await api.resumeSaved(viewing));
    setViewing(-1);
    await send_walkers();
  }

  /** Output level, straight to the engine (applied at the speaker, so it is immediate). */
  async function set_volume(v: number) {
    if (!_engine_url) return;
    _vol_at = Date.now();
    // state reads in this handler still see the old values: work from locals
    setVolume(v);
    const body: Record<string, unknown> = { volume: v };
    if (muted && v > 0.0) {
      // turning the volume up unmutes
      setMuted(false);
      body.muted = false;
    }
    await engine_post("/mix", body);
  }

  /** MRT2's sampling controls (the controls row), straight to the engine's /settings. */
  async function set_controls(body: Record<string, unknown>) {
    if (!_engine_url) return;
    const resp = await engine_post("/settings", body);
    if (resp.ok) setEngineStatus(await resp.json());
  }

  async function toggle_mute() {
    if (!_engine_url) return;
    _vol_at = Date.now();
    const want = !muted; // `muted` keeps its old value for the rest of this handler
    setMuted(want);
    await engine_post("/mix", { muted: want });
  }

  /** Quit: the server stops the engine, then itself (server/app.py shutdown_app).
   * The page stops polling and sending, and says so. */
  async function do_quit() {
    setQuitting(true);
    try {
      await api.shutdownApp();
    } catch (e) {
      setQuitting(false);
      setError("Quit failed: " + String(e));
      return;
    }
    _stopped = true;
    setStopped(true);
  }

  // ---- render ----------------------------------------------------------------------

  if (stopped) {
    return (
      <div className="atlas loading">
        <div className="stopped">
          <h1>SuperConductor Atlas has stopped</h1>
          <p>The audio engine and the app server shut down cleanly. Your session is saved.</p>
          <p className="hint">
            To start again, run <code>make</code> in the project folder, then reload this page.
          </p>
        </div>
      </div>
    );
  }
  if (session == null) {
    return (
      <div className="atlas loading">
        <p className="empty">Loading the atlas…</p>
      </div>
    );
  }

  // ---- derived view state ----
  const steps = session.steps;
  const last_idx = steps.length - 1;
  const viewing_ok = viewing >= 0 && viewing < session.saved.length;
  const shown_steps: StepView[] = viewing_ok ? session.saved[viewing].steps : steps;
  const focus_id = hover_id >= 0 ? hover_id : selected_id;
  const focus_back = focus_id >= BACK_BASE;
  let focus: ChordView | null = null;
  let chain: ChordView[] = [];
  let sounding_id = -1;
  let sounding_step = -1;
  // tentative chips: each side shows its hovered path, else its half-committed one
  const preview_before: ChordView[] = [];
  let preview_after: ChordView[] = [];
  if (search != null) {
    const by_id = node_map(search, typed);
    focus = focus_id in by_id ? by_id[focus_id] : null;
    chain = focus_id > 0 && focus_id !== BACK_BASE ? chain_in(by_id, focus_id) : [];
    const back_end = hover_id >= BACK_BASE ? hover_id : deepest_lock(by_id, locks, true);
    const fwd_end = hover_id > 0 && hover_id < BACK_BASE ? hover_id : deepest_lock(by_id, locks, false);
    if (back_end > BACK_BASE) for (const c of chain_in(by_id, back_end)) preview_before.unshift(c);
    if (fwd_end > 0) preview_after = chain_in(by_id, fwd_end);
  }

  // ---- the one timeline the engine should play now ----
  const lo = session.loop_from >= 0 ? session.loop_from : last_idx;
  const hi = session.loop_to >= 0 ? session.loop_to : session.loop_from >= 0 ? last_idx : lo;
  const want_chords: EngineChord[] = [];
  const want_slots: string[] = [];
  let want_role = "idle";
  let target = -1;
  if (search != null) {
    const by_now = node_map(search, typed);
    if (hover_id > 0 && hover_id !== BACK_BASE && hover_id in by_now) target = hover_id;
    else if (!play_all && selected_id > 0 && selected_id !== BACK_BASE && selected_id in by_now) target = selected_id;
    // (an alternative beside a kept chord only highlights on hover: a column that
    // already has its chord is never auditioned by hovering; click it to swap)
    if (target > 0) {
      want_role = "preview";
      const path = chain_in(by_now, target);
      if (target >= BACK_BASE) {
        // lead-ins in time order, then the progression up to the loop's end
        const upto = session.loop_to >= 0 ? session.loop_to : Math.max(0, session.loop_from);
        for (const c of path) {
          want_chords.unshift(engine_chord(c.rn, c.symbol, c.notes));
          want_slots.unshift("n" + String(c.id));
        }
        for (let i = 0; i < upto + 1; i++) {
          want_chords.push(engine_chord(steps[i].rn, steps[i].symbol, steps[i].notes));
          want_slots.push("s" + String(i));
        }
      } else {
        for (let i = lo; i < last_idx + 1; i++) {
          want_chords.push(engine_chord(steps[i].rn, steps[i].symbol, steps[i].notes));
          want_slots.push("s" + String(i));
        }
        for (const c of path) {
          want_chords.push(engine_chord(c.rn, c.symbol, c.notes));
          want_slots.push("n" + String(c.id));
        }
      }
    }
  }
  if (target < 0) {
    // the loop (or Play all) of the current progression, or of the committed one picked
    const play_steps = shown_steps;
    const mark = viewing_ok ? "p" : "s";
    const p_last = play_steps.length - 1;
    let p_lo = lo;
    let p_hi = hi;
    if (viewing_ok) {
      p_lo = view_from >= 0 ? view_from : 0;
      p_hi = view_to >= 0 ? view_to : p_last;
    }
    const first = play_all ? Math.max(0, play_steps.length - 16) : p_lo;
    const final = play_all ? p_last : p_hi;
    want_role = play_all ? "play" : "idle";
    for (let i = first; i < final + 1; i++) {
      want_chords.push(engine_chord(play_steps[i].rn, play_steps[i].symbol, play_steps[i].notes));
      want_slots.push(mark + String(i));
    }
  }
  _want = make_timeline(want_role, want_chords, want_slots);
  const want_id = _want.id;
  if (want_id !== _scheduled_id) {
    _scheduled_id = want_id;
    if (target > 0 && target === hover_id) {
      // hover: debounced, so sweeping the mouse across the map doesn't flood the engine
      schedule_push(150);
    } else {
      // a click or edit: send now (a microtask, so background-tab timer throttling can't delay it)
      if (_push_timer) clearTimeout(_push_timer);
      queueMicrotask(() => push_want());
    }
  }
  if (!(want_id in _slots)) {
    if (_slot_count > 300) {
      _slots = {};
      _slot_count = 0;
    }
    _slots[want_id] = want_slots;
    _slot_count += 1;
  }
  // what is sounding, read straight off the timeline the engine reports playing
  let saved_sounding = -1;
  if (playing_tid in _slots && playing_idx >= 0 && playing_idx < _slots[playing_tid].length) {
    const slot = _slots[playing_tid][playing_idx];
    if (slot.startsWith("s")) sounding_step = Math.trunc(Number(slot.slice(1)));
    else if (slot.startsWith("a")) sounding_step = -1;
    else if (slot.startsWith("p")) saved_sounding = Math.trunc(Number(slot.slice(1)));
    else sounding_id = Math.trunc(Number(slot.slice(1)));
  }

  // ---- the side panel's active tab ----
  let side_body: ReactNode = null;
  if (side_tab === "sound" && sc_url) {
    side_body = <SuperConductorInput view={sc_input} url={sc_url} />;
  } else if (side_tab === "sound") {
    side_body = (
      <SoundMixer
        texts={mix_texts}
        weights={mix_weights}
        engineStyle={engine_style}
        onText={(i, t) => setMixTexts(mix_texts.map((x, j) => (j === i ? t : x)))}
        onWeight={(i, v) => setMixWeights(mix_weights.map((x, j) => (j === i ? v : x)))}
        onCommit={commit_mix}
        onAdd={mix_add}
        onRemove={mix_remove}
        onReset={mix_reset}
      />
    );
  } else if (side_tab === "rules") {
    side_body = <Rules mode={session.mode} />;
  } else if (side_tab === "typing") {
    side_body = <TypingGuide />;
  } else {
    side_body = (
      <Inspector
        focusNode={focus}
        chain={chain}
        startRn={focus_back ? steps[0].rn : steps[last_idx].rn}
        backward={focus_back}
        isPinned={hover_id < 0 && selected_id >= 0}
        locked={locks.includes(selected_id)}
        story={story}
        storyLoading={story_loading}
        llmEnabled={session.llm_enabled}
        local={session.local}
        onCommitFirst={() => do_commit(false)}
        onCommitPath={() => do_commit(true)}
        onDescribe={do_describe}
      />
    );
  }

  // the top bar, and under it the controls row (MRT2's sampling controls)
  const header = (
    <div className="top-rows">
      <Header
        page={page}
        session={session}
        engineOk={engine_ok}
        engineInfo={engine_info}
        paused={paused}
        onPause={toggle_pause}
        restarting={restarting}
        onRestart={do_restart_engine}
        volume={volume}
        muted={muted}
        onVolume={set_volume}
        onMute={toggle_mute}
        quitting={quitting}
        onQuit={do_quit}
        walking={walking}
        onKey={change_key}
        onTempo={change_tempo}
        onTexture={change_texture}
        onLlm={toggle_llm}
        onSend={send_walkers}
      />
      <ControlsBar
        settings={engine_ok && engine_status ? engine_status.settings : null}
        sink={Boolean(engine_status && engine_status.sink)}
        onSend={set_controls}
      />
    </div>
  );

  // the Drums page keeps the top bar; the timeline above still plays underneath
  if (page === "drums") {
    return (
      <div className="atlas drums-view">
        {header}
        <DrumsPage
          settings={engine_ok && engine_status ? engine_status.settings : null}
          sink={Boolean(engine_status && engine_status.sink)}
          onSend={set_controls}
        />
        <Tooltip />
      </div>
    );
  }
  if (page === "bpm") {
    return (
      <div className="atlas drums-view">
        {header}
        <TempoPage status={engine_status} bpm={session.bpm} onSync={set_sync} />
      </div>
    );
  }

  return (
    <div className={"atlas" + (side_open ? "" : " side-closed")} style={{ "--side-w": String(side_w) + "px" } as CSSProperties}>
      {header}
      <main className="stage">
        {search != null ? (
          <ErrorBoundary fallback={<p className="empty stage-msg">The 3D view hit an error; reload the page to bring it back.</p>}>
            <Graph3D
              search={search}
              anim={anim}
              hoverId={hover_id}
              selectedId={selected_id}
              soundingId={sounding_id}
              soundingStep={sounding_step}
              loopFrom={session.loop_from}
              onStep={(i) => do_loop(i, false)}
              showAll={show_all}
              steps={steps}
              altHover={alt_hover}
              onAltHover={(i, j) => setAltHover([i, j])}
              onAltLeave={() => setAltHover([])}
              onAltPick={do_swap}
              onHover={on_hover}
              onLeave={on_leave}
              onSelect={on_select}
              locks={locks}
              typed={typed}
              onAddTyped={add_typed}
              onRemoveTyped={remove_typed}
              onStepTyped={(i, text) => {
                if (i === 0) do_add(text, "first");
                else do_replace_step(i, text);
              }}
              onRemoveStep={do_remove_step}
              saved={session.saved}
              viewing={viewing_ok ? viewing : -1}
              savedSounding={saved_sounding}
              onPickSaved={(k) => pick_saved(k === viewing ? -1 : k)}
            />
          </ErrorBoundary>
        ) : (
          <p className="empty stage-msg">Sending walkers…</p>
        )}
        <div className="stage-top">
          <div className="stage-status">
            <b>Graph view</b>
            <span>
              {anim >= 0
                ? "walkers on layer " + String(anim) + "…"
                : "searching before " +
                  steps[0].rn +
                  " and after " +
                  steps[last_idx].rn +
                  (session.saved.length > 0 ? " · " + String(session.saved.length) + " saved behind" : "")}
            </span>
            {error && <span className="err">{error}</span>}
          </div>
          <label className="stage-toggle" title="Also show the saved progressions, each on a plane behind the current one">
            <input type="checkbox" checked={show_all} onChange={() => setShowAll(!show_all)} />
            show all progressions
          </label>
          <label
            className="stage-toggle"
            title="Chords typed into the graph's columns are added even where the walkers' rules forbid the move (they still need a playable voicing)"
          >
            <input type="checkbox" checked={force_add} onChange={() => setForceAdd(!force_add)} />
            force-add typed chords
          </label>
          <label
            className="stage-toggle"
            title="Let the walkers use added-note chords: add9, add11 and 6 on the diatonic triads (where the added tone is in the scale). Typing also reads C6 as an added-sixth chord; add13 can always be typed."
          >
            <input type="checkbox" checked={session.extended} onChange={() => do_set_extended(!session.extended)} />
            add9 / add11 / 6 chords
          </label>
        </div>
        <div className={"legend" + (show_legend ? "" : " folded")} onClick={() => setShowLegend(!show_legend)} title="Click to fold or unfold the legend">
          <span>
            <i style={{ background: "var(--f-T)" }}></i>tonic
          </span>
          <span>
            <i style={{ background: "var(--f-TX)" }}></i>tonic ext.
          </span>
          <span>
            <i style={{ background: "var(--f-PD)" }}></i>pre-dominant
          </span>
          <span>
            <i style={{ background: "var(--f-D)" }}></i>dominant
          </span>
          <span>
            <i style={{ background: "var(--f-CAD64)" }}></i>cadential ⁶₄
          </span>
          <span>
            <i style={{ background: "var(--f-APP)" }}></i>applied
          </span>
          <span>
            <i style={{ background: "var(--f-MIX)" }}></i>mixture
          </span>
          <span>
            <i style={{ background: "var(--f-AUG6)" }}></i>aug. sixth
          </span>
          <span>
            <i style={{ background: "var(--f-NEAP)" }}></i>Neapolitan
          </span>
          <span className="legend-note">dashed edge: cadence beyond the beam · dashed box: typed by you</span>
        </div>
      </main>
      <SidePanel tab={side_tab} open={side_open} hidden={side_hidden} onTab={open_tab} onOpen={set_side_open} onHidden={set_side_hidden} onResize={set_side_w}>
        {side_body}
      </SidePanel>
      <Progression
        steps={shown_steps}
        soundingStep={viewing_ok ? saved_sounding : sounding_step}
        soundingId={sounding_id}
        previewBefore={viewing_ok ? [] : preview_before}
        previewAfter={viewing_ok ? [] : preview_after}
        loopFrom={viewing_ok ? view_from : session.loop_from}
        loopTo={viewing_ok ? view_to : session.loop_to}
        onLoop={viewing_ok ? view_loop : do_loop}
        canUndo={session.can_undo}
        notice={notice}
        keyName={session.key_name}
        onUndo={do_undo}
        onClear={do_clear}
        onPlay={() => {
          if (play_all) setPlayAll(false);
          else do_play();
        }}
        playAll={play_all}
        saved={session.saved}
        viewing={viewing_ok ? viewing : -1}
        onPick={pick_saved}
        onCommit={do_commit_progression}
        onRename={do_rename_saved}
        onDelete={do_delete_saved}
        onResume={do_resume_saved}
        keepPath={focus != null && focus.id !== 0 ? chain : []}
        onKeep={do_commit}
      />
      <Tooltip />
    </div>
  );
}
