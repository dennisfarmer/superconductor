"""SuperConductor Atlas - client shell. Owns all page state; sections are stateless.

Handler bodies live in frontend.impl.jac.
"""

import from react { useEffect }
import from "react-error-boundary" { ErrorBoundary }
import from .components.Header { Header }
import from .components.Progression { Progression }
import from .components.SoundMixer { SoundMixer }
import from .components.SuperConductorInput { SuperConductorInput }
import from .components.Graph3D { Graph3D }
import from .components.Inspector { Inspector }
import from .components.Rules { Rules }
import from .components.SidePanel { SidePanel }
import from .components.TypingGuide { TypingGuide }
import from .components.Tooltip { Tooltip }
import from .components.DrumGrid { DrumsPage }
import from .components.BeatLights { beat_sync }
import from .components.paths { node_map, chain_in, reachable, deepest_lock, lock_at, ancestor_at, is_back_id }

sv import from services.atlas {
    SessionView, SearchView, ChordView, PathStory, StepView,
    get_state, set_key, set_tempo, set_style_layers, set_texture, set_llm, explore,
    add_path, undo, clear_progression, describe, add_chords, restart_engine, set_loop, add_option,
    commit_progression, delete_saved, resume_saved, swap_step, shutdown_app
}

# Non-reactive module state: timers and values read from long-lived callbacks.
glob _search_id: int = -1,
     _engine_url: str = "",
     _engine_was_ok: bool = False,
     _restarting: bool = False,
     _bpm: float = 84.0,
     # set once the app has been quit: no more polling or timelines
     _stopped: bool = False,
     # when the volume was last changed here (the poll leaves it alone for a moment)
     _vol_at: float = 0.0;

# ---- what the engine should be playing -------------------------------------------
# The page is the single authority: every render works out the one timeline that
# should be sounding (`_want`), sends it when it changes, and the status poll keeps
# re-sending it until the engine reports playing it. Nothing else talks timelines to
# the engine, so playback cannot drift from what the page shows.
glob _want: dict[str, any] = {},
     _sent_id: str = "",
     _sent_at: float = 0.0,
     _push_timer: any = None,
     _play_seen: bool = False,
     _scheduled_id: str = "",
     # timeline id -> what each of its chords is on the page: "s3" (step 3) or "n120" (map node 120)
     _slots: dict[str, list[str]] = {},
     _slot_count: int = 0;


def engine_post(path: str, body: dict[str, any]) -> any {
    return fetch(_engine_url + path, {"method": "POST", "headers": {"Content-Type": "application/json"}, "body": JSON.stringify(body)});
}


"""Send the wanted timeline. An idle loop also ends any preview, so it takes over."""
async def push_want -> None {
    global _sent_id, _sent_at;
    if _stopped or not _engine_url or "id" not in _want {
        return;
    }
    want = _want;
    _sent_id = str(want["id"]);
    _sent_at = Date.now();
    try {
        await engine_post("/timeline", {"role": want["role"], "id": want["id"], "chords": want["chords"]});
        if want["role"] == "idle" {
            await engine_post("/release", {});
        }
    } except Exception {
        _sent_at = 0.0;   # the poll retries
    }
}


def schedule_push(delay: int) -> None {
    global _push_timer;
    if _push_timer {
        clearTimeout(_push_timer);
    }
    _push_timer = setTimeout(lambda { push_want(); }, delay);
}


"""A timeline whose id is its content, so any change of chords or voicing is a new id."""
def make_timeline(role: str, chords: list[dict[str, any]], slots: list[str]) -> dict[str, any] {
    tid = role;
    for (i, c) in enumerate(chords) {
        tid = tid + "|" + slots[i] + ":" + str(c["notes"]);
    }
    return {"id": tid, "role": role, "chords": chords};
}


def engine_chord(rn: str, symbol: str, notes: list[int]) -> dict[str, any] {
    return {"label": rn, "symbol": symbol, "notes": notes};
}


# Ids of the backward search tree start here (services.atlas.BACK_BASE).
glob BACK_BASE: int = 100000;

# The saved voice (VOICE.md).
glob DEFAULT_SOUND: str = "warm grand piano, intimate, slow";


"""Per-browser layout preferences (side panel tab, open, hidden tabs). Storage can
be missing or throw (private windows, previews): the page works without it."""
def pref_get(key: str, fallback: any) -> any {
    try {
        raw = localStorage.getItem("atlas." + key);
        return JSON.parse(raw) if raw else fallback;
    } except Exception {
        return fallback;
    }
}


def pref_set(key: str, value: any) -> None {
    try {
        localStorage.setItem("atlas." + key, JSON.stringify(value));
    } except Exception {}
}


def sleep_ms(ms: int) -> any {
    return new(Promise, lambda(resolve: any, reject: any) { setTimeout(resolve, ms); });
}


"""Nodes from the search start (exclusive) to `node_id` (inclusive), in either tree
(typed options included). For the backward tree this runs back in time."""
def chain_of(search: SearchView, typed: list[ChordView], node_id: int) -> list[ChordView] {
    return chain_in(node_map(search, typed), node_id);
}


def find_node(search: SearchView, typed: list[ChordView], node_id: int) -> ChordView | None {
    by_id = node_map(search, typed);
    return by_id[node_id] if node_id in by_id else None;
}


def:pub app -> JsxElement {
    has session: SessionView | None = None,
        search: SearchView | None = None,
        anim: int = -1,
        walking: bool = False,
        hover_id: int = -1,
        selected_id: int = -1,
        mix_texts: list[str] = [],
        mix_weights: list[float] = [],
        engine_style: str = "",
        # make run (SuperConductor sink): the client's client_midi URL, and its GET /input
        sc_url: str = "",
        sc_input: any = None,
        engine_ok: bool = False,
        engine_info: str = "engine offline",
        paused: bool = False,
        playing_role: str = "",
        playing_tid: str = "",
        playing_idx: int = -1,
        # the 3D view: every saved progression behind the current one, or only the current one
        show_all: bool = False,
        show_legend: bool = True,
        # the side panel: open tab, collapsed or not, tabs hidden from its bar
        side_tab: str = pref_get("side_tab", "inspect"),
        side_open: bool = pref_get("side_open", True),
        side_hidden: list[str] = pref_get("side_hidden", []),
        story: PathStory | None = None,
        story_loading: bool = False,
        error: str = "",
        notice: str = "",
        play_all: bool = False,
        # a saved progression being played (-1: the current one), and its loop range
        viewing: int = -1,
        view_from: int = -1,
        view_to: int = -1,
        # pins: one pinned node per column per tree (see components/paths)
        locks: list[int] = [],
        # chords typed into the map's columns, as tree nodes (status "custom")
        typed: list[ChordView] = [],
        restarting: bool = False,
        volume: float = 1.0,
        muted: bool = False,
        quitting: bool = False,
        stopped: bool = False,
        # the alternative of a kept chord under the mouse: [step index, alternative index]
        alt_hover: list[int] = [],
        # the page under the top bar: "harmony" or "drums" (/#drums)
        page: str = "drums" if window.location.hash == "#drums" else "harmony";

    can with entry {
        load();
    }

    useEffect(lambda {
        timer = setInterval(lambda { poll_engine(); }, 300);
        return lambda { clearInterval(timer); };
    }, []);

    useEffect(lambda {
        on_hash = lambda { page = "drums" if window.location.hash == "#drums" else "harmony"; };
        window.addEventListener("hashchange", on_hash);
        return lambda { window.removeEventListener("hashchange", on_hash); };
    }, []);

    def open_tab(t: str) -> None {
        side_tab = t;
        pref_set("side_tab", t);
    }

    def set_side_open(v: bool) -> None {
        side_open = v;
        pref_set("side_open", v);
    }

    def set_side_hidden(h: list[str]) -> None {
        side_hidden = h;
        pref_set("side_hidden", h);
    }

    # the stage changes width with the panel: make the 3D canvas re-measure at once
    useEffect(lambda {
        setTimeout(lambda { window.dispatchEvent(new(Event, "resize")); }, 30);
    }, [side_open]);

    # pinning a chord brings up its details
    useEffect(lambda {
        if selected_id >= 0 and side_open and side_tab != "inspect" {
            open_tab("inspect");
        }
    }, [selected_id]);

    async def load -> None;
    async def send_walkers -> None;
    async def poll_engine -> None;
    async def toggle_pause -> None;
    def on_hover(node_id: int) -> None;
    def on_leave -> None;
    async def on_select(node_id: int) -> None;
    async def do_commit(whole: bool) -> None;
    async def do_swap(index: int, cid: str) -> None;
    async def do_undo -> None;
    async def do_clear -> None;
    async def do_play -> None;
    async def change_key(tonic: str, mode: str) -> None;
    async def change_tempo(bpm: float, beats: int) -> None;
    async def change_texture(texture: str) -> None;
    async def toggle_llm(enabled: bool) -> None;
    async def commit_mix -> None;
    def mix_add(prompt: str) -> None;
    def mix_remove(i: int) -> None;
    async def mix_reset -> None;
    async def do_describe -> None;
    async def do_add(text: str, where: str) -> None;
    async def do_restart_engine -> None;
    async def set_volume(v: float) -> None;
    async def toggle_mute -> None;
    async def do_quit -> None;
    async def do_loop(index: int, extend: bool) -> None;
    def view_loop(index: int, extend: bool) -> None;
    def pick_saved(index: int) -> None;
    async def do_commit_progression -> None;
    async def do_delete_saved -> None;
    async def do_resume_saved -> None;
    async def add_typed(back: bool, layer: int, text: str) -> None;
    def remove_typed(ids: list[int]) -> None;
    def pin_deepest(back: bool, lock_ids: list[int], typed_now: list[ChordView]) -> None;

    if stopped {
        return <div className="atlas loading">
            <div className="stopped">
                <h1>SuperConductor Atlas has stopped</h1>
                <p>The audio engine and the app server shut down cleanly. Your session is saved.</p>
                <p className="hint">To start again, run <code>make</code> in the project folder, then reload this page.</p>
            </div>
        </div>;
    }
    if session is None {
        return <div className="atlas loading"><p className="empty">Loading the atlas…</p></div>;
    }

    # ---- derived view state ----
    steps = session.steps;
    last_idx = len(steps) - 1;
    viewing_ok = viewing >= 0 and viewing < len(session.saved);
    shown_steps: list[StepView] = session.saved[viewing].steps if viewing_ok else steps;
    # the loop repeats from this step (the clicked one, else the last chord)
    loop_idx = session.loop_from if session.loop_from >= 0 else last_idx;
    focus_id = hover_id if hover_id >= 0 else selected_id;
    focus_back = focus_id >= BACK_BASE;
    focus: ChordView | None = None;
    chain: list[ChordView] = [];
    sounding_id = -1;
    sounding_step = -1;
    # tentative chips: each side shows its hovered path, else its half-committed one
    preview_before: list[ChordView] = [];
    preview_after: list[ChordView] = [];
    if search is not None {
        by_id = node_map(search, typed);
        focus = by_id[focus_id] if focus_id in by_id else None;
        chain = chain_in(by_id, focus_id) if focus_id > 0 and focus_id != BACK_BASE else [];
        back_end = hover_id if hover_id >= BACK_BASE else deepest_lock(by_id, locks, True);
        fwd_end = hover_id if hover_id > 0 and hover_id < BACK_BASE else deepest_lock(by_id, locks, False);
        if back_end > BACK_BASE {
            for c in chain_in(by_id, back_end) {
                preview_before.insert(0, c);
            }
        }
        if fwd_end > 0 {
            preview_after = chain_in(by_id, fwd_end);
        }
    }

    # ---- the one timeline the engine should play now ----
    global _want, _slots, _slot_count, _scheduled_id;
    lo = session.loop_from if session.loop_from >= 0 else last_idx;
    hi = session.loop_to if session.loop_to >= 0 else (last_idx if session.loop_from >= 0 else lo);
    want_chords: list[dict[str, any]] = [];
    want_slots: list[str] = [];
    want_role = "idle";
    target = -1;
    if search is not None {
        by_now = node_map(search, typed);
        if hover_id > 0 and hover_id != BACK_BASE and hover_id in by_now {
            target = hover_id;
        } elif not play_all and selected_id > 0 and selected_id != BACK_BASE and selected_id in by_now {
            target = selected_id;
        }
        if len(alt_hover) == 2 and alt_hover[0] < len(steps) and alt_hover[1] < len(steps[alt_hover[0]].alts) {
            # an alternative of a kept chord: hear it in place, between its neighbours
            ai = alt_hover[0];
            alt = steps[ai].alts[alt_hover[1]];
            want_role = "preview";
            target = 0;
            if ai > 0 {
                want_chords.append(engine_chord(steps[ai - 1].rn, steps[ai - 1].symbol, steps[ai - 1].notes));
                want_slots.append("s" + str(ai - 1));
            }
            want_chords.append(engine_chord(alt.rn, alt.symbol, alt.notes));
            want_slots.append("a" + str(ai));
            if ai < last_idx {
                want_chords.append(engine_chord(steps[ai + 1].rn, steps[ai + 1].symbol, steps[ai + 1].notes));
                want_slots.append("s" + str(ai + 1));
            }
        } elif target > 0 {
            want_role = "preview";
            path = chain_in(by_now, target);
            if target >= BACK_BASE {
                # lead-ins in time order, then the progression up to the loop's end
                upto = session.loop_to if session.loop_to >= 0 else max(0, session.loop_from);
                for c in path {
                    want_chords.insert(0, engine_chord(c.rn, c.symbol, c.notes));
                    want_slots.insert(0, "n" + str(c.id));
                }
                for i in range(upto + 1) {
                    want_chords.append(engine_chord(steps[i].rn, steps[i].symbol, steps[i].notes));
                    want_slots.append("s" + str(i));
                }
            } else {
                for i in range(lo, last_idx + 1) {
                    want_chords.append(engine_chord(steps[i].rn, steps[i].symbol, steps[i].notes));
                    want_slots.append("s" + str(i));
                }
                for c in path {
                    want_chords.append(engine_chord(c.rn, c.symbol, c.notes));
                    want_slots.append("n" + str(c.id));
                }
            }
        }
    }
    if target < 0 {
        # the loop (or Play all) of the current progression, or of the committed one picked
        play_steps = shown_steps;
        mark = "p" if viewing_ok else "s";
        p_last = len(play_steps) - 1;
        p_lo = lo;
        p_hi = hi;
        if viewing_ok {
            p_lo = view_from if view_from >= 0 else 0;
            p_hi = view_to if view_to >= 0 else p_last;
        }
        first = max(0, len(play_steps) - 16) if play_all else p_lo;
        final = p_last if play_all else p_hi;
        want_role = "play" if play_all else "idle";
        for i in range(first, final + 1) {
            want_chords.append(engine_chord(play_steps[i].rn, play_steps[i].symbol, play_steps[i].notes));
            want_slots.append(mark + str(i));
        }
    }
    _want = make_timeline(want_role, want_chords, want_slots);
    want_id: str = _want["id"];
    if want_id != _scheduled_id {
        _scheduled_id = want_id;
        if (target > 0 and target == hover_id) or len(alt_hover) == 2 {
            # hover: debounced, so sweeping the mouse across the map doesn't flood the engine
            schedule_push(150);
        } else {
            # a click or edit: send now (a microtask, so background-tab timer throttling can't delay it)
            if _push_timer {
                clearTimeout(_push_timer);
            }
            queueMicrotask(lambda { push_want(); });
        }
    }
    if want_id not in _slots {
        if _slot_count > 300 {
            _slots = {};
            _slot_count = 0;
        }
        _slots[want_id] = want_slots;
        _slot_count += 1;
    }
    # what is sounding, read straight off the timeline the engine reports playing
    saved_sounding = -1;
    if playing_tid in _slots and playing_idx >= 0 and playing_idx < len(_slots[playing_tid]) {
        slot = _slots[playing_tid][playing_idx];
        if slot.startswith("s") {
            sounding_step = int(slot[1:]);
        } elif slot.startswith("a") {
            sounding_step = -1;
        } elif slot.startswith("p") {
            saved_sounding = int(slot[1:]);
        } else {
            sounding_id = int(slot[1:]);
        }
    }

    # ---- the side panel's active tab ----
    side_body: any = None;
    if side_tab == "sound" and sc_url {
        side_body = <SuperConductorInput view={sc_input} url={sc_url}/>;
    } elif side_tab == "sound" {
        side_body = <SoundMixer
            texts={mix_texts}
            weights={mix_weights}
            engineStyle={engine_style}
            onText={lambda i: int, t: str { mix_texts = [t if j == i else x for (j, x) in enumerate(mix_texts)]; }}
            onWeight={lambda i: int, v: float { mix_weights = [v if j == i else x for (j, x) in enumerate(mix_weights)]; }}
            onCommit={commit_mix}
            onAdd={mix_add}
            onRemove={mix_remove}
            onReset={mix_reset}
        />;
    } elif side_tab == "rules" {
        side_body = <Rules mode={session.mode}/>;
    } elif side_tab == "typing" {
        side_body = <TypingGuide/>;
    } else {
        side_body = <Inspector
            focusNode={focus}
            chain={chain}
            startRn={steps[0].rn if focus_back else steps[last_idx].rn}
            backward={focus_back}
            isPinned={hover_id < 0 and selected_id >= 0}
            locked={selected_id in locks}
            story={story}
            storyLoading={story_loading}
            llmEnabled={session.llm_enabled}
            onCommitFirst={lambda { do_commit(False); }}
            onCommitPath={lambda { do_commit(True); }}
            onDescribe={do_describe}
        />;
    }

    header = <Header
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
        />;

    # the Drums page keeps the top bar; the timeline above still plays underneath
    if page == "drums" {
        return <div className="atlas drums-view">
            {header}
            <DrumsPage/>
        </div>;
    }

    return <div className={"atlas" + ("" if side_open else " side-closed")}>
        {header}
        <main className="stage">
            {if search is not None {
                <ErrorBoundary fallback={<p className="empty stage-msg">The 3D view hit an error; reload the page to bring it back.</p>}>
                <Graph3D
                    search={search}
                    anim={anim}
                    hoverId={hover_id}
                    selectedId={selected_id}
                    soundingId={sounding_id}
                    soundingStep={sounding_step}
                    loopFrom={session.loop_from}
                    onStep={lambda i: int { do_loop(i, False); }}
                    showAll={show_all}
                    steps={steps}
                    altHover={alt_hover}
                    onAltHover={lambda i: int, j: int { alt_hover = [i, j]; play_all = False; }}
                    onAltLeave={lambda { alt_hover = []; }}
                    onAltPick={do_swap}
                    onHover={on_hover}
                    onLeave={on_leave}
                    onSelect={on_select}
                    locks={locks}
                    typed={typed}
                    onAddTyped={add_typed}
                    onRemoveTyped={remove_typed}
                    saved={session.saved}
                    viewing={viewing if viewing_ok else -1}
                    savedSounding={saved_sounding}
                    onPickSaved={lambda k: int { pick_saved(-1 if k == viewing else k); }}
                />
                </ErrorBoundary>
            } else {
                <p className="empty stage-msg">Sending walkers…</p>
            }}
            <div className="stage-top">
                <div className="stage-status">
                    <b>Graph view</b>
                    <span>{
                        ("walkers on layer " + str(anim) + "…") if anim >= 0
                        else "searching before " + steps[0].rn + " and after " + steps[last_idx].rn
                            + (" · " + str(len(session.saved)) + " saved behind" if len(session.saved) > 0 else "")
                    }</span>
                    {if error { <span className="err">{error}</span> }}
                </div>
                <label className="stage-toggle" title="Also show the saved progressions, each on a plane behind the current one">
                    <input type="checkbox" checked={show_all} onChange={lambda e: ChangeEvent { show_all = not show_all; }}/>
                    show all progressions
                </label>
            </div>
            <div className={"legend" + ("" if show_legend else " folded")} onClick={lambda e: MouseEvent { show_legend = not show_legend; }} title="Click to fold or unfold the legend">
                <span><i style={{"background": "var(--f-T)"}}></i>tonic</span>
                <span><i style={{"background": "var(--f-TX)"}}></i>tonic ext.</span>
                <span><i style={{"background": "var(--f-PD)"}}></i>pre-dominant</span>
                <span><i style={{"background": "var(--f-D)"}}></i>dominant</span>
                <span><i style={{"background": "var(--f-CAD64)"}}></i>cadential ⁶₄</span>
                <span><i style={{"background": "var(--f-APP)"}}></i>applied</span>
                <span><i style={{"background": "var(--f-MIX)"}}></i>mixture</span>
                <span><i style={{"background": "var(--f-AUG6)"}}></i>aug. sixth</span>
                <span><i style={{"background": "var(--f-NEAP)"}}></i>Neapolitan</span>
                <span className="legend-note">dashed edge: cadence beyond the beam · dashed box: typed by you</span>
            </div>
        </main>
        <SidePanel
            tab={side_tab}
            open={side_open}
            hidden={side_hidden}
            onTab={open_tab}
            onOpen={set_side_open}
            onHidden={set_side_hidden}
        >{side_body}</SidePanel>
        <Progression
            steps={shown_steps}
            soundingStep={saved_sounding if viewing_ok else sounding_step}
            soundingId={sounding_id}
            previewBefore={[] if viewing_ok else preview_before}
            previewAfter={[] if viewing_ok else preview_after}
            loopFrom={view_from if viewing_ok else session.loop_from}
            loopTo={view_to if viewing_ok else session.loop_to}
            onLoop={view_loop if viewing_ok else do_loop}
            canUndo={session.can_undo}
            notice={notice}
            keyName={session.key_name}
            onUndo={do_undo}
            onClear={do_clear}
            onPlay={lambda { if play_all { play_all = False; } else { do_play(); } }}
            playAll={play_all}
            onAdd={do_add}
            saved={session.saved}
            viewing={viewing if viewing_ok else -1}
            onPick={pick_saved}
            onCommit={do_commit_progression}
            onDelete={do_delete_saved}
            onResume={do_resume_saved}
            onGuide={lambda { open_tab("typing"); set_side_open(True); }}
        />
        <Tooltip/>
    </div>;
}
