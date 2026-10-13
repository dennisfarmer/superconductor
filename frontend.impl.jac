"""Handler bodies for the SuperConductor Atlas shell (frontend.cl.jac)."""

impl app.load -> None {
    global _engine_url;
    # jac-client classifies every endpoint here as a read-only "reader" and caches
    # repeat calls with the same arguments for 60 s, so e.g. "Loop it" on the pinned
    # node never reached the server. Every endpoint here changes state: no caching.
    globalThis.__jacCacheState__ = {"cache": {}, "order": [], "pending": {}, "config": {"maxEntries": 500, "defaultTtlMs": 0}};
    try {
        s = await get_state();
        session = s;
        mix_texts = s.style_texts;
        mix_weights = s.style_weights;
        _engine_url = s.engine_url;
    } except Exception as e {
        error = "could not load session: " + str(e);
        return;
    }
    await send_walkers();
}

"""Spawn the BeamSearch walker server-side, then replay its layers on the map."""
impl app.send_walkers -> None {
    global _search_id;
    walking = True;
    error = "";
    hover_id = -1;
    selected_id = -1;
    locks = [];
    typed = [];
    story = None;
    try {
        res = await explore();
        _search_id = res.search_id;
        anim = 0;
        search = res;
        for k in range(res.depth + 1) {
            anim = k;
            await sleep_ms(620);
        }
    } except Exception as e {
        error = "walkers failed: " + str(e);
    }
    anim = -1;
    walking = False;
}

impl app.poll_engine -> None {
    global _engine_was_ok, _sent_id, _bpm, _play_seen, _restarting;
    if _stopped or not _engine_url {
        return;
    }
    try {
        resp: any = await fetch(_engine_url + "/status");
        data: any = await resp.json();
        engine_ok = True;
        if not _engine_was_ok {
            # the engine (re)started: give it the sound and settings, and the timeline again
            _engine_was_ok = True;
            _restarting = False;
            restarting = False;
            _sent_id = "";
            session = await get_state();
        }
        ms: any = data.ms_per_frame;
        if data.sink {
            engine_info = str(data.model);   # make run: the sound comes from SuperConductor
            # the Sound tab shows what the SuperConductor client sends to MRT2
            # (a local: sc_url read here would be this timer's stale copy)
            client_midi = str(data.client_midi);
            sc_url = client_midi;
            try {
                sc_resp: any = await fetch(client_midi + "/input");
                sc_input = await sc_resp.json();
            } except Exception {
                sc_input = None;
            }
        } else {
            sc_url = "";
            engine_info = str(data.model) + " · " + (str(ms) + " ms/frame" if ms else "warming up");
        }
        engine_style = str(data.style);
        paused = bool(data.paused);
        if Date.now() - _vol_at > 1500 and data.volume is not None {
            volume = float(data.volume);
            muted = bool(data.muted);
        }
        if data.settings and data.settings.bpm {
            _bpm = float(data.settings.bpm);
        }
        playing: any = data.playing;
        if playing and data.settings {
            beat_sync(
                int(playing.beat), float(playing.beat_frac or 0), float(data.settings.bpm),
                int(data.settings.beats_per_chord), float(data.latency_ms or 0), not data.paused
            );
        } else {
            beat_sync(0, 0.0, _bpm, 4, 0.0, False);
        }
        now_tid = "";
        now_role = "";
        if playing {
            now_role = str(playing.role);
            now_tid = str(playing.timeline_id);
            playing_role = now_role;
            playing_tid = now_tid;
            playing_idx = int(playing.index);
        }
        # keep the engine on the wanted timeline: resend until it reports playing it
        if "id" in _want {
            want = _want;
            if want["role"] == "play" {
                if now_tid == want["id"] {
                    _play_seen = True;
                } elif _play_seen {
                    # "Play all" ran to the end: back to the loop
                    _play_seen = False;
                    play_all = False;
                    return;
                }
            }
            grace = 60000.0 / _bpm + 1500.0;
            if now_tid != want["id"] and (_sent_id != want["id"] or Date.now() - _sent_at > grace) {
                push_want();
            }
        }
    } except Exception {
        _engine_was_ok = False;
        engine_ok = False;
        engine_info = "engine restarting…" if _restarting else "engine offline";
        playing_role = "";
        beat_sync(0, 0.0, _bpm, 4, 0.0, False);
    }
}

"""Pause/resume the engine's audio directly (like the status poll)."""
impl app.toggle_pause -> None {
    if not _engine_url {
        return;
    }
    want = not paused;
    paused = want;
    await fetch(_engine_url + "/pause", {"method": "POST", "headers": {"Content-Type": "application/json"}, "body": JSON.stringify({"paused": want})});
}


"""Hover a map node: the page plays the path to it (see `_want` in frontend.cl.jac)."""
impl app.on_hover(node_id: int) -> None {
    hover_id = node_id;
    play_all = False;
}

"""Leave: playback falls back to the half-committed path, else the loop."""
impl app.on_leave -> None {
    hover_id = -1;
}

"""Click a map node to half-commit its column: hover there is ignored, and later
columns only offer paths through it. Click it again to release the column (and any
later locks that hung off it); click another chord in the column to move the lock."""
impl app.on_select(node_id: int) -> None {
    story = None;
    if search is None {
        return;
    }
    by_id = node_map(search, typed);
    if node_id not in by_id {
        return;
    }
    back = is_back_id(node_id);
    layer = by_id[node_id].layer;
    kept: list[int] = [];
    if node_id in locks {
        for l in locks {
            if is_back_id(l) != back or by_id[l].layer < layer {
                kept.append(l);
            }
        }
    } else {
        for l in locks {
            if is_back_id(l) != back or by_id[l].layer < layer {
                kept.append(l);
            } elif by_id[l].layer > layer and ancestor_at(by_id, l, layer) == node_id {
                kept.append(l);   # a later lock on a path through the new one still holds
            }
        }
        kept.append(node_id);
    }
    locks = kept;
    pin_deepest(back, kept, typed);
    hover_id = -1;
}

"""Make the deepest lock of a tree the pinned path (Inspector, chips, idle loop)."""
impl app.pin_deepest(back: bool, lock_ids: list[int], typed_now: list[ChordView]) -> None {
    if search is None {
        return;
    }
    by_id = node_map(search, typed_now);
    deepest = deepest_lock(by_id, lock_ids, back);
    if deepest < 0 {
        deepest = deepest_lock(by_id, lock_ids, not back);
    }
    selected_id = deepest;
}

"""Type a chord into column `layer` of one tree. It links from every chord in the
previous column you can still reach (only the locked one, if that column is locked)
that may legally move to it."""
impl app.add_typed(back: bool, layer: int, text: str) -> None {
    if search is None or not text.strip() {
        return;
    }
    cur: SearchView = search;
    by_id = node_map(cur, typed);
    start_id = BACK_BASE if back else 0;
    parents: list[int] = [];
    if layer == 1 {
        parents = [start_id];
    } else {
        held = lock_at(by_id, locks, back, layer - 1);
        if held >= 0 {
            parents = [held];
        } else {
            column: list[ChordView] = cur.back_nodes if back else cur.nodes;
            column = column + typed;
            for n in column {
                if n.layer == layer - 1 and is_back_id(n.id) == back and reachable(by_id, locks, n.id) {
                    parents.append(n.id);
                }
            }
        }
    }
    res = await add_option(_search_id, parents, text);
    error = res.notice;
    if res.notice {
        return;
    }
    # the typed chord first, then the mini walker's finds one column at a time
    shown: list[ChordView] = typed + [n for n in res.nodes if n.status == "custom"];
    typed = shown;
    grown = [n for n in res.nodes if n.status == "grown"];
    later = sorted(list(set([n.layer for n in grown])));
    for L in later {
        await sleep_ms(450);
        shown = shown + [n for n in grown if n.layer == L];
        typed = shown;
    }
}

"""Remove a typed option (all its links), anything typed after it, and their locks."""
impl app.remove_typed(ids: list[int]) -> None {
    gone: list[int] = [i for i in ids];
    changed = True;
    while changed {
        changed = False;
        for n in typed {
            if n.parent in gone and n.id not in gone {
                gone.append(n.id);
                changed = True;
            }
        }
    }
    remaining = [n for n in typed if n.id not in gone];
    typed = remaining;
    if any([l in gone for l in locks]) {
        held = [l for l in locks if l not in gone];
        locks = held;
        pin_deepest(selected_id >= BACK_BASE, held, remaining);
    }
    if hover_id in gone {
        hover_id = -1;
    }
}

"""Keep: put the hovered/pinned chord (or its whole path) into the progression. The
server keeps the other options beside each kept chord, so it can be swapped later."""
impl app.do_commit(whole: bool) -> None {
    target = hover_id if hover_id >= 0 else selected_id;
    if target <= 0 or _search_id < 0 {
        return;
    }
    s = await add_path(_search_id, target, whole);
    session = s;
    await send_walkers();
}

"""Swap the kept chord at step `index` for one of its alternatives."""
impl app.do_swap(index: int, cid: str) -> None {
    alt_hover = [];
    s = await swap_step(index, cid);
    session = s;
    notice = s.notice;
    await send_walkers();
}

impl app.do_undo -> None {
    s = await undo();
    session = s;
    await send_walkers();
}

impl app.do_clear -> None {
    s = await clear_progression();
    session = s;
    await send_walkers();
}

"""Play the progression once from the top (its last 16 chords), then back to the loop."""
impl app.do_play -> None {
    global _play_seen;
    _play_seen = False;
    hover_id = -1;
    play_all = True;
}

impl app.change_key(tonic: str, mode: str) -> None {
    if session is None {
        return;
    }
    cur: SessionView = session;
    tonics = cur.major_tonics if mode == "major" else cur.minor_tonics;
    t = tonic if tonic in tonics else tonics[0];
    s = await set_key(t, mode);
    session = s;
    await send_walkers();
}

impl app.change_tempo(bpm: float, beats: int) -> None {
    if not bpm or not beats {
        return;
    }
    session = await set_tempo(bpm, beats);
}

impl app.do_describe -> None {
    target = hover_id if hover_id >= 0 else selected_id;
    if target <= 0 {
        return;
    }
    story_loading = True;
    try {
        story = await describe(_search_id, target);
    } except Exception as e {
        error = "describe failed: " + str(e);
    }
    story_loading = False;
}

impl app.change_texture(texture: str) -> None {
    session = await set_texture(texture);
}

impl app.toggle_llm(enabled: bool) -> None {
    session = await set_llm(enabled);
}

"""Send the current layers + knob weights; the engine re-blends the style embedding."""
impl app.commit_mix -> None {
    if session is None or len(mix_texts) == 0 {
        return;
    }
    session = await set_style_layers(mix_texts, mix_weights);
}

impl app.mix_add(prompt: str) -> None {
    mix_texts = mix_texts + ["" if prompt == "custom" else prompt];
    mix_weights = mix_weights + [0.5];
    if prompt != "custom" {
        setTimeout(lambda { commit_mix(); }, 50);
    }
}

impl app.mix_remove(i: int) -> None {
    texts = [t for (j, t) in enumerate(mix_texts) if j != i];
    weights = [w for (j, w) in enumerate(mix_weights) if j != i];
    mix_texts = texts;
    mix_weights = weights;
    setTimeout(lambda { set_style_layers(texts, weights); }, 50);
}

impl app.mix_reset -> None {
    mix_texts = [DEFAULT_SOUND];
    mix_weights = [1.0];
    session = await set_style_layers([DEFAULT_SOUND], [1.0]);
}

"""Typed chords: "before" the first chord, "after" the last, or "replace" everything."""
impl app.do_add(text: str, where: str) -> None {
    before = len(session.steps);
    s = await add_chords(text, where);
    session = s;
    notice = s.notice;
    if s.notice {
        return;
    }
    await send_walkers();
}

"""Relaunch the audio engine process; poll_engine re-syncs it once it is back."""
impl app.do_restart_engine -> None {
    global _engine_was_ok, _restarting;
    _restarting = True;
    restarting = True;
    engine_ok = False;
    _engine_was_ok = False;
    engine_info = "engine restarting…";
    try {
        if not await restart_engine() {
            _restarting = False;
            restarting = False;
            error = "The audio engine did not stop when asked, so it was not restarted.";
        }
    } except Exception as e {
        _restarting = False;
        restarting = False;
        error = "engine restart failed: " + str(e);
    }
}

"""Progression clicks set the loop range. Click: loop from this chord to the end
(click the start again for the default, just the last chord). Shift-click: end the
loop at this chord, so any stretch can be looped."""
impl app.do_loop(index: int, extend: bool) -> None {
    cur = session.loop_from;
    start = index;
    end = -1;
    if extend and cur >= 0 and index != cur {
        start = min(cur, index);
        end = max(cur, index);
    } elif not extend and index == cur {
        start = -1;
    }
    session = await set_loop(start, end);
}

"""Loop range inside the committed progression being played (same clicks as for the
current one; kept on the page only)."""
impl app.view_loop(index: int, extend: bool) -> None {
    if extend and view_from >= 0 and index != view_from {
        view_to = max(view_from, index);
        view_from = min(view_from, index);
    } elif not extend and index == view_from {
        view_from = -1;
        view_to = -1;
    } else {
        view_from = index;
        view_to = -1;
    }
}

"""Play a committed progression (its loop takes over the engine); -1: back to the current one."""
impl app.pick_saved(index: int) -> None {
    viewing = index;
    view_from = -1;
    view_to = -1;
    play_all = False;
}

"""Save: shelve this progression (it moves behind in the graph view) and start a new one."""
impl app.do_commit_progression -> None {
    s = await commit_progression("");
    session = s;
    viewing = -1;
    await send_walkers();
}

impl app.do_delete_saved -> None {
    if viewing < 0 {
        return;
    }
    s = await delete_saved(viewing);
    session = s;
    viewing = -1;
}

"""Make the committed progression being played current again (the current one is
committed in its place unless it is only the opening tonic)."""
impl app.do_resume_saved -> None {
    if viewing < 0 {
        return;
    }
    s = await resume_saved(viewing);
    session = s;
    viewing = -1;
    await send_walkers();
}

"""Output level, straight to the engine (applied at the speaker, so it is immediate)."""
impl app.set_volume(v: float) -> None {
    global _vol_at;
    if not _engine_url {
        return;
    }
    _vol_at = Date.now();
    # state reads in this handler still see the old values: work from locals
    volume = v;
    body: dict[str, any] = {"volume": v};
    if muted and v > 0.0 {
        # turning the volume up unmutes
        muted = False;
        body["muted"] = False;
    }
    await engine_post("/mix", body);
}

impl app.toggle_mute -> None {
    global _vol_at;
    if not _engine_url {
        return;
    }
    _vol_at = Date.now();
    want = not muted;   # `muted` keeps its old value for the rest of this handler
    muted = want;
    await engine_post("/mix", {"muted": want});
}

"""Quit: the server stops the engine, then itself (services.atlas.shutdown_app).
The page stops polling and sending, and says so."""
impl app.do_quit -> None {
    global _stopped;
    quitting = True;
    try {
        await shutdown_app();
    } except Exception as e {
        quitting = False;
        error = "Quit failed: " + str(e);
        return;
    }
    _stopped = True;
    stopped = True;
}
