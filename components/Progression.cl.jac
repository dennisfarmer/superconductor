"""The dock along the bottom: the progression being written (or a saved one being
played) as a strip of chips, with the chord Magenta is sounding lit up, and the
progression actions, including the list of saved progressions. The strip scrolls
sideways (the mouse wheel scrolls it too) when it is wider than the dock.

The path under the mouse (or pinned) on the graph shows up as dashed chips on
whichever side it extends."""

import from react { useEffect }
sv import from ..services.atlas { StepView, ChordView, SavedView }


def chip_body(rn: str, symbol: str, cadence: str) -> list[any] {
    return [
        <div key="rn" className="rn">{rn}</div>,
        <div key="sym" className="sym">{symbol}</div>,
        (<span key="cad" className="cad">{cadence}</span>) if cadence else None
    ];
}


def:pub Progression(
    steps: list[StepView],
    soundingStep: int,
    soundingId: int,
    previewBefore: list[ChordView],
    previewAfter: list[ChordView],
    loopFrom: int,
    loopTo: int,
    onLoop: Callable[[int, bool], None],
    canUndo: bool,
    notice: str,
    keyName: str,
    onUndo: Callable[[], None],
    onClear: Callable[[], None],
    onPlay: Callable[[], None],
    playAll: bool,
    saved: list[SavedView],
    viewing: int,
    onPick: Callable[[int], None],
    onCommit: Callable[[str], None],
    onRename: Callable[[int, str], None],
    onDelete: Callable[[], None],
    onResume: Callable[[], None],
    keepPath: list[ChordView],
    onKeep: Callable[[bool], None]
) -> JsxElement {
    has more: bool = False,
        lib: bool = False,
        # the Save popover: open, and the name being typed
        naming: bool = False,
        name: str = "";

    # keep the sounding chip in view as playback moves along the strip
    useEffect(lambda {
        el: any = document.querySelector(".dock .chip.sounding");
        strip: any = document.querySelector(".dock .chips");
        if el and strip {
            left = el.offsetLeft - strip.offsetLeft;
            if left < strip.scrollLeft or left + el.offsetWidth > strip.scrollLeft + strip.clientWidth {
                strip.scrollTo({"left": left - 40, "behavior": "smooth"});
            }
        }
    }, [soundingStep, soundingId, len(steps)]);

    # a vertical mouse wheel scrolls the strip sideways
    def scroll_chips(e: any) -> None {
        strip: any = e.currentTarget;
        if e.deltaY != 0 and Math.abs(e.deltaY) > Math.abs(e.deltaX) and strip.scrollWidth > strip.clientWidth {
            strip.scrollLeft = strip.scrollLeft + e.deltaY;
        }
    }

    viewing_ok = viewing >= 0 and viewing < len(saved);

    def save -> None {
        onCommit(name);
        naming = False;
        name = "";
    }

    def rename(e: any) -> None {
        if e.target.value.strip() and e.target.value.strip() != saved[viewing].name {
            onRename(viewing, e.target.value);
        } else {
            e.target.value = saved[viewing].name;
        }
    }
    # the loop: loopFrom..loopTo, loopFrom..end, or (default) just the last chord
    lo = loopFrom if loopFrom >= 0 else len(steps) - 1;
    hi = loopTo if loopTo >= 0 else len(steps) - 1;
    # Play all covers the last 16 chords; while it runs, show the position in it
    play_first = max(0, len(steps) - 16);
    play_pos = (str(soundingStep - play_first + 1) + " / " + str(len(steps) - play_first)) if soundingStep >= play_first else "Playing";
    # the looped chords by name, e.g. "ii⁷ → V⁷ → I" (long loops: first … last)
    loop_names = "";
    if len(steps) > 0 and 0 <= lo and lo <= hi and hi < len(steps) {
        if hi - lo < 4 {
            for i in range(lo, hi + 1) {
                loop_names = loop_names + (" → " if i > lo else "") + steps[i].rn;
            }
        } else {
            loop_names = steps[lo].rn + " → … → " + steps[hi].rn + " (" + str(hi - lo + 1) + ")";
        }
    }

    return <footer className={"dock" + (" viewing" if viewing_ok else "")}>
        {if viewing_ok {
            <div className="dock-band">
                <b>Saved:</b>
                <input
                    className="saved-name"
                    key={"nm" + str(viewing) + saved[viewing].name}
                    defaultValue={saved[viewing].name}
                    title="Rename this progression (Enter to keep the name)"
                    onBlur={lambda e: FocusEvent { rename(e); }}
                    onKeyDown={lambda e: KeyboardEvent {
                        if e.key == "Enter" { e.target.blur(); }
                        elif e.key == "Escape" { e.target.value = saved[viewing].name; e.target.blur(); }
                    }}
                />
                <span>{saved[viewing].key_name + " · click a chord to loop from it, shift-click to end the loop"}</span>
                <span className="tb-sp"></span>
                <button className="btn primary" onClick={lambda e: MouseEvent { onPick(-1); }}>← Back to current</button>
                <button className="btn" title="Continue writing this progression (the current one is saved in its place)" onClick={lambda e: MouseEvent { onResume(); }}>Continue this one</button>
                <button className="btn ghost" title="Forget this saved progression" onClick={lambda e: MouseEvent { onDelete(); }}>Delete</button>
            </div>
        }}
        <div className="dock-row">
            <button
                className={"play-btn" + (" on" if playAll else "")}
                title={
                    "Playing the whole progression once, start to end. Afterwards it goes back to the loop by itself; click to go back now."
                    if playAll else
                    "Play the whole progression once, from the first chord to the last (up to 16 chords), then go back to the loop. Hovering a chord on the graph stops it."
                }
                onClick={lambda e: MouseEvent { onPlay(); }}
            >
                <span className="play-ic">{"■" if playAll else "▶"}</span>
                <span className="play-txt">{play_pos if playAll else "Play all"}</span>
            </button>
            <div className={"loop-now" + (" off" if playAll else "")} title="What Magenta repeats while you're not hovering anything. Click a chord to loop from it; shift-click another to end the loop there.">
                <span className="loop-label">Loop</span>
                <span className="loop-names">{loop_names}</span>
            </div>
            <div className="chips" onWheel={scroll_chips} title="Your progression. Click a chord to loop from it (click it again for the default loop); shift-click another to end the loop there. Dashed chords are the path you are hovering or have pinned on the graph, not yet kept.">
                {for c in previewBefore {
                    <div key={"pb" + str(c.id)} className="chip-wrap">
                        <div className={"chip tentative" + (" sounding" if c.id == soundingId else "")} title={c.rule}>
                            {chip_body(c.rn, c.symbol, c.cadence)}
                        </div>
                        <span className="arrow">→</span>
                    </div>
                }}
                {for (i, st) in enumerate(steps) {
                    <div key={"s" + str(i)} className="chip-wrap">
                        {if i > 0 { <span className="arrow">→</span> }}
                        <div
                            className={
                                "chip clickable" + (" current" if i == len(steps) - 1 and not viewing_ok else "")
                                + (" sounding" if i == soundingStep else "") + (" loop-start" if i == loopFrom else "")
                                + (" in-loop" if lo <= i and i <= hi and not playAll else "") + (" loop-end" if i == loopTo else "")
                            }
                            title={(
                                "Looping from here: click again for the default loop" if i == loopFrom
                                else "Click: loop from this chord · Shift-click: end the loop here"
                            ) + ("\n" + st.rule if st.rule else "")}
                            onClick={lambda e: MouseEvent { onLoop(i, e.shiftKey); }}
                        >
                            {if i == lo and not playAll { <span className="loop-tag">LOOP</span> }}
                            {chip_body(st.rn, st.symbol, st.cadence)}
                        </div>
                    </div>
                }}
                {for c in previewAfter {
                    <div key={"pa" + str(c.id)} className="chip-wrap">
                        <span className="arrow">→</span>
                        <div className={"chip tentative" + (" sounding" if c.id == soundingId else "")} title={c.rule}>
                            {chip_body(c.rn, c.symbol, c.cadence)}
                        </div>
                    </div>
                }}
            </div>

            {if not viewing_ok {
                <div className="dock-actions">
                    <button className="icon-btn" disabled={not canUndo} title="Undo" onClick={lambda e: MouseEvent { onUndo(); }}>↶</button>
                    <button
                        className="btn primary"
                        disabled={len(keepPath) == 0}
                        title={"Put it into the progression; the other options stay beside it, so you can swap later" if len(keepPath) > 0 else "Hover or pin a chord on the graph to keep it"}
                        onClick={lambda e: MouseEvent { onKeep(False); }}
                    >{"Keep " + keepPath[0].rn if len(keepPath) > 0 else "Keep"}</button>
                    {if len(keepPath) > 1 {
                        <button className="btn" title="Put every chord on this path into the progression at once" onClick={lambda e: MouseEvent { onKeep(True); }}>
                            {"Keep whole path (" + str(len(keepPath)) + ")"}
                        </button>
                    }}
                    <div className="pop-anchor">
                        <button
                            className={"btn" + (" on" if naming else "")}
                            disabled={len(steps) < 2}
                            title="Name this progression, shelve it (it moves behind in the graph view) and start a new one"
                            onClick={lambda e: MouseEvent { naming = not naming; more = False; lib = False; }}
                        >✓ Save</button>
                        {if naming {
                            <div className="menu up save-menu">
                                <input
                                    className="save-name"
                                    autoFocus={True}
                                    placeholder={"Progression " + str(len(saved) + 1)}
                                    value={name}
                                    onChange={lambda e: ChangeEvent { name = e.target.value; }}
                                    onKeyDown={lambda e: KeyboardEvent {
                                        if e.key == "Enter" { save(); }
                                        elif e.key == "Escape" { naming = False; }
                                    }}
                                />
                                <div className="confirm-actions">
                                    <button className="btn ghost" onClick={lambda e: MouseEvent { naming = False; }}>Cancel</button>
                                    <button className="btn primary" onClick={lambda e: MouseEvent { save(); }}>Save</button>
                                </div>
                            </div>
                        }}
                    </div>
                    <div className="pop-anchor">
                        <button className={"icon-btn" + (" on" if more else "")} title="More" onClick={lambda e: MouseEvent { more = not more; lib = False; naming = False; }}>⋯</button>
                        {if more {
                            <div className="menu up" onMouseLeave={lambda e: MouseEvent { more = False; }}>
                                <button disabled={len(steps) < 2} onClick={lambda e: MouseEvent { more = False; onClear(); }}>Clear progression</button>
                            </div>
                        }}
                    </div>
                </div>
            }}

            <div className="pop-anchor">
                <button
                    className={"btn ghost" + (" on" if lib or viewing_ok else "")}
                    disabled={len(saved) == 0}
                    title="Saved progressions: pick one to hear it and see it highlighted behind the current one in the graph"
                    onClick={lambda e: MouseEvent { lib = not lib; more = False; naming = False; }}
                >{"Saved " + str(len(saved)) + " ▴"}</button>
                {if lib {
                    <div className="menu up right" onMouseLeave={lambda e: MouseEvent { lib = False; }}>
                        {for p in saved {
                            <button
                                key={"sv" + str(p.index)}
                                className={"lib-item" + (" on" if p.index == viewing else "")}
                                onClick={lambda e: MouseEvent { lib = False; onPick(-1 if p.index == viewing else p.index); }}
                            >
                                <b>{p.name}</b>
                                <span>{p.key_name + " · " + str(len(p.steps)) + " chords"}</span>
                            </button>
                        }}
                        {if viewing_ok {
                            <button className="lib-item" onClick={lambda e: MouseEvent { lib = False; onPick(-1); }}><b>← Current progression</b></button>
                        }}
                    </div>
                }}
            </div>
        </div>
        {if notice { <div className="dock-notice">{notice}</div> }}
    </footer>;
}
