"""Top bar, left to right: key; transport (pause, tempo knob, the beat, beats per
chord, texture, and the Harmony - Drums page switch; the BPM label opens the BPM page); output (mute, volume); engine status and restart; Send walkers;
quit. The LLM switch sits in the menu beside the status pill."""

import from .Knob { Knob }
import from .BeatLights { BeatLights }
sv import from ..services.atlas { SessionView }

glob TEXTURES: list[list[str]] = [
    ["block", "block chords"],
    ["arpeggio", "arpeggio"],
    ["improvise", "improvise"]
];

# the tempo knob's range
glob BPM_MIN: float = 40.0,
     BPM_MAX: float = 200.0;


def bpm_to_knob(bpm: float) -> float {
    return (bpm - BPM_MIN) / (BPM_MAX - BPM_MIN);
}


def knob_to_bpm(v: float) -> float {
    return float(Math.round(BPM_MIN + v * (BPM_MAX - BPM_MIN)));
}


"""Speaker glyph: waves by level, a cross when muted."""
def speaker_icon(volume: float, muted: bool) -> JsxElement {
    return <svg className="ic" viewBox="0 0 20 20" width="18" height="18" aria-hidden="true">
        <path d="M3 7.5h3l4-3.5v12l-4-3.5H3z" fill="currentColor"/>
        {if muted {
            <path d="M13 7.5l5 5M18 7.5l-5 5" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round"/>
        } else {
            <g fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round">
                {if volume > 0.02 { <path d="M12.8 7.6a3.4 3.4 0 0 1 0 4.8"/> }}
                {if volume > 0.5 { <path d="M15.2 5.3a6.6 6.6 0 0 1 0 9.4"/> }}
            </g>
        }}
    </svg>;
}


def restart_icon -> JsxElement {
    return <svg className="ic" viewBox="0 0 20 20" width="16" height="16" aria-hidden="true">
        <path d="M15.5 10a5.5 5.5 0 1 1-1.6-3.9" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round"/>
        <path d="M14.6 2.8v3.8h-3.8" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round"/>
    </svg>;
}


def power_icon -> JsxElement {
    return <svg className="ic" viewBox="0 0 20 20" width="16" height="16" aria-hidden="true">
        <path d="M6.3 5.6a6 6 0 1 0 7.4 0" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round"/>
        <path d="M10 3v6.5" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round"/>
    </svg>;
}


def:pub Header(
    page: str,
    session: SessionView,
    engineOk: bool,
    engineInfo: str,
    paused: bool,
    onPause: Callable[[], None],
    restarting: bool,
    onRestart: Callable[[], None],
    volume: float,
    muted: bool,
    onVolume: Callable[[float], None],
    onMute: Callable[[], None],
    quitting: bool,
    onQuit: Callable[[], None],
    walking: bool,
    onKey: Callable[[str, str], None],
    onTempo: Callable[[float, int], None],
    onTexture: Callable[[str], None],
    onLlm: Callable[[bool], None],
    onSend: Callable[[], None]
) -> JsxElement {
    has menu_open: bool = False,
        confirm_quit: bool = False,
        # where an open menu sits on screen: the bar scrolls sideways, so menus are
        # fixed-position (a scrolling bar would clip them) and placed under their button
        menu_at: dict = {},
        # the tempo while the knob is being turned (-1: not turning)
        drag_bpm: float = -1.0;
    tonics = session.major_tonics if session.mode == "major" else session.minor_tonics;
    bpm_now = drag_bpm if drag_bpm >= 0.0 else session.bpm;

    def commit_tempo -> None {
        if drag_bpm >= 0.0 {
            if drag_bpm != session.bpm {
                onTempo(drag_bpm, session.beats_per_chord);
            }
            drag_bpm = -1.0;
        }
    }

    def place_menu(e: any) -> None {
        r: any = e.currentTarget.getBoundingClientRect();
        menu_at = {"top": str(r.bottom + 6) + "px", "right": str(max(8, window.innerWidth - r.right)) + "px"};
    }

    return <header className="topbar" onScroll={lambda e: UIEvent { menu_open = False; confirm_quit = False; }}>
        <div className="brand" title="Walkers search a graph of chord progressions; Magenta RT plays what you hover, pin and keep">SuperConductor Atlas</div>

        <div className="tb-group">
            <select className="tb-select key-select" title="Tonic of the key. Changing key transposes the progression and sends the walkers again." value={session.tonic} onChange={lambda e: ChangeEvent { onKey(e.target.value, session.mode); }}>
                {for t in tonics { <option key={t} value={t}>{t}</option> }}
            </select>
            <div className="seg" title="Major or minor mode">
                <button className={"on" if session.mode == "major" else ""} onClick={lambda e: MouseEvent { onKey(session.tonic, "major"); }}>major</button>
                <button className={"on" if session.mode == "minor" else ""} onClick={lambda e: MouseEvent { onKey(session.tonic, "minor"); }}>minor</button>
            </div>
        </div>

        <div className="tb-group transport">
            <button
                className={"icon-btn" + (" on" if not paused else "")}
                disabled={not engineOk}
                title={"Resume the audio" if paused else "Pause the audio (Magenta stops generating)"}
                onClick={lambda e: MouseEvent { onPause(); }}
            >{"▶" if paused else "⏸"}</button>
            <div className="tempo" title="Tempo: drag the knob up or down (or scroll on it), or type a value. Each chord lasts the beats shown by the lights.">
                <Knob
                    value={bpm_to_knob(bpm_now)}
                    size={30}
                    color="#8fb3ff"
                    onChange={lambda v: float { drag_bpm = knob_to_bpm(v); }}
                    onCommit={commit_tempo}
                />
                <input
                    className="tempo-in"
                    type="number"
                    key={"bpm" + str(bpm_now)}
                    defaultValue={str(int(bpm_now))}
                    onBlur={lambda e: FocusEvent {
                        if e.target.value and float(e.target.value) != session.bpm {
                            onTempo(min(BPM_MAX, max(BPM_MIN, float(e.target.value))), session.beats_per_chord);
                        }
                    }}
                    onKeyDown={lambda e: KeyboardEvent { if e.key == "Enter" { e.target.blur(); } }}
                />
                <a className={"unit bpm-link" + (" on" if page == "bpm" else "")} href="#bpm" title="The BPM page: sync with SuperConductor's tempo, and how messages reach the audio">BPM</a>
            </div>
            <BeatLights beats={session.beats_per_chord} running={engineOk and not paused}/>
            <label className="tb-field" title="Beats each chord lasts">
                <input
                    className="beats-in"
                    type="number"
                    min="1"
                    max="8"
                    value={str(session.beats_per_chord)}
                    onChange={lambda e: ChangeEvent { if e.target.value { onTempo(session.bpm, int(e.target.value)); } }}
                />
                <span className="unit">beats</span>
            </label>
            <select className="tb-select" title="How Magenta is asked to play each chord: held block chords, broken into an arpeggio, or improvise (the bass is given; the model chooses which chord tones to play, and when)" value={session.texture} onChange={lambda e: ChangeEvent { onTexture(e.target.value); }}>
                {for tx in TEXTURES { <option key={tx[0]} value={tx[0]}>{tx[1]}</option> }}
            </select>
            <nav className="page-switch">
                <a className={"on" if page != "drums" else ""} href="#" title="The Harmony page: the chord graph and the progression">Harmony</a>
                <span className="page-sep">-</span>
                <a className={"on" if page == "drums" else ""} href="#drums" title="The Drums page: when the model hits drums, as a 4/4 grid of -1 / 0 / 1">Drums</a>
            </nav>
        </div>

        <div className="tb-sp"></div>

        <div className={"tb-group output" + (" muted" if muted else "")}>
            <button
                className={"icon-btn" + (" on" if muted else "")}
                disabled={not engineOk}
                title={"Unmute" if muted else "Mute (the music keeps playing silently, so the beat stays in place)"}
                onClick={lambda e: MouseEvent { onMute(); }}
            >{speaker_icon(volume, muted)}</button>
            <input
                className="volume"
                type="range"
                min="0"
                max="100"
                step="1"
                disabled={not engineOk}
                value={str(int(Math.round(volume * 100.0)))}
                style={{"--fill": str(int(Math.round(volume * 100.0))) + "%"}}
                title={"Volume " + str(int(Math.round(volume * 100.0))) + "%"}
                onChange={lambda e: ChangeEvent { onVolume(float(e.target.value) / 100.0); }}
            />
        </div>

        <div className="tb-group engine">
            <span className="pill" title={"Audio engine (Magenta RT) at " + session.engine_url + ": model, and milliseconds to generate each 40 ms audio frame (under 40 keeps up with real time)"}>
                <span className={"dot " + ("on" if engineOk else "off")}></span>
                {engineInfo}
            </span>
            <button
                className={"icon-btn" + (" spin" if restarting else "")}
                disabled={restarting or quitting}
                title="Restart the audio engine (fixes glitching audio). It is asked to stop, then started again; the model takes a few seconds to load."
                onClick={lambda e: MouseEvent { onRestart(); }}
            >{restart_icon()}</button>
            <button className={"icon-btn" + (" on" if menu_open else "")} title="Engine options" onClick={lambda e: MouseEvent { place_menu(e); menu_open = not menu_open; }}>⋯</button>
            {if menu_open {
                <div className="menu fixed" style={menu_at} onMouseLeave={lambda e: MouseEvent { menu_open = False; }}>
                    <label title="LLM calls (Ollama) share the GPU with Magenta; switch off for glitch-free real time">
                        <input type="checkbox" checked={session.llm_enabled} onChange={lambda e: ChangeEvent { onLlm(not session.llm_enabled); }}/>
                        LLM path descriptions
                    </label>
                </div>
            }}
        </div>

        <button className="btn primary" disabled={walking} title="Run the beam search again, forwards from the last chord and backwards from the first, with the current settings" onClick={lambda e: MouseEvent { onSend(); }}>
            {"Walkers out…" if walking else "Send walkers"}
        </button>

        <div className="pop-anchor">
            <button
                className={"icon-btn quit" + (" on" if confirm_quit else "")}
                disabled={quitting}
                title="Quit SuperConductor Atlas: stop the audio engine and the app server"
                onClick={lambda e: MouseEvent { place_menu(e); confirm_quit = not confirm_quit; }}
            >{power_icon()}</button>
            {if confirm_quit {
                <div className="menu confirm fixed" style={menu_at}>
                    <p>Quit SuperConductor Atlas?</p>
                    <span>The audio engine and the app server shut down. Your session is kept for next time.</span>
                    <div className="confirm-actions">
                        <button className="btn ghost" onClick={lambda e: MouseEvent { confirm_quit = False; }}>Cancel</button>
                        <button className="btn danger" disabled={quitting} onClick={lambda e: MouseEvent { onQuit(); }}>{"Quitting…" if quitting else "Quit"}</button>
                    </div>
                </div>
            }}
        </div>
    </header>;
}
