"""The BPM page (/#bpm, from the BPM label in the top bar): tempo sync with the
SuperConductor server, and how Atlas's messages reach the audio.

Sync (POST /sync on the engine, see audio_engine/engine.py SinkEngine._syncer):
Atlas's tempo follows the server's heard tempo, Atlas's latency estimate is
calibrated against where the server hears the beat, and the server's beat one is
pinned to Atlas's. The engine reads the server's /tempo/status (it sends no CORS
headers, so the page can't) and reports it in its own /status as `sync`."""


"""A number with `digits` decimals, or – when it is missing (null or undefined)."""
def num(v: any, digits: int) -> str {
    if v is None or Number.isNaN(Number(v)) {
        return "–";
    }
    return Number(v).toFixed(digits);
}


def beat_of(beat: any, bar: any) -> str {
    if beat is None {
        return "–";
    }
    return str(int(beat) + 1) + " / " + str(bar or 4);
}


def:pub TempoPage(status: any, bpm: float, onSync: Callable[[bool], None]) -> JsxElement {
    sink = status is not None and status.sink;
    sync: any = status.sync if sink else None;
    srv: any = sync.server if sync and sync.ok else None;
    on = bool(sync and sync.on);
    playing: any = status.playing if status is not None else None;
    phase: any = sync.phase_beats if sync and sync.on else None;
    phase_ms: any = None if num(phase, 3) == "–" else phase * 60000.0 / bpm;

    return <div className="tempo-page">
        <header className="drums-head">
            <div>
                <h1>Tempo</h1>
                <p className="drums-sub">Atlas's beat clock and the audio you hear, side by side.</p>
            </div>
        </header>

        <section className="tp-sync">
            <button
                className={"btn " + ("primary" if not on else "ghost on")}
                disabled={not sink}
                title={"Follow the SuperConductor server's heard tempo, calibrate Atlas's latency against the beat the server hears, and pin the server's beat one to Atlas's. A manual tempo change turns it off." if sink else "Sync needs the SuperConductor mode (make run)."}
                onClick={lambda e: MouseEvent { onSync(not on); }}
            >{"Synced · stop" if on else "Sync to SuperConductor"}</button>
            <span className="tp-note">{
                "Needs make run: in make run-local there is no SuperConductor server." if not sink
                else ("server not reachable: " + str(sync.error) if sync and not sync.ok and sync.error
                else ("following the server's heard tempo and beat one" if on else "off: Atlas keeps its own tempo"))
            }</span>
        </section>

        <section className="tp-grid">
            <div className="tp-col">
                <h4>Atlas (this engine)</h4>
                <dl>
                    <dt title="The tempo Atlas writes its notes and drums at">bpm</dt><dd>{num(bpm, 1)}</dd>
                    <dt title="Estimated time from Atlas making a frame to it being heard: the client's buffer plus network. Sync calibrates it.">latency</dt><dd>{(str(sync.lead_ms) + " ms") if sync else "–"}</dd>
                    <dt title="The beat of Atlas's 4/4 bar being heard now, by Atlas's latency estimate">beat heard</dt><dd>{beat_of(playing.bar_beat if playing else None, 4)}</dd>
                </dl>
            </div>
            <div className="tp-col">
                <h4>SuperConductor server (/tempo)</h4>
                <dl>
                    <dt title="heard_bpm = model_bpm × speed: the tempo of the audio being heard">heard bpm</dt><dd>{num(srv.heard_bpm if srv else None, 1)}</dd>
                    <dt title="model_bpm: the tempo MRT2 is generating, estimated from the last 4-8 s of generated audio">model bpm</dt><dd>{num(srv.model_bpm if srv else None, 1)}</dd>
                    <dt title="speed: the time-stretch applied to each block before it is sent (heard = model × speed)">stretch</dt><dd>{("× " + num(srv.speed, 3)) if srv else "–"}</dd>
                    <dt title="free: no stretching · target: stretched toward an exact bpm · conduct: toward the conductor's beats">mode</dt><dd>{(str(srv.mode) + (" " + num(srv.target_bpm, 1) if srv.mode == "target" else (" " + num(srv.conductor_bpm, 1) if srv.mode == "conduct" else ""))) if srv else "–"}</dd>
                    <dt title="The beat being heard, from the server's beat grid of the heard audio; beat one detected, set by hand, or unknown">beat heard</dt><dd>{(beat_of(srv.beat, srv.beats_per_bar) + (" · beat one " + str(srv.downbeat_source) if srv.downbeat_source else " · beat one unknown")) if srv and srv.playing else "–"}</dd>
                </dl>
            </div>
            <div className="tp-col">
                <h4>Difference</h4>
                <dl>
                    <dt title="Where the server hears the beat minus where Atlas thinks it is heard, within one beat. Sync corrects Atlas's latency by it.">phase</dt><dd>{"–" if num(phase, 3) == "–" else (num(phase, 2) + " beat · " + num(phase_ms, 0) + " ms")}</dd>
                    <dt title="Whole beats between the server's beat one and Atlas's. Sync pins the server's beat one to Atlas's.">bar offset</dt><dd>{"–" if not sync or num(sync.bar_offset_beats, 0) == "–" else (num(sync.bar_offset_beats, 0) + " beats")}</dd>
                    <dt title="The tempo sync is heading for (the heard tempo, folded by octaves toward Atlas's)">sync target</dt><dd>{num(sync.target_bpm if sync else None, 1)}</dd>
                </dl>
            </div>
        </section>

        <section className="tp-explain">
            <h2>How Atlas's messages reach the audio</h2>

            <h3>Frames</h3>
            <p>MRT2 makes audio in frames of 40 ms. Each frame is generated from the style (SuperConductor's prompts), the audio before it, and that frame's two control inputs: NOTES (128 pitch slots) and DRUMS (one slot). An input only matters at the moment its frame is generated.</p>

            <h3>Messages</h3>
            <p>Atlas runs its own clock, one frame per 40 ms of wall time. When anything changes (a chord hovered or kept, a drum cell, the tempo), it writes the coming frames as a Pattern (one loop, or the progression once) and sends it: engine → HTTP → client_midi (port 8470) → websocket → server. The server applies it from the next frame it generates, starting the Pattern from its first step. Between changes, the server loops the Pattern on its own; Atlas sends nothing.</p>

            <h3>Why changes are heard late</h3>
            <p>The server generates ahead of playback. The client grants it a few blocks of credit (3 blocks × 5 frames = 600 ms locally, more over the cluster tunnel) and returns one each time a block finishes playing, so about that much audio is always generated but not yet heard. Audio that is generated is never regenerated or dropped: the model's state has already moved past it. So a message is heard about <b>lead = buffer + network ≈ 0.6 s</b> after it is sent, and nothing can change the audio already in the buffer.</p>

            <h3>Tempo changes</h3>
            <p>Atlas's tempo only sets how many frames each beat lasts in the Pattern: at 120 bpm a beat is 12.5 frames. A new tempo is a new Pattern, so it is heard after the same lead; the buffered audio keeps the old tempo. Atlas counts beats continuously, so a tempo change changes how fast the beat count moves, never where it is.</p>
            <p>The server can also time-stretch: it estimates MRT2's tempo (model bpm) from the last 4–8 s of generated audio, once a second, and stretches each block by a speed toward a target, so <b>heard bpm = model bpm × speed</b>. The speed moves toward its goal with a 1.5 s time constant and is applied as each block is generated, so the blocks already buffered keep the old speed. After Atlas changes tempo, the model plays the new tempo after the lead, but the estimate takes several seconds to catch up. Until it does, the speed is computed from the old tempo and the heard tempo overshoots.</p>
            <p>Stretching also bends the timing between the two clocks: the server's Pattern position advances one frame per generated frame, but at speed s those frames are heard at s × 25 per second, while Atlas's clock stays at 25 per second. Atlas's idea of what is being heard then drifts by (s − 1) beats per beat until the next resend.</p>

            <h3>What Sync does</h3>
            <p><b>Tempo:</b> Atlas's bpm is set to the server's heard bpm (folded to the nearest of ×½, ×1, ×2 of the current tempo, once it differs by 1 bpm). MRT2 then generates the heard tempo itself, so model bpm → heard bpm and speed → 1: no stretching, and the two clocks stop drifting apart.</p>
            <p><b>Latency:</b> twice a second Atlas compares where the server hears the beat (from its beat grid of the heard audio) with the frame Atlas thinks is heard now (made one lead ago). The part of a beat between them is an error in Atlas's latency estimate, and half of it is corrected each time. The music itself is not shifted: the beats the server hears come from Atlas's notes, so moving the notes would move the detected beats with them and never converge.</p>
            <p><b>Beat one:</b> a whole-beat difference means the server counts the bar from another beat. Atlas then waits until its own beat one is being heard and tells the server that this is beat one (POST /tempo/downbeat). From then on both count the same bar. Only 4/4 bars are lined up; if the server hears 3/4, only the beat phase is synced.</p>

            <h3>Delays at a glance</h3>
            <table>
                <tbody>
                    <tr><td>a note, drum or chord change</td><td>heard after the lead (~0.6 s)</td></tr>
                    <tr><td>a tempo change in Atlas</td><td>heard after the lead; the server's model bpm follows after 4–8 s</td></tr>
                    <tr><td>a new server target or conductor tempo</td><td>speed eases over ~1.5 s, from the next generated block; buffered blocks keep the old speed</td></tr>
                    <tr><td>Sync following the heard tempo</td><td>checked every 0.5 s, then heard after the lead</td></tr>
                </tbody>
            </table>
        </section>
    </div>;
}
