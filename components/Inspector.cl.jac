"""Details for the focused node of the search: the path, the SATB voicing the
walker chose, the three tiers of cost (syntax / voice leading / phrase), every
rule that fired, and actions (commit, audition, LLM description via byllm)."""

sv import from ..services.atlas { ChordView, PathStory }

glob VOICES: list[str] = ["S", "A", "T", "B"];


def bar_width(v: float) -> str {
    pct = v / 6.0 * 100.0;
    if pct < 0.0 {
        pct = 0.0;
    }
    if pct > 100.0 {
        pct = 100.0;
    }
    return str(pct) + "%";
}


"""`str.join` is not available in the client runtime, so paths are joined by hand."""
def arrow_join(parts: list[str]) -> str {
    out = "";
    for (i, p) in enumerate(parts) {
        out = out + (" → " if i > 0 else "") + p;
    }
    return out;
}


def:pub Inspector(
    focusNode: ChordView | None,
    chain: list[ChordView],
    startRn: str,
    backward: bool,
    isPinned: bool,
    locked: bool,
    story: PathStory | None,
    storyLoading: bool,
    llmEnabled: bool,
    onCommitFirst: Callable[[], None],
    onCommitPath: Callable[[], None],
    onDescribe: Callable[[], None]
) -> JsxElement {
    if focusNode is None {
        return <div className="card">
            <div className="card-h"><h2>Inspector</h2></div>
            <div className="card-b empty">
                Hover any chord on the map to hear the path to it looped by Magenta, starting from the
                chord you're on. Click to pin it: its column stops reacting to hover and later columns follow it; click again to unpin. Keep puts it into the progression.
            </div>
        </div>;
    }
    cv: ChordView = focusNode;
    # backwards, the chain runs back in time from the anchor: show it in time order
    path_rns: list[str] = [startRn] + [c.rn for c in chain];
    if backward {
        path_rns = [startRn];
        for c in chain {
            path_rns.insert(0, c.rn);
        }
    }
    satb = [cv.names[3], cv.names[2], cv.names[1], cv.names[0]];
    return <div className="card">
        <div className="card-h">
            <h2>{"Pinned" if isPinned else "Hovering"}</h2>
            <span className="sub">{"layer " + str(cv.layer) + " · " + cv.family}</span>
        </div>
        <div className="card-b">
            <div className="insp-title">
                <span className="rn">{cv.rn}</span>
                {if cv.alt_rn {
                    <span className="rn alt">{cv.alt_rn}</span>
                }}
                <span className="sym">{cv.symbol}</span>
                {if cv.cadence {
                    <span className="pill">{cv.cadence_name}</span>
                }}
            </div>
            {if cv.also {
                <div className="also">{cv.also}</div>
            }}
            <div className="path-line">{arrow_join(path_rns)}</div>
            {if cv.id != 0 {
                <div className="rule">{"Move: " + cv.rule}</div>
            }}
            <div className="satb">
                {for (i, nm) in enumerate(satb) {
                    <div key={"v" + str(i)}><b>{VOICES[i]}</b><span>{nm}</span></div>
                }}
            </div>
            {if cv.id != 0 {
                <div className="costs">
                    <span>syntax (×1.5)</span>
                    <div className="bar"><i style={{"width": bar_width(cv.syntax_cost * 1.5), "background": "#7bd88f"}}></i></div>
                    <span className="num">{f"{cv.syntax_cost:.2f}"}</span>
                    <span>voice leading</span>
                    <div className="bar"><i style={{"width": bar_width(cv.vl_cost), "background": "#8fb3ff"}}></i></div>
                    <span className="num">{f"{cv.vl_cost:.2f}"}</span>
                    <span>phrase</span>
                    <div className="bar"><i style={{"width": bar_width(cv.phrase_cost), "background": "#c792ea"}}></i></div>
                    <span className="num">{f"{cv.phrase_cost:.2f}"}</span>
                    <span>path total</span>
                    <div className="bar"><i style={{"width": bar_width(cv.total / 3.0), "background": "#ffcf66"}}></i></div>
                    <span className="num">{f"{cv.total:.2f}"}</span>
                </div>
            }}
            {if len(cv.marks) > 0 {
                <ul className="marks">
                    {for (i, m) in enumerate(cv.marks) {
                        <li key={"m" + str(i)} className={m.kind}>
                            <span>{m.text}</span>
                            {if m.cost != 0.0 {
                                <span className="c">{("+" if m.cost > 0 else "") + f"{m.cost:.1f}"}</span>
                            }}
                        </li>
                    }}
                </ul>
            }}
            {if cv.id != 0 {
                <div className="actions">
                    <button className="btn primary" title="Put it into the progression; the other options stay beside it, so you can swap later" onClick={lambda e: MouseEvent { onCommitFirst(); }}>
                        {"Keep " + chain[0].rn if len(chain) > 0 else "Keep"}
                    </button>
                    {if len(chain) > 1 {
                        <button className="btn" title="Put every chord on this path into the progression at once" onClick={lambda e: MouseEvent { onCommitPath(); }}>
                            {"Keep whole path (" + str(len(chain)) + ")"}
                        </button>
                    }}
                    <button className="btn ghost" disabled={storyLoading or not llmEnabled} title={"Ask the local LLM (Ollama) for a short description of this path" if llmEnabled else "The LLM is switched off (engine menu ⋯ in the top bar)"} onClick={lambda e: MouseEvent { onDescribe(); }}>
                        {"Describing…" if storyLoading else "✦ Describe this path (LLM)"}
                    </button>
                </div>
            }}
            {if story is not None and (story.title or story.explanation) {
                <div className="story">
                    {if story.title { <h3>{story.title}</h3> }}
                    {if story.mood { <p><i>{story.mood}</i></p> }}
                    <p>{story.explanation}</p>
                </div>
            }}
        </div>
    </div>;
}
