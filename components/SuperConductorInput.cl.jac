"""The Sound tab with the SuperConductor sink (make run): what the SuperConductor
client sends to MRT2 besides Atlas's notes and drums, live, as its camera overlay
shows it (client_midi GET /input, see MIDI_MSG_PROTOCOL.md). Read-only: the
objects in front of the camera set these.

Each row is one prompt in the style blend: its source (base prompt, test beat, the
object driving it, or a combo), its value, and its share of the blend. The server
averages the prompts' MusicCoCa embeddings by weight, so share = weight / sum."""


def pct(v: float) -> str {
    return str(int(Math.round(v * 100.0))) + "%";
}


def:pub SuperConductorInput(view: any, url: str) -> JsxElement {
    if view is None {
        return <div className="sc-input">
            <p className="empty">{"SuperConductor client not reachable at " + url + ". Start make server and make client in ../superconductor_client."}</p>
        </div>;
    }
    rows: list[any] = view.rows or [];
    keys: list[str] = Object.keys(view.controls or {});
    return <div className="sc-input">
        <div className="sc-head">
            <h4>MRT2 style input</h4>
            <span className={"dot " + ("on" if view.connected else "off")} title={"client_midi at " + url + (": connected to the MRT2 server" if view.connected else ": not connected to the MRT2 server")}></span>
        </div>
        <p className="sc-sub">From the SuperConductor client (camera objects), live. Atlas sends only the NOTES and DRUMS input.</p>
        {if len(rows) == 0 {
            <p className="empty">No prompts yet: put an object in front of the camera, or set a base prompt.</p>
        }}
        {for (i, r) in enumerate(rows) {
            <div key={"r" + str(i)} className="sc-row" style={{"--c": r.color or "#c8c8c8"}} title={r.prompt + " · " + r.source + " · value " + str(r.value) + " · " + pct(r.share) + " of the style blend"}>
                <span className="sc-prompt">{r.prompt}</span>
                <span className="sc-source">{r.source}</span>
                <span className="sc-bar"><i style={{"width": pct(min(1.0, max(0.0, r.value)))}}></i></span>
                <span className="sc-num">{f"{r.value:.2f}" + "  " + pct(r.share)}</span>
            </div>
        }}
        {if len(keys) > 0 {
            <div className="sc-controls" title="Sampling controls set by objects (temperature, top_k, cfg_musiccoca)">
                {for k in keys {
                    <span key={k}>{k + " " + f"{view.controls[k]:.2f}"}</span>
                }}
            </div>
        }}
    </div>;
}
