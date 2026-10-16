// The Sound tab with the SuperConductor sink (make run): what the SuperConductor
// client sends to MRT2 besides Atlas's notes and drums, live, as its camera overlay
// shows it (client_midi GET /input, see MIDI_MSG_PROTOCOL.md). Read-only: the
// objects in front of the camera set these.
//
// Each row is one prompt in the style blend: its source (base prompt, test beat, the
// object driving it, or a combo), its value, and its share of the blend. The server
// averages the prompts' MusicCoCa embeddings by weight, so share = weight / sum.

import type { CSSProperties } from "react";

export interface ScRow {
  prompt: string;
  source: string;
  value: number;
  share: number;
  color?: string;
}

export interface ScInputView {
  connected: boolean;
  rows?: ScRow[];
  controls?: Record<string, number>;
}

function pct(v: number): string {
  return String(Math.trunc(Math.round(v * 100.0))) + "%";
}

export function SuperConductorInput({ view, url }: { view: ScInputView | null; url: string }) {
  if (view == null) {
    return (
      <div className="sc-input">
        <p className="empty">
          {"SuperConductor client not reachable at " + url + ". Start make server and make client in ../superconductor_client."}
        </p>
      </div>
    );
  }
  const rows = view.rows || [];
  const controls = view.controls || {};
  const keys = Object.keys(controls);
  return (
    <div className="sc-input">
      <div className="sc-head">
        <h4>MRT2 style input</h4>
        <span
          className={"dot " + (view.connected ? "on" : "off")}
          title={"client_midi at " + url + (view.connected ? ": connected to the MRT2 server" : ": not connected to the MRT2 server")}
        ></span>
      </div>
      <p className="sc-sub">From the SuperConductor client (camera objects), live. Atlas sends only the NOTES and DRUMS input.</p>
      {rows.length === 0 && <p className="empty">No prompts yet: put an object in front of the camera, or set a base prompt.</p>}
      {rows.map((r, i) => (
        <div
          key={"r" + String(i)}
          className="sc-row"
          style={{ "--c": r.color || "#c8c8c8" } as CSSProperties}
          title={r.prompt + " · " + r.source + " · value " + String(r.value) + " · " + pct(r.share) + " of the style blend"}
        >
          <span className="sc-prompt">{r.prompt}</span>
          <span className="sc-source">{r.source}</span>
          <span className="sc-bar">
            <i style={{ width: pct(Math.min(1.0, Math.max(0.0, r.value))) }}></i>
          </span>
          <span className="sc-num">{r.value.toFixed(2) + "  " + pct(r.share)}</span>
        </div>
      ))}
      {keys.length > 0 && (
        <div className="sc-controls" title="Sampling controls set by objects (temperature, top_k, cfg_musiccoca)">
          {keys.map((k) => (
            <span key={k}>{k + " " + controls[k].toFixed(2)}</span>
          ))}
        </div>
      )}
    </div>
  );
}
