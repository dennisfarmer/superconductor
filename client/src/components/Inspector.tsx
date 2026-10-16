// Details for the focused node of the search: the path, the SATB voicing the
// walker chose, the three tiers of cost (syntax / voice leading / phrase), every
// rule that fired, and actions (commit, audition, LLM description).

import type { ChordView, PathStory } from "../api";

const VOICES = ["S", "A", "T", "B"];

function bar_width(v: number): string {
  let pct = (v / 6.0) * 100.0;
  if (pct < 0.0) pct = 0.0;
  if (pct > 100.0) pct = 100.0;
  return String(pct) + "%";
}

export interface InspectorProps {
  focusNode: ChordView | null;
  chain: ChordView[];
  startRn: string;
  backward: boolean;
  isPinned: boolean;
  locked: boolean;
  story: PathStory | null;
  storyLoading: boolean;
  llmEnabled: boolean;
  onCommitFirst: () => void;
  onCommitPath: () => void;
  onDescribe: () => void;
}

export function Inspector(props: InspectorProps) {
  const { focusNode, chain, startRn, backward, isPinned, story, storyLoading, llmEnabled, onCommitFirst, onCommitPath, onDescribe } = props;
  if (focusNode == null) {
    return (
      <div className="card">
        <div className="card-h">
          <h2>Inspector</h2>
        </div>
        <div className="card-b empty">
          Hover any chord on the map to hear the path to it looped by Magenta, starting from the chord you're on. Click to pin
          it: its column stops reacting to hover and later columns follow it; click again to unpin. Keep puts it into the
          progression.
        </div>
      </div>
    );
  }
  const cv = focusNode;
  // backwards, the chain runs back in time from the anchor: show it in time order
  let path_rns = [startRn, ...chain.map((c) => c.rn)];
  if (backward) {
    path_rns = [startRn];
    for (const c of chain) path_rns.unshift(c.rn);
  }
  const satb = [cv.names[3], cv.names[2], cv.names[1], cv.names[0]];
  return (
    <div className="card">
      <div className="card-h">
        <h2>{isPinned ? "Pinned" : "Hovering"}</h2>
        <span className="sub">{"layer " + String(cv.layer) + " · " + cv.family}</span>
      </div>
      <div className="card-b">
        <div className="insp-title">
          <span className="rn">{cv.rn}</span>
          {cv.alt_rn && <span className="rn alt">{cv.alt_rn}</span>}
          <span className="sym">{cv.symbol}</span>
          {cv.cadence && <span className="pill">{cv.cadence_name}</span>}
        </div>
        {cv.also && <div className="also">{cv.also}</div>}
        <div className="path-line">{path_rns.join(" → ")}</div>
        {cv.id !== 0 && <div className="rule">{"Move: " + cv.rule}</div>}
        <div className="satb">
          {satb.map((nm, i) => (
            <div key={"v" + String(i)}>
              <b>{VOICES[i]}</b>
              <span>{nm}</span>
            </div>
          ))}
        </div>
        {cv.id !== 0 && (
          <div className="costs">
            <span>syntax (×1.5)</span>
            <div className="bar">
              <i style={{ width: bar_width(cv.syntax_cost * 1.5), background: "#7bd88f" }}></i>
            </div>
            <span className="num">{cv.syntax_cost.toFixed(2)}</span>
            <span>voice leading</span>
            <div className="bar">
              <i style={{ width: bar_width(cv.vl_cost), background: "#8fb3ff" }}></i>
            </div>
            <span className="num">{cv.vl_cost.toFixed(2)}</span>
            <span>phrase</span>
            <div className="bar">
              <i style={{ width: bar_width(cv.phrase_cost), background: "#c792ea" }}></i>
            </div>
            <span className="num">{cv.phrase_cost.toFixed(2)}</span>
            <span>path total</span>
            <div className="bar">
              <i style={{ width: bar_width(cv.total / 3.0), background: "#ffcf66" }}></i>
            </div>
            <span className="num">{cv.total.toFixed(2)}</span>
          </div>
        )}
        {cv.marks.length > 0 && (
          <ul className="marks">
            {cv.marks.map((m, i) => (
              <li key={"m" + String(i)} className={m.kind}>
                <span>{m.text}</span>
                {m.cost !== 0.0 && <span className="c">{(m.cost > 0 ? "+" : "") + m.cost.toFixed(1)}</span>}
              </li>
            ))}
          </ul>
        )}
        {cv.id !== 0 && (
          <div className="actions">
            <button
              className="btn primary"
              title="Put it into the progression; the other options stay beside it, so you can swap later"
              onClick={() => onCommitFirst()}
            >
              {chain.length > 0 ? "Keep " + chain[0].rn : "Keep"}
            </button>
            {chain.length > 1 && (
              <button className="btn" title="Put every chord on this path into the progression at once" onClick={() => onCommitPath()}>
                {"Keep whole path (" + String(chain.length) + ")"}
              </button>
            )}
            <button
              className="btn ghost"
              disabled={storyLoading || !llmEnabled}
              title={llmEnabled ? "Ask the local LLM (Ollama) for a short description of this path" : "The LLM is switched off (engine menu ⋯ in the top bar)"}
              onClick={() => onDescribe()}
            >
              {storyLoading ? "Describing…" : "✦ Describe this path (LLM)"}
            </button>
          </div>
        )}
        {story != null && (story.title || story.explanation) && (
          <div className="story">
            {story.title && <h3>{story.title}</h3>}
            {story.mood && (
              <p>
                <i>{story.mood}</i>
              </p>
            )}
            <p>{story.explanation}</p>
          </div>
        )}
      </div>
    </div>
  );
}
