// The side panel beside the graph: one tab at a time (Inspect, Sound, Rules,
// Typing). It collapses to a rail of tab buttons, and tabs can be hidden from
// the bar. The page owns which tab is open; the body is passed in as children.

import { useState } from "react";
import type { MouseEvent, ReactNode } from "react";

// [id, label, icon, description]
const TABS: string[][] = [
  ["inspect", "Inspect", "◎", "The chord you are hovering or have pinned: its path, voicing, costs and rules"],
  ["sound", "Sound", "♫", "The sound Magenta plays: a blend of text prompts"],
  ["rules", "Rules", "§", "The harmony and voice-leading rules, with the function flow chart"],
  ["typing", "Typing", "⌨", "How to type chords and Roman numerals"],
];

export interface SidePanelProps {
  tab: string;
  open: boolean;
  hidden: string[];
  onTab: (tab: string) => void;
  onOpen: (open: boolean) => void;
  onHidden: (hidden: string[]) => void;
  onResize: (width: number) => void;
  children?: ReactNode;
}

export function SidePanel({ tab, open, hidden, onTab, onOpen, onHidden, onResize, children }: SidePanelProps) {
  const [editing, setEditing] = useState(false);

  const shown = TABS.filter((t) => !hidden.includes(t[0]) || t[0] === tab);

  /** Drag the panel's left border to resize it (kept between 260 px and 75% of the window). */
  function start_drag(e: MouseEvent) {
    e.preventDefault();
    document.body.classList.add("resizing");
    function move(ev: globalThis.MouseEvent) {
      onResize(Math.trunc(Math.max(260, Math.min(window.innerWidth * 0.75, window.innerWidth - ev.clientX))));
    }
    function up() {
      window.removeEventListener("mousemove", move);
      window.removeEventListener("mouseup", up);
      document.body.classList.remove("resizing");
    }
    window.addEventListener("mousemove", move);
    window.addEventListener("mouseup", up);
  }

  function toggle_hidden(tid: string) {
    if (hidden.includes(tid)) onHidden(hidden.filter((h) => h !== tid));
    else onHidden([...hidden, tid]);
  }

  if (!open) {
    return (
      <aside className="sidepanel rail">
        <button className="tab-btn" title="Open the panel" onClick={() => onOpen(true)}>
          «
        </button>
        {shown.map((t) => (
          <button
            key={t[0]}
            className={"tab-btn" + (t[0] === tab ? " on" : "")}
            title={t[1] + ": " + t[3]}
            onClick={() => {
              onTab(t[0]);
              onOpen(true);
            }}
          >
            {t[2]}
          </button>
        ))}
      </aside>
    );
  }

  return (
    <aside className="sidepanel">
      <div className="side-resize" title="Drag to resize the panel" onMouseDown={start_drag}></div>
      <nav className="tabs">
        {shown.map((t) => (
          <button key={t[0]} className={"tab" + (t[0] === tab ? " on" : "")} title={t[3]} onClick={() => onTab(t[0])}>
            {t[1]}
          </button>
        ))}
        <span className="tabs-sp"></span>
        <button className={"tab-btn" + (editing ? " on" : "")} title="Choose which tabs are shown" onClick={() => setEditing(!editing)}>
          ⋮
        </button>
        <button className="tab-btn" title="Collapse the panel" onClick={() => onOpen(false)}>
          »
        </button>
      </nav>
      {editing && (
        <div className="tab-picker">
          <span>Show tabs:</span>
          {TABS.map((t) => (
            <label key={t[0]}>
              <input type="checkbox" checked={!hidden.includes(t[0])} onChange={() => toggle_hidden(t[0])} />
              {t[1]}
            </label>
          ))}
        </div>
      )}
      <div className="side-body">{children}</div>
    </aside>
  );
}
