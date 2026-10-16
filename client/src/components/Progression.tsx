// The dock along the bottom: the progression being written (or a saved one being
// played) as a strip of chips, with the chord Magenta is sounding lit up, and the
// progression actions, including the list of saved progressions. The strip scrolls
// sideways (the mouse wheel scrolls it too) when it is wider than the dock.
//
// The path under the mouse (or pinned) on the graph shows up as dashed chips on
// whichever side it extends.

import { useEffect, useState } from "react";
import type { FocusEvent, WheelEvent } from "react";
import type { ChordView, SavedView, StepView } from "../api";

/** A chip's text; `alt` is the chord's other Roman numeral (VI for V/ii), shown under it. */
function chip_body(rn: string, symbol: string, cadence: string, alt = "") {
  return [
    <div key="rn" className="rn">
      {rn}
    </div>,
    alt ? (
      <div key="alt" className="alt">
        {alt}
      </div>
    ) : null,
    <div key="sym" className="sym">
      {symbol}
    </div>,
    cadence ? (
      <span key="cad" className="cad">
        {cadence}
      </span>
    ) : null,
  ];
}

export interface ProgressionProps {
  steps: StepView[];
  soundingStep: number;
  soundingId: number;
  previewBefore: ChordView[];
  previewAfter: ChordView[];
  loopFrom: number;
  loopTo: number;
  onLoop: (i: number, shift: boolean) => void;
  canUndo: boolean;
  notice: string;
  keyName: string;
  onUndo: () => void;
  onClear: () => void;
  onPlay: () => void;
  playAll: boolean;
  saved: SavedView[];
  viewing: number;
  onPick: (i: number) => void;
  onCommit: (name: string) => void;
  onRename: (i: number, name: string) => void;
  onDelete: () => void;
  onResume: () => void;
  keepPath: ChordView[];
  onKeep: (whole: boolean) => void;
}

export function Progression(props: ProgressionProps) {
  const { steps, soundingStep, soundingId, previewBefore, previewAfter, loopFrom, loopTo, onLoop, canUndo, notice } = props;
  const { onUndo, onClear, onPlay, playAll, saved, viewing, onPick, onCommit, onRename, onDelete, onResume, keepPath, onKeep } = props;
  const [more, setMore] = useState(false);
  const [lib, setLib] = useState(false);
  // the Save popover: open, and the name being typed
  const [naming, setNaming] = useState(false);
  const [name, setName] = useState("");

  // keep the sounding chip in view as playback moves along the strip
  useEffect(() => {
    const el = document.querySelector<HTMLElement>(".dock .chip.sounding");
    const strip = document.querySelector<HTMLElement>(".dock .chips");
    if (el && strip) {
      const left = el.offsetLeft - strip.offsetLeft;
      if (left < strip.scrollLeft || left + el.offsetWidth > strip.scrollLeft + strip.clientWidth) {
        strip.scrollTo({ left: left - 40, behavior: "smooth" });
      }
    }
  }, [soundingStep, soundingId, steps.length]);

  // a vertical mouse wheel scrolls the strip sideways
  function scroll_chips(e: WheelEvent<HTMLDivElement>) {
    const strip = e.currentTarget;
    if (e.deltaY !== 0 && Math.abs(e.deltaY) > Math.abs(e.deltaX) && strip.scrollWidth > strip.clientWidth) {
      strip.scrollLeft = strip.scrollLeft + e.deltaY;
    }
  }

  const viewing_ok = viewing >= 0 && viewing < saved.length;

  function save() {
    onCommit(name);
    setNaming(false);
    setName("");
  }

  function rename(e: FocusEvent<HTMLInputElement>) {
    if (e.target.value.trim() && e.target.value.trim() !== saved[viewing].name) onRename(viewing, e.target.value);
    else e.target.value = saved[viewing].name;
  }
  // the loop: loopFrom..loopTo, loopFrom..end, or (default) just the last chord
  const lo = loopFrom >= 0 ? loopFrom : steps.length - 1;
  const hi = loopTo >= 0 ? loopTo : steps.length - 1;
  // Play all covers the last 16 chords; while it runs, show the position in it
  const play_first = Math.max(0, steps.length - 16);
  const play_pos = soundingStep >= play_first ? String(soundingStep - play_first + 1) + " / " + String(steps.length - play_first) : "Playing";
  // the looped chords by name, e.g. "ii⁷ → V⁷ → I" (long loops: first … last)
  let loop_names = "";
  if (steps.length > 0 && 0 <= lo && lo <= hi && hi < steps.length) {
    if (hi - lo < 4) {
      for (let i = lo; i < hi + 1; i++) loop_names = loop_names + (i > lo ? " → " : "") + steps[i].rn;
    } else {
      loop_names = steps[lo].rn + " → … → " + steps[hi].rn + " (" + String(hi - lo + 1) + ")";
    }
  }

  return (
    <footer className={"dock" + (viewing_ok ? " viewing" : "")}>
      {viewing_ok && (
        <div className="dock-band">
          <b>Saved:</b>
          <input
            className="saved-name"
            key={"nm" + String(viewing) + saved[viewing].name}
            defaultValue={saved[viewing].name}
            title="Rename this progression (Enter to keep the name)"
            onBlur={(e) => rename(e)}
            onKeyDown={(e) => {
              if (e.key === "Enter") e.currentTarget.blur();
              else if (e.key === "Escape") {
                e.currentTarget.value = saved[viewing].name;
                e.currentTarget.blur();
              }
            }}
          />
          <span>{saved[viewing].key_name + " · click a chord to loop from it, shift-click to end the loop"}</span>
          <span className="tb-sp"></span>
          <button className="btn primary" onClick={() => onPick(-1)}>
            ← Back to current
          </button>
          <button className="btn" title="Continue writing this progression (the current one is saved in its place)" onClick={() => onResume()}>
            Continue this one
          </button>
          <button className="btn ghost" title="Forget this saved progression" onClick={() => onDelete()}>
            Delete
          </button>
        </div>
      )}
      <div className="dock-row">
        <button
          className={"play-btn" + (playAll ? " on" : "")}
          title={
            playAll
              ? "Playing the whole progression once, start to end. Afterwards it goes back to the loop by itself; click to go back now."
              : "Play the whole progression once, from the first chord to the last (up to 16 chords), then go back to the loop. Hovering a chord on the graph stops it."
          }
          onClick={() => onPlay()}
        >
          <span className="play-ic">{playAll ? "■" : "▶"}</span>
          <span className="play-txt">{playAll ? play_pos : "Play all"}</span>
        </button>
        <div
          className={"loop-now" + (playAll ? " off" : "")}
          title="What Magenta repeats while you're not hovering anything. Click a chord to loop from it; shift-click another to end the loop there."
        >
          <span className="loop-label">Loop</span>
          <span className="loop-names">{loop_names}</span>
        </div>
        <div
          className="chips"
          onWheel={scroll_chips}
          title="Your progression. Click a chord to loop from it (click it again for the default loop); shift-click another to end the loop there. Dashed chords are the path you are hovering or have pinned on the graph, not yet kept."
        >
          {previewBefore.map((c) => (
            <div key={"pb" + String(c.id)} className="chip-wrap">
              <div className={"chip tentative" + (c.id === soundingId ? " sounding" : "")} title={c.rule}>
                {chip_body(c.rn, c.symbol, c.cadence, c.alt_rn)}
              </div>
              <span className="arrow">→</span>
            </div>
          ))}
          {steps.map((st, i) => (
            <div key={"s" + String(i)} className="chip-wrap">
              {i > 0 && <span className="arrow">→</span>}
              <div
                className={
                  "chip clickable" +
                  (i === steps.length - 1 && !viewing_ok ? " current" : "") +
                  (i === soundingStep ? " sounding" : "") +
                  (i === loopFrom ? " loop-start" : "") +
                  (lo <= i && i <= hi && !playAll ? " in-loop" : "") +
                  (i === loopTo ? " loop-end" : "")
                }
                title={
                  (i === loopFrom ? "Looping from here: click again for the default loop" : "Click: loop from this chord · Shift-click: end the loop here") +
                  (st.also ? "\n" + st.also : "") +
                  (st.rule ? "\n" + st.rule : "")
                }
                onClick={(e) => onLoop(i, e.shiftKey)}
              >
                {i === lo && !playAll && <span className="loop-tag">LOOP</span>}
                {chip_body(st.rn, st.symbol, st.cadence, st.alt_rn)}
              </div>
            </div>
          ))}
          {previewAfter.map((c) => (
            <div key={"pa" + String(c.id)} className="chip-wrap">
              <span className="arrow">→</span>
              <div className={"chip tentative" + (c.id === soundingId ? " sounding" : "")} title={c.rule}>
                {chip_body(c.rn, c.symbol, c.cadence, c.alt_rn)}
              </div>
            </div>
          ))}
        </div>

        {!viewing_ok && (
          <div className="dock-actions">
            <button className="icon-btn" disabled={!canUndo} title="Undo" onClick={() => onUndo()}>
              ↶
            </button>
            <button
              className="btn primary"
              disabled={keepPath.length === 0}
              title={
                keepPath.length > 0
                  ? "Put it into the progression; the other options stay beside it, so you can swap later"
                  : "Hover or pin a chord on the graph to keep it"
              }
              onClick={() => onKeep(false)}
            >
              {keepPath.length > 0 ? "Keep " + keepPath[0].rn : "Keep"}
            </button>
            {keepPath.length > 1 && (
              <button className="btn" title="Put every chord on this path into the progression at once" onClick={() => onKeep(true)}>
                {"Keep whole path (" + String(keepPath.length) + ")"}
              </button>
            )}
            <div className="pop-anchor">
              <button
                className={"btn" + (naming ? " on" : "")}
                disabled={steps.length < 2}
                title="Name this progression, shelve it (it moves behind in the graph view) and start a new one"
                onClick={() => {
                  setNaming(!naming);
                  setMore(false);
                  setLib(false);
                }}
              >
                ✓ Save
              </button>
              {naming && (
                <div className="menu up save-menu">
                  <input
                    className="save-name"
                    autoFocus={true}
                    placeholder={"Progression " + String(saved.length + 1)}
                    value={name}
                    onChange={(e) => setName(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter") save();
                      else if (e.key === "Escape") setNaming(false);
                    }}
                  />
                  <div className="confirm-actions">
                    <button className="btn ghost" onClick={() => setNaming(false)}>
                      Cancel
                    </button>
                    <button className="btn primary" onClick={() => save()}>
                      Save
                    </button>
                  </div>
                </div>
              )}
            </div>
            <div className="pop-anchor">
              <button
                className={"icon-btn" + (more ? " on" : "")}
                title="More"
                onClick={() => {
                  setMore(!more);
                  setLib(false);
                  setNaming(false);
                }}
              >
                ⋯
              </button>
              {more && (
                <div className="menu up" onMouseLeave={() => setMore(false)}>
                  <button
                    disabled={steps.length < 2}
                    onClick={() => {
                      setMore(false);
                      onClear();
                    }}
                  >
                    Clear progression
                  </button>
                </div>
              )}
            </div>
          </div>
        )}

        <div className="pop-anchor">
          <button
            className={"btn ghost" + (lib || viewing_ok ? " on" : "")}
            disabled={saved.length === 0}
            title="Saved progressions: pick one to hear it and see it highlighted behind the current one in the graph"
            onClick={() => {
              setLib(!lib);
              setMore(false);
              setNaming(false);
            }}
          >
            {"Saved " + String(saved.length) + " ▴"}
          </button>
          {lib && (
            <div className="menu up right" onMouseLeave={() => setLib(false)}>
              {saved.map((p) => (
                <button
                  key={"sv" + String(p.index)}
                  className={"lib-item" + (p.index === viewing ? " on" : "")}
                  onClick={() => {
                    setLib(false);
                    onPick(p.index === viewing ? -1 : p.index);
                  }}
                >
                  <b>{p.name}</b>
                  <span>{p.key_name + " · " + String(p.steps.length) + " chords"}</span>
                </button>
              ))}
              {viewing_ok && (
                <button
                  className="lib-item"
                  onClick={() => {
                    setLib(false);
                    onPick(-1);
                  }}
                >
                  <b>← Current progression</b>
                </button>
              )}
            </div>
          )}
        </div>
      </div>
      {notice && <div className="dock-notice">{notice}</div>}
    </footer>
  );
}
