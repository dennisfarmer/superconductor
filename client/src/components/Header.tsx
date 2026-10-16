// Top bar, left to right: key; transport (pause, tempo knob, the beat, beats per
// chord, texture, and the Harmony - Drums page switch; the BPM label opens the BPM
// page); output (mute, volume); engine status and restart; Send walkers; quit. The
// LLM switch sits in the menu beside the status pill.

import { useState } from "react";
import type { CSSProperties, MouseEvent } from "react";
import { Knob } from "./Knob";
import { BeatLights } from "./BeatLights";
import type { SessionView } from "../api";

const TEXTURES: string[][] = [
  ["block", "block chords"],
  ["arpeggio", "arpeggio"],
  ["improvise", "improvise"],
];

// the tempo knob's range
const BPM_MIN = 40.0;
const BPM_MAX = 200.0;

function bpm_to_knob(bpm: number): number {
  return (bpm - BPM_MIN) / (BPM_MAX - BPM_MIN);
}

function knob_to_bpm(v: number): number {
  return Math.round(BPM_MIN + v * (BPM_MAX - BPM_MIN));
}

/** Speaker glyph: waves by level, a cross when muted. */
function speaker_icon(volume: number, muted: boolean) {
  return (
    <svg className="ic" viewBox="0 0 20 20" width="18" height="18" aria-hidden="true">
      <path d="M3 7.5h3l4-3.5v12l-4-3.5H3z" fill="currentColor" />
      {muted ? (
        <path d="M13 7.5l5 5M18 7.5l-5 5" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
      ) : (
        <g fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round">
          {volume > 0.02 && <path d="M12.8 7.6a3.4 3.4 0 0 1 0 4.8" />}
          {volume > 0.5 && <path d="M15.2 5.3a6.6 6.6 0 0 1 0 9.4" />}
        </g>
      )}
    </svg>
  );
}

function restart_icon() {
  return (
    <svg className="ic" viewBox="0 0 20 20" width="16" height="16" aria-hidden="true">
      <path d="M15.5 10a5.5 5.5 0 1 1-1.6-3.9" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" />
      <path d="M14.6 2.8v3.8h-3.8" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

function power_icon() {
  return (
    <svg className="ic" viewBox="0 0 20 20" width="16" height="16" aria-hidden="true">
      <path d="M6.3 5.6a6 6 0 1 0 7.4 0" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" />
      <path d="M10 3v6.5" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" />
    </svg>
  );
}

export interface HeaderProps {
  page: string;
  session: SessionView;
  engineOk: boolean;
  engineInfo: string;
  paused: boolean;
  onPause: () => void;
  restarting: boolean;
  onRestart: () => void;
  volume: number;
  muted: boolean;
  onVolume: (v: number) => void;
  onMute: () => void;
  quitting: boolean;
  onQuit: () => void;
  walking: boolean;
  onKey: (tonic: string, mode: string) => void;
  onTempo: (bpm: number, beats: number) => void;
  onTexture: (texture: string) => void;
  onLlm: (on: boolean) => void;
  onSend: () => void;
}

export function Header(props: HeaderProps) {
  const { page, session, engineOk, engineInfo, paused, onPause, restarting, onRestart, volume, muted, onVolume, onMute } = props;
  const { quitting, onQuit, walking, onKey, onTempo, onTexture, onLlm, onSend } = props;
  const [menu_open, setMenuOpen] = useState(false);
  const [confirm_quit, setConfirmQuit] = useState(false);
  // where an open menu sits on screen: the bar scrolls sideways, so menus are
  // fixed-position (a scrolling bar would clip them) and placed under their button
  const [menu_at, setMenuAt] = useState<CSSProperties>({});
  // the tempo while the knob is being turned (-1: not turning)
  const [drag_bpm, setDragBpm] = useState(-1.0);
  const tonics = session.mode === "major" ? session.major_tonics : session.minor_tonics;
  const bpm_now = drag_bpm >= 0.0 ? drag_bpm : session.bpm;

  function commit_tempo() {
    if (drag_bpm >= 0.0) {
      if (drag_bpm !== session.bpm) onTempo(drag_bpm, session.beats_per_chord);
      setDragBpm(-1.0);
    }
  }

  function place_menu(e: MouseEvent<HTMLElement>) {
    const r = e.currentTarget.getBoundingClientRect();
    setMenuAt({ top: String(r.bottom + 6) + "px", right: String(Math.max(8, window.innerWidth - r.right)) + "px" });
  }

  const vol = Math.trunc(Math.round(volume * 100.0));
  return (
    <header
      className="topbar"
      onScroll={() => {
        setMenuOpen(false);
        setConfirmQuit(false);
      }}
    >
      <div className="brand" title="Walkers search a graph of chord progressions; Magenta RT plays what you hover, pin and keep">
        SuperConductor Atlas
      </div>

      <div className="tb-group">
        <select
          className="tb-select key-select"
          title="Tonic of the key. Changing key transposes the progression and sends the walkers again."
          value={session.tonic}
          onChange={(e) => onKey(e.target.value, session.mode)}
        >
          {tonics.map((t) => (
            <option key={t} value={t}>
              {t}
            </option>
          ))}
        </select>
        <div className="seg" title="Major or minor mode">
          <button className={session.mode === "major" ? "on" : ""} onClick={() => onKey(session.tonic, "major")}>
            major
          </button>
          <button className={session.mode === "minor" ? "on" : ""} onClick={() => onKey(session.tonic, "minor")}>
            minor
          </button>
        </div>
      </div>

      <div className="tb-group transport">
        <button
          className={"icon-btn" + (!paused ? " on" : "")}
          disabled={!engineOk}
          title={paused ? "Resume the audio" : "Pause the audio (Magenta stops generating)"}
          onClick={() => onPause()}
        >
          {paused ? "▶" : "⏸"}
        </button>
        <div
          className="tempo"
          title="Tempo: drag the knob up or down (or scroll on it), or type a value. Each chord lasts the beats shown by the lights."
        >
          <Knob value={bpm_to_knob(bpm_now)} size={30} color="#8fb3ff" onChange={(v) => setDragBpm(knob_to_bpm(v))} onCommit={commit_tempo} />
          <input
            className="tempo-in"
            type="number"
            key={"bpm" + String(bpm_now)}
            defaultValue={String(Math.trunc(bpm_now))}
            onBlur={(e) => {
              if (e.target.value && parseFloat(e.target.value) !== session.bpm) {
                onTempo(Math.min(BPM_MAX, Math.max(BPM_MIN, parseFloat(e.target.value))), session.beats_per_chord);
              }
            }}
            onKeyDown={(e) => {
              if (e.key === "Enter") e.currentTarget.blur();
            }}
          />
          <a
            className={"unit bpm-link" + (page === "bpm" ? " on" : "")}
            href="#bpm"
            title="The BPM page: sync with SuperConductor's tempo, and how messages reach the audio"
          >
            BPM
          </a>
        </div>
        <BeatLights beats={session.beats_per_chord} running={engineOk && !paused} />
        <label className="tb-field" title="Beats each chord lasts">
          <input
            className="beats-in"
            type="number"
            min="1"
            max="8"
            value={String(session.beats_per_chord)}
            onChange={(e) => {
              if (e.target.value) onTempo(session.bpm, Math.trunc(Number(e.target.value)));
            }}
          />
          <span className="unit">beats</span>
        </label>
        <select
          className="tb-select"
          title="How Magenta is asked to play each chord: held block chords, broken into an arpeggio, or improvise (the bass is given; the model chooses which chord tones to play, and when)"
          value={session.texture}
          onChange={(e) => onTexture(e.target.value)}
        >
          {TEXTURES.map((tx) => (
            <option key={tx[0]} value={tx[0]}>
              {tx[1]}
            </option>
          ))}
        </select>
        <nav className="page-switch">
          <a className={page !== "drums" ? "on" : ""} href="#" title="The Harmony page: the chord graph and the progression">
            Harmony
          </a>
          <span className="page-sep">-</span>
          <a className={page === "drums" ? "on" : ""} href="#drums" title="The Drums page: when the model hits drums, as a 4/4 grid of -1 / 0 / 1">
            Drums
          </a>
        </nav>
      </div>

      <div className="tb-sp"></div>

      <div className={"tb-group output" + (muted ? " muted" : "")}>
        <button
          className={"icon-btn" + (muted ? " on" : "")}
          disabled={!engineOk}
          title={muted ? "Unmute" : "Mute (the music keeps playing silently, so the beat stays in place)"}
          onClick={() => onMute()}
        >
          {speaker_icon(volume, muted)}
        </button>
        <input
          className="volume"
          type="range"
          min="0"
          max="100"
          step="1"
          disabled={!engineOk}
          value={String(vol)}
          style={{ "--fill": String(vol) + "%" } as CSSProperties}
          title={"Volume " + String(vol) + "%"}
          onChange={(e) => onVolume(parseFloat(e.target.value) / 100.0)}
        />
      </div>

      <div className="tb-group engine">
        <span
          className="pill"
          title={
            "Audio engine (Magenta RT) at " +
            session.engine_url +
            ": model, and milliseconds to generate each 40 ms audio frame (under 40 keeps up with real time)"
          }
        >
          <span className={"dot " + (engineOk ? "on" : "off")}></span>
          {engineInfo}
        </span>
        <button
          className={"icon-btn" + (restarting ? " spin" : "")}
          disabled={restarting || quitting}
          title="Restart the audio engine (fixes glitching audio). It is asked to stop, then started again; the model takes a few seconds to load."
          onClick={() => onRestart()}
        >
          {restart_icon()}
        </button>
        <button
          className={"icon-btn" + (menu_open ? " on" : "")}
          title="Engine options"
          onClick={(e) => {
            place_menu(e);
            setMenuOpen(!menu_open);
          }}
        >
          ⋯
        </button>
        {menu_open && (
          <div className="menu fixed" style={menu_at} onMouseLeave={() => setMenuOpen(false)}>
            <label title="LLM calls (Ollama) share the GPU with Magenta; switch off for glitch-free real time">
              <input type="checkbox" checked={session.llm_enabled} onChange={() => onLlm(!session.llm_enabled)} />
              LLM path descriptions
            </label>
          </div>
        )}
      </div>

      <button
        className="btn primary"
        disabled={walking}
        title="Run the beam search again, forwards from the last chord and backwards from the first, with the current settings"
        onClick={() => onSend()}
      >
        {walking ? "Walkers out…" : "Send walkers"}
      </button>

      <div className="pop-anchor">
        <button
          className={"icon-btn quit" + (confirm_quit ? " on" : "")}
          disabled={quitting}
          title="Quit SuperConductor Atlas: stop the audio engine and the app server"
          onClick={(e) => {
            place_menu(e);
            setConfirmQuit(!confirm_quit);
          }}
        >
          {power_icon()}
        </button>
        {confirm_quit && (
          <div className="menu confirm fixed" style={menu_at}>
            <p>Quit SuperConductor Atlas?</p>
            <span>The audio engine and the app server shut down. Your session is kept for next time.</span>
            <div className="confirm-actions">
              <button className="btn ghost" onClick={() => setConfirmQuit(false)}>
                Cancel
              </button>
              <button className="btn danger" disabled={quitting} onClick={() => onQuit()}>
                {quitting ? "Quitting…" : "Quit"}
              </button>
            </div>
          </div>
        )}
      </div>
    </header>
  );
}
