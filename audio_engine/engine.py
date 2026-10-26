"""SuperConductor Atlas audio engine: Magenta RealTime 2, streaming forever.

The engine owns the model and the sound card. It never stops generating; what it
plays is steered by *timelines* posted over HTTP by the app server:

  idle     the current chord of the composer's progression, vamped
  preview  the path being auditioned (current chord -> candidate path), looped
  play     the committed progression, once, then back to idle

Timeline switches land on the next beat. If the new timeline starts with the
chord that is already sounding, playback jumps straight to the next chord, so
hovering an option means hearing the move *out of* the current chord right away.

Chords are sent as MIDI voicings. Each 40 ms model frame gets a 128-slot notes
vector (see magenta_rt.config.PIANOROLL_WITH_ONSETS):
  2 = onset, 1 = continuation, 3 = "on, model chooses onset/continuation"
On a chord change every pitch of the new voicing is sent as 2 (re-attacked, even
tones shared with the previous chord, so the whole chord is always heard). Within
a chord, frames use 3 ("auto" articulation) or 1 ("hold").

Textures (settings.texture) decide which pitches sound inside a chord:
  block      the voicing, as above
  arpeggio   bass held; upper chord tones broken into eighth notes, up and down over two octaves
  improvise  bass struck and held; every other chord-tone pitch (or chord.scale, if sent) in
             MIDI 36-96 is -1 ("model chooses"), everything else 0: Magenta improvises inside
             the harmony instead of playing a given voicing

Drums (settings.drum_grid, set on the Drums page): one value per 40 ms frame for
MRT2's drum_pianoroll input, -1 model chooses, 0 no hit, 1 hit. Modes: silent
(all 0), free (all -1) or grid: a 4/4 step grid (`bars` x 4 beats x `per_beat`
cells, each -1 / 0 / 1) on the beat clock. A hit cell is 1 on its first frame and
`after_hit` (0 or -1) on the rest of the cell.

Sink mode (--sink superconductor): no model and no sound card. The conductor runs
on a 40 ms clock and every timeline it enters is rendered into a Pattern (notes +
drums lanes, see superconductor_server/mrt2_server.py) and POSTed to the
SuperConductor client's client_midi (default http://localhost:9105), which plays it
through the MRT2 server with SuperConductor's prompts. /style doesn't apply there:
SuperConductor owns the prompts. The sampling controls (temperature, top_k,
cfg_scales) are POSTed to client_midi's /controls, which relays them to the server
as UpdateControls; in local mode they go straight into generate(). drum_prompt (0..1,
the Drums page) is the weight of the style prompt "drums", POSTed to client_midi's
/prompts, which adds it to the client's recipe; sink only (local mode ignores it).
Tempo sync (POST /sync, the BPM page): the sink follows the SuperConductor server's
heard tempo and lines its beat one up with the server's (SinkEngine._syncer).

HTTP API (JSON, CORS open, default port 9104):
  GET  /status
  POST /timeline {"role": "idle"|"preview"|"play", "chords": [{"label", "symbol", "notes"}], "id"}
  POST /release                      -> drop preview/play, back to idle
  POST /style    {"prompt": "..."}  or  {"layers": [{"text": "...", "weight": 0.7}, ...]}  (embedding blend)
  POST /settings {"bpm", "beats_per_chord", "articulation", "texture", "drum_grid", "gain",
                  "temperature", "top_k", "cfg_scales": {"musiccoca", "notes", "drums"}, "drum_prompt"}
                 temperature, top_k and cfg_scales are MRT2's generate() arguments, under
                 MRT2's names; cfg_scales may carry only some of its keys
  POST /mix {"volume": 0..1.5, "muted": bool}   output level, applied at the speaker (no buffer delay)
  POST /shutdown {}   finish the current chunk, close the audio stream and the HTTP server, exit
  POST /sync {"on": bool}   sink only: follow the SuperConductor server's tempo and beat one
  POST /output {"harmony": bool?, "drums": bool?}   sink only: whether Atlas sends that
                 page to the client. false resets it to the model default (NOTES or DRUMS
                 lane cleared, its cfg_scales key removed, and for drums the "drums" prompt
                 removed) and then leaves it alone, so another program can take it

Run:  python audio_engine/engine.py [--model mrt2_small] [--port 9104] [--record out.wav]
      python audio_engine/engine.py --sink superconductor [--client-midi http://localhost:9105] [--sc-server http://localhost:9000]
"""

import argparse
import json
import logging
import math
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

log = logging.getLogger("engine")

FRAME_RATE = 25
SAMPLE_RATE = 48_000
FRAME_SAMPLES = SAMPLE_RATE // FRAME_RATE  # 1920


@dataclass
class Chord:
    notes: list
    label: str = ""
    symbol: str = ""
    scale: list = field(default_factory=list)  # pitch classes available for melodic figuration


@dataclass
class Timeline:
    role: str
    chords: list
    id: str = ""


TEXTURES = ("block", "arpeggio", "improvise")
IMPROVISE_RANGE = (36, 96)  # MIDI pitches the improvise texture opens to the model


def drum_grid(body: dict | None = None) -> dict:
    """A validated drum grid (see the module docstring); missing fields get defaults."""
    body = body or {}
    mode = str(body.get("mode", "silent"))
    if mode not in ("silent", "free", "grid"):
        raise ValueError(f"unknown drum mode {mode}")
    bars = int(body.get("bars", 1))
    per_beat = int(body.get("per_beat", 2))
    if bars not in (1, 2, 4) or per_beat not in (1, 2, 3, 4):
        raise ValueError("drum grid: bars must be 1, 2 or 4 and per_beat 1, 2, 3 or 4")
    n = bars * 4 * per_beat
    cells = [int(c) for c in body.get("cells", [])][:n]
    cells += [-1] * (n - len(cells))
    if any(c not in (-1, 0, 1) for c in cells):
        raise ValueError("drum cells must be -1, 0 or 1")
    after_hit = int(body.get("after_hit", 0))
    if after_hit not in (-1, 0):
        raise ValueError("after_hit must be 0 or -1")
    return {"mode": mode, "bars": bars, "per_beat": per_beat, "cells": cells, "after_hit": after_hit}


# MRT2's cfg_scales keys and their range and step (magenta_rt/config.py, CFG_CONDITIONING_*)
CFG_SCALE_STEPS = {"musiccoca": 0.2, "notes": 0.2, "drums": 1.0}
CFG_MIN, CFG_MAX = -1.0, 7.0


def cfg_scales(body: dict, base: dict | None = None) -> dict:
    """`base` (default: the defaults) with the scales in `body` applied, each clamped
    to -1..7 and rounded to MRT2's step for it."""
    out = dict(base or Settings().cfg_scales)
    for key, value in body.items():
        if key not in CFG_SCALE_STEPS:
            raise ValueError(f"unknown cfg_scales key {key}; MRT2's are {list(CFG_SCALE_STEPS)}")
        step = CFG_SCALE_STEPS[key]
        out[key] = round(round(min(CFG_MAX, max(CFG_MIN, float(value))) / step) * step, 1)
    return out


# the Drums page's style prompt (Settings.drum_prompt is its weight)
DRUM_PROMPT = "drums"

# the drum grid's bar, and the bar the tempo sync lines up with the server's
BAR_BEATS = 4
# the tempo range the page allows (server/atlas.py set_tempo)
SYNC_BPM_MIN, SYNC_BPM_MAX = 40.0, 180.0


@dataclass
class Settings:
    bpm: float = 84.0
    beats_per_chord: int = 4
    articulation: str = "auto"  # auto | hold
    texture: str = "block"  # block | arpeggio | improvise
    drum_grid: dict = field(default_factory=drum_grid)  # silent: no drums, as before
    gain: float = 0.9
    # MRT2's sampling arguments, under MRT2's names (magenta_rt generate()). notes is 3.0
    # rather than MRT2's 1.0, so the chords are followed
    temperature: float = 1.3
    top_k: int = 40
    cfg_scales: dict = field(default_factory=lambda: {"musiccoca": 3.0, "notes": 3.0, "drums": 1.0})
    # weight 0..1 of the style prompt DRUM_PROMPT in the SuperConductor client's recipe (sink only)
    drum_prompt: float = 0.0

    @classmethod
    def from_dict(cls, d: dict) -> "Settings":
        d = dict(d)
        if "drums" in d:  # older presets: drums off | auto
            d["drum_grid"] = {"mode": "free" if d.pop("drums") == "auto" else "silent"}
        if "drum_grid" in d:
            d["drum_grid"] = drum_grid(d["drum_grid"])
        if "cfg_scales" in d:
            d["cfg_scales"] = cfg_scales(d["cfg_scales"])
        return cls(**d)

    def controls(self) -> dict:
        """The sampling controls, as an UpdateControls body (superconductor_server/mrt2_server.py)."""
        return {"temperature": self.temperature, "top_k": self.top_k, "cfg_scales": dict(self.cfg_scales)}

    def prompts(self) -> dict:
        """The extra style prompts, as a client_midi POST /prompts body."""
        return {DRUM_PROMPT: self.drum_prompt}

    def frames_per_beat(self) -> float:
        return FRAME_RATE * 60.0 / self.bpm


@dataclass
class FrameMeta:
    role: str
    timeline_id: str
    index: int
    label: str
    symbol: str
    beat: int
    beat_frac: float = 0.0  # how far through that beat this frame starts (0..1)
    drum: int = 0  # drum_pianoroll value for this frame
    drum_cell: int = -1  # drum grid cell this frame is in (-1: not in grid mode)
    beat_pos: float = 0.0  # beats since the conductor started (continuous across tempo changes)


class Conductor:
    """Decides, frame by frame, which chord is sounding and builds the notes vector."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.idle = Timeline("idle", [Chord([48, 55, 64, 72], "I", "C")], "idle")
        self.active: Timeline | None = None
        self.pending: Timeline | None = None
        self.release_pending = False
        self.index = 0
        self.frame = 0  # global frame counter
        # beats since start, advanced by 1 / frames_per_beat each frame: a tempo change
        # changes how fast it moves, never where it is (frame / frames_per_beat would jump)
        self.beat_pos = 0.0
        self.chord_start = 0
        self.current_notes: list = []
        self.entries = 0  # timelines entered so far (the sink re-renders on each)
        self.lock = threading.Lock()

    def clone(self) -> "Conductor":
        """A copy that can be stepped ahead (to render a timeline) without touching this one.
        Queued switches are left out: the copy plays on in the current timeline."""
        with self.lock:
            c = Conductor(self.settings)
            c.idle, c.active = self.idle, self.active
            c.index, c.frame, c.chord_start = self.index, self.frame, self.chord_start
            c.beat_pos = self.beat_pos
            c.current_notes = list(self.current_notes)
            c.entries = self.entries
            return c

    # --- control (HTTP threads) ---
    def set_timeline(self, tl: Timeline):
        log.info("timeline %s · %d chords · %s", tl.role, len(tl.chords), tl.id[:80])
        with self.lock:
            if tl.role == "idle" and self.active is not None:
                # not sounding now: safe to swap; it plays when the preview is released
                self.idle = tl
            else:
                # a new idle loop while the old one sounds must not replace it mid-chord
                # (its index belongs to the old loop): it is entered on the next beat
                self.pending = tl
            self.release_pending = False

    def release(self):
        with self.lock:
            if self.pending is not None and self.pending.role == "idle":
                return  # the new idle loop is already queued for the next beat
            self.pending = None
            self.release_pending = True

    # --- generation thread ---
    def _timeline(self) -> Timeline:
        return self.active if self.active is not None else self.idle

    def _chord_frames(self) -> int:
        s = self.settings
        return max(1, round(s.frames_per_beat() * s.beats_per_chord))

    def _beat_step(self) -> float:
        return 1.0 / self.settings.frames_per_beat()

    def _on_beat(self) -> bool:
        """This frame starts a beat (a whole number of beats falls inside it)."""
        return math.floor(self.beat_pos + 1e-9) != math.floor(self.beat_pos - self._beat_step() + 1e-9)

    def _enter(self, tl: Timeline):
        """Switch timelines; skip the first chord if it is the one already sounding."""
        if tl.role == "idle":
            self.idle = tl
        self.active = None if tl.role == "idle" else tl
        start = 0
        if len(tl.chords) > 1 and tl.chords[0].notes == self.current_notes:
            start = 1
        self.index = start
        self.chord_start = self.frame
        self.entries += 1
        self._begin_chord()

    def _begin_chord(self):
        self.current_notes = list(self._timeline().chords[self.index].notes)

    # --- textures ---
    def _hold(self, tokens: list, pitches: list, pos: int, sustain: int):
        # every chord change re-attacks the whole voicing: a common tone sent as a
        # continuation was struck seconds ago and has decayed, so it would be inaudible
        for p in pitches:
            tokens[p] = 2 if pos == 0 else sustain

    def _texture_tokens(self, pos: int, chord: Chord, timeline_id: str) -> list:
        s = self.settings
        tokens = [0] * 128
        sustain = 3 if s.articulation == "auto" else 1
        notes = self.current_notes
        tex = s.texture
        if tex == "block" or not notes:
            self._hold(tokens, notes, pos, sustain)
            return tokens
        eighth = max(1.0, s.frames_per_beat() / 2)
        step = int(pos / eighth)
        step_start = pos == 0 or int((pos - 1) / eighth) != step
        if tex == "arpeggio":
            bass, upper = notes[0], sorted(notes[1:])
            self._hold(tokens, [bass], pos, 1)
            ladder = sorted(set(upper + [p + 12 for p in upper if p + 12 <= 96]))
            pattern = ladder + ladder[-2:0:-1]  # up, then back down without repeating the ends
            p = pattern[step % len(pattern)]
            tokens[p] = 2 if step_start else 1
            return tokens
        if tex == "improvise":
            # the bass anchors the harmony; the model chooses which of the other
            # allowed pitches to play, and when
            classes = set(chord.scale) or {n % 12 for n in notes}
            lo, hi = IMPROVISE_RANGE
            for p in range(lo, hi + 1):
                if p % 12 in classes:
                    tokens[p] = -1
            tokens[notes[0]] = 2 if pos == 0 else 3
            return tokens
        self._hold(tokens, notes, pos, sustain)
        return tokens

    def _drum_token(self) -> tuple[int, int]:
        """(drum_pianoroll value, grid cell) for the current frame, on the beat clock."""
        g = self.settings.drum_grid
        if g["mode"] != "grid":
            return (0 if g["mode"] == "silent" else -1), -1
        k = math.floor(self.beat_pos * g["per_beat"] + 1e-9)
        cell = k % len(g["cells"])
        value = g["cells"][cell]
        if value == 1:
            first = self.frame == 0 or math.floor((self.beat_pos - self._beat_step()) * g["per_beat"] + 1e-9) != k
            value = 1 if first else g["after_hit"]
        return value, cell

    def next_frame(self) -> tuple[list, FrameMeta]:
        with self.lock:
            if self.frame == 0 and not self.current_notes:
                self.current_notes = list(self.idle.chords[0].notes)
            if self._on_beat() or self.frame == 0:
                if self.pending is not None:
                    tl, self.pending = self.pending, None
                    self._enter(tl)
                elif self.release_pending:
                    self.release_pending = False
                    self._enter(self.idle)
            tl = self._timeline()
            if self.frame - self.chord_start >= self._chord_frames():
                self.chord_start = self.frame
                self.index += 1
                if self.index >= len(tl.chords):
                    if tl.role == "play":
                        self.active = None
                        tl = self.idle
                        # the play timeline ends on the current chord: keep it sounding
                        self.index = len(tl.chords) - 1
                    else:
                        self.index = 0
                self._begin_chord()
            if self.index >= len(tl.chords):
                # never index past a timeline (a stale index would wedge generation)
                log.warning("index %d past %s timeline of %d chords; restarting it", self.index, tl.role, len(tl.chords))
                self.index = 0
                self.chord_start = self.frame
                self._begin_chord()
            pos = self.frame - self.chord_start
            chord = tl.chords[self.index]
            tokens = self._texture_tokens(pos, chord, tl.id)
            drum, drum_cell = self._drum_token()
            meta = FrameMeta(
                role=tl.role,
                timeline_id=tl.id,
                index=self.index,
                label=chord.label,
                symbol=chord.symbol,
                beat=int(pos / self.settings.frames_per_beat()),
                beat_frac=(pos / self.settings.frames_per_beat()) % 1.0,
                drum=drum,
                drum_cell=drum_cell,
                beat_pos=self.beat_pos,
            )
            self.frame += 1
            self.beat_pos += self._beat_step()
            return tokens, meta


class Engine:
    def __init__(self, model: str, buffer_frames: int, record: str | None, prompt: str, preset: dict | None = None):
        from magenta_rt import MagentaRT2StdMlxfn

        preset = preset or {}
        self.settings = Settings.from_dict(preset.get("settings", {}))
        self.conductor = Conductor(self.settings)
        if "idle_chord" in preset:
            c = preset["idle_chord"]
            self.conductor.idle = Timeline("idle", [Chord(list(c["notes"]), c.get("label", ""), c.get("symbol", ""))], "idle")
        model = preset.get("model", model)
        prompt = preset.get("prompt", prompt)
        # Audio moves in chunks of `buffer_frames` frames. The queue holds one finished
        # chunk while the sound card plays another, so the GPU works in bursts and then
        # sleeps, instead of waking for a single frame every 40 ms (which lets the GPU
        # clock down until each step takes the whole 40 ms).
        self.buffer_frames = buffer_frames
        self.queue: queue.Queue = queue.Queue(maxsize=1)
        self.playing_chunk: list = []
        self.playing_pos = 0
        self.model_name = model
        t0 = time.time()
        self.mrt = MagentaRT2StdMlxfn(size=model, temperature=self.settings.temperature, top_k=self.settings.top_k)
        log.info("model %s loaded in %.1fs", model, time.time() - t0)
        self.style_prompt = ""
        self.style_tokens: list = []
        self.style_cache: dict = {}  # exact saved tokens for single prompts (presets)
        self.embed_cache: dict = {}
        self.style_layers: list = []
        self.style_lock = threading.Lock()
        if "style_tokens" in preset:
            # exact saved tokens: no re-embedding, bit-identical style conditioning
            self.style_cache[prompt] = list(preset["style_tokens"])
        self._set_style_now(prompt)
        self.state = None
        self.underruns = 0
        self.paused = False  # silence out; the full buffer blocks the generator, so the GPU idles
        # output level, applied in the audio callback so a change is heard at once
        # (gain in Settings is the preset's level, baked in when a frame is generated)
        self.volume = 1.0
        self.muted = False
        self.out_latency_ms = 0.0  # the audio device's output latency, set once the stream opens
        self.step_ms: list = []
        self.now_playing: FrameMeta | None = None
        self.running = True
        self.record_path = record
        self.recorded: list = []

    def _embedding(self, text: str) -> np.ndarray:
        """MusicCoCa text embedding (LiteRT, independent of MLX), cached per prompt."""
        if text not in self.embed_cache:
            t0 = time.time()
            self.embed_cache[text] = np.asarray(self.mrt.embed_style(text, use_mapper=True), dtype=np.float32)
            log.info("embedded '%s' in %.0f ms", text, (time.time() - t0) * 1000)
        return self.embed_cache[text]

    def _set_layers_now(self, layers: list):
        """Blend several prompts: weighted mean of their embeddings, rescaled to the
        layers' mean norm, then RVQ-tokenized. One layer at weight > 0 = plain prompt."""
        from magenta_rt.config import MUSICCOCA

        layers = [(str(t).strip(), max(0.0, float(w))) for t, w in layers if str(t).strip()]
        active = [(t, w) for t, w in layers if w > 0] or layers[:1]
        with self.style_lock:
            if len(active) == 1 and active[0][0] in self.style_cache:
                tokens = self.style_cache[active[0][0]]
            else:
                embs = [self._embedding(t) for t, _ in active]
                weights = np.array([w for _, w in active] if len(active) > 1 else [1.0], dtype=np.float32)
                weights = weights / weights.sum()
                blend = sum(w * e for w, e in zip(weights, embs))
                target = float(np.mean([np.linalg.norm(e) for e in embs]))
                norm = float(np.linalg.norm(blend))
                if norm > 0:
                    blend = blend * (target / norm)
                tokens = self.mrt.tokenize_style(blend).tolist()[: MUSICCOCA.rvq_truncation_level]
            self.style_tokens = tokens
            total = sum(w for _, w in active) or 1.0
            self.style_layers = [{"text": t, "weight": w} for t, w in layers]
            self.style_prompt = " + ".join(
                f"{t} ({round(100 * w / total)}%)" if len(active) > 1 else t for t, w in active
            )
        log.info("style now: %s", self.style_prompt)

    def _set_style_now(self, prompt: str):
        self._set_layers_now([(prompt, 1.0)])

    def set_style(self, prompt: str):
        """Embed off the generation thread so the stream never stalls."""
        threading.Thread(target=self._set_style_now, args=(prompt,), daemon=True).start()

    def set_layers(self, layers: list):
        threading.Thread(target=self._set_layers_now, args=(layers,), daemon=True).start()

    def generate_one(self) -> tuple[np.ndarray, FrameMeta]:
        from magenta_rt.config import DRUM_PIANOROLL, MUSICCOCA, PIANOROLL_WITH_ONSETS

        tokens, meta = self.conductor.next_frame()
        s = self.settings
        cond = {
            MUSICCOCA.key: self.style_tokens,
            PIANOROLL_WITH_ONSETS.key: tokens,
            DRUM_PIANOROLL.key: [meta.drum],
        }
        t0 = time.perf_counter()
        wav, self.state = self.mrt.generate(
            conditioning=cond,
            cfg_scales=dict(s.cfg_scales),
            temperature=s.temperature,
            top_k=s.top_k,
            frames=1,
            state=self.state,
        )
        self.step_ms.append((time.perf_counter() - t0) * 1000)
        self.step_ms = self.step_ms[-100:]
        return (wav.samples.astype(np.float32) * s.gain), meta

    def generator_loop(self):
        # The bounded queue paces generation: put() blocks until the audio callback
        # takes the queued chunk, so this thread sleeps in the OS instead of polling.
        while self.running:
            try:
                item = [self.generate_one() for _ in range(self.buffer_frames)]
            except Exception:
                log.exception("generation failed")
                time.sleep(0.5)
                continue
            while self.running:
                try:
                    self.queue.put(item, timeout=0.5)
                    break
                except queue.Full:
                    pass

    def audio_callback(self, outdata, frames, _time, status):
        if status:
            log.debug("audio status: %s", status)
        if self.paused:
            outdata.fill(0)
            return
        if self.playing_pos >= len(self.playing_chunk):
            try:
                self.playing_chunk = self.queue.get_nowait()
                self.playing_pos = 0
            except queue.Empty:
                outdata.fill(0)
                self.underruns += 1
                return
        samples, meta = self.playing_chunk[self.playing_pos]
        self.playing_pos += 1
        # muted: keep consuming frames, so the timeline (and the beat) carry on silently
        if self.muted:
            outdata.fill(0)
        else:
            outdata[:] = samples[:frames] * self.volume
        self.now_playing = meta
        if self.record_path:
            self.recorded.append(samples)

    def status(self) -> dict:
        m = self.now_playing
        ms = self.step_ms
        return {
            "ok": True,
            "model": self.model_name,
            "style": self.style_prompt,
            "style_layers": self.style_layers,
            "ms_per_frame": round(float(np.mean(ms)), 1) if ms else None,
            "realtime_budget_ms": 1000 / FRAME_RATE,
            "underruns": self.underruns,
            "paused": self.paused,
            "volume": self.volume,
            "muted": self.muted,
            "latency_ms": self.out_latency_ms,
            "buffer_ms": (self.queue.qsize() * self.buffer_frames + len(self.playing_chunk) - self.playing_pos)
            * 1000 // FRAME_RATE,
            "settings": self.settings.__dict__,
            "playing": None
            if m is None
            else {
                "role": m.role,
                "timeline_id": m.timeline_id,
                "index": m.index,
                "label": m.label,
                "symbol": m.symbol,
                "beat": m.beat,
                "beat_frac": round(m.beat_frac, 3),
                "drum": m.drum,
                "drum_cell": m.drum_cell,
            },
        }

    def settings_changed(self):
        pass  # the local engine reads settings frame by frame

    def controls_changed(self):
        pass  # the same for temperature, top_k and cfg_scales

    def prompts_changed(self):
        pass  # drum_prompt: SuperConductor's recipe only

    def save_recording(self):
        if self.record_path and self.recorded:
            from magenta_rt.audio import Waveform

            Waveform(np.concatenate(self.recorded, axis=0), SAMPLE_RATE).write(self.record_path)
            log.info("recording written to %s", self.record_path)


def to_pattern(frames: list, loop: bool, settings: Settings, pattern_id: str) -> dict:
    """Per-frame (notes128, drum) values -> a Pattern body (superconductor_server/
    mrt2_server.py): consecutive frames are merged into steps. A notes step repeats
    its first frame with onsets (2) turned into holds (1); a drums step is `first`
    on its first frame and `rest` after."""
    notes_steps, drum_steps = [], []
    i = 0
    while i < len(frames):
        first = frames[i][0]
        later = [1 if v == 2 else v for v in first]
        j = i + 1
        while j < len(frames) and frames[j][0] == later:
            j += 1
        default = 0 if first.count(0) >= first.count(-1) else -1
        notes_steps.append({"frames": j - i, "default": default,
                            "pitches": {str(p): v for p, v in enumerate(first) if v != default}})
        i = j
    i = 0
    while i < len(frames):
        first = frames[i][1]
        rest = frames[i + 1][1] if i + 1 < len(frames) else (0 if first == 1 else first)
        j = i + 1
        while j < len(frames) and frames[j][1] == rest:
            j += 1
        drum_steps.append({"frames": j - i, "first": first, "rest": rest})
        i = j
    return {
        "id": pattern_id,
        "notes": {"loop": loop, "steps": notes_steps},
        "drums": {"loop": loop, "steps": drum_steps},
    }


def gate_pattern(pattern: dict | None, send_harmony: bool, send_drums: bool,
                 reset_harmony: bool, reset_drums: bool) -> dict | None:
    """The Pattern to POST to client_midi.

    A lane Atlas is still sending is copied through. A lane that was just released is
    set to null (the server clears it: every frame -1, MRT2's default, as with no
    Pattern). After that reset has been delivered, the lane is left out, so a later
    pattern doesn't put Atlas's values back and another program can set the lane.
    None means there is nothing to send (both lanes released, resets already delivered).
    """
    body: dict = {}
    if pattern:
        if pattern.get("id"):
            body["id"] = pattern["id"]
        if "notes" in pattern:
            body["notes"] = pattern["notes"]
        if "drums" in pattern:
            body["drums"] = pattern["drums"]
    if reset_harmony:
        body["notes"] = None
    elif not send_harmony:
        body.pop("notes", None)
    if reset_drums:
        body["drums"] = None
    elif not send_drums:
        body.pop("drums", None)
    if "notes" not in body and "drums" not in body:
        return None
    if reset_harmony or reset_drums:
        released = [name for name, on in (("harmony", reset_harmony), ("drums", reset_drums)) if on]
        body["id"] = "atlas release " + "+".join(released)
    return body


def controls_body(settings: Settings, send_harmony: bool, send_drums: bool,
                  reset_harmony: bool = False, reset_drums: bool = False) -> dict:
    """UpdateControls for the client. A released lane's cfg_scales key is left out
    (a reset sets it to null, which the server drops back to MRT2's default).
    musiccoca, temperature and top_k are the shared generate() controls and always go."""
    body = settings.controls()
    scales = dict(body["cfg_scales"])
    if reset_harmony:
        scales["notes"] = None
    elif not send_harmony:
        scales.pop("notes", None)
    if reset_drums:
        scales["drums"] = None
    elif not send_drums:
        scales.pop("drums", None)
    body["cfg_scales"] = scales
    return body


def merge_control_posts(old: dict, new: dict) -> dict:
    """Two unsent /controls bodies. A key in `new` wins, including null; a key only
    in `old` is kept, so a release's null isn't dropped by a later partial update."""
    scales = dict(old.get("cfg_scales") or {})
    scales.update(new.get("cfg_scales") or {})
    merged = {**old, **new}
    if "cfg_scales" in old or "cfg_scales" in new:
        merged["cfg_scales"] = scales
    return merged


class SinkEngine:
    """The engine without a model: the conductor runs on a 40 ms clock, and each
    timeline it enters (or a settings change) is rendered into a Pattern and sent
    to the SuperConductor client's client_midi, which plays it through the MRT2
    server. Same HTTP API as Engine, so the page and the app server don't change."""

    MAX_PLAY_FRAMES = FRAME_RATE * 600  # a "play" timeline is rendered to its end, up to 10 min

    # tempo sync (POST /sync, the BPM page): see _syncer
    SYNC_EVERY_S = 0.5
    SYNC_BPM_DEADBAND = 1.0  # apply the heard tempo once it differs by this much (bpm)
    SYNC_PHASE_DEADBAND = 0.04  # beats; smaller latency errors are left alone
    SYNC_PHASE_GAIN = 0.5  # share of a measured latency error corrected per check
    SYNC_PIN_EVERY_S = 4.0  # at most one beat-one pin per this many seconds

    def __init__(self, client_midi: str, lead_ms: float, preset: dict | None = None, sc_server: str = "http://localhost:9000"):
        preset = preset or {}
        self.settings = Settings.from_dict(preset.get("settings", {}))
        self.conductor = Conductor(self.settings)
        if "idle_chord" in preset:
            c = preset["idle_chord"]
            self.conductor.idle = Timeline("idle", [Chord(list(c["notes"]), c.get("label", ""), c.get("symbol", ""))], "idle")
        self.client_midi = client_midi.rstrip("/")
        # how long after the conductor makes a frame it is heard (the client's lead, plus
        # network); the sync calibrates it against what the server hears
        self.lead_s = max(0.0, lead_ms / 1000)
        self.history: deque = deque(maxlen=10 * FRAME_RATE)  # (wall time, meta) of recent frames
        self.sc_server = sc_server.rstrip("/")
        self.sync_on = False
        self.sync_state: dict = {}  # what the last check saw and did (the BPM page shows it)
        self._pin_wanted = False
        self._pin_at = 0.0
        self.dirty = False
        self.outbox: queue.Queue = queue.Queue(maxsize=1)  # the latest pattern wins
        # client_midi path -> body still to POST (/controls, /prompts); the latest per path wins
        self.posts: dict = {}
        self.posts_cv = threading.Condition()
        # harmony = the NOTES lane and cfg_scales.notes; drums = the DRUMS lane,
        # cfg_scales.drums and the "drums" prompt. Off: reset once, then stop sending.
        self._gate = threading.Lock()
        self.send_harmony = True
        self.send_drums = True
        self._harmony_reset_pending = False  # the next pattern must set notes to null
        self._drums_reset_pending = False
        self._harmony_cfg_pending = False  # the next /controls must set cfg_scales.notes to null
        self._drums_cfg_pending = False
        self._drums_prompt_pending = False  # the next /prompts must remove "drums"
        self.connected = False
        self.remote_model = None
        self.last_error = ""
        self.running = True
        # the page reads these from /status; the sound itself is SuperConductor's
        self.paused, self.volume, self.muted = False, 1.0, False
        self.style_prompt = ""
        threading.Thread(target=self._sender, daemon=True).start()
        threading.Thread(target=self._controls_sender, daemon=True).start()
        threading.Thread(target=self._syncer, daemon=True).start()

    # --- the parts of the Engine API that don't apply here ---
    def set_style(self, prompt: str):
        self.style_prompt = prompt

    def set_layers(self, layers: list):
        self.style_prompt = " + ".join(str(t) for t, _ in layers)

    def settings_changed(self):
        with self._gate:
            self.dirty = True

    def _post_latest(self, path: str, body: dict):
        with self.posts_cv:
            if path == "/controls" and path in self.posts:
                body = merge_control_posts(self.posts[path], body)
            self.posts[path] = body  # replaces an unsent older body (controls: merged)
            self.posts_cv.notify()

    def _controls_body(self) -> dict:
        return controls_body(
            self.settings, self.send_harmony, self.send_drums,
            self._harmony_cfg_pending, self._drums_cfg_pending,
        )

    def controls_changed(self):
        with self._gate:
            body = self._controls_body()
        self._post_latest("/controls", body)

    def prompts_changed(self):
        """The drums prompt. While drums are released this either finishes the reset
        (weight 0, which removes it) or sends nothing, so a knob move can't put it back."""
        with self._gate:
            if not self.send_drums:
                if not self._drums_prompt_pending:
                    return
                body = {DRUM_PROMPT: 0.0}
            else:
                body = self.settings.prompts()
        self._post_latest("/prompts", body)

    def set_output(self, harmony: bool | None = None, drums: bool | None = None):
        """Turn sending harmony or drums to the client on or off.

        Turning one off asks for a single reset (lane null, its cfg_scales key null,
        and for drums the prompt removed) and then leaves that lane out of later
        messages. Turning it on sends Atlas's current values again. The page's own
        state (the grid, the knobs) is kept, so taking control back restores it.
        """
        if harmony is None and drums is None:
            return
        post_prompts = False
        changed = False
        with self._gate:
            if harmony is not None and bool(harmony) != self.send_harmony:
                changed = True
                self.send_harmony = bool(harmony)
                self.dirty = True
                if self.send_harmony:
                    self._harmony_reset_pending = False
                    self._harmony_cfg_pending = False
                else:
                    self._harmony_reset_pending = True
                    self._harmony_cfg_pending = True
            if drums is not None and bool(drums) != self.send_drums:
                changed = True
                self.send_drums = bool(drums)
                self.dirty = True
                post_prompts = True
                if self.send_drums:
                    self._drums_reset_pending = False
                    self._drums_cfg_pending = False
                    self._drums_prompt_pending = False
                else:
                    self._drums_reset_pending = True
                    self._drums_cfg_pending = True
                    self._drums_prompt_pending = True
            body = self._controls_body() if changed else None
        if not changed or body is None:
            return
        self._post_latest("/controls", body)
        if post_prompts:
            self.prompts_changed()
        log.info(
            "client output: harmony %s, drums %s",
            "sending" if self.send_harmony else "released",
            "sending" if self.send_drums else "released",
        )

    def _ack_pattern(self, pattern: dict):
        with self._gate:
            if "notes" in pattern and pattern["notes"] is None:
                self._harmony_reset_pending = False
            if "drums" in pattern and pattern["drums"] is None:
                self._drums_reset_pending = False

    def _ack_post(self, path: str, body: dict):
        with self._gate:
            if path == "/controls":
                scales = body.get("cfg_scales") or {}
                if "notes" in scales and scales["notes"] is None:
                    self._harmony_cfg_pending = False
                if "drums" in scales and scales["drums"] is None:
                    self._drums_cfg_pending = False
            elif path == "/prompts" and not self.send_drums and body.get(DRUM_PROMPT, 1) == 0:
                self._drums_prompt_pending = False

    def _retry_post(self, path: str, body: dict):
        """Put a failed reset back unless a newer body for that path is already waiting."""
        with self._gate:
            scales = body.get("cfg_scales") or {}
            retry = (
                path == "/controls" and (
                    (self._harmony_cfg_pending and "notes" in scales and scales["notes"] is None)
                    or (self._drums_cfg_pending and "drums" in scales and scales["drums"] is None)
                )
            ) or (path == "/prompts" and self._drums_prompt_pending and not self.send_drums)
        if not retry:
            return
        with self.posts_cv:
            if path not in self.posts:
                self.posts[path] = body
                self.posts_cv.notify()
        time.sleep(1.0)

    # --- client_midi ---
    def _http(self, method: str, path: str, body=None) -> dict:
        from urllib.request import Request, urlopen

        data = None if body is None else json.dumps(body).encode()
        req = Request(self.client_midi + path, data=data, method=method, headers={"Content-Type": "application/json"})
        with urlopen(req, timeout=1.0) as resp:
            return json.loads(resp.read() or b"{}")

    def _sender(self):
        """Posts patterns (latest wins) and checks client_midi every 2 s; when the
        client comes (back) online, the current timeline is sent again."""
        while self.running:
            try:
                pattern = self.outbox.get(timeout=2.0)
            except queue.Empty:
                pattern = None
            try:
                if pattern is not None:
                    self._http("POST", "/pattern", pattern)
                    self._ack_pattern(pattern)
                    self.last_error = ""
                    continue
                st = self._http("GET", "/status")
                was = self.connected
                self.connected = bool(st.get("connected"))
                self.remote_model = st.get("model")
                if self.connected and not was:
                    log.info("SuperConductor client connected (%s)", self.remote_model)
                    with self._gate:
                        self.dirty = True
                    self.controls_changed()
                    self.prompts_changed()
            except Exception as e:  # client not running, or not connected to its server (503)
                if self.connected or not self.last_error:
                    log.info("client_midi at %s: %s", self.client_midi, e)
                self.connected = False
                self.last_error = str(e)
                if pattern is not None:
                    # a release that didn't land has to go again; a new session starts at
                    # the defaults, and a session that still has Atlas's lane needs the null
                    retry = False
                    with self._gate:
                        if self._harmony_reset_pending or self._drums_reset_pending:
                            self.dirty = True
                            retry = True
                    if retry:
                        time.sleep(1.0)

    def _controls_sender(self):
        """POSTs the sampling controls to client_midi's /controls (the client relays them
        to the server as UpdateControls) and drum_prompt to its /prompts; latest wins."""
        while self.running:
            with self.posts_cv:
                if not self.posts:
                    self.posts_cv.wait(timeout=1.0)
                posts, self.posts = self.posts, {}
            for path, body in posts.items():
                try:
                    self._http("POST", path, body)
                    self._ack_post(path, body)
                except Exception as e:  # not connected: a pending reset is tried again
                    log.info("client_midi %s: %s", path, e)
                    self._retry_post(path, body)

    def _render(self, tokens: list, meta: FrameMeta) -> dict:
        """The timeline the conductor is in, from the frame it just made: one full
        loop (idle / preview), or to its end (play)."""
        ahead = self.conductor.clone()
        frames = [(tokens, meta.drum)]
        if meta.role == "play":
            while len(frames) < self.MAX_PLAY_FRAMES:
                t, m = ahead.next_frame()
                if m.role != "play":
                    break
                frames.append((t, m.drum))
            loop = False
        else:
            length = len(ahead._timeline().chords) * ahead._chord_frames()
            for _ in range(length - 1):
                t, m = ahead.next_frame()
                frames.append((t, m.drum))
            loop = True
        return to_pattern(frames, loop, self.settings, f"atlas {meta.role} #{self.conductor.entries}")

    def _enqueue(self, pattern: dict):
        try:
            self.outbox.get_nowait()  # drop an unsent older pattern
        except queue.Empty:
            pass
        self.outbox.put(pattern)

    def run(self):
        """The conductor clock: one frame every 40 ms of wall time."""
        period = 1.0 / FRAME_RATE
        next_t = time.perf_counter()
        entries = -1
        while self.running:
            tokens, meta = self.conductor.next_frame()
            with self._gate:
                changed = self.conductor.entries != entries or self.dirty
                if changed:
                    entries = self.conductor.entries
                    self.dirty = False
                    send_h, send_d = self.send_harmony, self.send_drums
                    reset_h, reset_d = self._harmony_reset_pending, self._drums_reset_pending
            if changed:
                pattern = gate_pattern(self._render(tokens, meta), send_h, send_d, reset_h, reset_d)
                if pattern is not None:
                    self._enqueue(pattern)
            self.history.append((time.time(), meta))
            if self._pin_wanted:
                self._maybe_pin()
            next_t += period
            delay = next_t - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            elif delay < -0.5:
                next_t = time.perf_counter()  # fell far behind (laptop asleep): don't race to catch up

    def heard_meta(self, t: float | None = None) -> FrameMeta | None:
        """The frame being heard at wall time t (now): the one made lead_s earlier."""
        t = time.time() if t is None else t
        found = None
        for made, meta in reversed(self.history):
            found = meta
            if made <= t - self.lead_s:
                break
        return found

    def heard_beat_pos(self, t: float) -> float | None:
        """Beat position heard at wall time t, between frames (frames are 40 ms apart)."""
        heard_t = t - self.lead_s
        for made, meta in reversed(self.history):
            if made <= heard_t:
                return meta.beat_pos + (heard_t - made) * self.settings.bpm / 60.0
        return None

    def status(self) -> dict:
        m = self.heard_meta()
        return {
            "ok": True,
            "sink": "superconductor",
            "output": {"harmony": self.send_harmony, "drums": self.send_drums},
            "sync": {"on": self.sync_on, "server_url": self.sc_server, "lead_ms": round(self.lead_s * 1000), **self.sync_state},
            "connected": self.connected,
            "client_midi": self.client_midi,
            "model": f"SuperConductor · {self.remote_model}" if self.connected else "SuperConductor (client not connected)",
            "style": self.style_prompt,
            "style_layers": [],
            "ms_per_frame": None,
            "realtime_budget_ms": 1000 / FRAME_RATE,
            "underruns": 0,
            "paused": False,
            "volume": self.volume,
            "muted": self.muted,
            "latency_ms": 0.0,
            "buffer_ms": round(self.lead_s * 1000),
            "settings": self.settings.__dict__,
            "playing": None
            if m is None
            else {
                "role": m.role,
                "timeline_id": m.timeline_id,
                "index": m.index,
                "label": m.label,
                "symbol": m.symbol,
                "beat": m.beat,
                "beat_frac": round(m.beat_frac, 3),
                "drum": m.drum,
                "drum_cell": m.drum_cell,
                "bar_beat": math.floor(m.beat_pos) % BAR_BEATS,
                "bar_beat_frac": round(m.beat_pos % 1.0, 3),
            },
        }

    # --- tempo sync with the SuperConductor server's /tempo ---
    def set_sync(self, on: bool):
        self.sync_on = bool(on)
        self.sync_state = {}
        self._pin_wanted = False
        log.info("tempo sync %s (%s/tempo/status)", "on" if on else "off", self.sc_server)

    def _syncer(self):
        """Twice a second: read the server's /tempo/status (the BPM page shows it), and
        while sync is on:
        1. tempo: set this engine's bpm to the server's heard_bpm (the tempo the listener
           hears: MRT2's tempo x the server's time-stretch speed), folded by octaves to
           the nearest of x/2, x, 2x, once it differs by SYNC_BPM_DEADBAND;
        2. beat phase: compare where the server hears the bar (beat + beat_frac) with the
           frame this engine thinks is heard now (made lead_s ago). The fraction of a beat
           between them is an error in lead_s, which is corrected (moving the music
           instead would move the beats the server detects with it, never converging);
        3. beat one: a whole-beat difference means the server counts the bar from another
           beat. The server's beat one is then pinned (POST /tempo/downbeat) at the
           moment this engine's beat one is heard."""
        from urllib.request import urlopen

        while self.running:
            time.sleep(self.SYNC_EVERY_S)
            try:
                t0 = time.time()
                with urlopen(self.sc_server + "/tempo/status", timeout=1.0) as resp:
                    st = json.loads(resp.read())
                t = (t0 + time.time()) / 2
            except Exception as e:
                self.sync_state = {"ok": False, "error": f"{self.sc_server}/tempo/status: {e}"}
                continue
            state = {"ok": True, "error": "", "server": {k: st.get(k) for k in (
                "mode", "speed", "model_bpm", "heard_bpm", "target_bpm", "conductor_bpm", "playing",
                "beat", "beat_frac", "beats_per_bar", "downbeat_source", "meter_confidence")}}
            if not self.sync_on:
                self.sync_state = state
                continue
            # 1. tempo
            heard = st.get("heard_bpm")
            if heard:
                bpm = self.settings.bpm
                folded = min((heard / 2, heard, heard * 2), key=lambda x: abs(math.log(x / bpm)))
                folded = round(min(SYNC_BPM_MAX, max(SYNC_BPM_MIN, folded)), 1)
                state["target_bpm"] = folded
                if abs(folded - bpm) >= self.SYNC_BPM_DEADBAND:
                    self.settings.bpm = folded
                    self.settings_changed()
                    state["applied_at"] = time.time()
                    log.info("tempo sync: %.1f -> %.1f bpm (server hears %.1f)", bpm, folded, heard)
            # 2 + 3. beat phase and beat one
            mine = self.heard_beat_pos(t)
            if st.get("playing") and st.get("beat") is not None and st.get("beat_frac") is not None and mine is not None:
                bar = st.get("beats_per_bar") or BAR_BEATS
                theirs = st["beat"] + st["beat_frac"]
                d = theirs - (mine % bar)
                frac = (d + 0.5) % 1.0 - 0.5  # beats, in [-0.5, 0.5)
                whole = round(d - frac) % bar
                state["phase_beats"] = round(frac, 3)
                state["bar_offset_beats"] = whole
                if abs(frac) > self.SYNC_PHASE_DEADBAND:
                    seconds = frac * 60.0 / self.settings.bpm
                    self.lead_s = min(3.0, max(0.05, self.lead_s - self.SYNC_PHASE_GAIN * seconds))
                if bar == BAR_BEATS and whole != 0 and abs(frac) < 0.25:
                    self._pin_wanted = True
            else:
                state["phase_beats"] = None
                state["bar_offset_beats"] = None
            state["last_pin"] = self.sync_state.get("last_pin")
            self.sync_state = state

    def _maybe_pin(self):
        """Called every frame while a pin is wanted: once this engine's beat one is being
        heard, tell the server that the beat heard now is beat one."""
        m = self.heard_meta()
        if m is None or time.time() - self._pin_at < self.SYNC_PIN_EVERY_S:
            return
        if math.floor(m.beat_pos) % BAR_BEATS == 0 and m.beat_pos % 1.0 < 0.15:
            self._pin_wanted = False
            self._pin_at = time.time()

            def post():
                from urllib.request import Request, urlopen

                try:
                    req = Request(self.sc_server + "/tempo/downbeat", data=b"null", method="POST",
                                  headers={"Content-Type": "application/json"})
                    urlopen(req, timeout=1.0).close()
                    self.sync_state = {**self.sync_state, "last_pin": time.time()}
                    log.info("tempo sync: server beat one pinned to this engine's beat one")
                except Exception as e:
                    log.info("tempo sync: pinning beat one failed: %s", e)

            threading.Thread(target=post, daemon=True).start()


def prioritize_this_thread():
    """Ask macOS to schedule this thread as user-interactive (performance cores, first in
    line). Generation needs a CPU core every 40 ms to dispatch GPU work; background load
    (databases, indexers) otherwise stretches frames past the real-time budget."""
    try:
        import ctypes

        libc = ctypes.CDLL("/usr/lib/libSystem.dylib")
        QOS_CLASS_USER_INTERACTIVE = 0x21
        rc = libc.pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0)
        log.info("generation thread QoS: user-interactive (rc=%d)", rc)
    except Exception as e:  # not macOS, or unavailable
        log.info("could not raise thread QoS: %s", e)


def parse_timeline(body: dict) -> Timeline:
    chords = [
        Chord(
            notes=[int(n) for n in c["notes"]],
            label=c.get("label", ""),
            symbol=c.get("symbol", ""),
            scale=[int(p) % 12 for p in c.get("scale", [])],
        )
        for c in body.get("chords", [])
    ]
    if not chords:
        raise ValueError("timeline needs at least one chord")
    for c in chords:
        if not all(0 <= n < 128 for n in c.notes):
            raise ValueError(f"MIDI note out of range in {c.notes}")
    role = body.get("role", "preview")
    if role not in ("idle", "preview", "play"):
        raise ValueError(f"unknown role {role}")
    return Timeline(role=role, chords=chords, id=str(body.get("id", "")))


def make_handler(engine: Engine):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, code: int, payload: dict):
            data = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_OPTIONS(self):
            self._send(204, {})

        def do_GET(self):
            if self.path.startswith("/status"):
                self._send(200, engine.status())
            else:
                self._send(404, {"ok": False, "error": "not found"})

        def do_POST(self):
            try:
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length) or b"{}")
                if self.path == "/timeline":
                    engine.conductor.set_timeline(parse_timeline(body))
                elif self.path == "/pause":
                    engine.paused = bool(body.get("paused", True))
                elif self.path == "/mix":
                    if "volume" in body:
                        engine.volume = min(1.5, max(0.0, float(body["volume"])))
                    if "muted" in body:
                        engine.muted = bool(body["muted"])
                elif self.path == "/shutdown":
                    # answer first, then let the main loop wind down: it stops generating,
                    # closes the audio stream and shuts this server (see main)
                    self._send(200, {"ok": True, "stopping": True})
                    log.info("shutdown requested over the control API")
                    engine.running = False
                    return
                elif self.path == "/sync":
                    if not hasattr(engine, "set_sync"):
                        raise ValueError("tempo sync needs the SuperConductor sink (make run)")
                    engine.set_sync(bool(body.get("on", True)))
                elif self.path == "/output":
                    if not hasattr(engine, "set_output"):
                        raise ValueError("releasing harmony or drums needs the SuperConductor sink (make run)")
                    if "harmony" not in body and "drums" not in body:
                        raise ValueError("say harmony, drums, or both (true sends, false releases)")
                    for key in ("harmony", "drums"):
                        if key in body and not isinstance(body[key], bool):
                            raise ValueError(f"{key} must be true or false")
                    engine.set_output(harmony=body.get("harmony"), drums=body.get("drums"))
                elif self.path == "/release":
                    engine.conductor.release()
                elif self.path == "/style":
                    if "layers" in body:
                        engine.set_layers([(l["text"], l.get("weight", 1.0)) for l in body["layers"]])
                    else:
                        engine.set_style(str(body["prompt"]))
                elif self.path == "/settings":
                    s = engine.settings
                    if "temperature" in body and not float(body["temperature"]) > 0:
                        raise ValueError("temperature must be > 0")
                    if "top_k" in body and int(body["top_k"]) < 1:
                        raise ValueError("top_k must be >= 1")
                    if "drum_prompt" in body and not 0.0 <= float(body["drum_prompt"]) <= 1.0:
                        raise ValueError("drum_prompt must be 0..1")
                    scales = cfg_scales(body.get("cfg_scales") or {}, s.cfg_scales)
                    for k in ("bpm", "temperature", "gain"):
                        if k in body:
                            setattr(s, k, float(body[k]))
                    if "top_k" in body:
                        s.top_k = int(body["top_k"])
                    s.cfg_scales = scales
                    if {"temperature", "top_k", "cfg_scales"} & body.keys():
                        engine.controls_changed()
                    if "drum_prompt" in body:
                        s.drum_prompt = round(float(body["drum_prompt"]), 3)
                        engine.prompts_changed()
                    if not body.keys() - {"temperature", "top_k", "cfg_scales", "drum_prompt"}:
                        self._send(200, engine.status())
                        return
                    if "beats_per_chord" in body:
                        s.beats_per_chord = int(body["beats_per_chord"])
                    if "texture" in body:
                        if body["texture"] not in TEXTURES:
                            raise ValueError(f"unknown texture {body['texture']}")
                        s.texture = str(body["texture"])
                    if "articulation" in body:
                        s.articulation = str(body["articulation"])
                    if "drums" in body:  # older clients: off | auto
                        s.drum_grid = drum_grid({**s.drum_grid, "mode": "free" if body["drums"] == "auto" else "silent"})
                    if "drum_grid" in body:
                        s.drum_grid = drum_grid(body["drum_grid"])
                    engine.settings_changed()
                else:
                    self._send(404, {"ok": False, "error": "not found"})
                    return
                self._send(200, engine.status())
            except Exception as e:
                self._send(400, {"ok": False, "error": str(e)})

    return Handler


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="mrt2_small")
    ap.add_argument("--port", type=int, default=9104)
    ap.add_argument("--buffer-frames", type=int, default=8, help="frames per audio chunk (40 ms each); one chunk plays while the next is generated")
    ap.add_argument("--prompt", default="warm grand piano, intimate, slow")
    ap.add_argument("--device", default=None, help="sounddevice output device name or index")
    ap.add_argument("--record", default=None, help="also write everything played to this wav on exit")
    ap.add_argument("--preset", default=None, help="preset name in audio_engine/presets/ or a path to a preset JSON (see VOICE.md)")
    ap.add_argument("--sink", default="local", choices=["local", "superconductor"],
                    help="local: run Magenta here; superconductor: send notes/drums to the SuperConductor client instead")
    ap.add_argument("--client-midi", default="http://localhost:9105", help="SuperConductor client_midi URL (--sink superconductor)")
    ap.add_argument("--sc-server", default="http://localhost:9000",
                    help="--sink superconductor: the MRT2 server's HTTP address, for tempo sync (its /tempo/status)")
    ap.add_argument("--lead-ms", type=float, default=600.0,
                    help="--sink superconductor: how far SuperConductor's audio trails the conductor (its buffer)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", force=True)

    import signal

    preset = None
    if args.preset:
        from pathlib import Path

        path = Path(args.preset)
        if not path.exists():
            path = Path(__file__).parent / "presets" / f"{args.preset}.json"
        preset = json.loads(path.read_text())
        log.info("preset %s loaded from %s", preset.get("name", path.stem), path)

    if args.sink == "superconductor":
        sink = SinkEngine(args.client_midi, args.lead_ms, preset, args.sc_server)
        server = ThreadingHTTPServer(("0.0.0.0", args.port), make_handler(sink))
        threading.Thread(target=server.serve_forever, daemon=True).start()

        def stop_sink(_sig, _frame):
            sink.running = False

        signal.signal(signal.SIGINT, stop_sink)
        signal.signal(signal.SIGTERM, stop_sink)
        log.info("sink: notes/drums to %s; control API on http://127.0.0.1:%d", args.client_midi, args.port)
        try:
            sink.run()
        finally:
            server.shutdown()
            server.server_close()
            log.info("engine stopped")
        return

    import sounddevice as sd

    prioritize_this_thread()
    # MLX streams are per-thread: the model is loaded and run on the main thread.
    # The HTTP server runs on a background thread; the audio callback on PortAudio's.
    engine = Engine(args.model, args.buffer_frames, args.record, args.prompt, preset)
    server = ThreadingHTTPServer(("0.0.0.0", args.port), make_handler(engine))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    engine.queue.put([engine.generate_one() for _ in range(args.buffer_frames)])

    device = int(args.device) if args.device and args.device.isdigit() else args.device
    stream = sd.OutputStream(
        samplerate=SAMPLE_RATE,
        channels=2,
        dtype="float32",
        blocksize=FRAME_SAMPLES,
        callback=engine.audio_callback,
        device=device,
    )

    def stop(_sig, _frame):
        engine.running = False

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    log.info("streaming; control API on http://127.0.0.1:%d", args.port)
    with stream:
        engine.out_latency_ms = round(float(stream.latency) * 1000.0, 1)
        try:
            engine.generator_loop()
        except KeyboardInterrupt:
            pass
        finally:
            engine.running = False
            server.shutdown()
            server.server_close()
            engine.save_recording()
            log.info("engine stopped")


if __name__ == "__main__":
    main()
