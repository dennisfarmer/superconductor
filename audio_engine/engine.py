"""SuperConductor Atlas audio engine: Magenta RealTime 2, streaming forever.

The engine owns the model and the sound card. It never stops generating; what it
plays is steered by *timelines* posted over HTTP by the Jac backend:

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
SuperConductor client's client_midi (default http://localhost:8470), which plays it
through the MRT2 server with SuperConductor's prompts. /style, temperature and
cfg_style don't apply there: SuperConductor owns the sound.

HTTP API (JSON, CORS open, default port 8765):
  GET  /status
  POST /timeline {"role": "idle"|"preview"|"play", "chords": [{"label", "symbol", "notes"}], "id"}
  POST /release                      -> drop preview/play, back to idle
  POST /style    {"prompt": "..."}  or  {"layers": [{"text": "...", "weight": 0.7}, ...]}  (embedding blend)
  POST /settings {"bpm", "beats_per_chord", "articulation", "texture", "drum_grid", "cfg_notes", "cfg_style", "temperature", "gain"}
  POST /mix {"volume": 0..1.5, "muted": bool}   output level, applied at the speaker (no buffer delay)
  POST /shutdown {}   finish the current chunk, close the audio stream and the HTTP server, exit

Run:  python audio_engine/engine.py [--model mrt2_small] [--port 8765] [--record out.wav]
      python audio_engine/engine.py --sink superconductor [--client-midi http://localhost:8470]
"""

import argparse
import json
import logging
import queue
import threading
import time
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
    cfg = min(7.0, max(-1.0, float(body.get("cfg", 1.0))))
    return {"mode": mode, "bars": bars, "per_beat": per_beat, "cells": cells, "after_hit": after_hit, "cfg": cfg}


@dataclass
class Settings:
    bpm: float = 84.0
    beats_per_chord: int = 4
    articulation: str = "auto"  # auto | hold
    texture: str = "block"  # block | arpeggio | improvise
    drum_grid: dict = field(default_factory=drum_grid)  # silent: no drums, as before
    cfg_notes: float = 3.0
    cfg_style: float = 3.0
    temperature: float = 1.2
    gain: float = 0.9

    @classmethod
    def from_dict(cls, d: dict) -> "Settings":
        d = dict(d)
        if "drums" in d:  # older presets: drums off | auto
            d["drum_grid"] = {"mode": "free" if d.pop("drums") == "auto" else "silent"}
        if "drum_grid" in d:
            d["drum_grid"] = drum_grid(d["drum_grid"])
        return cls(**d)

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

    def _on_beat(self) -> bool:
        fpb = self.settings.frames_per_beat()
        return int(self.frame / fpb) != int((self.frame - 1) / fpb)

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
        frames_per_cell = self.settings.frames_per_beat() / g["per_beat"]
        k = int(self.frame / frames_per_cell)
        cell = k % len(g["cells"])
        value = g["cells"][cell]
        if value == 1:
            first = self.frame == 0 or int((self.frame - 1) / frames_per_cell) != k
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
            )
            self.frame += 1
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
        self.mrt = MagentaRT2StdMlxfn(size=model, temperature=self.settings.temperature, top_k=int(preset.get("top_k", 40)))
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
            cfg_scales={"musiccoca": s.cfg_style, "notes": s.cfg_notes, "drums": s.drum_grid["cfg"]},
            temperature=s.temperature,
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
        "cfg_notes": settings.cfg_notes,
        "cfg_drums": settings.drum_grid["cfg"],
    }


class SinkEngine:
    """The engine without a model: the conductor runs on a 40 ms clock, and each
    timeline it enters (or a settings change) is rendered into a Pattern and sent
    to the SuperConductor client's client_midi, which plays it through the MRT2
    server. Same HTTP API as Engine, so the page and the Jac app don't change."""

    MAX_PLAY_FRAMES = FRAME_RATE * 600  # a "play" timeline is rendered to its end, up to 10 min

    def __init__(self, client_midi: str, lead_ms: float, preset: dict | None = None):
        preset = preset or {}
        self.settings = Settings.from_dict(preset.get("settings", {}))
        self.conductor = Conductor(self.settings)
        if "idle_chord" in preset:
            c = preset["idle_chord"]
            self.conductor.idle = Timeline("idle", [Chord(list(c["notes"]), c.get("label", ""), c.get("symbol", ""))], "idle")
        self.client_midi = client_midi.rstrip("/")
        self.lead_frames = max(0, round(lead_ms / 1000 * FRAME_RATE))
        self.history: list = []  # metas of the last lead_frames + 1 frames; the oldest is "playing"
        self.dirty = False
        self.outbox: queue.Queue = queue.Queue(maxsize=1)  # the latest pattern wins
        self.connected = False
        self.remote_model = None
        self.last_error = ""
        self.running = True
        # the page reads these from /status; the sound itself is SuperConductor's
        self.paused, self.volume, self.muted = False, 1.0, False
        self.style_prompt = ""
        threading.Thread(target=self._sender, daemon=True).start()

    # --- the parts of the Engine API that don't apply here ---
    def set_style(self, prompt: str):
        self.style_prompt = prompt

    def set_layers(self, layers: list):
        self.style_prompt = " + ".join(str(t) for t, _ in layers)

    def settings_changed(self):
        self.dirty = True

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
                    self.last_error = ""
                    continue
                st = self._http("GET", "/status")
                was = self.connected
                self.connected = bool(st.get("connected"))
                self.remote_model = st.get("model")
                if self.connected and not was:
                    log.info("SuperConductor client connected (%s)", self.remote_model)
                    self.dirty = True
            except Exception as e:  # client not running, or not connected to its server (503)
                if self.connected or not self.last_error:
                    log.info("client_midi at %s: %s", self.client_midi, e)
                self.connected = False
                self.last_error = str(e)

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

    def run(self):
        """The conductor clock: one frame every 40 ms of wall time."""
        period = 1.0 / FRAME_RATE
        next_t = time.perf_counter()
        entries = -1
        while self.running:
            tokens, meta = self.conductor.next_frame()
            if self.conductor.entries != entries or self.dirty:
                entries, self.dirty = self.conductor.entries, False
                pattern = self._render(tokens, meta)
                try:
                    self.outbox.get_nowait()  # drop an unsent older pattern
                except queue.Empty:
                    pass
                self.outbox.put(pattern)
            self.history = (self.history + [meta])[-(self.lead_frames + 1):]
            next_t += period
            delay = next_t - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            elif delay < -0.5:
                next_t = time.perf_counter()  # fell far behind (laptop asleep): don't race to catch up

    def status(self) -> dict:
        m = self.history[0] if self.history else None
        return {
            "ok": True,
            "sink": "superconductor",
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
            "buffer_ms": self.lead_frames * 1000 // FRAME_RATE,
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
                elif self.path == "/release":
                    engine.conductor.release()
                elif self.path == "/style":
                    if "layers" in body:
                        engine.set_layers([(l["text"], l.get("weight", 1.0)) for l in body["layers"]])
                    else:
                        engine.set_style(str(body["prompt"]))
                elif self.path == "/settings":
                    s = engine.settings
                    for k in ("bpm", "cfg_notes", "cfg_style", "temperature", "gain"):
                        if k in body:
                            setattr(s, k, float(body[k]))
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
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--buffer-frames", type=int, default=8, help="frames per audio chunk (40 ms each); one chunk plays while the next is generated")
    ap.add_argument("--prompt", default="warm grand piano, intimate, slow")
    ap.add_argument("--device", default=None, help="sounddevice output device name or index")
    ap.add_argument("--record", default=None, help="also write everything played to this wav on exit")
    ap.add_argument("--preset", default=None, help="preset name in audio_engine/presets/ or a path to a preset JSON (see VOICE.md)")
    ap.add_argument("--sink", default="local", choices=["local", "superconductor"],
                    help="local: run Magenta here; superconductor: send notes/drums to the SuperConductor client instead")
    ap.add_argument("--client-midi", default="http://localhost:8470", help="SuperConductor client_midi URL (--sink superconductor)")
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
        sink = SinkEngine(args.client_midi, args.lead_ms, preset)
        server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(sink))
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
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(engine))
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
