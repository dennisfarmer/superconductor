# VOICE: the saved "warm piano" sound

Saved 2026-09-26. This is the Magenta RealTime 2 sound that played from the audio engine's
idle vamp on a C major chord. It is stored as a preset:

```bash
python audio_engine/engine.py --preset warm_piano
```

The preset is `audio_engine/presets/warm_piano.json`. With no `--preset`, `audio_engine/engine.py`
uses the same values as its defaults, except for the C-major voicing noted below.

Two runs were audible, and both used **identical sound parameters**:

| Run | C-major voicing (MIDI → notes) | Notes |
|---|---|---|
| **A/B test** (`ab.py`, the one saved as the preset) | `[48, 55, 64, 72]` → C3 G3 E4 C5 | Engine's built-in idle chord. Clean playback, 0 underruns. Third on top of the inner voices, open spacing. |
| **Live engine** (driven by the Jac app) | `[48, 52, 60, 67]` → C3 E3 C4 G4 | The app's "home voicing" for I (`theory/voicing.jac: home_voicing`). The first time it played it stuttered, from GPU contention with the headless browser. On a re-run it played clean with 0 underruns. |

To use the live-engine voicing in the preset instead, set `"idle_chord.notes"` to `[48, 52, 60, 67]`.

---

## 1. Environment

| Item | Value |
|---|---|
| Machine | MacBook Pro, Apple M3 Pro (MLX, Metal GPU) |
| Conda env | `sc_env_jaclang`, Python 3.12 |
| `magenta-rt` | 2.0.3 (`magenta-rt[mlx]`) |
| `mlx` | **0.32.0**, pinned. The `.mlxfn` format breaks at 0.32.1, and 0.32.0 loads files exported by 0.31.x and 0.32.0. |
| Model files | `~/Documents/Magenta/magenta-rt-v2/models/mrt2_small/` (`mrt2_small.mlxfn`, header says exported by MLX 0.31.1; `mrt2_small_state.safetensors`) |
| Style model | MusicCoCa (`~/Documents/Magenta/magenta-rt-v2/resources/musiccoca`), runs on LiteRT/XNNPACK on the CPU |
| Audio output | `sounddevice` (PortAudio), default output device |

## 2. Model construction

```python
from magenta_rt import MagentaRT2StdMlxfn
mrt = MagentaRT2StdMlxfn(
    size="mrt2_small",      # exported .mlxfn, 230M params
    temperature=1.2,        # also passed explicitly on every generate() call
    top_k=40,               # library default
    # cfg_scales left at the library default here; the real values are passed per call (below)
    # warmup_steps=5 (library default): 5 dummy steps with all conditioning masked
)
```

## 3. Style (MusicCoCa) conditioning

| Parameter | Value |
|---|---|
| Text prompt | `"warm grand piano, intimate, slow"` |
| Embedding call | `mrt.embed_style(prompt, use_mapper=True)`, with defaults `pool_across_time=True`, `seed=0` |
| Tokenisation | `mrt.tokenize_style(embedding)`, truncated to 12 RVQ levels (`MUSICCOCA.rvq_truncation_level`) |
| **Exact style tokens** | `[635, 654, 757, 449, 524, 160, 633, 479, 203, 628, 10, 184]` |

The tokens are deterministic for this prompt, which I checked by embedding it twice. The preset
stores them, so the style conditioning is bit-identical even if MusicCoCa changes.

## 4. Per-frame sampling parameters

`mrt.generate(conditioning=..., cfg_scales=..., temperature=1.2, frames=1, state=state)` is called
once per 40 ms frame, and the streaming state is threaded from call to call.

| Parameter | Value |
|---|---|
| `temperature` | **1.2** (the library CLI default is 1.3) |
| `top_k` | **40** |
| CFG, style (`musiccoca`) | **3.0** |
| CFG, notes | **3.0** (the library's `generate` CLI defaults to 0.1 for notes in its function signature and 1.0 on its command line; 3.0 makes the model follow the chord closely) |
| CFG, drums | 1.0 |
| Drums conditioning (`drum_pianoroll_tokens`) | `[0]` = no drums ("off") |
| Output gain | 0.9, applied after the library's own int16 conversion (internal gain 0.5, then ÷32768) |

## 5. Notes (MIDI-like) conditioning: `pianoroll_with_onsets_tokens`

This is a 128-slot vector per frame, one slot per MIDI pitch. Pitches not in the chord get `0` (off).

| Frame | Value for each chord pitch |
|---|---|
| First frame of a chord | `2` (onset) for new pitches; `1` (continuation) for pitches that were also in the previous chord (common tones held) |
| Every later frame of the chord | **`3`** ("on, the model chooses onset or continuation"). This is **articulation = "auto"**, and it lets the model re-strike and shape the chord itself. `"hold"` would send `1` instead. |

During the saved sound, only the single idle chord was sounding, as a continuous vamp. Its first frame had
all four pitches as onsets (`2`), and every frame after that was `3`.

## 6. Timing

| Parameter | Value |
|---|---|
| Frame rate | 25 frames/s (40 ms/frame), 1920 samples/frame |
| Sample rate / channels | 48 kHz, stereo, float32 |
| Tempo | 84 bpm, 4 beats per chord, so 80 frames (3.2 s) per chord slot. This only matters when chords change; timeline switches are quantised to beats. |
| Engine buffer | 6 frames (240 ms) generated ahead of playback |
| PortAudio block size | 1920 samples (exactly one model frame per callback) |

## 7. Streaming state

- Each run started from the model's **initial state** (`mrt2_small_state.safetensors`), after the 5 warm-up steps. There was no audio prefill or continuation.
- The recurrent state then carries forward frame by frame, so the texture you hear develops from the vamp's own history. The A/B run had generated about 150 unplayed frames (6 s of history) before the audible part started. The live-engine run was audible from the first buffered frame.
- Sampling is stochastic (temperature 1.2, top-k 40), and no RNG seed is fixed. The *parameters* reproduce the sound's character, but a rerun won't produce the exact same waveform.

## 8. Code location

- Engine and defaults: `audio_engine/engine.py` (`Settings` dataclass, `Conductor.next_frame`, `Engine.generate_one`)
- Preset: `audio_engine/presets/warm_piano.json`
- The Jac app sends the same style prompt and tempo on load (`services/atlas.jac`, `Session` defaults), so the app's sound matches this preset.

## 9. Hiss: the `warm_piano_clean` variant

The saved voice's temperature (1.2) produces a faint constant hiss. I measured one held chord,
changing one setting at a time:

| Temperature | Energy in the 8–20 kHz band |
|---|---|
| 1.2 (saved voice) | 0.06% |
| **0.9** | **0.01%** (about 6× less) |
| 0.7 | 0.02% |

`audio_engine/presets/warm_piano_clean.json` is identical to `warm_piano` except `temperature: 0.9`.
Use `make restart PRESET=warm_piano_clean`, or `make restart PRESET=warm_piano` for the original.
