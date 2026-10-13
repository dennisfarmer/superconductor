# SuperConductor Atlas

A harmony explorer that drives SuperConductor's music model. You build a chord
progression on a 3D graph: beam-search walkers (Jac) propose what could come before
and after it, scored on harmonic syntax, voice leading and cadences. Hovering a chord
plays the path to it, clicking keeps it, and the Drums page sets when drum hits fall.
Atlas decides **which pitches and when**; SuperConductor's prompts decide **what it
sounds like**.

## Run

```bash
bash setup_env.sh          # once: creates the conda env sc_env_jaclang
make run                   # SuperConductor mode (default), app on http://localhost:8000
make run-local             # standalone: local Magenta RT 2, no SuperConductor needed
```

For `make run`, start `make server` and `make client` in `../superconductor_client`
first (Atlas starts anyway and connects once they are up). `make start` / `make
start-local` run in the background; `make stop`, `make status` and `make logs` work
for both.

## How it connects to SuperConductor

```
browser (Harmony / Drums pages)
   │  Jac app (main.jac, services/atlas.jac): progression, walkers, session
   ▼
audio_engine/engine.py --sink superconductor        (port 8765)
   │  HTTP POST /pattern                            (MIDI message protocol)
   ▼
superconductor_client  client_midi                  (port 8470)
   │  websocket "Pattern"
   ▼
mrt2_server.py  ──▶  audio, with SuperConductor's prompts
```

In SuperConductor mode the engine runs no model. It keeps a 40 ms frame clock, and
each time playback changes (a new chord hovered, kept or played, or new settings)
it renders what should sound into one Pattern and POSTs it to client_midi. Hovering
and the loop are sent as looping lanes; "Play all" is sent once (`loop: false`), and
SuperConductor then falls back to the loop. The format is the established protocol in
[`MIDI_MSG_PROTOCOL.md`](https://github.com/dennisfarmer/superconductor/blob/main/MIDI_MSG_PROTOCOL.md)
(`../superconductor_client/MIDI_MSG_PROTOCOL.md` locally).

## What Atlas changes in the MRT2 input

MRT2 takes, every 40 ms frame, a style embedding plus two control inputs. Atlas sets
only the two control inputs and their guidance strengths. **The style (text prompts)
stays with SuperConductor.**

| MRT2 input | Set by Atlas | Values |
|---|---|---|
| **NOTES** (128 pitch slots) | the voiced chord progression, by texture | see below |
| **DRUMS** (one slot) | the Drums page | `-1` model chooses, `0` no hit, `1` hit |
| `cfg_notes` | engine setting (default 3.0) | how strictly the model follows the notes |
| `cfg_drums` | "Follow" slider on the Drums page | how strictly it follows the drum lane |
| style / prompts | no (only in `make run-local`) | |

**NOTES, by texture** (the top bar's texture menu):

- **block chords:** on each chord change every pitch of the voicing is `2` (onset),
  then `3` (on, the model picks the articulation; `1`, held, with articulation set
  to "hold") until the next chord. All other pitches are `0`.
- **arpeggio:** the bass is held; the upper chord tones are broken into eighth notes
  (`2` per note), up and down over two octaves.
- **improvise:** the bass is struck (`2`), then `3`. Every pitch in MIDI 36–96 whose
  pitch class is in the chord is `-1`, so the model may play it, and everything else
  is `0`. The model
  improvises inside the harmony instead of playing a given voicing.

Chord lengths come from the tempo and "beats" in the top bar.

**DRUMS** (Drums page): *Free* is `-1` on every frame, *Silent* is `0`, and *Grid* is
a 4/4 step grid on the same beat clock. A hit cell is `1` on its first frame and `0`
for the rest of the cell; free and rest cells hold `-1` / `0` for the whole cell.
The model picks the drum sounds; the lane only says when hits happen.

## Configuration

- `walker.toml`: beam width, depth, menu size and the scoring dials of the walkers.
  It is read on each request, so an edit applies the next time the walkers are sent.
- `Makefile` variables: `PORT`, `ENGINE_PORT`, `CLIENT_MIDI`, `PRESET` (local mode).
- More detail: `docs/ENGINE.md` (engine), `docs/CRITERIA.md` (how chords are scored),
  `docs/VOICE.md` (local-mode sound presets).
