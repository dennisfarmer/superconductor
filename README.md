# SuperConductor Atlas

A harmony explorer that drives SuperConductor's music model. You build a chord
progression on a 3D graph: beam-search walkers propose what could come before
and after it, scored on harmonic syntax, voice leading and cadences. Hovering a chord
plays the path to it, clicking keeps it, and the Drums page sets when drum hits fall.
Atlas decides **which pitches and when**; SuperConductor's prompts decide **what it
sounds like**.

## Run

```bash
bash setup_env.sh          # once: creates the conda env sc_env_atlas
make run                   # SuperConductor mode (default), app on http://localhost:9103
make run-local             # standalone: local Magenta RT 2, no SuperConductor needed
make dev                   # development: API reloads on change, client hot-reloads
```

The client is built into `client/dist` (needs Node and npm) whenever its sources
changed; `make run` and `make start` do this first.

For `make run`, start `make server` and `make client` in `../superconductor_client`
first (Atlas starts anyway and connects once they are up). Restarting the engine and the
LLM path descriptions are only offered in local mode (`make run-local`, the app's
`--local`): with SuperConductor there is no model here to restart. `make start` / `make
start-local` run in the background; `make stop`, `make status` and `make logs` work
for both.

## How it connects to SuperConductor

```
browser (Harmony / Drums pages)
   │  app: React client (client/) + FastAPI server (server/app.py, server/atlas.py):
   │  progression, walkers, session
   ▼
audio_engine/engine.py --sink superconductor        (port 9104)
   │  HTTP POST /pattern, POST /controls            (MIDI message protocol)
   ▼
superconductor_client  client_midi                  (port 9105)
   │  websocket "Pattern", "UpdateControls" (same bodies)
   ▼
mrt2_server.py  ──▶  audio, with SuperConductor's prompts
```

In SuperConductor mode the engine runs no model. It keeps a 40 ms frame clock, and
each time playback changes (a new chord hovered, kept or played, or new settings)
it renders what should sound into one Pattern and POSTs it to client_midi. Hovering
and the loop are sent as looping lanes; "Play all" is sent once (`loop: false`), and
SuperConductor then falls back to the loop. The sampling controls (the controls row) go
separately, to `/controls`, so turning a knob doesn't resend the Pattern. The format is the established protocol in
[`MIDI_MSG_PROTOCOL.md`](https://github.com/dennisfarmer/superconductor/blob/main/MIDI_MSG_PROTOCOL.md)
(`../superconductor_client/MIDI_MSG_PROTOCOL.md` locally).

## What Atlas changes in the MRT2 input

MRT2 takes, every 40 ms frame, a style embedding plus two control inputs, and is
sampled with `generate()`'s `cfg_scales`, `temperature` and `top_k`. Atlas sets the two
control inputs and the sampling controls. **The style (text prompts) stays with
SuperConductor.**

| MRT2 input | Set by Atlas | Values |
|---|---|---|
| **NOTES** (128 pitch slots) | the voiced chord progression, by texture | see below |
| **DRUMS** (one slot) | the Drums page | `-1` model chooses, `0` no hit, `1` hit |
| `cfg_scales.musiccoca` | the controls row under the top bar | how strictly the model follows the style (MRT2 default 3.0) |
| `cfg_scales.notes` | the controls row | how strictly it follows NOTES (Atlas starts at 3.0; MRT2 default 1.0) |
| `cfg_scales.drums` | the controls row | how strictly it follows DRUMS (-1 to 7 in whole steps; default 1.0) |
| `temperature`, `top_k` | the controls row | sampling (MRT2 defaults 1.3 and 40; the `warm_piano` preset uses 1.2) |
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

Chord lengths come from the tempo and "beats" in the top bar. The **BPM page** (click "BPM"
in the top bar) can sync Atlas's tempo and beat one to the tempo the SuperConductor server
hears, and explains the delays between a message and the audio.

**DRUMS** (Drums page): *Free* is `-1` on every frame, *Silent* is `0`, and *Grid* is
a 4/4 step grid on the same beat clock. A hit cell is `1` on its first frame and `0`
for the rest of the cell; free and rest cells hold `-1` / `0` for the whole cell.
The model picks the drum sounds; the lane only says when hits happen.

## Configuration

- `walker.toml`: beam width, depth, menu size and the scoring dials of the walkers.
  It is read on each request, so an edit applies the next time the walkers are sent.
- `Makefile` variables: `PORT`, `ENGINE_PORT`, `CLIENT_MIDI`, `SC_SERVER` (the MRT2 server's
  HTTP address, for tempo sync), `PRESET` (local mode).
- Environment: `ATLAS_ENGINE_URL` (where the app reaches the engine, default
  `http://127.0.0.1:9104`), `ATLAS_ENGINE_PUBLIC_URL` (where the page reaches it, since it
  sends timelines and reads `/status` itself; default the same, `/engine` behind the
  SuperConductor container's gateway), `ATLAS_DATA`, `OLLAMA_URL`, `LLM_MODEL`.
- More detail: `docs/ENGINE.md` (engine), `docs/CRITERIA.md` (how chords are scored),
  `docs/VOICE.md` (local-mode sound presets).

## Layout

- `server/theory/`: the music theory (vocabulary, voicing, syntax, voice-leading rules,
  cadences, phrase costs).
- `server/graph/`: the harmony graph, grown lazily, and the `BeamSearch` walker.
- `server/atlas.py`: the session and every endpoint; `server/app.py` serves them as
  `POST /function/<name>`, plus `/healthz` and the built client. The session is saved
  to `data/session.json` (`ATLAS_DATA` sets the folder).
- `server/narrate.py`: the optional LLM description of a path (Ollama, `LLM_MODEL`).
- `client/src/`: the React + TypeScript page. `api-types.ts` is generated from the
  server's OpenAPI schema with `npm --prefix client run gen-types` (API on :8763).
- `tests/`: `pytest tests/` checks that the session file saves and loads.
