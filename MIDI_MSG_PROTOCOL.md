# MIDI message protocol (client_midi)

How another program (a MIDI file player, Harmonic Atlas, a controller, ...) gives
SuperConductor's music model its **NOTES** and **DRUMS** input. The SuperConductor
client keeps providing the prompts, so it decides *what it sounds like*. Your program
decides *which pitches and when*, and *when drum hits happen*.

```
your program ──HTTP POST /pattern──▶ superconductor_client (client_midi, port 8470)
                                         │  websocket "Pattern" (same body)
                                         ▼
                                     mrt2_server.py ──▶ audio back to superconductor_client
```

Start `make server` and `make client` (or `client-remote`) in `superconductor_client`.
client_midi listens on `http://localhost:8470` once the client is running. The port is
`midi_port` in `superconductor/collab.toml`, and `0` turns it off.

## Endpoints

| Request | Body | Reply |
|---|---|---|
| `POST /pattern` | a Pattern (below), or `null` | `{"ok": true}` once forwarded; `400` if the body isn't a JSON object or `null`; `503` if the client isn't connected to the server yet |
| `GET /status` | | `{"connected": bool, "model": "mrt2_small", "last_pattern_id": ...}` |

CORS is open, so a browser page can send patterns too. The server validates the pattern
in full. If it is invalid, the HTTP reply is still `{"ok": true}`, because the message was
forwarded, and the client console prints `MRT2 server: Pattern: <reason>`. Music keeps
playing with the previous pattern.

## What the model takes

MRT2 generates audio in **frames of 40 ms** (25 per second). Besides the text style, every
frame takes two inputs:

**NOTES**: 128 slots, one per MIDI pitch (0–127), each one of:

| value | meaning |
|---|---|
| `-1` | no instruction: the model decides whether this pitch sounds |
| `0` | silent |
| `1` | held: still sounding from before |
| `2` | onset: a new attack in this frame |
| `3` | on: the model decides whether it's a new attack or held |

**DRUMS**: one slot:

| value | meaning |
|---|---|
| `-1` | no instruction: the model decides |
| `0` | no drum hit |
| `1` | a drum hit in this frame |

There is one drum lane. It controls *when* hits happen; the model chooses the drum
sound (kick, snare, hats, ...). There is no velocity or instrument input: the model
picks instrument, loudness and expression from the style.

With no pattern, every slot of both lanes is `-1`, which is how SuperConductor plays
without a note source.

## The Pattern

```json
{
  "id": "my-player song.mid",
  "notes": {"loop": false, "steps": [
    {"frames": 25, "pitches": {"48": 2, "64": 2, "67": 2}, "default": 0},
    {"frames": 13, "pitches": {"48": 1, "64": 1, "67": 1, "72": 2}, "default": 0}
  ]},
  "drums": {"loop": false, "steps": [
    {"frames": 9, "first": 1, "rest": 0},
    {"frames": 9, "first": 1, "rest": 0}
  ]},
  "cfg_notes": 3.0,
  "cfg_drums": 1.0
}
```

Every field is optional.

| field | meaning |
|---|---|
| `notes` | the NOTES lane: `{"loop": bool, "steps": [...]}`. Left out: the lane is unchanged. `null`: cleared (all `-1`). |
| `drums` | the DRUMS lane, the same way |
| `cfg_notes` | how strongly the model follows the notes (classifier-free guidance, `-1` to `7`, in steps of 0.2). It stays set until changed. |
| `cfg_drums` | the same for drums (`-1` to `7`, steps of 1) |
| `id` | any string; printed by the server and shown in `GET /status` |

A body of `null` clears both lanes.

### Steps

A lane is a list of steps, and each step lasts `frames` × 40 ms (`frames` ≥ 1, default 1).

**Notes step:** `{"frames": n, "pitches": {"<midi>": value, ...}, "default": value}`
- `pitches` sets some slots; every other slot gets `default` (`-1` if omitted).
- Frame 1 of the step uses the values as written. In frames 2..n, every `2` (onset) becomes
  `1` (held), so a step is a note event with a duration: one attack, then held.
- Pitch keys may be strings or numbers.

**Drums step:** `{"frames": n, "first": value, "rest": value}`
- `first` is the value of the step's first frame (default `-1`); `rest` is the value of the
  remaining frames.
- `rest` defaults to `0` when `first` is `1` (one hit, then nothing), otherwise to `first`.

### Lanes, loops and timing

- The two lanes are **independent**. Each has its own steps, length, loop and position, so one
  program can own the notes while another owns the drums. Send only the lane you own.
- `"loop": true` repeats the lane forever. `"loop": false` plays it once, then the lane
  returns to the **last looping steps it was given**, or to all `-1` if there were none.
- A new lane **replaces** the old one and starts from its first step **at the next frame
  the server generates**. It does not wait for the previous lane to finish.
- **Latency:** the server generates a few blocks ahead of playback (`credits` in
  `collab.toml`, 3 × 200 ms by default; more over the cluster tunnel), so a new pattern is
  heard about **0.6 s after it arrives** locally. There is no position feedback. If you need to
  line up with what is heard (e.g. a cursor in your UI), assume this fixed lead.
- The server's tempo control (`/tempo` on the server) time-stretches the audio. With a
  driven drum lane or a MIDI file at its own tempo, keep it off (`POST /tempo/free`, the
  default).

## Streaming a MIDI file

Because a new pattern replaces the current lane, send a whole song (or a whole section) as
**one** play-once pattern. A 3-minute file is 4,500 frames, but run-length-encoded steps keep
it to a few hundred steps, which is a small JSON message. To restart, seek or switch songs,
send a new pattern starting at the new position. To stop, send `{"notes": null, "drums":
null}`.

Converting a file:
1. Get every note as `(start_s, end_s, pitch, channel)` in seconds, with tempo changes applied.
2. Frame `f` covers `[f × 0.04, (f + 1) × 0.04)` s. A note sounds in frames
   `round(start_s × 25)` to `max(that + 1, round(end_s × 25))` (exclusive): `2` in its first
   frame, `1` after.
3. Choose what the pitches the file doesn't play should be:
   - `default: 0`: play the file as written (only its notes sound);
   - `default: -1`: the file's notes are guaranteed, and the model may add its own;
   - file notes as `3` or `-1` instead of `2`/`1`: the model chooses the articulation, or
     whether to play them at all.
4. Drums: General MIDI channel 10 (index 9). Mark `1` in the frame where any drum note
   starts and `0` elsewhere. With `-1` instead of `0`, the model may add fills between your
   hits. Leave the drum notes out of the notes lane.
5. Merge consecutive frames into steps, as below.

Reference converter (needs `pip install mido`):

```python
import json, sys, urllib.request
import mido

FPS = 25

def midi_to_pattern(path, default=0, drums_between=0, loop=False):
    notes, hits, active = [], [], {}
    t = 0.0
    for msg in mido.MidiFile(path):          # iterating applies the tempo map; msg.time is in seconds
        t += msg.time
        if msg.type not in ("note_on", "note_off"):
            continue
        on = msg.type == "note_on" and msg.velocity > 0
        if msg.channel == 9:                 # GM drums -> the drum lane
            if on:
                hits.append(round(t * FPS))
            continue
        key = (msg.channel, msg.note)
        if on:
            active[key] = t
        elif key in active:
            notes.append((active.pop(key), t, msg.note))
    end = max([round(e * FPS) for _, e, _ in notes] + [h + 1 for h in hits] + [1])

    frames = [[default] * 128 for _ in range(end)]
    for start, stop, pitch in notes:
        a = round(start * FPS)
        for f in range(a, max(a + 1, round(stop * FPS))):
            if frames[f][pitch] != 2:        # an attack in this frame wins over a held note
                frames[f][pitch] = 2 if f == a else 1
    drums = [drums_between] * end
    for h in hits:
        drums[h] = 1

    def note_steps():
        steps, i = [], 0
        while i < end:
            later = [1 if v == 2 else v for v in frames[i]]
            j = i + 1
            while j < end and frames[j] == later:
                j += 1
            steps.append({"frames": j - i, "default": default,
                          "pitches": {str(p): v for p, v in enumerate(frames[i]) if v != default}})
            i = j
        return steps

    def drum_steps():
        steps, i = [], 0
        while i < end:
            rest = drums[i + 1] if i + 1 < end else drums_between
            j = i + 1
            while j < end and drums[j] == rest:
                j += 1
            steps.append({"frames": j - i, "first": drums[i], "rest": rest})
            i = j
        return steps

    body = {"id": path, "notes": {"loop": loop, "steps": note_steps()}}
    if hits:
        body["drums"] = {"loop": loop, "steps": drum_steps()}
    return body

def send(body, url="http://localhost:8470/pattern"):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=2) as r:
        return json.loads(r.read())

if __name__ == "__main__":
    print(send(midi_to_pattern(sys.argv[1])))
```

Pitches 0–127 all count, but the model is trained on musical ranges. Very low or high notes
may be ignored.

## Live input (a controller or keyboard)

For input that changes as it happens, send the current state each time it changes: an
attack frame, then a long held step, as a looping lane. When a C-major chord goes down:

```json
{"notes": {"loop": true, "steps": [
  {"frames": 1, "pitches": {"60": 2, "64": 2, "67": 2}, "default": -1},
  {"frames": 100000, "pitches": {"60": 1, "64": 1, "67": 1}, "default": -1}]}}
```

When the keys go up, send the new state, e.g. `{"notes": null}` to hand everything back to
the model. Looping (rather than `loop: false`) makes this the lane's fallback too, so a
play-once pattern sent later returns to it. A single one-frame looping step would re-attack
the chord every 40 ms. Expect the ~0.6 s lead between a key press and hearing it.

## Recipes

| Goal | Notes lane | Drums lane |
|---|---|---|
| Play the file as written | file notes `2`/`1`, `default: 0` | GM drum hits `1`, else `0` |
| File melody, the model accompanies | file notes `2`/`1`, `default: -1` | `-1` |
| Improvise inside a chord | bass `2` then `3`; other chord-tone pitches `-1`; `default: 0` | — |
| Model's notes over a fixed groove | `null` (all `-1`) | your hits `1`, `rest: 0` |
| Groove with fills | — | hits `1`, `rest: -1` |
| No drums at all | — | one step `{"frames": 1, "first": 0}`, `loop: true` |

## Quick test

```bash
curl -s localhost:8470/status
# a drum hit every second
curl -s -X POST localhost:8470/pattern -d '{"drums":{"loop":true,"steps":[{"frames":25,"first":1}]}}'
# back to free
curl -s -X POST localhost:8470/pattern -d 'null'
```

The server-side reference is the module docstring of `superconductor_server/mrt2_server.py`
(`Pattern`, `expand_lane`, `Lane`).
