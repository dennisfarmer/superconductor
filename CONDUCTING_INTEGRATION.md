# Conducting Integration

Four processes run at once. The conducting process sends tempo requests directly to the server over HTTP:

| Process | Branch / folder | Port |
|---|---|---|
| server (MRT2 + tempo) | `server` / `superconductor_server` | 9100 (9000 through the Lighthouse SSH tunnel) |
| client | `main` / `superconductor_client` | 8467 (objects page) |
| describe | `describe` / `superconductor_describe` | 9200 |
| conducting | (new) | sends requests to the server |

### Requests

The conducting process sends one of two kinds of input: beats, or an exact bpm.

| Request | Body | Use |
|---|---|---|
| `POST /tempo/beat` | `{"t": seconds}` | one beat per detected downbeat of the hand |
| `POST /tempo/target` | `{"bpm": 120}` | an exact tempo, if the conducting process computes the bpm itself |
| `POST /tempo/free` | `null` | stop conducting; the music eases back to MRT2's own tempo |
| `GET /tempo/status` | | `mode, speed, model_bpm, heard_bpm, conductor_bpm, ...` |

Set `t` from the conducting process's own monotonic clock, measured when the beat happened, not when the request was sent. The server only uses the intervals between beats, so network delay doesn't matter.

```python
import time, requests

SERVER = "http://localhost:9100"

def on_beat():  # called by the hand tracker
    requests.post(f"{SERVER}/tempo/beat", json={"t": time.monotonic()}, timeout=0.5)
```

Sending requests from a background thread or queue keeps the tracking loop from blocking.

### How the server responds

- Beats are smoothed by a Kalman filter on the beat period (`BeatTracker` in `tempo.py`). A single missed or doubled beat is ignored.
- After 3s without beats, the server holds the current tempo. To return to MRT2's own tempo, send `/tempo/free`.
- The speed moves gradually toward `target / model_bpm` (time constant 1.5s). It is clamped to 0.75–1.3 and can follow at half or double time.
- `model_bpm` needs about 4s of audio first. Until then, requests are accepted but the speed doesn't change.
- Music only plays while a client is connected.

To test without a conducting process, use the conductor page at `http://localhost:9100/conduct` (space bar = one beat).

### Possible extensions

- **Speed from hand movement magnitude:** larger or faster movements speed the music up, smaller ones slow it down. This needs a new mode in `TempoController` (e.g. `POST /tempo/nudge {"amount": ...}`) that changes the target relative to the current `heard_bpm`, rather than an absolute bpm.
