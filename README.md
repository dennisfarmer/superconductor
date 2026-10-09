# SuperConductor MIDI Player

Plays a MIDI file through SuperConductor's music model. The player turns the file's notes into the model's NOTES / DRUMS input (40 ms frames) and sends them to the running `superconductor_client`, which forwards them to the MRT2 server. The client's prompts (objects, combos, base prompt) still decide what it sounds like; the file decides which pitches play and when. The message format is described in `superconductor_client/MIDI_MSG_PROTOCOL.md`.

```
midi_player.py ──POST /pattern──▶ superconductor_client (client_midi, :8470) ──▶ mrt2_server.py
```

### Startup

1. In `superconductor_client`: `make server` and `make client` (or `client-remote`). client_midi listens on port 8470 (`midi_port` in `collab.toml`).
2. Here: `make player`, then open http://localhost:8475.

Needs `aiohttp` and `mido` (`pip install mido` in sc_env).

### The page

- **Songs:** drop a `.mid` file onto the page or click to upload it. Uploads are kept in `songs/` (not committed), so they're there next time. `make player ARGS="path/to/song.mid"` adds files from the command line.
- **Play / Stop**, **⏮** (from the beginning), and click the bar to play from that point.
- **Tempo:** in bpm. The file's first tempo is the reference (e.g. 120 bpm → 90 bpm plays at 0.75×); a file's own tempo changes are kept, scaled by the same factor. **file tempo** goes back to the written tempo.
- **Other notes:** *silent* plays the file as written (every pitch the file doesn't play is off); *free* lets the model add its own notes around the file's.
- **Loop** and **Follow notes** (`cfg_notes`: how strongly the music follows the notes, -1 to 7).

A song is sent as one play-once Pattern from the current position. Changing the tempo, mode, loop or position while playing re-sends the rest of the song from where it is. Stop clears the lanes the player set. General MIDI drums (channel 10) go to the drum lane. A song without drums leaves the drum lane alone, so another program can own it.

### Timing

- What you hear lags the page's bar by the client's lead (about 0.6 s locally, more over the cluster tunnel).
- The bar is a clock started when the song was sent. If the music is paused on the objects page, the server holds the song's position, but the bar keeps moving. Press Play again (it re-sends from the bar's position) after resuming.
- Keep the server's tempo control off (`POST /tempo/free`, the default): it would time-stretch the audio on top of the player's tempo.

### HTTP API

| Request | Body | Reply |
|---|---|---|
| `GET /api/state` | | songs, the selected song, `playing`, `position_s`, `speed`, settings, client status |
| `POST /api/upload` | multipart form, field `file` | the song's info; `400` if mido can't read it |
| `POST /api/play` | `{song?, start_s?, bpm?, mode?, loop?, cfg_notes?}` | state; `502` if client_midi refused it or isn't running |
| `POST /api/stop` | | state |
| `POST /api/settings` | `{bpm?, mode?, loop?, cfg_notes?}` | state (re-sent if playing) |
| `DELETE /api/songs/{name}` | | state |

`mode` is `"written"` or `"accompany"`, and `bpm: null` means the file's tempo. Other flags: `python midi_player.py --help` (`--port`, `--client_url`, `--songs_dir`).
