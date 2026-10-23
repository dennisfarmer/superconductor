# OSC messages

One copy of the SC Playback plugin (or app) sends these, with **Send OSC** on. They go to the OSC host and port set in the plugin (default `127.0.0.1:7000`), as one bundle, **Per second** times a second (15, 30, 60 or 120; default 60).

The sound measurements come from the samples the plugin is outputting at that moment, so they follow what is heard. The tempo comes from the MRT2 server's `GET /tempo/status`, read 5 times a second, with the beat moved forward smoothly in between.

## Sound

| Address | Type | Meaning |
|---|---|---|
| `/sc/level` | float 0–1 | Loudness over the last ~21 ms (-80 dB → 0, 0 dB → 1) |
| `/sc/rms_db` | float | The same, in dBFS (-80 … 0) |
| `/sc/peak` | float 0–1 | Highest sample over the last ~21 ms |
| `/sc/bands` | N floats 0–1 | Frequency bands, low to high, spaced evenly in pitch from 40 Hz to 16 kHz. N is **Bands** (4, 8, 16 or 32) |
| `/sc/flux` | float | How much the spectrum just grew (spectral flux) |
| `/sc/onset` | float | The flux on the update where a hit was detected, otherwise 0. At most one hit every 80 ms |

## Tempo

| Address | Type | Meaning |
|---|---|---|
| `/sc/tempo/reachable` | int 0/1 | The MRT2 server answered the last poll |
| `/sc/tempo/playing` | int 0/1 | Music is being heard |
| `/sc/tempo/bpm` | float | The tempo being heard (0 if unknown) |
| `/sc/tempo/beat_known` | int 0/1 | The server has found the beat. While it is 0, beat and beat_frac stay at 0 |
| `/sc/tempo/beat` | int | Beat within the bar, 0 = beat one |
| `/sc/tempo/beat_frac` | float 0–1 | How far through that beat |
| `/sc/tempo/beats_per_bar` | int | 3 or 4 (4 while the server isn't sure) |
| `/sc/tempo/bar_phase` | float 0–1 | How far through the bar: (beat + beat_frac) / beats_per_bar |

## In TouchDesigner

An **OSC In CHOP** on port 7000 makes one channel per address (`sc/level`, `sc/tempo/bar_phase`, …). `/sc/bands` carries several values, so it becomes one channel per band.
