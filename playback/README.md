# SuperConductor Playback VST3 Plugin

A second way to hear SuperConductor, next to the camera page: a plugin that plays the music stream inside a DAW or TouchDesigner, and the same thing as a small app. The camera page keeps working as before.

```
MRT2 server :9100 ─► client ─► ws://localhost:8470/audio ─┬─► camera page (unchanged)
                                                           ├─► SuperConductor Playback VST3 Plugin in a DAW
                                                           ├─► SuperConductor Playback VST3 Plugin in TouchDesigner (Audio VST CHOP)
                                                           └─► SuperConductor Playback VST3 Plugin app ─► speakers / BlackHole
                                    one copy also sends OSC ─► TouchDesigner: loudness, bands, hits, tempo
```

## Build

Needs CMake 3.22+ and a C++20 compiler (Xcode on Mac, Visual Studio 2022 on Windows, GCC/Clang on Linux). JUCE and IXWebSocket are downloaded on the first build.

```bash
make playback-build   # from the repo root: plugins + app, Release
make playback         # build if needed, then open the app
```

The plugins are copied to your plugin folders after each build (Mac: `~/Library/Audio/Plug-Ins/VST3` and `Components`). Builds: VST3 everywhere, AU on Mac, LV2 on Linux, plus the Standalone app.

## Use

1. Start the server and client as usual (`make server`, `make client` in client-web).
2. Load **SuperConductor Playback VST3 Plugin** in a DAW, in TouchDesigner, or open the app. It connects to `ws://localhost:8470/audio`.
3. Pick **Clock** (this copy sets the pace; the camera page goes quiet) or **Monitor** (plays along). Turn on **Send OSC** in one copy.

Settings are saved with the DAW or TouchDesigner project. Details: `docs/routing.md` (where the sound goes, per OS) and `docs/osc.md` (every OSC message).

## Folders

| Folder | What |
|---|---|
| `shared/` | Everything except the plugin wrapper, used by every format |
| `shared/StreamClient` | The connection to `/audio`: clock / monitor, one ack per block played, underruns, reconnecting |
| `shared/AudioQueue` | The received audio, passed from the network thread to the audio thread without locks |
| `shared/Resampler` | 48 kHz stream → the host's rate (a pass-through at 48 kHz), plus a monitor's small speed corrections |
| `shared/Player` | Ties those together. Starts, stops, and keeps a monitor in step |
| `shared/Features` | Loudness, frequency bands, hits, measured on the samples as they're output |
| `shared/TempoPoller` | Tempo and beat from the MRT2 server's `/tempo/status` |
| `shared/OscSender` | Sends the features and tempo, at 15–120 bundles a second |
| `plugin/` | The JUCE plugin and its settings window. Builds VST3 / AU / LV2 / Standalone |
| `docs/` | Routing and OSC reference |

The audio thread only copies from the queue; the network, acks, analysis and OSC all run on their own threads.
