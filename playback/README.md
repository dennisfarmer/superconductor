# SuperConductor Playback VST3 Plugin

A plugin that plays the server's music stream inside a DAW or TouchDesigner, and the same thing as a small app. It is the player: it connects to the server's playback websocket itself.

```
MRT2 server
  ws://localhost:9000/stream/playback   local (`make server`)
  ws://localhost:9000/stream/playback   SSH tunnel (`ssh -N -L 9000:lh2300:9000`)
        ├─► camera page, Start sound
        ├─► SuperConductor Playback VST3 Plugin in a DAW
        ├─► SuperConductor Playback VST3 Plugin in TouchDesigner (Audio VST CHOP)
        └─► SuperConductor Playback VST3 Plugin app ─► speakers / BlackHole
                                    one copy also sends OSC ─► TouchDesigner: loudness, bands, hits, tempo
```

One of those connects at a time. A second playback socket is closed.

## Build

Needs CMake 3.22+ and a C++20 compiler (Xcode on Mac, Visual Studio 2022 on Windows, GCC/Clang on Linux). JUCE and IXWebSocket are downloaded on the first build.

```bash
make playback-build    # from the repo root: plugins + app, Release
make playback          # build if needed, ask where the music server is, open the app
make playback-local    # server on this computer (port 9000)
make playback-remote   # server through the SSH tunnel (port 9000)
make playback-cluster  # server through the container gateway
```

`make playback` asks the same question as `make client` and `make audio`. The app does not talk to the client, so it is not asked where the client is. The addresses are `addresses.toml`. `playback-cluster` uses `ws://server.localhost:8080/stream/playback`.

The plugins are copied to your plugin folders after each build (Mac: `~/Library/Audio/Plug-Ins/VST3` and `Components`). Builds: VST3 everywhere, AU on Mac, LV2 on Linux, plus the Standalone app.

## Use

1. Start the server and the client (`make server`, `make client`). The client opens `ws://localhost:9000/stream/input` and does not play. Through the SSH tunnel, `make client-remote` opens `ws://localhost:9000/stream/input`.
2. Load **SuperConductor Playback VST3 Plugin** in a DAW or in TouchDesigner, or open the app with `make playback`. The app uses the address you picked. A plugin inside a DAW still opens `ws://localhost:9000/stream/playback` until you change its stream field; `make playback-remote` and `make playback-cluster` only affect the app they launch.
3. Turn on **Send OSC** in one copy.

The page's Start sound and `make audio` open the same playback socket. Close the other player first, or the server rejects this one.

Settings are saved with the DAW or TouchDesigner project. Details: `docs/routing.md` (where the sound goes, per OS) and `docs/osc.md` (every OSC message).

## Folders

| Folder | What |
|---|---|
| `shared/` | Everything except the plugin wrapper, used by every format |
| `shared/StreamClient` | The connection to `/stream/playback`: Credits once, ReceivedChunk per block played, reconnecting |
| `shared/AudioQueue` | The received audio, passed from the network thread to the audio thread without locks |
| `shared/Resampler` | 48 kHz stream → the host's rate (a pass-through at 48 kHz) |
| `shared/Player` | Ties those together. Starts and stops with the stream |
| `shared/Features` | Loudness, frequency bands, hits, measured on the samples as they're output |
| `shared/TempoPoller` | Tempo and beat from the MRT2 server's `/tempo/status` |
| `shared/OscSender` | Sends the features and tempo, at 15–120 bundles a second |
| `plugin/` | The JUCE plugin and its settings window. Builds VST3 / AU / LV2 / Standalone |
| `docs/` | Routing and OSC reference |

The audio thread only copies from the queue; the network, ReceivedChunk, analysis and OSC all run on their own threads.
