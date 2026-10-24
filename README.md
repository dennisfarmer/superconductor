# SuperConductor client-web

```
audio_receiver.py          optional speaker playback of WS /audio (make audio)
superconductor_client/     one installable package
  client.py                sends messages to the music server, on a clock
  tracking/                the camera page and object tracking
    library/               saved objects
    var/                   the browser detector and other generated files
  conducting/              MediaPipe conducting
  describe/                the describe server
  midi/                    the MIDI player
client_container/          the Singularity image (Docker builds it)
docs/                      how to run it, and the protocols
```

`make client` starts the client together with tracking, conducting, describe and midi. Each one also has its own command: `sc-client`, `sc-tracking`, `sc-conduct`, `sc-describe`, `sc-midi`.

The MRT2 server, Atlas, playback and TouchDesigner still live in their own checkouts. `make worktrees` creates those.

Start here: [docs/README.md](docs/README.md). The container: [client_container/README.md](client_container/README.md).
