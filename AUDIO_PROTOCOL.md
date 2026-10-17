# Audio protocol (`WS /audio`)

The music comes from the MRT2 server, already tempo-stretched, over the client's server connection. The client process (`magenta_remote.py`) plays none of it itself. It forwards every block to whoever listens on client_midi's `WS /audio`:
- `ws://localhost:8470/audio` natively
- `ws://localhost:8080/audio` through the container's gateway

The camera page is one such player and `sc-audio` (`superconductor/audio_receiver.py`) is another. A later receiver (TouchDesigner, a DAW, stem separation) connects the same way, without changes to the server or the client.

## Connecting

```
ws://<client>/audio?role=clock&name=browser
```

- **`role=clock`** (the default): this player sets the pace. It answers `ack` once it has played each block, and only then does the client send the server a `ReceivedChunk`, which lets the server generate the next block. This keeps the server's credits, and its beat grid, tied to what is actually heard.
- **`role=monitor`**: this player gets the same blocks but never acks. Its `ack`s are ignored.
- **`name`**: how the player is shown to the others (for example "sound plays in sc-audio").

There is one clock at a time: the newest connection that asked for `role=clock`. Earlier ones become monitors (the camera page then mutes itself). When the clock disconnects, the next newest clock-asker becomes the clock again. For example, stopping `sc-audio` hands playback back to the page. While no clock is connected nothing is acked, so the server stops once its credits are used up.

## Client → player

- **Audio (binary):** `[uint32 LE sequence number][float32 LE samples]`. The samples are interleaved stereo at 48 kHz, one block per message (`frames_per_block` × 40 ms, 200 ms by default).
- **Role (text):** `{"type": "role", "role": "clock" | "monitor", "clock": "<name of the clock>" | null}`. Sent on connecting and whenever the clock changes. A player that becomes the clock gets the blocks not acked yet sent again, so it should empty its queue when it receives `role: clock`.
- **Pause (text):** `{"type": "pause"}`. The music was paused, and the silence that follows isn't an underrun.

## Player → client (the clock only)

- `{"type": "ack"}`: one block finished playing.
- `{"type": "underrun"}`: the player ran out of audio. It's counted in the client's stats; send it once per stretch of silence.
