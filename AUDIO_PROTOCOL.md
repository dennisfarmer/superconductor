# Playback (`GET /stream/playback`)

The plugin receives the music directly from the MRT2 server. The client process does not forward audio.

- Local server: `ws://localhost:9100/stream/playback`
- SSH tunnel (`ssh -N -L 9000:lh2300:9100`): `ws://localhost:9000/stream/playback`

This socket is one player at a time. The camera page (Start sound), `make audio` (`audio_receiver.py`), and this plugin each open it. A second connection is closed with "Session already active".

The client process uses the other socket, `GET /stream/input`, for recipe, controls, pattern, and pause:

- Local: `ws://localhost:9100/stream/input` (`make client`)
- SSH tunnel: `ws://localhost:9000/stream/input` (`make client-remote`)

Generation waits until a player is attached, that player has granted a lead, and a recipe has been set. The lead is that many blocks. A new recipe, controls, or pattern applies on the next block. Blocks already sent still play.

## Server → player

- **Ready** (text), once on connect: `{"type": "Ready", "body": {"model", "backend", "sample_rate", "frames_per_block", "host"}}`
- **Stats** (text), about once a second: `{"type": "Stats", "body": {"rtf", "gen_ms_per_frame"}}` (`rtf` > 1 means generating faster than real time)
- **Error** (text): `{"type": "Error", "body": {"error": "..."}}`
- **Audio** (binary), one block per message (`frames_per_block` × 40 ms, 200 ms by default, 48 kHz stereo):

```
[4 bytes: uint32 LE block sequence number][rest: float32 LE samples, interleaved stereo]
```

## Player → server

On connect the plugin sends `{"type": "Credits", "body": {"credits": 3}}`. That sets how many blocks may be in flight. After each block it finishes, it sends `{"type": "ReceivedChunk", "body": null}`, which grants one more block.

`UpdateRecipe`, `UpdateControls`, `Pattern`, `Pause`, `Resume`, and `EndSession` on this socket get an `Error` and change nothing.

When the player disconnects, generation stops and the lead is cleared. The recipe, controls, pattern, and model state stay, and the block sequence keeps climbing. The next player continues from there.
