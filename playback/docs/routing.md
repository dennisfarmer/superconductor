# Getting the sound where you want it

The plugin connects to the server's playback websocket (`../../AUDIO_PROTOCOL.md`).

- Local server: `ws://localhost:9000/stream/playback`
- SSH tunnel (`ssh -N -L 9000:lh2300:9000`): `ws://localhost:9000/stream/playback`

One player at a time. The camera page's Start sound, `make audio`, and this plugin each open that socket. A second connection is closed with "Session already active".

Set the DAW (and TouchDesigner) to **48 kHz**: the stream is 48 kHz, so the plugin then passes it through unchanged. Other rates work too; the plugin converts them.

## The player

This copy is the player. On connect it sends `Credits` (a lead of 3 blocks). After each block it finishes, it sends `ReceivedChunk`, which lets the server generate the next one. There is no separate monitor connection: the server sends the audio to this socket only.

Only one copy should send OSC. The plugin sends it by default inside TouchDesigner, where it's used, and not elsewhere; change it with **Send OSC** in the plugin's window.

## In a DAW

Put **SuperConductor Playback VST3 Plugin** on an instrument track. It plays as long as the DAW is processing that track, whether or not the transport is running. To record it, record-arm the track, or route it to an audio track.

## In TouchDesigner

**Audio VST CHOP** → Plugin: `~/Library/Audio/Plug-Ins/VST3/SuperConductor Playback VST3 Plugin.vst3` (Mac) or `C:\Program Files\Common Files\VST3\SuperConductor Playback VST3 Plugin.vst3` (Windows). Its output is the stereo stream as CHOP channels. Open the plugin's window from the CHOP to set the stream URL and OSC.

## Without a DAW: the app

`SuperConductor Playback VST3 Plugin.app` (built next to the plugins, or `make playback`) plays the stream on any output device. Choose it under **Options → Audio/MIDI Settings**:

- **Speakers / headphones:** the built-in output.
- **A program that can't load plugins:** a virtual device, which that program records from:
  - Mac: **BlackHole 2ch** (installed on this machine).
  - Windows: **VB-CABLE** (VB-Audio).
  - Linux: none needed. With PipeWire or JACK, connect the app's output to the program's input in qpwgraph or Carla.
