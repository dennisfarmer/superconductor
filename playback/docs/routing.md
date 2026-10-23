# Getting the sound where you want it

The plugin connects to the client's `ws://localhost:8470/audio` itself (`../../AUDIO_PROTOCOL.md`), so any program that can load a VST3 (or AU, or LV2) plugin gets the sound directly, with no virtual audio device.

Set the DAW (and TouchDesigner) to **48 kHz**: the stream is 48 kHz, so the plugin then passes it through unchanged. Other rates work too; the plugin converts them.

## Clock or monitor

Exactly one player paces the server: the **clock**. It tells the client each time it has played a block, and that's what lets the server generate the next one. Every other player is a **monitor**: it gets the same audio and plays along, adjusting its speed by a few hundredths of a percent to stay in step.

- The newest player that asked to be the clock is the clock. The camera page asks too, so it goes quiet while the plugin is the clock and plays again when the plugin closes.
- With the plugin in a DAW and in TouchDesigner at the same time, set one to **Clock** and the other to **Monitor**.
- Turn **Send OSC** on in only one of them.

## In a DAW

Put **SC Playback** on an instrument track. It plays as long as the DAW is processing that track, whether or not the transport is running. To record it, record-arm the track, or route it to an audio track.

## In TouchDesigner

**Audio VST CHOP** → Plugin: `~/Library/Audio/Plug-Ins/VST3/SC Playback.vst3` (Mac) or `C:\Program Files\Common Files\VST3\SC Playback.vst3` (Windows). Its output is the stereo stream as CHOP channels. Open the plugin's window from the CHOP to set Clock/Monitor and OSC.

## Without a DAW: the app

`SC Playback.app` (built next to the plugins, or `make playback`) plays the stream on any output device. Choose it under **Options → Audio/MIDI Settings**:

- **Speakers / headphones:** the built-in output.
- **A program that can't load plugins:** a virtual device, which that program records from:
  - Mac: **BlackHole 2ch** (installed on this machine).
  - Windows: **VB-CABLE** (VB-Audio).
  - Linux: none needed. With PipeWire or JACK, connect the app's output to the program's input in qpwgraph or Carla.
