"""sc-audio: plays the music on this machine's sound device.

Connects to the client's WS /audio (client_midi, see AUDIO_PROTOCOL.md) and plays
every block it receives. As the clock (--role clock, the default) it acknowledges
each block once it has played, which is what lets the server generate the next one;
the camera page then goes quiet. As a monitor (--role monitor) it only plays along.
Stopping it hands playback back to the camera page.

It runs where the speakers are: with the client on lh2300, that's the laptop,
through the SSH tunnel to the gateway (--url ws://localhost:8080/audio).

    sc-audio [--url ws://localhost:8470/audio] [--role clock|monitor] [--device NAME]
    sc-audio --list-devices
"""
import asyncio
import json
from collections import deque

import click
import numpy as np

SAMPLE_RATE = 48000
CHANNELS = 2


async def play(url, role, device):
    import sounddevice as sd
    import websockets

    loop = asyncio.get_running_loop()
    buffer = deque()  # np.ndarray blocks of shape (n, 2)
    played = asyncio.Queue()  # one entry per block the callback finished
    state = {"started": False, "underruns": 0, "role": None}

    def callback(outdata, frames, time_info, status):
        filled = 0
        while filled < frames and buffer:
            block = buffer[0]
            n = min(len(block), frames - filled)
            outdata[filled:filled + n] = block[:n]
            filled += n
            if n == len(block):
                buffer.popleft()
                loop.call_soon_threadsafe(played.put_nowait, None)
            else:
                buffer[0] = block[n:]
        if filled < frames:
            outdata[filled:].fill(0)
            if state["started"]:  # don't count the silence before the first block
                state["underruns"] += 1
                state["started"] = False
                loop.call_soon_threadsafe(played.put_nowait, "underrun")

    stream = sd.OutputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, callback=callback,
                             blocksize=1024, device=device)
    stream.start()
    sep = "&" if "?" in url else "?"
    target = f"{url}{sep}role={role}&name=sc-audio"
    try:
        while True:
            try:
                async with websockets.connect(target, compression=None, max_size=None) as ws:
                    print(f"connected to {url} as {role}")

                    async def acks():
                        while True:
                            event = await played.get()
                            if state["role"] != "clock":
                                continue
                            await ws.send('{"type": "underrun"}' if event == "underrun" else '{"type": "ack"}')

                    sender = asyncio.create_task(acks())
                    try:
                        async for message in ws:
                            if isinstance(message, bytes):  # [uint32 LE seq][float32 LE samples (n, 2)]
                                buffer.append(np.frombuffer(message[4:], dtype="<f4").reshape(-1, CHANNELS))
                                state["started"] = True
                                continue
                            msg = json.loads(message)
                            if msg.get("type") == "role":
                                if msg["role"] == "clock" and state["role"] != "clock":
                                    # the blocks not acked yet come again: start from an empty queue
                                    buffer.clear()
                                    while not played.empty():
                                        played.get_nowait()
                                if msg["role"] != state["role"]:
                                    print("playing as the clock" if msg["role"] == "clock" else
                                          f"monitoring (the clock is {msg.get('clock') or 'nobody'})")
                                state["role"] = msg["role"]
                            elif msg.get("type") == "pause":
                                state["started"] = False  # the silence while paused isn't an underrun
                    finally:
                        sender.cancel()
            except (OSError, websockets.exceptions.WebSocketException) as e:
                print(f"{url}: {e}; retrying")
            state["role"] = None
            buffer.clear()
            await asyncio.sleep(1.0)
    finally:
        stream.stop()
        stream.close()


@click.command()
@click.option("--url", default="ws://localhost:8470/audio",
              help="The client's WS /audio (through the gateway: ws://localhost:8080/audio).")
@click.option("--role", default="clock", type=click.Choice(["clock", "monitor"]),
              help="clock: play and pace the server (the camera page goes quiet); monitor: only play along.")
@click.option("--device", default=None, help="Output device (name or index; see --list-devices).")
@click.option("--list-devices", is_flag=True, help="List the sound devices and exit.")
def main(url, role, device, list_devices):
    if list_devices:
        import sounddevice as sd

        print(sd.query_devices())
        return
    if device is not None and device.isdigit():
        device = int(device)
    try:
        asyncio.run(play(url, role, device))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
