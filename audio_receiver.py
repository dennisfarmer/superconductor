"""sc-audio: plays the music on this machine's sound device.

Connects to the server's GET /stream/playback and plays every block it receives.
Sends Credits once, then ReceivedChunk after each block it finishes, which is
what lets the server generate the next one. One playback connection at a time.

The playback address comes from superconductor_client/addresses.toml, the same
question `make client` asks. `--where NAME` skips it.

    python audio_receiver.py                       # ask
    python audio_receiver.py --where local         # this computer, port 9100
    python audio_receiver.py --where remote        # SSH tunnel, port 9100

    python audio_receiver.py --list-devices
    make audio
"""
import asyncio
import json
import socket
import sys
from collections import deque

import click
import numpy as np

SAMPLE_RATE = 48000
CHANNELS = 2


async def play(url, credits, device):
    import sounddevice as sd
    import websockets

    loop = asyncio.get_running_loop()
    buffer = deque()  # np.ndarray blocks of shape (n, 2)
    played = asyncio.Queue()  # one entry per block the callback finished
    state = {"started": False, "underruns": 0}

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

    stream = sd.OutputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, callback=callback,
                             blocksize=1024, device=device)
    stream.start()
    try:
        while True:
            try:
                async with websockets.connect(url, compression=None, max_size=None) as ws:
                    print(f"connected to {url}")
                    await ws.send(json.dumps({"type": "Credits", "body": {"credits": credits}}))

                    async def acks():
                        while True:
                            await played.get()
                            await ws.send('{"type":"ReceivedChunk","body":null}')

                    sender = asyncio.create_task(acks())
                    try:
                        async for message in ws:
                            if isinstance(message, bytes):  # [uint32 LE seq][float32 LE samples (n, 2)]
                                buffer.append(np.frombuffer(message[4:], dtype="<f4").reshape(-1, CHANNELS))
                                state["started"] = True
                                continue
                            msg = json.loads(message)
                            if msg.get("type") == "Error":
                                print(f"server: {(msg.get('body') or {}).get('error')}")
                    finally:
                        sender.cancel()
            except (OSError, websockets.exceptions.WebSocketException) as e:
                print(f"{url}: {e}; retrying")
            buffer.clear()
            await asyncio.sleep(1.0)
    finally:
        stream.stop()
        stream.close()


@click.command()
@click.option("--where", default=None, metavar="NAME",
              help="Which music server, a name in addresses.toml. Omit to be asked.")
@click.option("--url", default=None,
              help="Playback websocket, instead of --where.")
@click.option("--credits", default=3, show_default=True,
              help="Blocks the server stays ahead of this player.")
@click.option("--device", default=None, help="Output device (name or index; see --list-devices).")
@click.option("--list-devices", is_flag=True, help="List output devices and exit.")
def main(where, url, credits, device, list_devices):
    if list_devices:
        import sounddevice as sd

        devices = sd.query_devices()
        print(sd.DeviceList(d for d in devices if d["max_output_channels"] > 0))
        return
    if url and where:
        raise SystemExit("pass --where or --url, not both")
    if not url:
        from superconductor_client.addresses import AddressError, choose_place, load_place
        if not where:
            where = choose_place(socket.gethostname(), sys.stdin.isatty(), input)
            if where is None:
                raise SystemExit(2)
        try:
            url = load_place(where).browser.playback
        except AddressError as exc:
            raise SystemExit(str(exc)) from exc
    if device is not None and device.isdigit():
        device = int(device)
    try:
        asyncio.run(play(url, credits, device))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
