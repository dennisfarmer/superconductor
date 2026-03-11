import asyncio
import json
import numpy as np
import websockets
import sounddevice as sd

SAMPLE_RATE = 48000
CHANNELS = 2

async def main():
    uri = "ws://localhost:9000/stream"

    async with websockets.connect(uri) as ws:
        print("Connected")

        await ws.send(json.dumps({"type": "StartSession", "body": None}))

        dummy_style = np.random.randn(768).astype(np.float32).tolist()

        await ws.send(json.dumps({
            "type": "UpdateControl",
            "body": {"style": dummy_style}
        }))

        await ws.send(json.dumps({
            "type": "UpdatePlayback",
            "body": {"state": "PLAYING"}
        }))

        print("Streaming... Press Ctrl+C to stop.")

        try:
            while True:
                message = await ws.recv()

                if isinstance(message, bytes):
                    audio = np.frombuffer(message, dtype="<f4")
                    stereo = audio.reshape(-1, 2)

                    # PLAY immediately
                    sd.play(stereo, SAMPLE_RATE, blocking=False)

                    # Ack so server sends next chunk
                    await ws.send(json.dumps({
                        "type": "ReceivedChunk",
                        "body": None
                    }))

        except KeyboardInterrupt:
            print("Stopping...")

            await ws.send(json.dumps({
                "type": "UpdatePlayback",
                "body": {"state": "STOPPED"}
            }))

            await ws.send(json.dumps({
                "type": "EndSession",
                "body": None
            }))

asyncio.run(main())