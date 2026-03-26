import asyncio
import queue
import struct
import threading
import time

import numpy as np
import pyaudio
import requests
import websockets
import time 
import json

SERVER_URL = "ws://localhost:9000/stream"
INFO_URL = "http://localhost:9000/stream_info"
STYLE_URL = "http://localhost:9000/style"

def get_stream_info():
    resp = requests.get(INFO_URL)
    resp.raise_for_status()
    return resp.json()

def get_style_embedding(prompt: str) -> list:
    resp = requests.post(STYLE_URL, data=prompt, headers={"Content-Type": "text/plain"})
    resp.raise_for_status()
    return resp.json()

def playback_worker(audio_queue: queue.Queue, sample_rate: int, num_channels: int):
    time.sleep(10)
    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paFloat32,
        channels=num_channels,
        rate=sample_rate,
        output=True,
    )
    print("[Playback] Audio stream opened.")

    while True:
        chunk = audio_queue.get()
        if chunk is None: 
            print("[Playback] Received stop signal.")
            break
        stream.write(chunk.tobytes())
        audio_queue.task_done()

    stream.stop_stream()
    stream.close()
    p.terminate()
    print("[Playback] Audio stream closed.")


async def receive_chunks(audio_queue: queue.Queue, sample_rate: int, num_channels: int, style_prompt: str):
    style_embedding = get_style_embedding(style_prompt)
    print(f"[WS] Got style embedding, length={len(style_embedding)}")

    async with websockets.connect(SERVER_URL) as ws:
        await ws.send('{"type": "StartSession", "body": {}}')
        print("[WS] Sent StartSession")

        control_msg = json.dumps({
            "type": "UpdateControl",
            "body": {
                "style": style_embedding,
                "generation_kwargs": {
                    "temperature": 1.0,
                }
            }
        })
        await ws.send(control_msg)
        print("[WS] Sent UpdateControl with style embedding")

        await ws.send('{"type": "UpdatePlayback", "body": {"state": "PLAYING"}}')
        print("[WS] Sent UpdatePlayback PLAYING")

        chunk_count = 0
        try:
            async for message in ws:
                if isinstance(message, bytes):
                    num_floats = len(message) // 4
                    samples = np.array(struct.unpack(f"<{num_floats}f", message), dtype=np.float32)

                    if num_channels > 1:
                        samples = samples.reshape(-1, num_channels)

                    audio_queue.put(samples)
                    chunk_count += 1
                    print(f"[WS] Received chunk #{chunk_count}, samples={num_floats}")
                    await ws.send('{"type": "ReceivedChunk", "body": {}}')

        except websockets.ConnectionClosed:
            print("[WS] Connection closed.")

        finally:
            try:
                await ws.send('{"type": "EndSession", "body": {}}')
                print("[WS] Sent EndSession")
            except Exception:
                pass

    audio_queue.put(None)


def main():
    style_prompt = "upbeat jazzy piano"  

    info = get_stream_info()
    sample_rate = info["sample_rate"]
    num_channels = info["num_channels"]
    chunk_length = info["chunk_length"]
    print(f"[Main] Stream info: sample_rate={sample_rate}, channels={num_channels}, chunk_length={chunk_length}s")

    audio_queue = queue.Queue(maxsize=10)  

    player_thread = threading.Thread(
        target=playback_worker,
        args=(audio_queue, sample_rate, num_channels),
        daemon=True,
    )
    player_thread.start()
    asyncio.run(receive_chunks(audio_queue, sample_rate, num_channels, style_prompt))
    audio_queue.join()
    audio_queue.put(None)  
    player_thread.join()
    print("[Main] Done.")


if __name__ == "__main__":
    main()