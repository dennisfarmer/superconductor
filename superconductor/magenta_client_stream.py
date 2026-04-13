import asyncio
import json
import threading

import numpy as np
import requests
import sounddevice as sd
import websockets
from collections import deque

#buffer length = 10
#going back to 10 chunks before
#queue: store both state and music
#queue on server side vs on client side

SAMPLE_RATE = 48000
CHANNELS = 2

# TODO: 
# try just dropping everything in the queue see if there's inconsistency


# when i call update_recipe, client finish playing the current chunk AND the next chunk, then drop everything after that 
# when i call update_recipe, scheduler rewind the internal state of the model back to the next next chunk and keeps generating
# what if new ones come in but it;s not the updated version
# do i need to store ID so that it's easier? if so what ID do we use? uuid?


def get_style_embedding(text):
    r = requests.post("http://localhost:9000/style", data=text)
    return r.json()


class MagentaClient:

    def __init__(self, uri="ws://localhost:9000/stream"):
        self.uri = uri
        self.ws = None
        self.loop = None
        self.thread = None
        self.connected = False
        self.audio_buffer = deque()
        self.waiting_for_chunk = False

        # cache embeddings so we don't repeatedly call /style
        self.embedding_cache = {}

    # ---------- public API ----------

    def start(self):
        """Start websocket client in background thread"""
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()

    def update_recipe(self, recipe):
        """
        recipe example:
        {"Rock":0.6,"Guitar":0.8,"Jazz":0.3}
        """

        if not self.connected:
            print("MagentaClient not connected yet")
            return

        embedding = self._recipe_to_embedding(recipe)

        if embedding is None:
            return

        print("updating recipe")

        self._drop_buffer()
        print(f"[AFTER UPDATE] buffer size = {len(self.audio_buffer)}")

        asyncio.run_coroutine_threadsafe(
            self.ws.send(json.dumps({
                "type": "UpdateControl",
                "body": {
                    "style": embedding.tolist()
                }
            })),
            self.loop
        )

    # ---------- internal ----------

    def _run_loop(self):
        """Create asyncio loop in thread"""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self._connect())

    async def _connect(self):

        async with websockets.connect(self.uri) as ws:

            self.ws = ws
            self.connected = True

            print("Connected to MagentaRT")

            await ws.send(json.dumps({
                "type": "StartSession",
                "body": None
            }))

            style_embedding = get_style_embedding("jazz")

            await ws.send(json.dumps({
                "type": "UpdateControl",
                "body": {"style": style_embedding}
            }))

            await ws.send(json.dumps({
                "type": "UpdatePlayback",
                "body": {"state": "PLAYING"}
            }))
            
            self.stream = sd.OutputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                callback=self._audio_callback,
                blocksize=1024
            )

            self.stream.start()

            await self._receive_audio()

    def _get_cached_embedding(self, prompt):

        if prompt not in self.embedding_cache:
            emb = np.array(get_style_embedding(prompt), dtype=np.float32)
            self.embedding_cache[prompt] = emb

        return self.embedding_cache[prompt]

    async def stop(self):
        print("Stopping Magenta client...")

        if self.ws is None:
            return

        try:
            if not self.ws.closed:
                await self.ws.send(json.dumps({
                    "type": "EndSession"
                }))
        except Exception as e:
            print("Send EndSession error:", e)

        try:
            if not self.ws.closed:
                await self.ws.close()
        except Exception as e:
            print("Close WS error:", e)

        self.ws = None
        self.connected = False
        
    def _recipe_to_embedding(self, recipe):

        embeddings = []
        weights = []

        for prompt, weight in recipe.items():

            if weight <= 0:
                continue

            emb = self._get_cached_embedding(prompt)

            embeddings.append(emb)
            weights.append(weight)

        if not embeddings:
            return None

        embeddings = np.stack(embeddings)
        weights = np.array(weights, dtype=np.float32)

        weights = weights / weights.sum()

        final_embedding = np.sum(embeddings * weights[:, None], axis=0)

        return final_embedding

    async def _receive_audio(self):

        while True:

            message = await self.ws.recv()

            if isinstance(message, bytes):

                audio = np.frombuffer(message, dtype="<f4")
                stereo = audio.reshape(-1, 2)

                self.audio_buffer.append(stereo)
                self.waiting_for_chunk = False
                print(f"[RECEIVE] chunk | buffer size = {len(self.audio_buffer)}")
    

    def _audio_callback(self, outdata, frames, time, status):

        if len(self.audio_buffer) == 0:
            outdata.fill(0)
            return

        chunk = self.audio_buffer[0]  # 👈 peek, DO NOT pop yet

        if len(chunk) <= frames:
            # use entire chunk
            outdata[:len(chunk)] = chunk
            outdata[len(chunk):].fill(0)

            self.audio_buffer.popleft()  # 👈 NOW remove it
            #print(f"[PLAY] consumed full chunk | buffer = {len(self.audio_buffer)}")

        else:
            # use only part of chunk
            outdata[:] = chunk[:frames]

            # keep the remaining part
            self.audio_buffer[0] = chunk[frames:]
            #print(f"[PLAY] partial chunk | buffer = {len(self.audio_buffer)}")

        # request logic (unchanged, just moved here cleanly)
        if len(self.audio_buffer) < 10 and not self.waiting_for_chunk:
            print(f"[REQUEST] buffer low ({len(self.audio_buffer)}) → requesting more")
            asyncio.run_coroutine_threadsafe(
                self.ws.send(json.dumps({
                    "type": "ReceivedChunk",
                    "body": None
                })),
                self.loop
            )
            self.waiting_for_chunk = True
    
    def _drop_buffer(self):
        # keep current + next chunk for smoothness
        keep = 2

        while len(self.audio_buffer) > keep:
            self.audio_buffer.pop()

        print(f"[DROP] buffer trimmed to {len(self.audio_buffer)}")



