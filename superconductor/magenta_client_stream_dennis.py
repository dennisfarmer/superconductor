import asyncio
import json
import threading
import uuid

import numpy as np
import sounddevice as sd
import websockets
from collections import deque

#buffer length = 10
#going back to 10 chunks before
#queue: store both state and music
#queue on server side vs on client side

SAMPLE_RATE = 48000
CHANNELS = 2

#########################################
#########################################
#########################################
# Dennis:
# this version does a few things differently to support a websocket-based scheduler implementation:
# - no longer handles style embeddings since having the scheduler makes it so the client doesn't have
#       to manage style/state/etc.. 
# - ==> replaces UpdateControl with UpdateRecipe
# - also recieves the audio id as a UUID that is the first 16 bytes of recieved audio chunks
#       this is used for being able to tell the scheduler what chunk is currently being played
#       at what time offset for chunk generation scheduling purposes
# - currently the time offset info is communicated to the scheduler through the UpdateRecipe message type
#       (seems sufficient because that is likely the only time it is useful to know)

# websocket is what the original server used and also it makes more sense than 
# using an API approach (that I'm more used to) since we're streaming audio
#########################################
#########################################
#########################################

class MagentaClient:

    def __init__(self, uri="ws://localhost:9000/stream"):
        self.uri = uri
        self.ws = None
        self.loop = None
        self.thread = None
        self.connected = False
        self.audio_buffer = deque()
        self.waiting_for_chunk = False

        # Updated by the audio callback on every tick: (chunk_id, offset_seconds)
        # where offset_seconds is how far the playback bar has advanced into
        # the chunk currently at the front of audio_buffer. Read by
        # update_recipe so the scheduler can pick an accurate fork point.
        # None until the first chunk starts playing.
        self._playback_position = None

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

        if not recipe:
            return

        print("updating recipe")

        self._drop_buffer()
        print(f"[AFTER UPDATE] buffer size = {len(self.audio_buffer)}")

        ##############################################################
        ##############################################################
        body = {"recipe": recipe}
        pos = self._playback_position
        if pos is not None:
            body["chunk_id"] = pos[0]
            body["offset_seconds"] = pos[1]

        asyncio.run_coroutine_threadsafe(
            self.ws.send(json.dumps({
                "type": "UpdateRecipe",
                "body": body
            })),
            self.loop
        )
        ##############################################################
        ##############################################################

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

            ##############################################################
            ##############################################################
            await ws.send(json.dumps({
                "type": "UpdateRecipe",
                "body": {"recipe": {"jazz": 1.0}}
            }))
            ##############################################################
            ##############################################################

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

    async def _receive_audio(self):

        while True:

            message = await self.ws.recv()

            if isinstance(message, bytes):

                # Frame layout: [16-byte UUID][float32 LE samples...]
                chunk_id = str(uuid.UUID(bytes=message[:16]))
                audio = np.frombuffer(message[16:], dtype="<f4")
                stereo = audio.reshape(-1, 2)

                self.audio_buffer.append((chunk_id, stereo))
                self.waiting_for_chunk = False
                print(f"[RECEIVE] chunk | buffer size = {len(self.audio_buffer)}")


    def _audio_callback(self, outdata, frames, time, status):

        if len(self.audio_buffer) == 0:
            self._playback_position = None 
            outdata.fill(0)
            return

        chunk_id, chunk = self.audio_buffer[0]  # 👈 peek, DO NOT pop yet

        # how many frames of THIS chunk will actually be played this tick
        played = min(len(chunk), frames)

        # advance the playback-bar tracker for update_recipe / UpdateProgress.
        # if the front chunk id changed since last tick, the old chunk just
        # finished and we're starting fresh at offset 0 for the new one.
        prev = self._playback_position
        if prev is None or prev[0] != chunk_id:
            new_offset = played / SAMPLE_RATE
        else:
            new_offset = prev[1] + played / SAMPLE_RATE
        self._playback_position = (chunk_id, new_offset)

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
            self.audio_buffer[0] = (chunk_id, chunk[frames:])
            #print(f"[PLAY] partial chunk | buffer = {len(self.audio_buffer)}")

        # request logic
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



