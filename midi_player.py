"""SuperConductor MIDI player: plays a MIDI file through SuperConductor's music model.

It reads a .mid file, turns its notes into a NOTES / DRUMS Pattern (40 ms frames,
see superconductor_client/MIDI_MSG_PROTOCOL.md) and POSTs it to the running
superconductor_client (client_midi, http://localhost:8470/pattern), which forwards
it to the MRT2 server. The client's prompts still decide what it sounds like; the
file decides which pitches play and when.

A song is sent as one play-once Pattern from the current position. The tempo is
applied here, by scaling the file's times, so a tempo change (or a seek) while
playing re-sends the rest of the song from the current position. Stop clears the
lanes this player set.

Open http://localhost:8475 to upload .mid files and play them. Uploads are kept
in --songs_dir.

HTTP API (used by the page):
  GET    /api/state            songs, what is playing, position, client status
  POST   /api/upload           multipart form, field "file": a .mid file
  POST   /api/play             {song, start_s?, bpm?, mode?, loop?, cfg_notes?}
  POST   /api/stop             null
  POST   /api/settings         {bpm?, mode?, loop?, cfg_notes?}  applied now if playing
  DELETE /api/songs/{name}
"""
import argparse
import asyncio
import json
import re
import shutil
import time
from pathlib import Path

import aiohttp
import mido
from aiohttp import web

FPS = 25  # MRT2 frames per second (40 ms each)
PAGE = Path(__file__).parent / "player.html"
DRUM_CHANNEL = 9  # General MIDI channel 10
# notes lane `default`: what the pitches the file doesn't play do
MODES = {"written": 0,  # only the file's notes sound
         "accompany": -1}  # the file's notes sound, and the model may add its own
MIN_SPEED, MAX_SPEED = 0.25, 4.0


class Song:
    """A MIDI file as notes and drum hits in seconds (the file's own tempo map applied)."""

    def __init__(self, path):
        self.path = Path(path)
        self.name = self.path.name
        midi = mido.MidiFile(self.path)
        self.notes = []  # (start_s, end_s, pitch)
        self.hits = []  # drum hit times (s)
        self.bpm = None  # first tempo in the file
        self.time_signature = None
        active = {}  # (channel, pitch) -> start_s
        t = 0.0
        for msg in midi:  # iterating applies the tempo map; msg.time is in seconds
            t += msg.time
            if msg.type == "set_tempo" and self.bpm is None:
                self.bpm = mido.tempo2bpm(msg.tempo)
            elif msg.type == "time_signature" and self.time_signature is None:
                self.time_signature = f"{msg.numerator}/{msg.denominator}"
            if msg.type not in ("note_on", "note_off"):
                continue
            on = msg.type == "note_on" and msg.velocity > 0
            if msg.channel == DRUM_CHANNEL:
                if on:
                    self.hits.append(t)
                continue
            key = (msg.channel, msg.note)
            if key in active:
                self.notes.append((active.pop(key), t, msg.note))  # note off, or re-struck while held
            if on:
                active[key] = t
        self.notes += [(start, t, pitch) for (_, pitch), start in active.items()]  # never released
        self.notes.sort()
        self.bpm = self.bpm or 120.0  # the MIDI default
        self.length_s = max([e for _, e, _ in self.notes] + [h + 1 / FPS for h in self.hits] + [0.0])

    def info(self):
        pitches = [p for _, _, p in self.notes]
        return {"name": self.name, "length_s": round(self.length_s, 2), "bpm": round(self.bpm, 2),
                "time_signature": self.time_signature, "notes": len(self.notes), "drum_hits": len(self.hits),
                "range": [min(pitches), max(pitches)] if pitches else None}

    def pattern(self, start_s=0.0, speed=1.0, default=0, loop=False):
        """The song from `start_s` (file seconds) at `speed` x its tempo, as a Pattern.

        Frame f covers [f, f + 1) x 40 ms of playback, i.e. file time
        start_s + [f, f + 1) x 0.04 x speed. A note sounding at start_s is struck again
        at frame 0. With `loop`, the lanes repeat from start_s to the end."""
        def frame(t):
            return round((t - start_s) / speed * FPS)

        end = max(1, frame(self.length_s))
        notes = [[default] * 128 for _ in range(end)]
        for start, stop, pitch in self.notes:
            if stop <= start_s and start < start_s:
                continue
            a = max(0, frame(start))
            if a >= end:
                continue
            for f in range(a, min(end, max(a + 1, frame(stop)))):
                if notes[f][pitch] != 2:  # an attack in this frame wins over a held note
                    notes[f][pitch] = 2 if f == a else 1

        body = {"id": f"superconductor_midi {self.name} @{start_s:.1f}s x{speed:.2f}",
                "notes": {"loop": loop, "steps": _note_steps(notes, default)}}
        hits = [frame(h) for h in self.hits if h >= start_s]
        if hits:
            drums = [0] * end
            for h in hits:
                if h < end:
                    drums[h] = 1
            body["drums"] = {"loop": loop, "steps": _drum_steps(drums)}
        return body


def _note_steps(frames, default):
    """Run-length encode per-frame NOTES values: a step is its first frame, then that
    frame with onsets (2) as held (1) for as long as nothing changes."""
    steps, i, end = [], 0, len(frames)
    while i < end:
        later = [1 if v == 2 else v for v in frames[i]]
        j = i + 1
        while j < end and frames[j] == later:
            j += 1
        steps.append({"frames": j - i, "default": default,
                      "pitches": {str(p): v for p, v in enumerate(frames[i]) if v != default}})
        i = j
    return steps


def _drum_steps(drums):
    """Run-length encode per-frame DRUMS values as {first, rest} steps."""
    steps, i, end = [], 0, len(drums)
    while i < end:
        rest = drums[i + 1] if i + 1 < end else 0
        j = i + 1
        while j < end and drums[j] == rest:
            j += 1
        steps.append({"frames": j - i, "first": drums[i], "rest": rest})
        i = j
    return steps


class Player:
    """Song library, playback position and the connection to client_midi."""

    def __init__(self, songs_dir, client_url):
        self.songs_dir = Path(songs_dir)
        self.songs_dir.mkdir(parents=True, exist_ok=True)
        self.client_url = client_url.rstrip("/")
        self.songs = {}  # name -> Song
        self.errors = {}  # name -> why the file can't be read
        for path in sorted(self.songs_dir.glob("*.mid")) + sorted(self.songs_dir.glob("*.midi")):
            self._load(path)
        self.settings = {"bpm": None, "mode": "written", "loop": False, "cfg_notes": 3.0}
        self.song = None  # selected Song
        self.playing = False
        self._start_s = 0.0  # file position at `_since`
        self._since = 0.0  # time.time() the current Pattern was sent
        self._lanes = set()  # lanes this player set, cleared on stop
        self.client = {"connected": False}
        self.message = ""

    def _load(self, path):
        try:
            song = Song(path)
        except Exception as e:  # pylint: disable=broad-exception-caught
            self.errors[path.name] = str(e)
            return None
        self.songs[song.name] = song
        self.errors.pop(song.name, None)
        return song

    def add(self, src):
        """copy a .mid file into the library"""
        src = Path(src)
        dest = self.songs_dir / src.name
        if src.resolve() != dest.resolve():
            shutil.copyfile(src, dest)
        return self._load(dest)

    def speed(self):
        if self.song is None or not self.settings["bpm"]:
            return 1.0
        return min(MAX_SPEED, max(MIN_SPEED, self.settings["bpm"] / self.song.bpm))

    def position(self):
        """file seconds being generated now (heard ~credits x 200 ms later)"""
        if self.song is None:
            return 0.0
        if not self.playing:
            return self._start_s
        pos = self._start_s + (time.time() - self._since) * self.speed()
        if pos >= self.song.length_s:
            if self.settings["loop"]:
                span = self.song.length_s - self._start_s
                return self._start_s + (pos - self._start_s) % span if span > 0 else self._start_s
            self.playing, self._start_s = False, 0.0  # finished: the lanes went back to their fallback
            return 0.0
        return pos

    async def _post(self, body):
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
            async with session.post(f"{self.client_url}/pattern", json=body) as r:
                reply = await r.text()
                if r.status != 200:
                    raise RuntimeError(f"client_midi {r.status}: {reply.strip()}")

    async def play(self, start_s=0.0):
        song = self.song
        start_s = min(max(0.0, start_s), max(0.0, song.length_s - 1 / FPS))
        body = song.pattern(start_s, self.speed(), MODES[self.settings["mode"]], self.settings["loop"])
        body["cfg_notes"] = self.settings["cfg_notes"]
        # a lane this player set before but the song doesn't use (drums) is handed back
        for lane in self._lanes - body.keys():
            body[lane] = None
        await self._post(body)
        self._lanes = {k for k in ("notes", "drums") if body.get(k) is not None}
        self.playing, self._start_s, self._since = True, start_s, time.time()
        steps = sum(len(body[k]["steps"]) for k in self._lanes)
        self.message = (f"sent {song.name} from {start_s:.1f}s at x{self.speed():.2f} "
                        f"({steps} steps, {len(json.dumps(body)) / 1024:.0f} KB)")

    async def stop(self):
        position = self.position()
        if self._lanes:
            await self._post({"id": "superconductor_midi stop", **{lane: None for lane in self._lanes}})
        self._lanes = set()
        self.playing, self._start_s = False, position
        self.message = "stopped"

    async def poll_client(self):
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=1)) as session:
                async with session.get(f"{self.client_url}/status") as r:
                    self.client = await r.json()
        except Exception:  # pylint: disable=broad-exception-caught
            self.client = {"connected": False, "unreachable": True}

    def state(self):
        position = self.position()
        return {"songs": [s.info() for s in sorted(self.songs.values(), key=lambda s: s.name.lower())],
                "errors": self.errors,
                "song": self.song.name if self.song else None,
                "playing": self.playing, "position_s": round(position, 2), "speed": round(self.speed(), 3),
                "settings": self.settings, "client": self.client, "client_url": self.client_url,
                "message": self.message}


class PlayerServer:
    def __init__(self, player, port):
        self.player = player
        self.port = port
        self.app = web.Application(client_max_size=32 * 1024 ** 2)
        self.app.router.add_get("/", lambda r: web.FileResponse(PAGE))
        self.app.router.add_get("/api/state", self._state)
        self.app.router.add_post("/api/upload", self._upload)
        self.app.router.add_post("/api/play", self._play)
        self.app.router.add_post("/api/stop", self._stop)
        self.app.router.add_post("/api/settings", self._settings)
        self.app.router.add_delete("/api/songs/{name}", self._delete)
        self.app.on_startup.append(self._start_polling)

    async def _start_polling(self, app):
        async def poll():
            while True:
                await self.player.poll_client()
                await asyncio.sleep(1.0)
        app["poll"] = asyncio.create_task(poll())

    async def _state(self, request):
        return web.json_response(self.player.state())

    async def _upload(self, request):
        reader = await request.multipart()
        field = await reader.next()
        while field is not None and field.name != "file":
            field = await reader.next()
        if field is None or not field.filename:
            raise web.HTTPBadRequest(text="no file")
        name = re.sub(r"[^\w .()'&,-]", "_", Path(field.filename).name).strip() or "upload.mid"
        if not name.lower().endswith((".mid", ".midi")):
            name += ".mid"
        dest = self.player.songs_dir / name
        dest.write_bytes(await field.read(decode=False))
        song = self.player._load(dest)  # pylint: disable=protected-access
        if song is None:
            error = self.player.errors.pop(name, "unreadable")
            dest.unlink(missing_ok=True)
            raise web.HTTPBadRequest(text=f"{name} is not a MIDI file mido can read: {error}")
        return web.json_response(song.info())

    async def _apply(self, body):
        """settings from a request body (bpm null = the file's own tempo)"""
        s = self.player.settings
        if "bpm" in body:
            s["bpm"] = float(body["bpm"]) if body["bpm"] not in (None, "") else None
        if body.get("mode") in MODES:
            s["mode"] = body["mode"]
        if "loop" in body:
            s["loop"] = bool(body["loop"])
        if body.get("cfg_notes") not in (None, ""):
            s["cfg_notes"] = round(min(7.0, max(-1.0, float(body["cfg_notes"]))) / 0.2) * 0.2

    async def _play(self, request):
        body = await request.json()
        song = self.player.songs.get(body.get("song")) or self.player.song
        if song is None:
            raise web.HTTPBadRequest(text="no song")
        if song is not self.player.song:
            self.player.song, self.player._start_s = song, 0.0  # pylint: disable=protected-access
            self.player.settings["bpm"] = None  # a new song starts at its own tempo
        await self._apply(body)
        start = body.get("start_s")
        return await self._send(self.player.play(self.player.position() if start is None else float(start)))

    async def _stop(self, request):
        return await self._send(self.player.stop())

    async def _settings(self, request):
        position = self.player.position()  # at the old speed
        await self._apply(await request.json())
        if self.player.playing:  # re-send the rest of the song with the new settings
            return await self._send(self.player.play(position))
        return web.json_response(self.player.state())

    async def _send(self, coro):
        try:
            await coro
        except Exception as e:  # pylint: disable=broad-exception-caught
            self.player.message = f"not sent: {e}" if str(e) else f"not sent: {type(e).__name__}"
            return web.json_response(self.player.state(), status=502)
        return web.json_response(self.player.state())

    async def _delete(self, request):
        name = request.match_info["name"]
        song = self.player.songs.pop(name, None)
        if song is None:
            raise web.HTTPNotFound()
        if self.player.song is song:
            if self.player.playing:
                await self._send(self.player.stop())
            self.player.song = None
        song.path.unlink(missing_ok=True)
        return web.json_response(self.player.state())

    def run(self):
        print(f"MIDI player on http://localhost:{self.port} -> {self.player.client_url}/pattern")
        web.run_app(self.app, port=self.port, print=None)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="*", help=".mid files to add to the library")
    parser.add_argument("--port", type=int, default=8475)
    parser.add_argument("--client_url", default="http://localhost:8470",
                        help="client_midi of the running superconductor_client")
    parser.add_argument("--songs_dir", default=str(Path(__file__).parent / "songs"),
                        help="where uploaded .mid files are kept")
    args = parser.parse_args()
    player = Player(args.songs_dir, args.client_url)
    for path in args.files:
        if player.add(path) is None:
            print(f"skipped {path}: {player.errors.get(Path(path).name)}")
    PlayerServer(player, args.port).run()


if __name__ == "__main__":
    main()
