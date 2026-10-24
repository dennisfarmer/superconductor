# SuperConductor Frontend

## Browser conducting (with the SuperConductor stack)

`conducting_web/` runs the live conductor in the browser, next to client-web's camera page.
The browser owns the camera. MediaPipe's HandLandmarker runs in a Web Worker and sends each
frame's hand landmarks to `conducting_web/server.py`. The server passes them through the
unchanged `StopGestureDetector`, `CycleTempoEstimator` and `ServerTempoPlayer`, which sends
`POST /tempo/target` / `/tempo/free` to the MRT2 server. The page draws what the OpenCV
preview drew (hands, fingertip trail, status lines). Its status line shows each latency:

- tracking fps and the page round trip;
- MediaPipe's time;
- the conductor's time per frame;
- the HTTP POST round trip, and frame→server;
- the server's target and heard BPM.

```bash
# in superconductor_client_web
make server            # MRT2 server, :9100
make client            # camera page http://localhost:8467 (the music plays there)
make conduct           # conducting page http://localhost:8480 (make conduct-remote: via the :9000 tunnel)
```

The client-web container starts this same process. Open it at http://localhost:8080/conduct/
(the camera needs `http://localhost`). On the machine running the container it is also
http://127.0.0.1:8480, and `ws://127.0.0.1:8480/ws/hands/monitor` is unchanged.

Both pages can use the same webcam, or two different ones. Frames come straight from the
camera track (`MediaStreamTrackProcessor`), so both pages keep running in background tabs.
`make web-deps` downloads `@mediapipe/tasks-vision` and `hand_landmarker.task` into `var/web`,
and `make conduct` runs it for you. `make test` runs the unit tests in `tests/`.

`ws://localhost:8480/ws/hands/monitor` is listen-only (TouchDesigner reads it): each frame's
hands message as the page sent it (`seq`, `t_ms`, `width`, `height`, `hands`: 21 points per
hand, x and y 0–1, `handedness`), plus `conduct`, the conductor's reply to that frame.

The Python-only live conductor below is unchanged.

Run live camera conducting against the tempo server (Linux/macOS):

```bash
python3 -m pip install -r superconductor/requirements-live.txt
python3 superconductor/new_tempo_detector.py --server-url http://localhost:9100
# Through the Lighthouse SSH tunnel:
python3 superconductor/new_tempo_detector.py --server-url http://localhost:9000
```

Server mode is now the default and follows
[CONDUCTING_INTEGRATION.md](https://github.com/dennisfarmer/superconductor/blob/server/CONDUCTING_INTEGRATION.md).
Run the server, client, and describe processes separately as described there.
The audio client must be connected for music to play. This conducting process
needs a camera and preview display, but no music file, FFmpeg, or Bungee build
in server mode.

Detected BPM is sent as `POST /tempo/target {"bpm": ...}`. This uses the exact-BPM
API rather than sending whole-gesture cycles as individual beats. A background
worker keeps the latest target, sends at most four target requests per second,
ignores changes under 2 BPM, and refreshes the target every two seconds while
valid tracking continues. Requests time out after 0.5 seconds and retry after
failures. The camera loop does not wait for HTTP. Status is polled roughly once
per second and connection failures appear in the preview.

Tracking gaps hold the last target. A confirmed stop gesture or **F** sends
`POST /tempo/free` with JSON `null` and releases conducting control; **R**
starts a fresh gesture-estimation window. Exit also attempts `/tempo/free`,
warning if the server cannot confirm it. **Free does not mute server audio**:
music eases back to MRT2's own tempo. The documented API has no pause endpoint.
The server handles smoothing (1.5 s), speed limits (0.75–1.3), and half/double
time. It needs about four seconds of model audio before changing speed. Local
Bungee smoothing and the fixed 80 BPM track reference do not apply to server mode.

### Optional local Bungee playback

The previous standalone camera/music workflow remains available:

```bash
python3 -m pip install -r superconductor/requirements-live.txt
# Ubuntu/Debian (macOS: install FFmpeg and the Xcode command-line tools)
sudo apt install ffmpeg build-essential git
bash superconductor/build_bungee.sh
python3 superconductor/new_tempo_detector.py --playback local
```

In local mode, the default input is camera 0 and the default music is the uploaded
`testmusic-80.mp3` beside the script. A preview draws your hand movement as it
happens and displays the detected BPM and playback speed. Press Q or Escape,
or close the preview, to stop.

The Bach recording `testmusic-80.mp3` uses a source tempo of **80 BPM**
by default, skipping automatic analysis. Override the source tempo if needed:

```bash
python3 superconductor/new_tempo_detector.py --playback local --camera 0 --music-bpm 80
```

Use `--music` to select another track. The earlier `testmusic.mp3` retains its 190 BPM reference. Other tracks are analyzed once using
their first 90 seconds unless `--music-bpm` is supplied. Automatic estimates
can be incorrect, so use the known source BPM when available.

Playback continuously time-stretches at
`detected BPM / original music BPM`, preserving pitch using
[Bungee Basic](https://github.com/bungee-audio-stretch/bungee).
Rate changes retain the playhead; the track loops at its end. Music starts at
normal speed when the live camera loop begins, including during tempo warmup.
Playback speed follows tempo changes with a 2.5-second exponential smoothing
time constant in the audio rendering thread, independently of camera frame rate.
Target changes smaller than 2 BPM are ignored, and playback tempo changes by
at most 4 BPM per second. This favors steady sound over fast gesture response. Missing or stopped hand motion holds the current speed without
pausing unless an intentional stop cue is confirmed (see below). Q, Escape, window close, or a fatal device error still ends playback.

Conduct a repeating **4/4 pattern**. The default estimator window requires five
seconds of initial motion and updates every 0.25 seconds. Changes take several
seconds to settle; live playback does not remove that estimation delay. The
five-second window accommodates 80 BPM conducting. Use `--tempo-window 3`
for a faster response when conducting around 100 BPM or above. Matching tempo does
not align the music's individual beats or downbeat to the gesture. The source
track is assumed to have a steady tempo.

Camera timestamps use a monotonic clock, and capture retains only the latest
frame, so processing does not accumulate a video backlog or store the session
in memory. Camera, display, audio output, FFmpeg, and the compiled Bungee bridge must be available on the
machine running the script.

Explicit video paths still use the offline annotation pipeline:

```bash
python3 superconductor/new_tempo_detector.py superconductor/42test.mp4
```

![diagram](media/diagram.jpeg)

## Folder layout

- `live/`: camera conducting (tempo, stop gesture, server and local playback).
- `annotate/`: offline video annotation.
- `bench/`: landmark and tempo benchmarks.
- `conducting_web/`: browser conducting page.
- `native/`: C++ Bungee bridge.
- `scripts/`: MediaPipe download and the manual camera check (`python3 scripts/test_camera.py`).
- `tests/`: unit tests.
- `new_tempo_detector.py`: command-line entry, live camera or a video file.
- Input videos, `testmusic.mp3`, and `hand_landmarker.task`: kept in this directory.
- `outputs/annotations/`: generated videos and frame metadata.
- `outputs/benchmarks/`: benchmark reports, results, and reusable landmark caches.
- `outputs/import-metadata/`: preserved Windows download metadata.
- `media/`: documentation images.

Offline annotation writes to `outputs/annotations/` beside the input video.
Benchmark tools write to `outputs/benchmarks/` beside the input video (the cycle
benchmark defaults to this project's outputs folder). Explicit output paths
still work. Generated outputs remain ignored by Git.

### Partially green camera image under WSL

Both camera entry points request MJPEG at 640×480, 30 FPS, with OpenCV color
conversion enabled. This reduces USB transport bandwidth compared with raw
YUYV. A partially green frame can indicate incomplete camera data; this setting
is a workaround, not proof that transport was the cause. Unsupported cameras
warn and retain their negotiated format.

Close any running camera preview, then restart the conductor or run
`python3 scripts/test_camera.py` from this directory. The standalone check prints the actual
negotiated format; check that it says `MJPG`. If corruption persists, the USB
forwarding/driver path needs further diagnosis even if Windows capture works.

### Audio startup under WSLg

When `PULSE_SERVER` points into `/mnt/wslg/`, ffplay uses SDL’s PulseAudio
output. FFmpeg decodes the music once to 48 kHz stereo; a persistent Bungee
Basic stream performs all time-stretching. ffplay only plays the resulting PCM.
There is no mpv dependency.

### Building and testing Bungee

Run `bash superconductor/build_bungee.sh` from the workspace root. It downloads
Bungee at revision `8cb6977d0c1a1b411ac320493b3c7f5182ed2d22` and its
submodules, installs CMake 3.31.6 locally, and builds the native bridge in the
ignored `.build/` directory. It requires a C++20 compiler, Python with pip, Git,
and network access on first setup. Build artifacts need rebuilding on a new
machine. Bungee Basic is MPL-2.0; its license and source remain in `.build/bungee/`.
This integration uses the open-source Basic edition, not commercial Bungee Pro.

The worker keeps one stretcher alive across speed changes and track loops,
with bounded render-ahead and continuous pitch preservation. Full decoded
audio stays in RAM (approximately 23 MB per minute at 48 kHz stereo). This
backend can still produce artifacts at extreme ratios; it does not guarantee
transparent stretching of every recording. Linux/WSLg is tested; macOS has
not been hardware-tested.

```bash
python3 -m unittest discover -s tests -t . -p 'test_*.py'
```

Native tests check pitch, duration, rate changes, looping, and shutdown. The
end-to-end test uses silence and SDL dummy output, without camera or speakers.

HTTP integration tests use an isolated localhost fixture, not the live MRT2 server:

```bash
python3 -m unittest discover -s tests -t . -p 'test_server_playback.py'
```

### Intentional stops

Ordinary tracking gaps still hold playback speed. Deliberate stop cues bypass
slow tempo filtering and latch playback stopped:

- **Closed fist:** uses 3D finger folding and reach along the palm to accommodate
  inward-facing fists. Requires four curled fingers, or three clearly curled
  fingers and one uncertain folded finger, with no clearly extended fingers.
  Clear fists require at least 180 ms; ambiguous or inferred inward-facing
  poses require a continuous 450 ms hold. Both require at least four frames.
- **Rapid motion cutoff:** a clearly tracked, open hand must have recent
  sustained palm movement and a cycle-correlation score of at least 0.75,
  followed quickly by at least 320 ms of near-stillness (six or more frames).
  The correlation score is not a calibrated probability of an intentional stop.

Missing/ambiguous hands, large position jumps, and tracking gaps invalidate
pending confirmation. The motion cue uses palm movement, not falling estimated
BPM, to avoid interpreting normal tempo changes as a stop. A 20 ms audio fade
reduces clicks; already queued audio and device buffering add cutoff latency.
In local mode the playhead holds after the fade, and the overlay identifies the stop reason.
In server mode the same cue releases tempo control via `/tempo/free`; it does not mute music.
Press **R** to resume from the held position with a fresh tempo-estimation window.
Q or Escape still exits. Open the hand before resuming to avoid triggering the
fist cue again.

These are heuristic thresholds covered by synthetic tests, not validated
real-world gesture accuracy. Fist poses, viewpoint, and landmark jitter may
require tuning with the actual camera. Slow deceleration alone does not stop
playback.
