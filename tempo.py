"""Adaptive playback speed for the MRT2 server.

MRT2 picks its own tempo, so the server measures it and time-stretches the
generated audio (changing the speed, not the pitch) toward a target tempo:

    model audio --+--> TempoEstimator ("model bpm", the tempo MRT2 is playing)
                  +--> Stretcher(speed) --> client      heard bpm = model bpm * speed

The target tempo comes either from the test page (an exact bpm) or from the
conductor's beats, smoothed by a small Kalman filter (`BeatTracker`). The speed
is never set directly: every block it moves a fraction of the way toward
`target / model bpm`, so tempo changes are gradual and audio is never skipped.

The heard audio also gets a beat grid (`BeatGrid`): where the beats are, which
one is beat one and whether a bar has 3 or 4 beats, aligned with what the
listener hears through the client's block acknowledgements. Beats per bar and
beat one can also be set by hand.

Only numpy, so it also runs in the Lighthouse venv.
"""

import threading
import time
from collections import Counter, deque

import numpy as np

SAMPLE_RATE = 48000

MIN_SPEED, MAX_SPEED = 0.75, 1.3  # speeding up needs rtf > speed
SPEED_TIME_CONSTANT = 1.5  # seconds for the speed to move ~63% of the way to its goal
BEAT_TIMEOUT = 3.0  # seconds without beats before the conductor tempo is dropped
DEFAULT_BEATS_PER_BAR = 4  # 4/4 until set otherwise (None would detect it)


class TempoEstimator:
    """Tempo of the last few seconds of audio: onset strength + autocorrelation.

    1. mono, downsampled 4x to 12kHz
    2. onset envelope = spectral flux (how much louder each frequency got
       since the previous 5ms frame), summed over frequencies
    3. autocorrelation of the envelope: the strongest repeat period between
       60 and 200 bpm, mildly preferring tempos near 110 bpm
    The result is the median of the last few estimates, so one bad window
    doesn't move it.
    """

    DEC = 4
    RATE = SAMPLE_RATE // DEC
    FRAME, HOP = 512, 64
    WINDOW_S, MIN_S = 8.0, 4.0  # analysed audio; audio needed for a first estimate
    EVERY_S = 1.0  # analyse once per second of audio
    PRIOR_BPM, PRIOR_OCTAVES = 110.0, 1.0

    def __init__(self):
        self._ring = np.zeros(int(self.WINDOW_S * self.RATE), dtype=np.float32)
        self._filled = 0
        self._since = 0
        self._recent = []
        self.pushed = 0  # 12kHz samples pushed so far
        self.bpm = None  # median of recent estimates

    def push(self, samples):
        """`samples`: (n, 2) float32 at 48kHz. True if it analysed the window."""
        n = len(samples) // self.DEC * self.DEC
        if n == 0:
            return False
        mono = samples[:n].mean(axis=1).reshape(-1, self.DEC).mean(axis=1)
        mono = mono[-len(self._ring):]
        self._ring = np.roll(self._ring, -len(mono))
        self._ring[-len(mono):] = mono
        self._filled = min(len(self._ring), self._filled + len(mono))
        self._since += len(mono)
        self.pushed += n // self.DEC
        if self._since >= self.EVERY_S * self.RATE and self._filled >= self.MIN_S * self.RATE:
            self._since = 0
            bpm = self.estimate(self._ring[-self._filled:])
            if bpm is not None:
                self._recent = (self._recent + [bpm])[-5:]
                self.bpm = float(np.median(self._recent))
            return True
        return False

    def window(self):
        """The analysed audio (12kHz mono) and the time of its end, in seconds of
        audio pushed since this estimator was created."""
        return self._ring[-self._filled:], self.pushed / self.RATE

    @classmethod
    def spectrogram(cls, audio):
        """Magnitude spectra of 43ms frames every 5ms: (frames, FRAME // 2 + 1)."""
        frames = np.lib.stride_tricks.sliding_window_view(audio, cls.FRAME)[::cls.HOP]
        return np.abs(np.fft.rfft(frames * np.hanning(cls.FRAME), axis=1))

    @classmethod
    def onset_envelope(cls, mag):
        """Spectral flux of `mag` (or of some of its frequency bins), with the
        slowly changing loudness removed. Value i is the onset at frame i + 1."""
        flux = np.maximum(np.diff(np.log1p(100 * mag), axis=0), 0).sum(axis=1)
        k = int(cls.RATE / cls.HOP / 2)
        return np.maximum(flux - np.convolve(flux, np.ones(k) / k, mode="same"), 0)

    @classmethod
    def frame_time(cls, i):
        """Time of frame i's centre, in seconds from the start of the audio."""
        return (i * cls.HOP + cls.FRAME / 2) / cls.RATE

    @classmethod
    def estimate(cls, audio):
        flux = cls.onset_envelope(cls.spectrogram(audio))
        env_rate = cls.RATE / cls.HOP
        if not flux.any():
            return None
        spec_ac = np.fft.rfft(flux, 2 * len(flux))
        ac = np.fft.irfft(spec_ac * np.conj(spec_ac))[:len(flux)]
        lags = np.arange(int(env_rate * 60 / 200), int(env_rate * 60 / 60) + 1)
        bpms = 60 * env_rate / lags
        prior = np.exp(-0.5 * (np.log2(bpms / cls.PRIOR_BPM) / cls.PRIOR_OCTAVES) ** 2)
        score = ac[lags] * prior
        i = int(np.argmax(score))
        lag = float(lags[i])
        if 0 < i < len(score) - 1:  # parabolic interpolation between lags
            a, b, c = score[i - 1], score[i], score[i + 1]
            if a - 2 * b + c < 0:
                lag += 0.5 * (a - c) / (a - 2 * b + c)
        return 60 * env_rate / lag


class BeatGrid:
    """Where the beats are, which one is beat one and how many beats a bar has,
    from a `TempoEstimator`'s window of heard audio. Times are seconds of heard
    audio since the session started. Once a second:

    1. beat phase: the onset envelope is sampled once per beat period, starting
       at each possible offset (a comb over offsets); the offset whose samples
       are loudest, recent beats counting more, is where the beats are. The
       circular median of the last 5 phases is kept.
    2. the beats are numbered continuously: a new grid keeps the numbers of the
       old grid's nearest beats, so beat one can be voted on across windows.
    3. meter and beat one: each beat gets an accent (onset strength + bass onset
       + chord change since the previous beat, each normalised). For 3 and 4
       beats per bar and each candidate beat one, the contrast is the mean
       accent of the beats in that position minus that of the others. The best
       candidate of each window is a vote; the last 6 votes are counted.
    """

    METERS = (3, 4)
    PHASES, VOTES = 5, 6  # windows in the phase median; in the meter vote
    HALF_LIFE_S = 3.0  # a beat's weight in the comb halves every 3s back
    MIN_VOTES, MIN_SHARE = 3, 0.5  # votes and share of them needed to report a meter
    BASS_HZ = 200.0
    CHROMA_HZ = (200.0, 3000.0)

    def __init__(self):
        self.ref_t = None  # time of beat number `ref_n`
        self.ref_n = 0
        self.period = None
        self._phases = []  # (beat time, period) of recent windows
        self._votes = []  # per window {"m": best meter, 3: beat one mod 3, 4: beat one mod 4}

    def position(self, t):
        """Beat number at time t, as a float: beat n starts at n.0."""
        return None if self.ref_t is None else self.ref_n + (t - self.ref_t) / self.period

    def meter(self, pinned=None):
        """(beats per bar, beat one's number mod beats per bar, share of the votes).
        The beats per bar is `pinned` if given, else the most voted one; the share
        is that of the votes agreeing on both."""
        if not self._votes:
            return pinned, None, None
        m = pinned or Counter(v["m"] for v in self._votes).most_common(1)[0][0]
        b = Counter(v[m] for v in self._votes).most_common(1)[0][0]
        agree = sum(v[m] == b and (pinned is not None or v["m"] == m) for v in self._votes)
        return m, b, agree / len(self._votes)

    def sure(self, share):
        return share is not None and len(self._votes) >= self.MIN_VOTES and share >= self.MIN_SHARE

    def analyse(self, audio, end_t, bpm):
        """`audio`: 12kHz mono ending at `end_t`; `bpm`: the heard tempo."""
        E = TempoEstimator
        if bpm is None:
            return
        env_rate = E.RATE / E.HOP
        start_t = end_t - len(audio) / E.RATE
        mag = E.spectrogram(audio)
        env = E.onset_envelope(mag)
        if not env.any():
            return
        freqs = np.fft.rfftfreq(E.FRAME, 1 / E.RATE)
        bass = E.onset_envelope(mag[:, (freqs > 0) & (freqs < self.BASS_HZ)])
        smooth = lambda x: np.convolve(x, np.hanning(7) / np.hanning(7).sum(), mode="same")
        env, bass = smooth(env), smooth(bass)
        env_time = lambda i: start_t + E.frame_time(np.asarray(i) + 1)  # env[i]: onset at frame i + 1

        # 1. comb over offsets: the offset whose beats are loudest
        period = 60.0 / bpm
        p = period * env_rate
        offsets = np.arange(int(np.ceil(p)))
        idx = np.round(offsets[:, None] + np.arange(int(len(env) / p) + 1)[None, :] * p).astype(int)
        w = 0.5 ** ((len(env) - idx) / env_rate / self.HALF_LIFE_S) * (idx < len(env))
        score = (env[np.minimum(idx, len(env) - 1)] * w).sum(axis=1) / w.sum(axis=1)
        best = offsets[int(np.argmax(score))]
        beat_t = float(env_time(best + np.floor((len(env) - 1 - best) / p) * p))  # last beat
        self._phases = (self._phases + [(beat_t, period)])[-self.PHASES:]
        # each phase moved (with its own period) to the beat nearest beat_t
        diffs = [(t + round((beat_t - t) / q) * q - beat_t + period / 2) % period - period / 2
                 for t, q in self._phases]
        beat_t += float(np.median(diffs))

        # 2. continuous numbering
        n = 0 if self.ref_t is None else self.ref_n + int(round((beat_t - self.ref_t) / self.period))
        self.ref_t, self.ref_n, self.period = beat_t, n, period

        # 3. accents of the beats that have a whole beat before and after them in the window
        frame_t = start_t + E.frame_time(np.arange(len(mag)))
        env_t = env_time(np.arange(len(env)))
        band = (freqs >= self.CHROMA_HZ[0]) & (freqs <= self.CHROMA_HZ[1])
        pitch_class = np.round(12 * np.log2(freqs[band] / 440.0) + 69).astype(int) % 12
        chroma = mag[:, band] @ np.eye(12)[pitch_class]
        nums, feats = [], []
        j = 1
        while beat_t - (j + 1) * period >= start_t:
            t = beat_t - j * period
            near = np.abs(env_t - t) < 0.1 * period
            before = chroma[(frame_t >= t - period) & (frame_t < t)].mean(axis=0)
            after = chroma[(frame_t >= t) & (frame_t < t + period)].mean(axis=0)
            change = 1 - before @ after / (np.linalg.norm(before) * np.linalg.norm(after) + 1e-9)
            nums.append(n - j)
            feats.append((env[near].max(), bass[near].max(), change))
            j += 1
        if len(nums) < 2 * max(self.METERS):
            return
        feats = np.array(feats)
        accent = ((feats - feats.mean(axis=0)) / (feats.std(axis=0) + 1e-9)).sum(axis=1)
        nums = np.array(nums)
        vote, best_contrast = {}, -np.inf
        for m in self.METERS:
            contrast = [accent[(nums - b) % m == 0].mean() - accent[(nums - b) % m != 0].mean()
                        for b in range(m)]
            vote[m] = int(np.argmax(contrast))
            if max(contrast) > best_contrast:
                vote["m"], best_contrast = m, max(contrast)
        self._votes = (self._votes + [vote])[-self.VOTES:]


class BeatTracker:
    """Conductor tempo from beat times: a 1-D Kalman filter on the beat period.

    Each interval between two beats is a noisy measurement of the period. The
    filter trusts a new interval more when it is unsure of the period
    (start-up) and less once it has settled, which smooths hand jitter but
    still follows a gradual tempo change. An interval far from the current
    period (a missed or doubled beat) is ignored, unless two in a row agree,
    which means the conductor really changed tempo.
    """

    MEASUREMENT_VAR = 0.04 ** 2  # human timing jitter, ~40ms
    PROCESS_VAR = 0.015 ** 2  # how much the real period may drift per beat
    OUTLIER = 0.3  # an interval more than 30% off the period is an outlier

    def __init__(self):
        self.reset()

    def reset(self):
        self.period = None
        self.var = None
        self.last_beat = None
        self._outliers = []

    def beat(self, t):
        last, self.last_beat = self.last_beat, t
        if last is None:
            return
        x = t - last
        if not 0.25 <= x <= 2.0:  # outside 30-240 bpm: a pause, start over
            self.period = None
            return
        if self.period is None:
            self.period, self.var = x, self.MEASUREMENT_VAR
            return
        if abs(x - self.period) > self.OUTLIER * self.period:
            self._outliers.append(x)
            if len(self._outliers) >= 2 and abs(self._outliers[-1] - self._outliers[-2]) < self.OUTLIER * x:
                self.period, self.var = x, self.MEASUREMENT_VAR
                self._outliers = []
            return
        self._outliers = []
        self.var += self.PROCESS_VAR  # predict
        gain = self.var / (self.var + self.MEASUREMENT_VAR)  # update
        self.period += gain * (x - self.period)
        self.var *= 1 - gain

    @property
    def bpm(self):
        return None if self.period is None else 60.0 / self.period


class Stretcher:
    """Streaming WSOLA time-stretch: changes the speed, keeps the pitch.

    Cuts the input into overlapping 43ms pieces and lays them out again with a
    fixed spacing in the output, while walking through the input faster or
    slower (`speed`). Each piece is shifted by up to +-11ms so that it lines up
    with the previous one (the most similar waveform), which avoids clicks.
    """

    N, HS, DELTA, DEC = 2048, 1024, 512, 4  # piece, output hop, max shift, search decimation

    def __init__(self):
        self._window = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(self.N) / self.N)).astype(np.float32)[:, None]
        self._in = np.zeros((0, 2), dtype=np.float32)
        self._base = 0  # input sample index of self._in[0]
        self._pos = 0.0  # nominal input position of the next piece
        self._prev = None  # input position of the previous piece
        self._acc = np.zeros((self.N, 2), dtype=np.float32)

    def process(self, samples, speed):
        self._in = np.concatenate([self._in, samples])
        end = self._base + len(self._in)
        out = []
        while True:
            if self._prev is None:
                if self.N > end:
                    break
                start = 0
            else:
                a = int(round(self._pos))
                nat = self._prev + self.HS  # where the previous piece naturally continues
                if max(a + self.DELTA, nat) + self.N > end:
                    break
                start = self._best_start(a - self.DELTA, a + self.DELTA, nat)
            piece = self._in[start - self._base:start - self._base + self.N]
            self._acc += piece * self._window
            out.append(self._acc[:self.HS].copy())
            self._acc = np.concatenate([self._acc[self.HS:], np.zeros((self.HS, 2), np.float32)])
            self._prev = start
            self._pos += self.HS * speed
            keep = min(int(round(self._pos)) - self.DELTA, self._prev + self.HS)
            if keep > self._base:
                self._in = self._in[keep - self._base:]
                self._base = keep
        return np.concatenate(out) if out else np.zeros((0, 2), np.float32)

    def _best_start(self, lo, hi, nat):
        mono = lambda s, n: self._in[s - self._base:s - self._base + n].mean(axis=1)
        template = mono(nat, self.N)

        def similarity(region, tmpl):
            corr = np.correlate(region, tmpl, mode="valid")
            energy = np.convolve(region ** 2, np.ones(len(tmpl)), mode="valid")
            return corr / np.sqrt(energy + 1e-9)

        # coarse search on every DEC-th sample, then refine around the best
        region = mono(lo, hi - lo + self.N)
        coarse = similarity(region[::self.DEC], template[::self.DEC])
        best = lo + int(np.argmax(coarse)) * self.DEC
        flo, fhi = max(lo, best - self.DEC), min(hi, best + self.DEC)
        fine = similarity(mono(flo, fhi - flo + self.N), template)
        return flo + int(np.argmax(fine))


def _fold(ratio):
    """Pick ratio, 2x or 1/2x, whichever is closest to 1 (half/double-time)."""
    return min((ratio * f for f in (0.5, 1.0, 2.0)), key=lambda r: abs(np.log(r)))


class TempoController:
    """Thread-safe: the generator thread calls `process` and `played`; the HTTP
    handlers call `beat`, `set_target`, `set_free`, `set_meter`, `downbeat`,
    `clear_override` and `status`.

    Heard position: the client acknowledges each block when it has played it, so
    the last acknowledged block's end, plus the time since its acknowledgement,
    is where the listener is (minus the sound card's latency, which the pages
    let the user nudge). The beat grid is in the same time base."""

    def __init__(self):
        self._lock = threading.Lock()
        self._beats = BeatTracker()
        self.mode = "free"  # free (speed 1) | target (exact bpm) | conduct (beats)
        self.target_bpm = None
        self.speed = 1.0
        self._last_beat_wall = 0.0
        self.reset()

    def reset(self):
        """New session: new audio, so new tempo estimates."""
        with self._lock:
            self._model = TempoEstimator()  # what MRT2 generates
            self._output = TempoEstimator()  # what the listener hears (a check)
            self._stretcher = Stretcher()
            self._grid = BeatGrid()  # beats of the heard audio
            self._sent = 0  # stretched samples sent
            self._block_ends = deque()  # sample end of each block sent but not yet played
            self._anchor = None  # (wall time, sample end) of the last played block
            self._meter_pin = DEFAULT_BEATS_PER_BAR  # beats per bar set by hand (None = detect)
            self._downbeat_pin = None  # beat number marked as beat one by hand
            self.speed = 1.0

    def beat(self, t=None):
        with self._lock:
            self._beats.beat(time.time() if t is None else t)
            self._last_beat_wall = time.time()
            if self._beats.bpm is not None:
                self.mode = "conduct"

    def set_target(self, bpm):
        with self._lock:
            self.mode, self.target_bpm = "target", float(bpm)

    def set_free(self):
        with self._lock:
            self.mode, self.target_bpm = "free", None
            self._beats.reset()

    def _goal_speed(self):
        if self.mode == "conduct":
            if time.time() - self._last_beat_wall > BEAT_TIMEOUT:
                return self.speed  # conductor stopped: hold the current tempo
            target = self._beats.bpm
        elif self.mode == "target":
            target = self.target_bpm
        else:
            return 1.0
        if target is None or self._model.bpm is None:  # still calibrating
            return self.speed
        return float(np.clip(_fold(target / self._model.bpm), MIN_SPEED, MAX_SPEED))

    def process(self, samples):
        """Model audio in, stretched audio out. Called once per generated block."""
        with self._lock:
            self._model.push(samples)
            dt = len(samples) / SAMPLE_RATE
            goal = self._goal_speed()
            self.speed += (goal - self.speed) * min(1.0, dt / SPEED_TIME_CONSTANT)
            if goal == 1.0 and abs(self.speed - 1.0) < 1e-3:
                self.speed = 1.0  # at exactly 1 the stretcher passes the audio through unchanged
            out = self._stretcher.process(samples, self.speed)
            if self._output.push(out) and self._model.bpm is not None:
                self._grid.analyse(*self._output.window(), self._model.bpm * self.speed)
            self._sent += len(out)
            self._block_ends.append(self._sent)
            return out

    def played(self, t):
        """The client finished playing the oldest unplayed block at wall time t."""
        with self._lock:
            if self._block_ends:
                self._anchor = (t, self._block_ends.popleft())

    def _heard(self):
        """(seconds of audio heard so far, whether audio is playing now)."""
        if self._anchor is None:
            return None, False
        t, end = self._anchor
        heard = end / SAMPLE_RATE + time.time() - t
        sent = self._sent / SAMPLE_RATE
        return min(heard, sent), heard < sent

    def set_meter(self, beats):
        """Pin the beats per bar (3 or 4), or None to detect it."""
        with self._lock:
            self._meter_pin = None if beats is None else int(beats)

    def downbeat(self):
        """The beat heard now (the nearest one) is beat one."""
        with self._lock:
            heard, _ = self._heard()
            if heard is not None and self._grid.ref_t is not None:
                self._downbeat_pin = int(round(self._grid.position(heard)))

    def clear_override(self):
        """Back to the default 4/4 and the detected beat one."""
        with self._lock:
            self._meter_pin, self._downbeat_pin = DEFAULT_BEATS_PER_BAR, None

    def _bar(self):
        """Fields of `status` about the beat the listener is on."""
        heard, playing = self._heard()
        m, b, share = self._grid.meter(self._meter_pin)
        source = None
        if self._downbeat_pin is not None:
            m = m or 4
            b, source = self._downbeat_pin % m, "manual"
        elif b is not None and (self._meter_pin or self._grid.sure(share)):
            source = "detected"
        elif not self._meter_pin:
            m = None  # not sure yet
        pos = None if heard is None else self._grid.position(heard)
        if pos is None or not playing:
            beat = frac = None
        else:
            n = int(np.floor(pos))
            beat = (n - (b if source else 0)) % (m or 4)
            frac = pos - n
        return {
            "playing": playing,
            "beat": beat,  # index within the bar, 0 = beat one (beats_per_bar or 4 if unknown)
            "beat_frac": frac,  # 0-1 through that beat
            "beats_per_bar": m,  # 3 | 4 | None (not sure yet)
            "meter_confidence": share,  # share of the recent windows agreeing on meter and beat one
            "meter_source": None if m is None else ("manual" if self._meter_pin else "detected"),
            "downbeat_source": source,  # detected | manual | None (beat one unknown)
        }

    def status(self):
        with self._lock:
            model = self._model.bpm
            return {
                "mode": self.mode,
                "speed": self.speed,
                "stretching": self.speed != 1.0,  # the heard audio is sped up / slowed down
                "conductor_holding": self.mode == "conduct" and time.time() - self._last_beat_wall > BEAT_TIMEOUT,
                "model_bpm": model,  # tempo MRT2 is generating
                "heard_bpm": None if model is None else model * self.speed,
                "measured_output_bpm": self._output.bpm,  # independent check of heard_bpm
                "target_bpm": self.target_bpm,
                "conductor_bpm": self._beats.bpm,
                **self._bar(),
            }
