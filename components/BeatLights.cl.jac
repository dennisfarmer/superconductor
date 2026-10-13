"""The beat: one light per beat of the chord, flashing on each quarter note as it is
heard. The engine reports which beat its output is on and how far into it (plus the
audio device's latency); `beat_sync` turns each report into a time anchor, and an
animation-frame loop lights the dots from that anchor between reports. The dots are
updated directly, so the page does not re-render 60 times a second."""

import from react { useEffect, useRef }

# when beat `beat` was (or will be) heard, in performance.now() ms
glob _sync: dict[str, any] = {"anchor": 0.0, "beat": 0, "period": 500.0, "beats": 4, "running": False},
     _lights_el: any = None,
     _raf: any = None;


"""Called on every engine status poll. `frac` is how far through `beat` the frame
being handed to the device is; it reaches the speaker `latency_ms` later."""
def:pub beat_sync(beat: int, frac: float, bpm: float, beats: int, latency_ms: float, running: bool) -> None {
    period = 60000.0 / max(bpm, 1.0);
    anchor = performance.now() + latency_ms - frac * period;
    old = _sync;
    if running and old["running"] and abs(old["period"] - period) < 1.0 and old["beats"] == beats {
        # same tempo: nudge towards the new reading instead of jumping, so poll jitter
        # (about one 40 ms frame) does not make the flashes stutter
        k = Math.round((anchor - old["anchor"]) / period);
        expected = old["anchor"] + k * period;
        diff = anchor - expected;
        if abs(diff) < period * 0.25 and ((old["beat"] + k) % beats + beats) % beats == beat {
            anchor = expected + diff * 0.3;
        }
    }
    _sync["anchor"] = anchor;
    _sync["beat"] = beat;
    _sync["period"] = period;
    _sync["beats"] = max(beats, 1);
    _sync["running"] = running;
}


def light_frame(t: any) -> None {
    global _raf;
    el: any = _lights_el;
    if el {
        dots: any = el.children;
        n = dots.length;
        idx = -1;
        phase = 1.0;
        if _sync["running"] {
            elapsed = performance.now() - _sync["anchor"];
            k = Math.floor(elapsed / _sync["period"]);
            idx = ((_sync["beat"] + k) % n + n) % n;
            phase = elapsed / _sync["period"] - k;
        }
        for i in range(n) {
            d: any = dots[i];
            lit = i == idx;
            # a sharp flash at the start of the beat that fades over the first half of it
            d.style.opacity = str(max(0.0, 1.0 - phase * 1.6) * 0.75 + 0.25) if lit else "";
            d.classList.toggle("lit", lit);
            d.classList.toggle("flash", lit and phase < 0.14);
        }
    }
    _raf = requestAnimationFrame(light_frame);
}


def:pub BeatLights(beats: int, running: bool) -> JsxElement {
    global _lights_el;
    ref: any = useRef(None);

    useEffect(lambda {
        global _lights_el, _raf;
        _lights_el = ref.current;
        if not _raf {
            _raf = requestAnimationFrame(light_frame);
        }
        return lambda {
            global _raf, _lights_el;
            if _raf {
                cancelAnimationFrame(_raf);
            }
            _raf = None;
            _lights_el = None;
        };
    }, []);

    return <div
        ref={ref}
        className={"beat-lights" + ("" if running else " idle")}
        title={"The beat, as you hear it: " + str(beats) + " quarter notes per chord; the first light is the chord change" if running else "The beat (the audio is paused)"}
    >
        {for i in range(max(beats, 1)) {
            <span key={"b" + str(i)} className={"beat-dot" + (" down" if i == 0 else "")}></span>
        }}
    </div>;
}
