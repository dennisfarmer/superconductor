#pragma once

// The beat being heard, from the MRT2 server's GET /tempo/status (tempo.py on the
// server branch): heard_bpm, playing, beat (0 = beat one), beat_frac, beats_per_bar.
// Polled a few times a second on its own thread; now() moves the beat forward at the
// heard tempo between polls, so the beat position runs smoothly.

#include <atomic>
#include <chrono>
#include <mutex>
#include <string>
#include <thread>

namespace sc
{
struct TempoState
{
    bool reachable = false;  // the last poll got an answer
    bool playing = false;
    bool beatKnown = false;  // the server has a beat grid (beat / beat_frac not null)
    double bpm = 0;          // heard_bpm (0 if unknown)
    int beat = 0;            // within the bar, 0 = beat one
    double beatFrac = 0;     // 0-1 through that beat
    int beatsPerBar = 4;     // 3 | 4 (4 while the server isn't sure)
    double barPhase = 0;     // 0-1 through the bar
};

class TempoPoller
{
public:
    TempoPoller();
    ~TempoPoller();

    void setServer (const std::string& baseUrl);  // e.g. http://localhost:9100
    TempoState now() const;

private:
    void loop();

    std::thread thread;
    std::atomic<bool> running { true };
    mutable std::mutex lock;
    std::string url;
    TempoState polled;
    std::chrono::steady_clock::time_point polledAt;
};
} // namespace sc
