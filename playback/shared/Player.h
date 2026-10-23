#pragma once

// One player: the stream, the queue it fills, and the audio-thread side that reads it
// out at the host's rate. The plugin (and its Standalone app) own one of these.
//
// As the clock it plays whatever is queued (the server's lead is the buffer). As a
// monitor it gets the same blocks, but its sound card runs on a different clock than
// the one acking, so it keeps a small cushion and speeds up or slows down by a few
// hundredths of a percent to hold it (`drift`).

#include <atomic>
#include <string>

#include "AudioQueue.h"
#include "Features.h"
#include "Resampler.h"
#include "StreamClient.h"

namespace sc
{
class Player
{
public:
    Player();

    void connect (const std::string& url, bool wantsClock, const std::string& name);
    void disconnect();

    // audio thread
    void prepare (double hostSampleRate, int maxBlock);
    void render (float* left, float* right, int frames);

    struct Status
    {
        bool connected = false, clock = false, playing = false;
        std::string clockName, error;
        double bufferedMs = 0, driftPercent = 0;
        int64_t blocks = 0, underruns = 0;
    };
    Status status() const;

    FeatureTap& featureTap() { return tap; }
    double hostSampleRate() const { return hostRate.load(); }

private:
    AudioQueue queue;
    StreamClient client { queue };
    Resampler resampler;
    FeatureTap tap;

    std::atomic<double> hostRate { 48000.0 };
    double baseRatio = 1.0;
    std::atomic<double> drift { 1.0 };
    std::atomic<bool> playing { false };

    // the monitor's cushion: lowest fill over a window, in stream samples
    int windowMinFill = 0;
    double windowFrames = 0;
};
} // namespace sc
