#pragma once

// One player: the stream, the queue it fills, and the audio-thread side that reads it
// out at the host's rate. The plugin (and its Standalone app) own one of these.
//
// It plays whatever is queued. The server's lead is the buffer.

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

    void connect (const std::string& url, const std::string& name);
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
    std::atomic<bool> playing { false };
};
} // namespace sc
