#pragma once

// Reads the 48 kHz stream out of an AudioQueue at the host's rate. `ratio` is the
// number of stream samples per output sample: 48000 / host rate, times a small drift
// correction when this player is a monitor (see Player).

#include <juce_audio_basics/juce_audio_basics.h>

#include "AudioQueue.h"

namespace sc
{
class Resampler
{
public:
    void prepare (int maxOutputFrames, double maxRatio);
    void reset();

    // fills left/right with `frames` samples; returns how many came from the queue (the
    // rest are silence: the queue ran dry)
    int read (AudioQueue& queue, float* left, float* right, int frames, double ratio);

private:
    juce::LagrangeInterpolator interpLeft, interpRight;
    std::vector<float> inLeft, inRight;
};
} // namespace sc
