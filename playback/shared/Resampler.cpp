#include "Resampler.h"

#include <algorithm>
#include <cmath>

namespace sc
{
void Resampler::prepare (int maxOutputFrames, double maxRatio)
{
    const auto n = (size_t) std::ceil (maxOutputFrames * maxRatio) + 16;
    inLeft.assign (n, 0.0f);
    inRight.assign (n, 0.0f);
    reset();
}

void Resampler::reset()
{
    interpLeft.reset();
    interpRight.reset();
}

int Resampler::read (AudioQueue& queue, float* left, float* right, int frames, double ratio)
{
    const auto capacity = (int) inLeft.size();
    const auto wanted = std::min (capacity, (int) std::ceil (frames * ratio) + 4);
    const auto got = queue.peek (inLeft.data(), inRight.data(), wanted);

    // as many output samples as the input in hand can make without running out
    int produce = frames;
    if (got < wanted)
        produce = std::clamp ((int) std::floor ((got - 2) / ratio), 0, frames);

    int used = 0;
    if (produce > 0)
    {
        used = interpLeft.process (ratio, inLeft.data(), left, produce, got, 0);
        interpRight.process (ratio, inRight.data(), right, produce, got, 0);
        queue.consume (used);
    }
    if (produce < frames)
    {
        std::fill (left + produce, left + frames, 0.0f);
        std::fill (right + produce, right + frames, 0.0f);
    }
    return produce;
}
} // namespace sc
