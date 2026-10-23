#include "AudioQueue.h"

#include <algorithm>

namespace sc
{
static int nextPowerOfTwo (int n)
{
    int p = 1;
    while (p < n)
        p <<= 1;
    return p;
}

AudioQueue::AudioQueue (int capacityFrames)
    : capacity (nextPowerOfTwo (capacityFrames)),
      left ((size_t) capacity),
      right ((size_t) capacity)
{
}

bool AudioQueue::push (const float* interleaved, int frames)
{
    const auto w = writePos.load (std::memory_order_relaxed);
    const auto r = readPos.load (std::memory_order_acquire);
    if (w + frames - r > capacity)
        return false;
    const auto blockIndex = blocksWritten.load (std::memory_order_relaxed);
    if (blockIndex - blocksRead.load (std::memory_order_acquire) >= maxBlocks)
        return false;

    const int mask = capacity - 1;
    for (int i = 0; i < frames; ++i)
    {
        const auto at = (size_t) ((w + i) & mask);
        left[at] = interleaved[2 * i];
        right[at] = interleaved[2 * i + 1];
    }
    blockEnds[(size_t) (blockIndex % maxBlocks)] = w + frames;
    writePos.store (w + frames, std::memory_order_release);
    blocksWritten.store (blockIndex + 1, std::memory_order_release);
    return true;
}

void AudioQueue::flush()
{
    flushTo.store (writePos.load (std::memory_order_relaxed), std::memory_order_release);
}

void AudioQueue::applyFlush()
{
    const auto to = flushTo.exchange (-1, std::memory_order_acq_rel);
    if (to < 0)
        return;
    if (to > readPos.load (std::memory_order_relaxed))
        readPos.store (to, std::memory_order_release);
    // the flushed blocks are dropped, not played
    auto b = blocksRead.load (std::memory_order_relaxed);
    const auto written = blocksWritten.load (std::memory_order_acquire);
    while (b < written && blockEnds[(size_t) (b % maxBlocks)] <= to)
        ++b;
    blocksRead.store (b, std::memory_order_release);
}

int AudioQueue::available() const
{
    const auto to = flushTo.load (std::memory_order_acquire);
    const auto r = std::max (readPos.load (std::memory_order_relaxed), to);
    return (int) std::max<int64_t> (0, writePos.load (std::memory_order_acquire) - r);
}

int AudioQueue::peek (float* l, float* r, int frames) const
{
    const auto to = flushTo.load (std::memory_order_acquire);
    const auto start = std::max (readPos.load (std::memory_order_relaxed), to);
    const auto n = (int) std::min<int64_t> (frames, writePos.load (std::memory_order_acquire) - start);
    const int mask = capacity - 1;
    for (int i = 0; i < n; ++i)
    {
        const auto at = (size_t) ((start + i) & mask);
        l[i] = left[at];
        r[i] = right[at];
    }
    return std::max (0, n);
}

void AudioQueue::consume (int frames)
{
    applyFlush();
    const auto r = std::min (readPos.load (std::memory_order_relaxed) + frames,
                             writePos.load (std::memory_order_acquire));
    readPos.store (r, std::memory_order_release);

    auto b = blocksRead.load (std::memory_order_relaxed);
    const auto written = blocksWritten.load (std::memory_order_acquire);
    int64_t finished = 0;
    while (b < written && blockEnds[(size_t) (b % maxBlocks)] <= r)
    {
        ++b;
        ++finished;
    }
    blocksRead.store (b, std::memory_order_release);
    if (finished > 0)
        played.fetch_add (finished, std::memory_order_acq_rel);
}
} // namespace sc
