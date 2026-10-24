#pragma once

// The received audio, between the network thread (push) and the audio thread (peek /
// consume). One writer, one reader, no locks. Positions are absolute sample counts, so
// "has this block finished playing" is a plain comparison.
//
// The queue also remembers where each block ends: a block counts as played once the
// reader is past its end, which is what this player answers with ReceivedChunk
// (AUDIO_PROTOCOL.md).

#include <atomic>
#include <cstdint>
#include <vector>

namespace sc
{
class AudioQueue
{
public:
    explicit AudioQueue (int capacityFrames = 1 << 20);  // ~22 s at 48 kHz

    // network thread -------------------------------------------------------------
    // one block of interleaved stereo; false if it didn't fit (the reader stalled)
    bool push (const float* interleaved, int frames);
    // drop everything queued so far. A new connection does not replay blocks already sent.
    void flush();

    // audio thread ---------------------------------------------------------------
    int available() const;
    // copy up to `frames` without consuming them; returns how many were copied
    int peek (float* left, float* right, int frames) const;
    // consume `frames`; blocks that end inside them count as played
    void consume (int frames);

    // any thread -----------------------------------------------------------------
    // blocks played since the start (monotonic; flushed blocks don't count)
    int64_t blocksPlayed() const { return played.load(); }

private:
    void applyFlush();

    const int capacity;  // a power of two
    std::vector<float> left, right;
    std::atomic<int64_t> writePos { 0 }, readPos { 0 };
    std::atomic<int64_t> flushTo { -1 };

    static constexpr int maxBlocks = 4096;
    std::vector<int64_t> blockEnds = std::vector<int64_t> (maxBlocks);
    std::atomic<int64_t> blocksWritten { 0 }, blocksRead { 0 };
    std::atomic<int64_t> played { 0 };
};
} // namespace sc
