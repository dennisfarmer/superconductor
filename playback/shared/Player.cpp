#include "Player.h"

#include <algorithm>
#include <limits>

namespace sc
{
namespace
{
constexpr double streamRate = StreamClient::sampleRate;
constexpr int monitorStart = (int) (0.25 * streamRate);  // a monitor starts with this much queued
constexpr int monitorTarget = (int) (0.05 * streamRate);  // and keeps at least this much
constexpr int monitorMaxLag = (int) (1.0 * streamRate);   // beyond this it skips ahead
constexpr double driftWindowSeconds = 2.0;
constexpr double maxDrift = 0.003;  // +-0.3 %
} // namespace

Player::Player() = default;

void Player::connect (const std::string& url, bool wantsClock, const std::string& name)
{
    client.connect (url, wantsClock, name);
}

void Player::disconnect()
{
    client.disconnect();
}

void Player::prepare (double sampleRate, int maxBlock)
{
    hostRate = sampleRate;
    baseRatio = streamRate / sampleRate;
    resampler.prepare (maxBlock, baseRatio * (1.0 + maxDrift));
    tap.prepare (sampleRate);
    playing = false;
    drift = 1.0;
    windowMinFill = std::numeric_limits<int>::max();
    windowFrames = 0;
}

void Player::render (float* left, float* right, int frames)
{
    const bool isClock = client.isClock();
    auto avail = queue.available();

    if (! playing.load())
    {
        if (avail < (isClock ? 1 : monitorStart))
        {
            std::fill (left, left + frames, 0.0f);
            std::fill (right, right + frames, 0.0f);
            tap.push (left, right, frames);
            return;
        }
        playing = true;
        resampler.reset();
    }

    if (! isClock && avail > monitorMaxLag)
    {
        // the host stopped calling us for a while (or the clock raced ahead): catch up
        queue.consume (avail - monitorStart);
        avail = monitorStart;
    }

    const auto ratio = baseRatio * (isClock ? 1.0 : drift.load());
    const auto got = resampler.read (queue, left, right, frames, ratio);
    tap.push (left, right, frames);

    if (got < frames)
    {
        playing = false;
        if (! client.isPaused())
            client.noteUnderrun();  // only sent while we are the clock
        windowMinFill = std::numeric_limits<int>::max();
        windowFrames = 0;
        return;
    }

    if (isClock)
    {
        drift = 1.0;
        return;
    }
    windowMinFill = std::min (windowMinFill, queue.available());
    windowFrames += frames;
    if (windowFrames >= driftWindowSeconds * hostRate.load())
    {
        // more cushion than needed: read a little faster; less: a little slower
        const auto error = (windowMinFill - monitorTarget) / streamRate;
        const auto wanted = 1.0 + std::clamp (0.02 * error, -maxDrift, maxDrift);
        drift = 0.7 * drift.load() + 0.3 * wanted;
        windowMinFill = std::numeric_limits<int>::max();
        windowFrames = 0;
    }
}

Player::Status Player::status() const
{
    Status s;
    s.connected = client.isConnected();
    s.clock = client.isClock();
    s.playing = playing.load();
    s.clockName = client.clockName();
    s.error = client.lastError();
    s.bufferedMs = 1000.0 * queue.available() / streamRate;
    s.driftPercent = 100.0 * (drift.load() - 1.0);
    s.blocks = client.blocksReceived();
    s.underruns = client.underrunCount();
    return s;
}
} // namespace sc
