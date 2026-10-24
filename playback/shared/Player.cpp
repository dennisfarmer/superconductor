#include "Player.h"

#include <algorithm>

namespace sc
{
namespace
{
constexpr double streamRate = StreamClient::sampleRate;
} // namespace

Player::Player() = default;

void Player::connect (const std::string& url, const std::string& name)
{
    client.connect (url, name);
}

void Player::disconnect()
{
    client.disconnect();
}

void Player::prepare (double sampleRate, int maxBlock)
{
    hostRate = sampleRate;
    baseRatio = streamRate / sampleRate;
    resampler.prepare (maxBlock, baseRatio);
    tap.prepare (sampleRate);
    playing = false;
}

void Player::render (float* left, float* right, int frames)
{
    const auto avail = queue.available();

    if (! playing.load())
    {
        if (avail < 1)
        {
            std::fill (left, left + frames, 0.0f);
            std::fill (right, right + frames, 0.0f);
            tap.push (left, right, frames);
            return;
        }
        playing = true;
        resampler.reset();
    }

    const auto got = resampler.read (queue, left, right, frames, baseRatio);
    tap.push (left, right, frames);

    if (got < frames)
    {
        playing = false;
        if (! client.isPaused())
            client.noteUnderrun();
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
    s.driftPercent = 0;
    s.blocks = client.blocksReceived();
    s.underruns = client.underrunCount();
    return s;
}
} // namespace sc
