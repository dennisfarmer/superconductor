#include "TempoPoller.h"

#include <cmath>

#include <ixwebsocket/IXHttpClient.h>
#include <juce_core/juce_core.h>

namespace sc
{
namespace
{
constexpr auto pollEvery = std::chrono::milliseconds (200);
}

TempoPoller::TempoPoller()
{
    thread = std::thread ([this] { loop(); });
}

TempoPoller::~TempoPoller()
{
    running = false;
    thread.join();
}

void TempoPoller::setServer (const std::string& baseUrl)
{
    std::lock_guard<std::mutex> l (lock);
    url = baseUrl;
    while (! url.empty() && url.back() == '/')
        url.pop_back();
}

TempoState TempoPoller::now() const
{
    std::lock_guard<std::mutex> l (lock);
    auto s = polled;
    if (s.playing && s.beatKnown && s.bpm > 0)
    {
        const auto elapsed = std::chrono::duration<double> (std::chrono::steady_clock::now() - polledAt).count();
        const auto pos = s.beatFrac + elapsed * s.bpm / 60.0;
        const auto whole = (int) std::floor (pos);
        s.beat = (s.beat + whole) % s.beatsPerBar;
        s.beatFrac = pos - whole;
    }
    s.barPhase = (s.beat + s.beatFrac) / s.beatsPerBar;
    return s;
}

void TempoPoller::loop()
{
    ix::HttpClient http;
    while (running.load())
    {
        const auto started = std::chrono::steady_clock::now();
        std::string base;
        {
            std::lock_guard<std::mutex> l (lock);
            base = url;
        }

        TempoState s;
        if (! base.empty())
        {
            auto args = http.createRequest();
            args->connectTimeout = 1;
            args->transferTimeout = 1;
            const auto response = http.get (base + "/tempo/status", args);
            if (response && response->statusCode == 200)
            {
                const auto json = juce::JSON::parse (juce::String (response->body));
                s.reachable = true;
                s.playing = (bool) json["playing"];
                s.bpm = json["heard_bpm"].isVoid() ? 0.0 : (double) json["heard_bpm"];
                s.beatsPerBar = json["beats_per_bar"].isInt() || json["beats_per_bar"].isDouble()
                                    ? juce::jlimit (1, 16, (int) json["beats_per_bar"])
                                    : 4;
                s.beatKnown = ! json["beat"].isVoid() && ! json["beat_frac"].isVoid();
                if (s.beatKnown)
                {
                    s.beat = (int) json["beat"] % s.beatsPerBar;
                    s.beatFrac = (double) json["beat_frac"];
                }
            }
        }
        {
            std::lock_guard<std::mutex> l (lock);
            polled = s;
            polledAt = std::chrono::steady_clock::now();  // when the answer arrived
        }
        std::this_thread::sleep_until (started + pollEvery);
    }
}
} // namespace sc
