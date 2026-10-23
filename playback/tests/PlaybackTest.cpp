// Tests for playback/shared, with no sound device and no DAW.
//
//   SCPlaybackTest unit
//       the queue (acks, flush) and the resampler, offline
//   SCPlaybackTest live [--url ws://localhost:8470/audio] [--monitor] [--seconds 20]
//                       [--osc-port 7000] [--server http://localhost:9100]
//       a Player against the real client, its "sound card" a thread that pulls 512
//       samples at a time at 48 kHz in real time (nothing is played). Prints the status
//       once a second; exits 1 if it never played or had underruns while the clock.

#include <chrono>
#include <cmath>
#include <iostream>
#include <thread>

#include <juce_core/juce_core.h>

#include "AudioQueue.h"
#include "OscSender.h"
#include "Player.h"
#include "Resampler.h"

namespace
{
int failures = 0;

void check (bool ok, const char* what)
{
    std::cout << (ok ? "  ok   " : "  FAIL ") << what << "\n";
    if (! ok)
        ++failures;
}

std::vector<float> block (int frames, float value)
{
    return std::vector<float> ((size_t) frames * 2, value);
}

int unitTests()
{
    std::cout << "AudioQueue\n";
    {
        sc::AudioQueue q (4096);
        auto b = block (100, 0.5f);
        q.push (b.data(), 100);
        q.push (b.data(), 100);
        check (q.available() == 200, "two blocks queued");
        std::vector<float> l (300), r (300);
        check (q.peek (l.data(), r.data(), 300) == 200 && l[0] == 0.5f && r[199] == 0.5f, "peek copies both channels");
        q.consume (150);
        check (q.blocksPlayed() == 1, "a block counts as played once the reader passes its end");
        q.consume (50);
        check (q.blocksPlayed() == 2 && q.available() == 0, "second block played");

        q.push (b.data(), 100);
        q.flush();
        check (q.available() == 0, "flush empties the queue");
        q.push (b.data(), 100);
        check (q.available() == 100, "blocks after a flush are kept");
        q.consume (100);
        check (q.blocksPlayed() == 3, "flushed blocks are not counted as played");
    }

    std::cout << "Resampler\n";
    for (double hostRate : { 48000.0, 44100.0, 96000.0 })
    {
        sc::AudioQueue q (1 << 18);
        const int seconds = 2, inFrames = 48000 * seconds;
        std::vector<float> sine ((size_t) inFrames * 2);
        for (int i = 0; i < inFrames; ++i)
            sine[(size_t) (2 * i)] = sine[(size_t) (2 * i + 1)] = std::sin (2.0f * 3.14159265f * 1000.0f * (float) i / 48000.0f);
        q.push (sine.data(), inFrames);

        sc::Resampler rs;
        rs.prepare (512, 48000.0 / hostRate);
        std::vector<float> l (512), r (512);
        int produced = 0, crossings = 0;
        float last = 0;
        while (true)
        {
            const auto got = rs.read (q, l.data(), r.data(), 512, 48000.0 / hostRate);
            for (int i = 0; i < got; ++i)
            {
                if (last < 0 && l[(size_t) i] >= 0)
                    ++crossings;
                last = l[(size_t) i];
            }
            produced += got;
            if (got < 512)
                break;
        }
        const auto expectedFrames = hostRate * seconds;
        const auto measuredHz = crossings / (double) seconds;
        const auto label = juce::String (hostRate / 1000.0, 1) + " kHz: " + juce::String (produced) + " samples, 1 kHz sine reads "
                           + juce::String (measuredHz, 1) + " Hz";
        check (std::abs (produced - expectedFrames) < 16 && std::abs (measuredHz - 1000) < 2, label.toRawUTF8());
    }

    std::cout << (failures == 0 ? "all passed\n" : "FAILED\n");
    return failures == 0 ? 0 : 1;
}

int liveTest (const juce::StringArray& args)
{
    auto option = [&args] (const juce::String& name, const juce::String& fallback)
    {
        const auto i = args.indexOf (name);
        return i >= 0 && i + 1 < args.size() ? args[i + 1] : fallback;
    };
    const auto url = option ("--url", "ws://localhost:8470/audio").toStdString();
    const bool clock = ! args.contains ("--monitor");
    const auto seconds = option ("--seconds", "20").getIntValue();

    sc::Player player;
    player.prepare (48000.0, 512);
    player.connect (url, clock, clock ? "SCPlaybackTest (clock)" : "SCPlaybackTest (monitor)");

    sc::OscSender osc (player.featureTap());
    sc::OscSender::Settings oscSettings;
    oscSettings.enabled = args.contains ("--osc-port");
    oscSettings.port = option ("--osc-port", "7000").getIntValue();
    oscSettings.serverUrl = option ("--server", "http://localhost:9100").toStdString();
    osc.apply (oscSettings);

    // the "sound card": 512 samples every 512 / 48000 s, on a steady schedule
    std::atomic<bool> running { true };
    std::atomic<double> peak { 0.0 };
    std::thread card ([&]
    {
        std::vector<float> l (512), r (512);
        const auto period = std::chrono::nanoseconds ((int64_t) (512 * 1e9 / 48000));
        auto next = std::chrono::steady_clock::now();
        while (running.load())
        {
            player.render (l.data(), r.data(), 512);
            for (auto v : l)
                peak = std::max (peak.load(), (double) std::abs (v));
            next += period;
            std::this_thread::sleep_until (next);
        }
    });

    bool everPlayed = false;
    int64_t underrunsAtStart = -1;
    for (int s = 1; s <= seconds; ++s)
    {
        std::this_thread::sleep_for (std::chrono::seconds (1));
        const auto st = player.status();
        everPlayed = everPlayed || st.playing;
        if (st.playing && underrunsAtStart < 0)
            underrunsAtStart = st.underruns;
        std::cout << s << "s  " << (st.connected ? (st.clock ? "clock  " : "monitor") : "offline")
                  << "  clock=" << (st.clockName.empty() ? "-" : st.clockName)
                  << "  playing=" << st.playing << "  blocks=" << st.blocks
                  << "  buffered=" << juce::String (st.bufferedMs, 0) << "ms"
                  << "  speed=" << juce::String (st.driftPercent, 3) << "%"
                  << "  underruns=" << st.underruns << "  peak=" << juce::String (peak.exchange (0.0), 3)
                  << "  osc=" << osc.bundlesSent()
                  << (st.error.empty() ? "" : "  error=" + st.error) << "\n";
    }
    running = false;
    card.join();
    player.disconnect();

    const auto st = player.status();
    const auto underruns = underrunsAtStart < 0 ? 0 : st.underruns - underrunsAtStart;
    std::cout << (everPlayed ? "played" : "never played") << ", " << underruns << " underruns while playing\n";
    return everPlayed && (! clock || underruns == 0) ? 0 : 1;
}
} // namespace

int main (int argc, char** argv)
{
    juce::StringArray args;
    for (int i = 1; i < argc; ++i)
        args.add (argv[i]);
    if (args.isEmpty() || args[0] == "unit")
        return unitTests();
    if (args[0] == "live")
        return liveTest (args);
    std::cerr << "usage: SCPlaybackTest unit | live [--url URL] [--monitor] [--seconds N] [--osc-port P] [--server URL]\n";
    return 2;
}
