#include "OscSender.h"

#include <chrono>
#include <thread>

namespace sc
{
OscSender::OscSender (FeatureTap& t) : juce::Thread ("sc-osc"), tap (t)
{
    startThread();
}

OscSender::~OscSender()
{
    stopThread (1000);
}

void OscSender::apply (const Settings& s)
{
    std::lock_guard<std::mutex> l (lock);
    pending = s;
    changed = true;
}

void OscSender::run()
{
    Features features;
    double preparedRate = 0;
    int preparedBands = -1;
    std::vector<float> samples (Features::fftSize);
    auto last = std::chrono::steady_clock::now();
    auto next = last;  // a steady schedule: late ticks don't push the later ones back

    while (! threadShouldExit())
    {
        {
            std::lock_guard<std::mutex> l (lock);
            if (changed)
            {
                settings = pending;
                changed = false;
                osc.disconnect();
                sending = settings.enabled && osc.connect (settings.host, settings.port);
                tempo.setServer (settings.enabled ? settings.serverUrl : std::string());
            }
        }

        if (sending.load())
        {
            const auto bands = juce::jlimit (1, 64, settings.bands);
            if (tap.sampleRate() != preparedRate || bands != preparedBands)
            {
                preparedRate = tap.sampleRate();
                preparedBands = bands;
                features.prepare (preparedRate, preparedBands);
            }

            const auto now = std::chrono::steady_clock::now();
            const auto dt = std::chrono::duration<double> (now - last).count();
            last = now;

            juce::OSCBundle bundle;
            if (tap.latest (samples.data(), Features::fftSize))
            {
                const auto f = features.analyse (samples.data(), dt);
                bundle.addElement (juce::OSCMessage ("/sc/level", f.level));
                bundle.addElement (juce::OSCMessage ("/sc/rms_db", f.rmsDb));
                bundle.addElement (juce::OSCMessage ("/sc/peak", f.peak));
                juce::OSCMessage bandsMessage ("/sc/bands");
                for (auto b : f.bands)
                    bandsMessage.addFloat32 (b);
                bundle.addElement (bandsMessage);
                bundle.addElement (juce::OSCMessage ("/sc/flux", f.flux));
                bundle.addElement (juce::OSCMessage ("/sc/onset", f.onset));
            }

            const auto t = tempo.now();
            bundle.addElement (juce::OSCMessage ("/sc/tempo/reachable", (juce::int32) t.reachable));
            bundle.addElement (juce::OSCMessage ("/sc/tempo/playing", (juce::int32) t.playing));
            bundle.addElement (juce::OSCMessage ("/sc/tempo/bpm", (float) t.bpm));
            bundle.addElement (juce::OSCMessage ("/sc/tempo/beat_known", (juce::int32) t.beatKnown));
            bundle.addElement (juce::OSCMessage ("/sc/tempo/beat", (juce::int32) t.beat));
            bundle.addElement (juce::OSCMessage ("/sc/tempo/beat_frac", (float) t.beatFrac));
            bundle.addElement (juce::OSCMessage ("/sc/tempo/beats_per_bar", (juce::int32) t.beatsPerBar));
            bundle.addElement (juce::OSCMessage ("/sc/tempo/bar_phase", (float) t.barPhase));

            if (osc.send (bundle))
                sent.fetch_add (1);
        }

        next += std::chrono::microseconds (1000000 / juce::jlimit (1, 240, settings.rate));
        const auto now = std::chrono::steady_clock::now();
        if (next < now)
            next = now;  // fell behind (e.g. the machine slept): start over from now
        while (! threadShouldExit() && std::chrono::steady_clock::now() < next)
        {
            const auto remaining = std::chrono::duration_cast<std::chrono::microseconds> (next - std::chrono::steady_clock::now());
            if (remaining.count() > 1500)
                wait ((int) (remaining.count() / 1000) - 1);  // sleep, then spin the last ~1 ms
            else
                std::this_thread::yield();
        }
    }
}
} // namespace sc
