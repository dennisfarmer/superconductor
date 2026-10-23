#pragma once

// Sends the sound's features and the beat to TouchDesigner (or anything) over OSC,
// `rate` times a second (default 60), as one bundle. The analysis runs on this thread,
// never the audio thread, so a high rate costs CPU but can't make the sound stutter.
// Addresses: docs/osc.md.
//
// Only one copy of the plugin should send (a switch in the plugin), so TouchDesigner
// doesn't get every value twice.

#include <atomic>
#include <mutex>
#include <string>

#include <juce_osc/juce_osc.h>

#include "Features.h"
#include "TempoPoller.h"

namespace sc
{
class OscSender : private juce::Thread
{
public:
    explicit OscSender (FeatureTap& tap);
    ~OscSender() override;

    struct Settings
    {
        bool enabled = false;
        std::string host = "127.0.0.1";
        int port = 7000;
        int bands = 8;
        int rate = 60;  // bundles per second
        std::string serverUrl = "http://localhost:9100";  // for /tempo/status
    };
    void apply (const Settings& settings);

    bool isSending() const { return sending.load(); }
    int64_t bundlesSent() const { return sent.load(); }

private:
    void run() override;

    FeatureTap& tap;
    TempoPoller tempo;
    juce::OSCSender osc;

    std::mutex lock;
    Settings settings, pending;
    bool changed = false;
    std::atomic<bool> sending { false };
    std::atomic<int64_t> sent { 0 };
};
} // namespace sc
