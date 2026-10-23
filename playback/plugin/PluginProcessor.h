#pragma once

#include <juce_audio_processors/juce_audio_processors.h>

#include "OscSender.h"
#include "Player.h"

// The plugin: plays the stream on its output bus. Its settings (stream URL, clock or
// monitor, OSC) are saved with the host's project; they are not automatable parameters.
class SCPlaybackProcessor final : public juce::AudioProcessor
{
public:
    struct Settings
    {
        juce::String url = "ws://localhost:8470/audio";
        bool clock = true;
        sc::OscSender::Settings osc;
    };

    SCPlaybackProcessor();
    ~SCPlaybackProcessor() override;

    Settings getSettings() const;
    void setSettings (const Settings& settings);  // reconnects if the stream changed
    sc::Player::Status playerStatus() const { return player.status(); }
    bool oscSending() const { return osc.isSending(); }
    float outputPeak() const { return peak.load(); }
    juce::String playerName() const;

    void prepareToPlay (double sampleRate, int samplesPerBlock) override;
    void releaseResources() override {}
    bool isBusesLayoutSupported (const BusesLayout& layouts) const override;
    void processBlock (juce::AudioBuffer<float>&, juce::MidiBuffer&) override;
    using AudioProcessor::processBlock;

    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override { return true; }

    const juce::String getName() const override { return JucePlugin_Name; }
    bool acceptsMidi() const override { return true; }
    bool producesMidi() const override { return false; }
    bool isMidiEffect() const override { return false; }
    double getTailLengthSeconds() const override { return 0.0; }

    int getNumPrograms() override { return 1; }
    int getCurrentProgram() override { return 0; }
    void setCurrentProgram (int) override {}
    const juce::String getProgramName (int) override { return {}; }
    void changeProgramName (int, const juce::String&) override {}

    void getStateInformation (juce::MemoryBlock& destData) override;
    void setStateInformation (const void* data, int sizeInBytes) override;

private:
    sc::Player player;
    sc::OscSender osc { player.featureTap() };

    mutable std::mutex settingsLock;
    Settings settings;
    int maxBlock = 512;
    std::atomic<float> peak { 0.0f };

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (SCPlaybackProcessor)
};
