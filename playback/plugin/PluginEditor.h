#pragma once

#include "PluginProcessor.h"

// Settings and status. A text field applies when you press Return or click away.
class SCPlaybackEditor final : public juce::AudioProcessorEditor, private juce::Timer
{
public:
    explicit SCPlaybackEditor (SCPlaybackProcessor&);

    void paint (juce::Graphics&) override;
    void resized() override;

private:
    void timerCallback() override;
    void load();   // processor settings -> controls
    void apply();  // controls -> processor settings

    void addField (juce::Label& label, const juce::String& text, juce::TextEditor& field);

    SCPlaybackProcessor& processor;

    juce::Label urlLabel, roleLabel, oscLabel, hostLabel, portLabel, bandsLabel, rateLabel, serverLabel;
    juce::TextEditor url, host, port, server;
    juce::ComboBox role, bands, rate;
    juce::ToggleButton oscEnabled { "Send OSC (only one copy should)" };
    juce::Label status;
    float meter = 0.0f;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (SCPlaybackEditor)
};
