#include "PluginEditor.h"

namespace
{
constexpr int rowHeight = 24, gap = 6, labelWidth = 110, margin = 12;
const int bandChoices[] = { 4, 8, 16, 32 };
const int rateChoices[] = { 15, 30, 60, 120 };
} // namespace

SCPlaybackEditor::SCPlaybackEditor (SCPlaybackProcessor& p) : AudioProcessorEditor (&p), processor (p)
{
    addField (urlLabel, "Stream", url);
    addField (hostLabel, "OSC host", host);
    addField (portLabel, "OSC port", port);
    addField (serverLabel, "MRT2 server", server);
    port.setInputRestrictions (5, "0123456789");

    oscLabel.setText ("OSC", juce::dontSendNotification);
    addAndMakeVisible (oscLabel);
    oscEnabled.onClick = [this] { apply(); };
    addAndMakeVisible (oscEnabled);

    bandsLabel.setText ("Bands", juce::dontSendNotification);
    addAndMakeVisible (bandsLabel);
    for (auto b : bandChoices)
        bands.addItem (juce::String (b), b);
    bands.onChange = [this] { apply(); };
    addAndMakeVisible (bands);

    rateLabel.setText ("Per second", juce::dontSendNotification);
    addAndMakeVisible (rateLabel);
    for (auto r : rateChoices)
        rate.addItem (juce::String (r), r);
    rate.onChange = [this] { apply(); };
    addAndMakeVisible (rate);

    status.setJustificationType (juce::Justification::topLeft);
    status.setFont (juce::FontOptions (13.0f));
    addAndMakeVisible (status);

    load();
    setSize (520, 374);
    startTimerHz (10);
}

void SCPlaybackEditor::addField (juce::Label& label, const juce::String& text, juce::TextEditor& field)
{
    label.setText (text, juce::dontSendNotification);
    addAndMakeVisible (label);
    field.onReturnKey = [this] { apply(); };
    field.onFocusLost = [this] { apply(); };
    addAndMakeVisible (field);
}

void SCPlaybackEditor::load()
{
    const auto s = processor.getSettings();
    url.setText (s.url, false);
    oscEnabled.setToggleState (s.osc.enabled, juce::dontSendNotification);
    host.setText (s.osc.host, false);
    port.setText (juce::String (s.osc.port), false);
    bands.setSelectedId (s.osc.bands, juce::dontSendNotification);
    rate.setSelectedId (s.osc.rate, juce::dontSendNotification);
    server.setText (s.osc.serverUrl, false);
}

void SCPlaybackEditor::apply()
{
    auto s = processor.getSettings();
    s.url = url.getText().trim();
    s.osc.enabled = oscEnabled.getToggleState();
    s.osc.host = host.getText().trim().toStdString();
    s.osc.port = juce::jlimit (1, 65535, port.getText().getIntValue());
    s.osc.bands = bands.getSelectedId() > 0 ? bands.getSelectedId() : s.osc.bands;
    s.osc.rate = rate.getSelectedId() > 0 ? rate.getSelectedId() : s.osc.rate;
    s.osc.serverUrl = server.getText().trim().toStdString();
    processor.setSettings (s);
}

void SCPlaybackEditor::timerCallback()
{
    const auto st = processor.playerStatus();
    juce::String text;
    if (! st.connected)
        text << "Not connected" << (st.error.empty() ? juce::String() : ": " + juce::String (st.error)) << "\n";
    else
        text << "Connected\n";
    text << (st.playing ? "Playing" : "Waiting for audio")
         << "   buffered " << juce::String (st.bufferedMs, 0) << " ms";
    text << "\nBlocks " << juce::String (st.blocks) << "   underruns " << juce::String (st.underruns)
         << "\nOSC " << (processor.oscSending() ? "sending" : "off")
         << "   as " << processor.playerName();
    status.setText (text, juce::dontSendNotification);

    meter = juce::jmax (processor.outputPeak(), meter * 0.8f);
    repaint (getLocalBounds().removeFromBottom (margin + 10));
}

void SCPlaybackEditor::paint (juce::Graphics& g)
{
    g.fillAll (getLookAndFeel().findColour (juce::ResizableWindow::backgroundColourId));
    auto bar = getLocalBounds().reduced (margin).removeFromBottom (8).toFloat();
    g.setColour (juce::Colours::black.withAlpha (0.35f));
    g.fillRoundedRectangle (bar, 3.0f);
    g.setColour (meter > 0.98f ? juce::Colours::orangered : juce::Colours::limegreen);
    g.fillRoundedRectangle (bar.withWidth (bar.getWidth() * juce::jlimit (0.0f, 1.0f, meter)), 3.0f);
}

void SCPlaybackEditor::resized()
{
    auto area = getLocalBounds().reduced (margin);
    auto row = [&area] { auto r = area.removeFromTop (rowHeight); area.removeFromTop (gap); return r; };
    auto place = [] (juce::Rectangle<int> r, juce::Label& label, juce::Component& field)
    {
        label.setBounds (r.removeFromLeft (labelWidth));
        field.setBounds (r);
    };

    place (row(), urlLabel, url);
    place (row(), oscLabel, oscEnabled);
    place (row(), hostLabel, host);
    place (row(), portLabel, port);
    {
        auto r = row();
        auto left = r.removeFromLeft (r.getWidth() / 2);
        place (left.withTrimmedRight (gap), bandsLabel, bands);
        rateLabel.setBounds (r.removeFromLeft (80));
        rate.setBounds (r);
    }
    place (row(), serverLabel, server);
    area.removeFromTop (gap);
    area.removeFromBottom (8 + gap);
    status.setBounds (area);
}
