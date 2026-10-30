#include "PluginProcessor.h"

#include "PluginEditor.h"

#include <cstdlib>

namespace
{
// `make playback` sets these. They win over the address saved with the project,
// so the app opens the address in addresses.toml.
void applyLaunchAddress (SCPlaybackProcessor::Settings& settings)
{
    if (const char* url = std::getenv ("SC_PLAYBACK_URL"); url != nullptr && url[0] != '\0')
        settings.url = juce::String (url).trim();
    if (const char* http = std::getenv ("SC_SERVER_HTTP"); http != nullptr && http[0] != '\0')
        settings.osc.serverUrl = juce::String (http).trim().toStdString();
}


// JUCE doesn't recognise TouchDesigner as a host; its executable name does
juce::String hostName()
{
    const auto exe = juce::File::getSpecialLocation (juce::File::hostApplicationPath).getFileNameWithoutExtension();
    if (exe.containsIgnoreCase ("TouchDesigner"))
        return "TouchDesigner";
    const auto known = juce::String (juce::PluginHostType().getHostDescription());
    return known != "Unknown" ? known : exe;
}
} // namespace

SCPlaybackProcessor::SCPlaybackProcessor()
    : AudioProcessor (BusesProperties().withOutput ("Output", juce::AudioChannelSet::stereo(), true))
{
    // OSC is used inside TouchDesigner, so that copy sends it unless told otherwise.
    const bool inTouchDesigner = wrapperType != wrapperType_Standalone && hostName() == "TouchDesigner";
    settings.osc.enabled = inTouchDesigner;
    applyLaunchAddress (settings);
    setSettings (settings);
}

SCPlaybackProcessor::~SCPlaybackProcessor()
{
    player.disconnect();
}

juce::String SCPlaybackProcessor::playerName() const
{
    // shown in this window's status line
    if (wrapperType == wrapperType_Standalone)
        return "SuperConductor Playback VST3 Plugin";
    return "SuperConductor Playback VST3 Plugin in " + hostName();
}

SCPlaybackProcessor::Settings SCPlaybackProcessor::getSettings() const
{
    std::lock_guard<std::mutex> l (settingsLock);
    return settings;
}

void SCPlaybackProcessor::setSettings (const Settings& s)
{
    std::lock_guard<std::mutex> l (settingsLock);
    const bool reconnect = s.url != settings.url || ! playerStatus().connected;
    settings = s;
    if (reconnect)
        player.connect (s.url.toStdString(), playerName().toStdString());
    osc.apply (s.osc);
}

void SCPlaybackProcessor::prepareToPlay (double sampleRate, int samplesPerBlock)
{
    maxBlock = juce::jmax (1, samplesPerBlock);
    player.prepare (sampleRate, maxBlock);
}

bool SCPlaybackProcessor::isBusesLayoutSupported (const BusesLayout& layouts) const
{
    return layouts.getMainOutputChannelSet() == juce::AudioChannelSet::stereo()
           && layouts.getMainInputChannelSet().isDisabled();
}

void SCPlaybackProcessor::processBlock (juce::AudioBuffer<float>& buffer, juce::MidiBuffer& midi)
{
    juce::ScopedNoDenormals noDenormals;
    midi.clear();
    if (buffer.getNumChannels() < 2)
    {
        buffer.clear();
        return;
    }

    auto* left = buffer.getWritePointer (0);
    auto* right = buffer.getWritePointer (1);
    const auto total = buffer.getNumSamples();
    // a host may hand us more than it announced in prepareToPlay
    for (int done = 0; done < total;)
    {
        const auto n = juce::jmin (maxBlock, total - done);
        player.render (left + done, right + done, n);
        done += n;
    }
    for (int ch = 2; ch < buffer.getNumChannels(); ++ch)
        buffer.clear (ch, 0, total);

    peak = juce::jmax (buffer.getMagnitude (0, 0, total), buffer.getMagnitude (1, 0, total));
}

juce::AudioProcessorEditor* SCPlaybackProcessor::createEditor()
{
    return new SCPlaybackEditor (*this);
}

void SCPlaybackProcessor::getStateInformation (juce::MemoryBlock& destData)
{
    const auto s = getSettings();
    juce::ValueTree state ("SCPlayback");
    state.setProperty ("url", s.url, nullptr);
    state.setProperty ("oscEnabled", s.osc.enabled, nullptr);
    state.setProperty ("oscHost", juce::String (s.osc.host), nullptr);
    state.setProperty ("oscPort", s.osc.port, nullptr);
    state.setProperty ("bands", s.osc.bands, nullptr);
    state.setProperty ("oscRate", s.osc.rate, nullptr);
    state.setProperty ("serverUrl", juce::String (s.osc.serverUrl), nullptr);
    if (auto xml = state.createXml())
        copyXmlToBinary (*xml, destData);
}

void SCPlaybackProcessor::setStateInformation (const void* data, int sizeInBytes)
{
    const auto xml = getXmlFromBinary (data, sizeInBytes);
    if (xml == nullptr)
        return;
    const auto state = juce::ValueTree::fromXml (*xml);
    if (! state.hasType ("SCPlayback"))
        return;
    Settings s;
    s.url = state.getProperty ("url", s.url).toString();
    s.osc.enabled = state.getProperty ("oscEnabled", s.osc.enabled);
    s.osc.host = state.getProperty ("oscHost", juce::String (s.osc.host)).toString().toStdString();
    s.osc.port = state.getProperty ("oscPort", s.osc.port);
    s.osc.bands = state.getProperty ("bands", s.osc.bands);
    s.osc.rate = state.getProperty ("oscRate", s.osc.rate);
    s.osc.serverUrl = state.getProperty ("serverUrl", juce::String (s.osc.serverUrl)).toString().toStdString();
    applyLaunchAddress (s);
    setSettings (s);
}

juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new SCPlaybackProcessor();
}
