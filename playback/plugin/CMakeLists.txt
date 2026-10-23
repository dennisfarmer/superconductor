# The SC plugin: an instrument with no input that plays the SuperConductor stream.
# VST3 (Mac / Windows / Linux), AU (Mac), LV2 (Linux), and a Standalone app (pick the
# output device: speakers, BlackHole, VB-CABLE, ...). Built plugins are copied to the
# user's plugin folders (~/Library/Audio/Plug-Ins/{VST3,Components} on a Mac).
set(SC_FORMATS VST3 Standalone)
if(APPLE)
    list(APPEND SC_FORMATS AU)
elseif(UNIX)
    list(APPEND SC_FORMATS LV2)
endif()

juce_add_plugin(SCPlayback
    COMPANY_NAME "SuperConductor"
    BUNDLE_ID edu.umich.superconductor.playback
    PLUGIN_MANUFACTURER_CODE Supc
    PLUGIN_CODE Scpb
    FORMATS ${SC_FORMATS}
    PRODUCT_NAME "SuperConductor Playback VST3 Plugin"
    IS_SYNTH TRUE
    NEEDS_MIDI_INPUT TRUE      # some hosts only put instruments that take MIDI on instrument tracks
    NEEDS_MIDI_OUTPUT FALSE
    IS_MIDI_EFFECT FALSE
    EDITOR_WANTS_KEYBOARD_FOCUS TRUE
    COPY_PLUGIN_AFTER_BUILD TRUE
    LV2URI "https://github.com/dennisfarmer/superconductor/playback"
    NEEDS_WEB_BROWSER FALSE
    NEEDS_CURL FALSE)

target_sources(SCPlayback PRIVATE PluginEditor.cpp PluginProcessor.cpp)

target_compile_definitions(SCPlayback PUBLIC
    JUCE_WEB_BROWSER=0
    JUCE_USE_CURL=0
    JUCE_VST3_CAN_REPLACE_VST2=0
    JUCE_DISPLAY_SPLASH_SCREEN=0)

target_link_libraries(SCPlayback
    PRIVATE
        sc_shared
        juce::juce_audio_utils
        juce::juce_dsp
        juce::juce_osc
    PUBLIC
        juce::juce_recommended_config_flags
        juce::juce_recommended_lto_flags
        juce::juce_recommended_warning_flags)
