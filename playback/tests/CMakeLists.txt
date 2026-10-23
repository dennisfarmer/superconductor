# SCPlaybackTest: playback/shared without a sound device (tests/PlaybackTest.cpp).
juce_add_console_app(SCPlaybackTest PRODUCT_NAME "SCPlaybackTest")
target_sources(SCPlaybackTest PRIVATE PlaybackTest.cpp)
target_compile_definitions(SCPlaybackTest PRIVATE JUCE_WEB_BROWSER=0 JUCE_USE_CURL=0)
target_link_libraries(SCPlaybackTest
    PRIVATE
        sc_shared
        juce::juce_audio_basics
        juce::juce_dsp
        juce::juce_osc
    PUBLIC
        juce::juce_recommended_config_flags
        juce::juce_recommended_warning_flags)
