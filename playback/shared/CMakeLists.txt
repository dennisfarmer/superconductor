# The parts every target shares: the stream, the queue, resampling, features, tempo,
# OSC. An INTERFACE library, so its sources compile inside each target with that
# target's JUCE modules (juce_audio_basics, juce_dsp, juce_osc).
add_library(sc_shared INTERFACE)
target_sources(sc_shared INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/AudioQueue.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Features.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/OscSender.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Player.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Resampler.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/StreamClient.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/TempoPoller.cpp)
target_include_directories(sc_shared INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(sc_shared INTERFACE ixwebsocket)
