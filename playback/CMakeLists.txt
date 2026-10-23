# SuperConductor playback: the SC plugin (VST3 / AU / LV2) and its Standalone app.
#
#   cmake -S playback -B playback/build -DCMAKE_BUILD_TYPE=Release
#   cmake --build playback/build --config Release
#
# (or `make playback-build` from the repo root). JUCE and IXWebSocket are fetched at
# configure time, at the versions pinned below.
cmake_minimum_required(VERSION 3.22)
project(SuperConductorPlayback VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_OSX_DEPLOYMENT_TARGET "11.0" CACHE STRING "")

include(FetchContent)

FetchContent_Declare(juce
    GIT_REPOSITORY https://github.com/juce-framework/JUCE.git
    GIT_TAG 9.0.3
    GIT_SHALLOW ON)

# the WebSocket (ws://.../audio) and the HTTP poll of /tempo/status; plain ws://, no TLS
set(USE_TLS OFF CACHE BOOL "" FORCE)
set(USE_ZLIB OFF CACHE BOOL "" FORCE)
set(IXWEBSOCKET_INSTALL OFF CACHE BOOL "" FORCE)
FetchContent_Declare(ixwebsocket
    GIT_REPOSITORY https://github.com/machinezone/IXWebSocket.git
    GIT_TAG v12.0.1
    GIT_SHALLOW ON)

FetchContent_MakeAvailable(juce ixwebsocket)

add_subdirectory(shared)
add_subdirectory(plugin)
