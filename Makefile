# SuperConductor Playback VST3 Plugin (playback/README.md)
#
#   make playback-build    build the plugins (VST3 / AU / LV2) and the Standalone app
#   make playback          build if needed, then ask where the music server is, then open the app
#   make playback-local    server on this computer (port 9000)
#   make playback-remote   server through the SSH tunnel (port 9000)
#   make playback-cluster  server through the container gateway (server.localhost:8080)

BUILD_DIR := playback/build
APP_DIR := $(BUILD_DIR)/plugin/SCPlayback_artefacts/Release/Standalone

.PHONY: playback-build playback playback-local playback-remote playback-cluster

playback-build:
	cmake -S playback -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=Release
	cmake --build $(BUILD_DIR) --config Release --parallel

playback: playback-build
	playback/launch.sh

playback-local: playback-build
	playback/launch.sh --where local

playback-remote: playback-build
	playback/launch.sh --where remote

playback-cluster: playback-build
	playback/launch.sh --where cluster
