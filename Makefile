# SuperConductor Playback VST3 Plugin (playback/README.md)
#
#   make playback-build    build the plugins (VST3 / AU / LV2) and the Standalone app
#   make playback          build if needed, then open the app on the music server (localhost:9000)

BUILD_DIR := playback/build
APP_DIR := $(BUILD_DIR)/plugin/SCPlayback_artefacts/Release/Standalone

.PHONY: playback-build playback

playback-build:
	cmake -S playback -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=Release
	cmake --build $(BUILD_DIR) --config Release --parallel

playback: playback-build
	playback/launch.sh
