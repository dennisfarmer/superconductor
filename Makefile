# SC Playback (playback/README.md)
#
#   make playback-build   build the plugins (VST3 / AU / LV2) and the Standalone app
#   make playback         build if needed, then open the Standalone app

BUILD_DIR := playback/build
APP_DIR := $(BUILD_DIR)/plugin/SCPlayback_artefacts/Release/Standalone

.PHONY: playback-build playback

playback-build:
	cmake -S playback -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=Release
	cmake --build $(BUILD_DIR) --config Release --parallel

playback: playback-build
ifeq ($(shell uname),Darwin)
	open "$(APP_DIR)/SC Playback.app"
else
	"$(APP_DIR)/SC Playback"
endif
