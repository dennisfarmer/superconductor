# SuperConductor in TouchDesigner (touchdesigner/README.md)
#
#   make touchdesigner-build             optional: rebuild touchdesigner/superconductor_inputs.tox from
#                                        build_inputs.py (the committed .tox is already built)
#   make visualize                       check the stack (cluster, through the SSH tunnel), open the
#                                        Playback app, the two pages, and the visualizer
#   make visualize VISUALIZER=<name>     visualizers/<name>/<name>.toe
#   make touchdesigner                   only the visualizer, in its output window;
#                                        PERFORM=0 opens it in the editor instead

VISUALIZER ?= overlay_example_1
PERFORM ?= 1
TOE := touchdesigner/visualizers/$(VISUALIZER)/$(VISUALIZER).toe

.PHONY: touchdesigner-build visualize touchdesigner

touchdesigner-build:
	python3 touchdesigner/tools/build_inputs_tox.py

visualize:
	touchdesigner/run_visualizer.sh $(VISUALIZER)

touchdesigner:
	@test -f "$(TOE)" || { echo "no visualizer at $(TOE)"; exit 1; }
	open -n -a TouchDesigner $(if $(filter 1,$(PERFORM)),--env SC_PERFORM=1) "$(CURDIR)/$(TOE)"
