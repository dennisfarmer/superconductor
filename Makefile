# SuperConductor in TouchDesigner (touchdesigner/README.md)
#
#   make visualize                       the stack + overlay_example_1 in its output window
#   make visualize VISUALIZER=<name>     the stack + visualizers/<name>/<name>.toe
#   make touchdesigner                   only the visualizer (the stack started elsewhere), in its
#                                        output window; PERFORM=0 opens it in the editor instead

VISUALIZER ?= overlay_example_1
PERFORM ?= 1
TOE := touchdesigner/visualizers/$(VISUALIZER)/$(VISUALIZER).toe

.PHONY: visualize touchdesigner

visualize:
	touchdesigner/run_visualizer.sh $(VISUALIZER)

touchdesigner:
	@test -f "$(TOE)" || { echo "no visualizer at $(TOE)"; exit 1; }
	open -n -a TouchDesigner $(if $(filter 1,$(PERFORM)),--env SC_PERFORM=1) "$(CURDIR)/$(TOE)"
