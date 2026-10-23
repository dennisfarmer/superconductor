# SuperConductor in TouchDesigner (touchdesigner/README.md)
#
#   make visualize                       the stack + overlay_example_1 in its output window
#   make visualize VISUALIZER=<name>     the stack + visualizers/<name>/<name>.toe

VISUALIZER ?= overlay_example_1

.PHONY: visualize

visualize:
	touchdesigner/run_visualizer.sh $(VISUALIZER)
