# SuperConductor in TouchDesigner (touchdesigner/README.md)
#
#   make visualize                       ask where the client is, then where the music server is
#   make visualize-local                 client and music server on this computer
#   make visualize-remote                client here, music server through the SSH tunnel (port 9100)
#   make visualize-cluster               both on the cluster; this computer uses the gateway (:8080)
#   make visualize VISUALIZER=<name>     visualizers/<name>/<name>.toe
#   make touchdesigner                   only the visualizer (the stack started elsewhere), in its
#                                        output window; PERFORM=0 opens it in the editor instead

VISUALIZER ?= overlay_example_1
PERFORM ?= 1
TOE := touchdesigner/visualizers/$(VISUALIZER)/$(VISUALIZER).toe

.PHONY: visualize visualize-local visualize-remote visualize-cluster touchdesigner

visualize:
	touchdesigner/run_visualizer.sh $(VISUALIZER)

visualize-local:
	touchdesigner/run_visualizer.sh --where local $(VISUALIZER)

visualize-remote:
	touchdesigner/run_visualizer.sh --where remote $(VISUALIZER)

visualize-cluster:
	touchdesigner/run_visualizer.sh --where cluster $(VISUALIZER)

touchdesigner:
	@test -f "$(TOE)" || { echo "no visualizer at $(TOE)"; exit 1; }
	open -n -a TouchDesigner $(if $(filter 1,$(PERFORM)),--env SC_PERFORM=1) "$(CURDIR)/$(TOE)"
