"""Object tracking (YOLO + BoT-SORT) and object -> music parameter mapping."""
from superconductor.object_tracking.calibration import Calibration
from superconductor.object_tracking.commands import Commands
from superconductor.object_tracking.embedder import Embedder
from superconductor.object_tracking.identity import Identity, IdentityRegistry
from superconductor.object_tracking.library import ObjectLibrary
from superconductor.object_tracking.tracker import ObjectTracker, TrackedObject
from superconductor.object_tracking.mapper import Combo, Parameter, ParameterMapper
