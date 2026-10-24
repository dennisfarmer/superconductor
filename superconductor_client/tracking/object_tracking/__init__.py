"""Object tracking (YOLO + BoT-SORT) and object -> music parameter mapping."""
from superconductor_client.tracking.object_tracking.calibration import Calibration
from superconductor_client.tracking.object_tracking.embedder import Embedder
from superconductor_client.tracking.object_tracking.identity import Identity, IdentityRegistry
from superconductor_client.tracking.object_tracking.library import ObjectLibrary
from superconductor_client.tracking.object_tracking.tracker import ObjectTracker, TrackedObject
from superconductor_client.tracking.object_tracking.mapper import Combo, Parameter, ParameterMapper
