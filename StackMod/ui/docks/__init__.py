"""Docks package containing OBS-style dock widgets."""
from .modes_dock import ModesDock
from .sources_dock import SourcesDock
from .tuning_dock import TuningDock
from .overlay_dock import OverlayDock
from .controls_dock import ControlsDock

__all__ = [
    "ModesDock",
    "SourcesDock",
    "TuningDock",
    "OverlayDock",
    "ControlsDock",
]
