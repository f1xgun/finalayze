# config/modes.py — re-export for backwards compatibility.
# WorkMode and ModeManager are now canonical in src/finalayze/core/modes.py.
from __future__ import annotations

from finalayze.core.modes import ModeManager, WorkMode

__all__ = ["ModeManager", "WorkMode"]
