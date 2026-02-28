# config/modes.py — DEPRECATED re-export kept only for legacy callers.
# Import directly from finalayze.core.modes to avoid circular dependency risk.
# This file will be removed in a future cleanup.
from __future__ import annotations

from finalayze.core.modes import ModeManager, WorkMode

__all__ = ["ModeManager", "WorkMode"]
