"""Work mode definitions for Finalayze.

See docs/architecture/OVERVIEW.md for mode descriptions.
"""

from __future__ import annotations

from enum import StrEnum


class WorkMode(StrEnum):
    """Operating modes with progressive risk."""

    DEBUG = "debug"
    SANDBOX = "sandbox"
    TEST = "test"
    REAL = "real"
