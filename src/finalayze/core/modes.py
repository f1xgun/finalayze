"""Work mode management for Finalayze (Layer 0 / Layer 1 boundary).

``WorkMode`` lives at Layer 0 alongside exceptions; ``ModeManager`` acts as
a thin runtime service that can be consumed by upper layers.

See docs/architecture/DEPENDENCY_LAYERS.md for layering rules.
"""

from __future__ import annotations

import os
from enum import StrEnum

from finalayze.core.exceptions import ModeError


class WorkMode(StrEnum):
    """Operating modes with progressive risk.

    DEBUG   -- read-only, no order submission
    SANDBOX -- paper trading / simulated orders
    TEST    -- automated test orders (broker sandbox)
    REAL    -- live trading with real money
    """

    DEBUG = "debug"
    SANDBOX = "sandbox"
    TEST = "test"
    REAL = "real"

    def can_submit_orders(self) -> bool:
        """Return True if this mode allows submitting orders."""
        return self != WorkMode.DEBUG

    def requires_confirmation(self) -> bool:
        """Return True if this mode requires explicit confirmation before use."""
        return self == WorkMode.REAL

    def is_real_trading(self) -> bool:
        """Return True if this mode involves real money."""
        return self == WorkMode.REAL


class ModeManager:
    """Runtime manager that tracks and validates mode transitions.

    Transitioning to ``WorkMode.REAL`` requires the environment variable
    ``FINALAYZE_REAL_CONFIRMED`` to be set to ``"true"`` (case-insensitive).
    """

    def __init__(self, initial_mode: WorkMode = WorkMode.DEBUG) -> None:
        self._mode: WorkMode = initial_mode

    @property
    def current_mode(self) -> WorkMode:
        """Return the active work mode."""
        return self._mode

    def transition_to(self, mode: WorkMode) -> None:
        """Transition to *mode*, enforcing safety guards where applicable.

        Raises:
            ModeError: When transitioning to ``WorkMode.REAL`` without the
                ``FINALAYZE_REAL_CONFIRMED=true`` environment variable.
        """
        if mode == WorkMode.REAL:
            confirmed = os.getenv("FINALAYZE_REAL_CONFIRMED", "").lower()
            if confirmed != "true":
                msg = "Set FINALAYZE_REAL_CONFIRMED=true to enable real trading"
                raise ModeError(msg)
        self._mode = mode
