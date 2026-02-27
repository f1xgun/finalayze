"""Daily and weekly loss limit tracker (Layer 4).

Tracks equity drawdowns within a day and across a rolling week. When either
threshold is breached, the tracker signals that new entries should be
suppressed until the cooldown period expires.

See docs/architecture/DEPENDENCY_LAYERS.md for layering rules.
"""

from __future__ import annotations

from datetime import datetime  # noqa: TC003
from decimal import Decimal

_DEFAULT_DAILY_LIMIT_PCT = 3.0
_DEFAULT_WEEKLY_LIMIT_PCT = 5.0
_DEFAULT_COOLDOWN_DAYS = 1


class LossLimitTracker:
    """Track daily and weekly losses, halting trading when limits are breached."""

    def __init__(
        self,
        daily_loss_limit_pct: float = _DEFAULT_DAILY_LIMIT_PCT,
        weekly_loss_limit_pct: float = _DEFAULT_WEEKLY_LIMIT_PCT,
        cooldown_days: int = _DEFAULT_COOLDOWN_DAYS,
    ) -> None:
        self._daily_limit = Decimal(str(daily_loss_limit_pct)) / Decimal(100)
        self._weekly_limit = Decimal(str(weekly_loss_limit_pct)) / Decimal(100)
        self._cooldown_days = cooldown_days

        self._day_start_equity: Decimal = Decimal(0)
        self._week_start_equity: Decimal = Decimal(0)
        self._halted_until: datetime | None = None

    def reset_day(self, dt: datetime, equity: Decimal) -> None:
        """Reset the daily baseline at the start of a new trading day."""
        self._day_start_equity = equity
        # Clear halt if cooldown has passed
        if self._halted_until is not None and dt >= self._halted_until:
            self._halted_until = None

    def reset_week(self, dt: datetime, equity: Decimal) -> None:  # noqa: ARG002
        """Reset the weekly baseline at the start of a new trading week."""
        self._week_start_equity = equity

    def is_halted(self, dt: datetime, current_equity: Decimal) -> bool:
        """Check if trading should be halted due to loss limits.

        Args:
            dt: Current timestamp.
            current_equity: Current portfolio equity.

        Returns:
            True if trading should be halted.
        """
        # Check existing cooldown
        if self._halted_until is not None and dt < self._halted_until:
            return True

        # Check daily loss
        if self._day_start_equity > 0:
            daily_loss = (self._day_start_equity - current_equity) / self._day_start_equity
            if daily_loss >= self._daily_limit:
                return True

        # Check weekly loss
        if self._week_start_equity > 0:
            weekly_loss = (self._week_start_equity - current_equity) / self._week_start_equity
            if weekly_loss >= self._weekly_limit:
                return True

        return False
