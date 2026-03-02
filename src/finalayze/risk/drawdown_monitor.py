"""Rolling peak-to-trough drawdown monitor (Layer 4).

Tracks portfolio equity against a rolling peak and triggers when the
drawdown from peak exceeds a configurable threshold (default 12%).
This is a rolling monitor -- it does NOT reset on calendar boundaries.
"""

from __future__ import annotations

from decimal import Decimal

_ZERO = Decimal(0)


class DrawdownMonitor:
    """Monitors portfolio drawdown and triggers at threshold.

    A new equity peak updates the baseline.  Once triggered, the flag
    stays set until :meth:`reset` is called explicitly (e.g. at the
    start of a new backtest run).
    """

    _DEFAULT_THRESHOLD = Decimal("0.12")  # 12%

    def __init__(self, threshold: Decimal = _DEFAULT_THRESHOLD) -> None:
        self._threshold = threshold
        self._peak_equity: Decimal = _ZERO
        self._current_equity: Decimal = _ZERO
        self._triggered = False

    def update(self, current_equity: Decimal) -> bool:
        """Update with current equity.

        Returns ``True`` if drawdown threshold is breached on this call.
        """
        self._current_equity = current_equity

        if current_equity > self._peak_equity:
            self._peak_equity = current_equity
            self._triggered = False

        if self._peak_equity > _ZERO:
            drawdown = (self._peak_equity - current_equity) / self._peak_equity
            if drawdown >= self._threshold:
                self._triggered = True
                return True
        return False

    @property
    def triggered(self) -> bool:
        """Whether the drawdown threshold has been breached."""
        return self._triggered

    @property
    def current_drawdown(self) -> Decimal:
        """Current drawdown percentage from peak (as a Decimal ratio)."""
        if self._peak_equity <= _ZERO:
            return _ZERO
        return (self._peak_equity - self._current_equity) / self._peak_equity

    def reset(self) -> None:
        """Reset for new backtest run."""
        self._peak_equity = _ZERO
        self._current_equity = _ZERO
        self._triggered = False
