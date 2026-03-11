"""Bond-equity correlation tracker for MOEX portfolio (Layer 4).

MOEX bonds and equities correlate during stress (both driven by CBR/sanctions/oil).
This module computes trailing correlation and triggers risk adjustments when
correlation exceeds thresholds.

Plan requirements (Task 5.5):
- Compute trailing 60-day bond-equity correlation (IMOEX returns vs OFZ price returns)
- If correlation > 0.5: reduce combined position limit, increase cash buffer
- If correlation > 0.7: further reduce position limit, higher cash buffer
- Add bonds to correlation position limit
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal

_DEFAULT_WINDOW = 60  # 60 trading days
_HIGH_CORRELATION_THRESHOLD = Decimal("0.5")
_CRITICAL_CORRELATION_THRESHOLD = Decimal("0.7")
_MIN_OBSERVATIONS = 20  # Minimum observations for reliable correlation estimate
_DENOM_EPSILON = 1e-15  # Guard against zero-variance division

# Risk adjustment parameters
_HIGH_POSITION_LIMIT_MULTIPLIER = 0.75
_CRITICAL_POSITION_LIMIT_MULTIPLIER = 0.5
_NORMAL_POSITION_LIMIT_MULTIPLIER = 1.0
_HIGH_CASH_BUFFER_PCT = 0.05  # 5%
_CRITICAL_CASH_BUFFER_PCT = 0.10  # 10%
_NO_CASH_BUFFER = 0.0
_MIN_DATA_POINTS = 2  # Pearson needs at least 2 points


@dataclass(frozen=True)
class CorrelationStatus:
    """Snapshot of bond-equity correlation state."""

    correlation: float  # trailing 60-day Pearson correlation
    is_high: bool  # correlation > 0.5
    is_critical: bool  # correlation > 0.7
    position_limit_multiplier: float  # 1.0 = normal, 0.75 = high, 0.5 = critical
    cash_buffer_pct: float  # suggested cash buffer increase (0.0 = no increase)
    window_size: int  # actual number of observations in the window


def compute_pearson_correlation(x: list[float], y: list[float]) -> float:
    """Compute Pearson correlation coefficient between two series.

    Returns 0.0 if insufficient data (< 2 points) or zero variance in either series.
    Pure Python implementation — no numpy dependency needed for this computation.

    Uses population formula (divides by N, not N-1) consistent with the existing
    correlation code in ``finalayze.risk.correlation``.
    """
    n = min(len(x), len(y))
    if n < _MIN_DATA_POINTS:
        return 0.0

    # Trim to equal length (use last n elements)
    xa = x[-n:]
    ya = y[-n:]

    mean_x = sum(xa) / n
    mean_y = sum(ya) / n

    cov = sum((a - mean_x) * (b - mean_y) for a, b in zip(xa, ya, strict=True)) / n
    var_x = sum((a - mean_x) ** 2 for a in xa) / n
    var_y = sum((b - mean_y) ** 2 for b in ya) / n

    denom = (var_x * var_y) ** 0.5
    if denom < _DENOM_EPSILON:
        return 0.0

    corr = cov / denom

    if not math.isfinite(corr):
        return 0.0

    # Clamp to [-1, 1] to guard against floating-point overshoot
    clamped: float = max(-1.0, min(1.0, corr))
    return clamped


class BondEquityCorrelationTracker:
    """Track trailing correlation between bond and equity returns.

    Usage::

        tracker = BondEquityCorrelationTracker(window=60)
        for bar in daily_bars:
            tracker.update(bond_return=..., equity_return=...)
            status = tracker.status()
            if status.is_high:
                # Reduce position limits
                ...
    """

    def __init__(self, window: int = _DEFAULT_WINDOW) -> None:
        self._window = window
        self._bond_returns: list[float] = []
        self._equity_returns: list[float] = []

    def update(self, bond_return: float, equity_return: float) -> None:
        """Add a new daily return observation."""
        self._bond_returns.append(bond_return)
        self._equity_returns.append(equity_return)
        # Keep only the last ``window`` observations
        if len(self._bond_returns) > self._window:
            self._bond_returns = self._bond_returns[-self._window :]
            self._equity_returns = self._equity_returns[-self._window :]

    def status(self) -> CorrelationStatus:
        """Compute current correlation status with risk-adjustment parameters."""
        corr = self.correlation()
        corr_decimal = Decimal(str(corr))

        is_critical = corr_decimal > _CRITICAL_CORRELATION_THRESHOLD
        is_high = corr_decimal > _HIGH_CORRELATION_THRESHOLD

        if is_critical:
            multiplier = _CRITICAL_POSITION_LIMIT_MULTIPLIER
            buffer = _CRITICAL_CASH_BUFFER_PCT
        elif is_high:
            multiplier = _HIGH_POSITION_LIMIT_MULTIPLIER
            buffer = _HIGH_CASH_BUFFER_PCT
        else:
            multiplier = _NORMAL_POSITION_LIMIT_MULTIPLIER
            buffer = _NO_CASH_BUFFER

        return CorrelationStatus(
            correlation=corr,
            is_high=is_high,
            is_critical=is_critical,
            position_limit_multiplier=multiplier,
            cash_buffer_pct=buffer,
            window_size=len(self._bond_returns),
        )

    def correlation(self) -> float:
        """Compute trailing Pearson correlation.

        Returns 0.0 if fewer than ``_MIN_OBSERVATIONS`` data points are available.
        """
        if len(self._bond_returns) < _MIN_OBSERVATIONS:
            return 0.0
        return compute_pearson_correlation(self._bond_returns, self._equity_returns)

    def reset(self) -> None:
        """Reset all observations."""
        self._bond_returns = []
        self._equity_returns = []
