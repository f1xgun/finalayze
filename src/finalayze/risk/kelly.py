"""Rolling Kelly position sizing estimator (Layer 4)."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from decimal import Decimal

_MIN_TRADES_FOR_KELLY = 20
_DEFAULT_WINDOW = 50
_DEFAULT_FRACTION = 0.25  # quarter-Kelly
_MIN_KELLY_FRACTION = Decimal("0.005")  # 0.5% minimum
_FIXED_FRACTIONAL = Decimal("0.01")  # 1% before enough data


@dataclass
class TradeRecord:
    """Minimal trade record for Kelly computation."""

    pnl: Decimal
    pnl_pct: Decimal


class RollingKelly:
    """Estimate Kelly fraction from a rolling window of recent trades."""

    def __init__(self, window: int = _DEFAULT_WINDOW, fraction: float = _DEFAULT_FRACTION) -> None:
        self._window = window
        self._fraction = fraction
        self._trades: deque[TradeRecord] = deque(maxlen=window)

    def update(self, trade: TradeRecord) -> None:
        """Append a trade to the rolling window."""
        self._trades.append(trade)

    @property
    def trade_count(self) -> int:
        """Number of trades currently in the window."""
        return len(self._trades)

    def optimal_fraction(self) -> Decimal:
        """Return the dampened Kelly fraction, or fixed fractional if insufficient data."""
        if len(self._trades) < _MIN_TRADES_FOR_KELLY:
            return _FIXED_FRACTIONAL

        # Exclude break-even trades (pnl == 0) from win/loss classification
        wins = [t for t in self._trades if t.pnl > 0]
        losses = [t for t in self._trades if t.pnl < 0]
        if not wins or not losses:
            return _FIXED_FRACTIONAL

        total_decisive = len(wins) + len(losses)
        win_rate = len(wins) / total_decisive
        avg_win = sum(float(t.pnl_pct) for t in wins) / len(wins)
        avg_loss = abs(sum(float(t.pnl_pct) for t in losses) / len(losses))

        if avg_loss == 0:
            return _FIXED_FRACTIONAL

        ratio = avg_win / avg_loss
        kelly = (win_rate * ratio - (1 - win_rate)) / ratio

        if kelly <= 0:
            return _MIN_KELLY_FRACTION

        return Decimal(str(round(kelly * self._fraction, 4)))
