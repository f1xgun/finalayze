"""Rolling Kelly position sizing estimator (Layer 4)."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal

_MIN_TRADES_FOR_KELLY = 10
_MIN_KELLY_BLEND_TRADES = 50
_DEFAULT_WINDOW = 50
_DEFAULT_FRACTION = 0.25  # quarter-Kelly
_MIN_KELLY_FRACTION = Decimal("0.01")  # 1% minimum
_FIXED_FRACTIONAL = Decimal("0.01")  # 1% before enough data
_FOUR_DP = Decimal("0.0001")
_KILL_THRESHOLD = 3


@dataclass
class TradeRecord:
    """Minimal trade record for Kelly computation."""

    pnl: Decimal
    pnl_pct: Decimal


class RollingKelly:
    """Estimate Kelly fraction from a rolling window of recent trades."""

    def __init__(self, window: int = _DEFAULT_WINDOW, fraction: float = _DEFAULT_FRACTION) -> None:
        self._window = window
        self._fraction = Decimal(str(fraction))
        self._trades: deque[TradeRecord] = deque(maxlen=window)
        self._consecutive_negative_windows: int = 0
        self._should_halt: bool = False

    def update(self, trade: TradeRecord) -> None:
        """Append a trade to the rolling window."""
        self._trades.append(trade)

    @property
    def trade_count(self) -> int:
        """Number of trades currently in the window."""
        return len(self._trades)

    def _compute_raw_kelly(self) -> Decimal | None:
        """Compute raw dampened Kelly fraction from trade history.

        Returns None if there is insufficient decisive data (no wins or no losses,
        or avg_loss == 0), or if Kelly is non-positive (negative expectancy).
        Returns the dampened Kelly fraction otherwise.
        """
        wins = [t for t in self._trades if t.pnl > 0]
        losses = [t for t in self._trades if t.pnl < 0]
        if not wins or not losses:
            return None

        total_decisive = Decimal(len(wins) + len(losses))
        win_rate = Decimal(len(wins)) / total_decisive
        avg_win = sum(t.pnl_pct for t in wins) / Decimal(len(wins))
        avg_loss = abs(sum(t.pnl_pct for t in losses) / Decimal(len(losses)))

        if avg_loss == 0:
            return None

        ratio = avg_win / avg_loss
        one = Decimal(1)
        kelly = (win_rate * ratio - (one - win_rate)) / ratio

        if kelly <= 0:
            return Decimal(0)

        return (kelly * self._fraction).quantize(_FOUR_DP, rounding=ROUND_HALF_UP)

    @property
    def should_halt(self) -> bool:
        """Return True if the kill switch has been triggered.

        The kill switch activates when _compute_raw_kelly() returns 0
        (negative expectancy) for _KILL_THRESHOLD consecutive full windows.
        It resets when a positive expectancy window is observed.
        """
        return self._should_halt

    def optimal_fraction(self) -> Decimal:  # noqa: PLR0911
        """Return the dampened Kelly fraction with graduated blending.

        - <10 trades: fixed 1%
        - 10-50 trades: linear blend from fixed to pure Kelly
        - >50 trades: pure dampened Kelly
        - Floor of 1% when Kelly is positive (0 stays 0 for negative expectancy)
        - Kill switch: 3 consecutive full-window negative expectancy -> halt
        """
        # If kill switch is active, return zero immediately
        if self._should_halt:
            return Decimal(0)

        n = len(self._trades)

        if n < _MIN_TRADES_FOR_KELLY:
            return _FIXED_FRACTIONAL

        raw = self._compute_raw_kelly()
        if raw is None:
            return _FIXED_FRACTIONAL

        # Track consecutive negative-expectancy full windows for kill switch
        if raw == 0 and n >= self._window:
            self._consecutive_negative_windows += 1
            if self._consecutive_negative_windows >= _KILL_THRESHOLD:
                self._should_halt = True
                return Decimal(0)
            return _FIXED_FRACTIONAL / 2

        # Positive expectancy resets the kill switch counter
        if raw > 0:
            self._consecutive_negative_windows = 0
            self._should_halt = False

        # Negative expectancy — use reduced fixed fractional to allow recovery.
        if raw == 0:
            return _FIXED_FRACTIONAL / 2

        if n < _MIN_KELLY_BLEND_TRADES:
            # Graduated blend: w goes from 0 at 20 trades to 1 at 50 trades
            w = Decimal(n - _MIN_TRADES_FOR_KELLY) / Decimal(
                _MIN_KELLY_BLEND_TRADES - _MIN_TRADES_FOR_KELLY
            )
            blended = _FIXED_FRACTIONAL * (1 - w) + raw * w
            return max(blended, _MIN_KELLY_FRACTION).quantize(_FOUR_DP, rounding=ROUND_HALF_UP)

        # Pure Kelly with floor
        return max(raw, _MIN_KELLY_FRACTION)
