"""Ruble-oil decorrelation regime signal for MOEX market.

When the historical correlation between RUB and oil breaks down,
it signals geopolitical / sanctions stress.  This provider monitors
the rolling Pearson correlation of log returns and maps it to
MarketRegime levels:

  corr > 0.3   NORMAL    (scale=1.0, longs allowed)
  0.1 < corr <= 0.3  ELEVATED  (scale=0.5, longs allowed but suppress mean-reversion)
  corr <= 0.1  CRISIS    (scale=0.25, block new longs)
"""

from __future__ import annotations

import math
from decimal import Decimal
from typing import TYPE_CHECKING

from finalayze.risk.regime import MarketRegime, RegimeState

if TYPE_CHECKING:
    from finalayze.core.schemas import Candle

# Correlation thresholds
_CORR_NORMAL_LOWER = 0.3
_CORR_ELEVATED_LOWER = 0.1

# Default rolling window for correlation
_DEFAULT_WINDOW = 60

# Minimum observations for a meaningful Pearson correlation
_MIN_OBSERVATIONS = 2

# Near-zero denominator guard for Pearson calculation
_DENOM_EPSILON = 1e-15

# Position scales (mirror regime.py constants)
_SCALE_NORMAL = Decimal("1.0")
_SCALE_ELEVATED = Decimal("0.5")
_SCALE_CRISIS = Decimal("0.25")


def compute_rub_oil_correlation(
    rub_candles: list[Candle],
    oil_candles: list[Candle],
    window: int = _DEFAULT_WINDOW,
) -> float:
    """Compute Pearson correlation of log returns between RUB and oil series.

    Args:
        rub_candles: Candle series for USDRUB (or similar RUB proxy).
        oil_candles: Candle series for Brent (or similar oil benchmark).
        window: Number of return observations to use.

    Returns:
        Pearson correlation coefficient in [-1, 1].
        Returns 1.0 (assume normal) if insufficient data.
    """
    # Need window+1 candles to compute `window` returns
    min_len = min(len(rub_candles), len(oil_candles))
    if min_len < window + 1:
        return 1.0

    # Take the last window+1 candles from each series
    rub_recent = rub_candles[-(window + 1) :]
    oil_recent = oil_candles[-(window + 1) :]

    # Compute log returns
    rub_returns: list[float] = []
    oil_returns: list[float] = []
    for i in range(1, window + 1):
        rub_prev = float(rub_recent[i - 1].close)
        rub_curr = float(rub_recent[i].close)
        oil_prev = float(oil_recent[i - 1].close)
        oil_curr = float(oil_recent[i].close)

        if rub_prev <= 0 or rub_curr <= 0 or oil_prev <= 0 or oil_curr <= 0:
            continue

        rub_returns.append(math.log(rub_curr / rub_prev))
        oil_returns.append(math.log(oil_curr / oil_prev))

    n = len(rub_returns)
    if n < _MIN_OBSERVATIONS:
        return 1.0

    # Pearson correlation
    mean_r = sum(rub_returns) / n
    mean_o = sum(oil_returns) / n

    cov = sum((rub_returns[i] - mean_r) * (oil_returns[i] - mean_o) for i in range(n))
    var_r = sum((r - mean_r) ** 2 for r in rub_returns)
    var_o = sum((o - mean_o) ** 2 for o in oil_returns)

    denom = math.sqrt(var_r * var_o)
    if denom < _DENOM_EPSILON:
        return 1.0

    return cov / denom


class RubOilRegimeSignal:
    """Regime provider based on ruble-oil decorrelation.

    Conforms to the ``RegimeProvider`` protocol.  The ``candles`` argument
    to ``get_regime`` is the asset candle series (e.g. GAZP), but internally
    the provider uses its own stored RUB and oil candle data.

    Args:
        rub_candles: USDRUB (or equivalent) candle history.
        oil_candles: Brent (or equivalent) candle history.
        window: Rolling correlation window (default 60).
    """

    def __init__(
        self,
        rub_candles: list[Candle],
        oil_candles: list[Candle],
        window: int = _DEFAULT_WINDOW,
    ) -> None:
        self._rub_candles = rub_candles
        self._oil_candles = oil_candles
        self._window = window

    def get_regime(
        self,
        candles: list[Candle],  # noqa: ARG002
        bar_index: int,  # noqa: ARG002
    ) -> RegimeState:
        """Determine MOEX regime from ruble-oil correlation.

        Args:
            candles: Asset candle history (unused -- kept for protocol).
            bar_index: Current bar index (unused -- kept for protocol).

        Returns:
            RegimeState reflecting the current ruble-oil correlation regime.
        """
        corr = compute_rub_oil_correlation(
            self._rub_candles,
            self._oil_candles,
            window=self._window,
        )

        if corr <= _CORR_ELEVATED_LOWER:
            return RegimeState(
                regime=MarketRegime.CRISIS,
                allow_new_longs=False,
                position_scale=_SCALE_CRISIS,
            )

        if corr <= _CORR_NORMAL_LOWER:
            return RegimeState(
                regime=MarketRegime.ELEVATED,
                allow_new_longs=True,
                position_scale=_SCALE_ELEVATED,
            )

        return RegimeState(
            regime=MarketRegime.NORMAL,
            allow_new_longs=True,
            position_scale=_SCALE_NORMAL,
        )
