"""Market regime detection and provider protocol.

Defines MarketRegime enum, RegimeState dataclass, and the RegimeProvider
protocol used by BacktestEngine to gate entries based on market conditions.

VIX thresholds:
  <15  LOW_VOL   (scale=1.0)
  15-20 NORMAL   (scale=1.0)
  20-30 ELEVATED (scale=0.5)
  >30  CRISIS    (scale=0.25)

MOEX realized vol thresholds:
  <25% LOW_VOL/NORMAL
  25-40% ELEVATED
  >40% CRISIS
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from enum import StrEnum
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from datetime import datetime

    from finalayze.core.schemas import Candle

# VIX thresholds
_VIX_LOW_VOL_UPPER = Decimal(15)
_VIX_NORMAL_UPPER = Decimal(20)
_VIX_ELEVATED_UPPER = Decimal(30)

# VIX momentum threshold: if current - 5day_sma > this, upgrade regime
_VIX_MOMENTUM_THRESHOLD = Decimal(5)
_VIX_MOMENTUM_WINDOW = 5

# MOEX realized vol thresholds (annualized)
_MOEX_NORMAL_UPPER = Decimal("0.25")
_MOEX_ELEVATED_UPPER = Decimal("0.40")

# Annualization factor for daily returns
_ANNUALIZATION_FACTOR = Decimal(252)
_SIX_DP = Decimal("0.000001")

# Position scales per regime
_SCALE_LOW_VOL = Decimal("1.0")
_SCALE_NORMAL = Decimal("1.0")
_SCALE_ELEVATED = Decimal("0.5")
_SCALE_CRISIS = Decimal("0.25")


class MarketRegime(StrEnum):
    """Broad market regime classification."""

    LOW_VOL = "low_vol"
    NORMAL = "normal"
    ELEVATED = "elevated"
    CRISIS = "crisis"


@dataclass(frozen=True, slots=True)
class RegimeState:
    """Snapshot of current market regime with trading constraints.

    Attributes:
        regime: The current market regime classification.
        allow_new_longs: Whether new long positions are permitted.
        position_scale: Multiplier for position sizing (0.10 to 1.0).
        vix_value: The VIX value used to compute this state (None if unavailable).
    """

    regime: MarketRegime
    allow_new_longs: bool
    position_scale: Decimal
    vix_value: Decimal | None = None

    @staticmethod
    def normal() -> RegimeState:
        """Convenience factory for a normal regime state."""
        return RegimeState(
            regime=MarketRegime.NORMAL,
            allow_new_longs=True,
            position_scale=_SCALE_NORMAL,
        )

    @staticmethod
    def crisis() -> RegimeState:
        """Convenience factory for a crisis regime state."""
        return RegimeState(
            regime=MarketRegime.CRISIS,
            allow_new_longs=False,
            position_scale=Decimal("0.10"),
        )


def compute_regime_state(
    vix_value: Decimal,
    sma200_above: bool = True,
) -> RegimeState:
    """Compute regime state from VIX value and SMA200 trend.

    Logic:
      - CRISIS: always block longs
      - ELEVATED + below SMA200: block longs
      - Otherwise: allow longs

    Args:
        vix_value: Current VIX level (or realized vol proxy for MOEX).
        sma200_above: Whether price is above the 200-day SMA.

    Returns:
        A RegimeState with regime, position_scale, and allow_new_longs set.
    """
    if vix_value > _VIX_ELEVATED_UPPER:
        regime = MarketRegime.CRISIS
        scale = _SCALE_CRISIS
    elif vix_value > _VIX_NORMAL_UPPER:
        regime = MarketRegime.ELEVATED
        scale = _SCALE_ELEVATED
    elif vix_value >= _VIX_LOW_VOL_UPPER:
        regime = MarketRegime.NORMAL
        scale = _SCALE_NORMAL
    else:
        regime = MarketRegime.LOW_VOL
        scale = _SCALE_LOW_VOL

    # allow_longs logic
    if regime == MarketRegime.CRISIS or (regime == MarketRegime.ELEVATED and not sma200_above):
        allow_longs = False
    else:
        allow_longs = True

    return RegimeState(
        regime=regime,
        allow_new_longs=allow_longs,
        position_scale=scale,
        vix_value=vix_value,
    )


class RegimeProvider(Protocol):
    """Protocol for objects that can determine the current market regime.

    Implementations receive the candle history up to the current bar
    (no look-ahead) and return a RegimeState describing trading constraints.
    """

    def get_regime(self, candles: list[Candle], bar_index: int) -> RegimeState: ...


class StaticRegimeProvider:
    """Always returns a fixed regime state. Useful for testing."""

    def __init__(self, state: RegimeState) -> None:
        self._state = state

    def get_regime(
        self,
        candles: list[Candle],  # noqa: ARG002
        bar_index: int,  # noqa: ARG002
    ) -> RegimeState:
        """Return the pre-configured static regime state."""
        return self._state


def compute_realized_vol(
    candles: list[Candle],
    window: int = 20,
) -> Decimal:
    """Compute annualized realized volatility from daily close-to-close returns.

    Args:
        candles: List of candles ordered chronologically.
        window: Number of daily returns to use.

    Returns:
        Annualized volatility as Decimal, or Decimal(0) if insufficient data.
    """
    # Need at least window+1 candles to compute `window` returns
    if len(candles) < window + 1:
        return Decimal(0)

    # Take the last window+1 candles
    recent = candles[-(window + 1) :]
    returns: list[Decimal] = []
    for i in range(1, len(recent)):
        prev_close = recent[i - 1].close
        if prev_close == 0:
            continue
        daily_ret = (recent[i].close - prev_close) / prev_close
        returns.append(daily_ret)

    if not returns:
        return Decimal(0)

    n = Decimal(len(returns))
    mean = sum(returns) / n
    variance = sum((r - mean) ** 2 for r in returns) / n

    daily_vol = Decimal(str(math.sqrt(float(variance))))
    annualized = daily_vol * Decimal(str(math.sqrt(float(_ANNUALIZATION_FACTOR))))
    return annualized.quantize(_SIX_DP, rounding=ROUND_HALF_UP)


def compute_moex_regime_state(
    realized_vol: Decimal,
) -> RegimeState:
    """Compute regime state for MOEX using realized volatility.

    MOEX thresholds (annualized):
      <25% LOW_VOL/NORMAL (scale=1.0)
      25-40% ELEVATED (scale=0.5)
      >40% CRISIS (scale=0.25)

    No SMA200 filter for MOEX. CRISIS always blocks longs.

    Args:
        realized_vol: Annualized realized volatility (e.g. 0.25 for 25%).

    Returns:
        A RegimeState with regime, position_scale, and allow_new_longs.
    """
    if realized_vol > _MOEX_ELEVATED_UPPER:
        regime = MarketRegime.CRISIS
        scale = _SCALE_CRISIS
    elif realized_vol >= _MOEX_NORMAL_UPPER:
        regime = MarketRegime.ELEVATED
        scale = _SCALE_ELEVATED
    elif realized_vol >= Decimal("0.15"):
        regime = MarketRegime.NORMAL
        scale = _SCALE_NORMAL
    else:
        regime = MarketRegime.LOW_VOL
        scale = _SCALE_LOW_VOL

    allow_longs = regime != MarketRegime.CRISIS

    return RegimeState(
        regime=regime,
        allow_new_longs=allow_longs,
        position_scale=scale,
    )


class HMMRegimeProvider:
    """Regime provider using a Hidden Markov Model with periodic retraining.

    Wraps HMMRegimeDetector and conforms to the RegimeProvider protocol.
    If insufficient data (<252 candles), returns NORMAL regime.
    Retrains periodically according to retrain_frequency (default 21 bars).
    """

    _HMM_MIN_DATA_POINTS = 252  # mirrors HMMRegimeDetector._MIN_DATA_POINTS

    def __init__(self, retrain_frequency: int = 21) -> None:
        from finalayze.risk.hmm_regime import HMMRegimeDetector  # noqa: PLC0415

        self._detector = HMMRegimeDetector(retrain_frequency=retrain_frequency)
        self._last_train_bar: int = -1

    def get_regime(self, candles: list[Candle], bar_index: int) -> RegimeState:
        """Get regime state, retraining the HMM periodically.

        Args:
            candles: Full candle history.
            bar_index: Current bar index.

        Returns:
            RegimeState from HMM prediction, or NORMAL if insufficient data.
        """
        if len(candles) < self._HMM_MIN_DATA_POINTS:
            return RegimeState.normal()

        # Use only candles up to bar_index (no look-ahead)
        available = candles[: bar_index + 1]
        if len(available) < self._HMM_MIN_DATA_POINTS:
            return RegimeState.normal()

        # Retrain periodically
        if bar_index - self._last_train_bar >= self._detector._retrain_frequency:
            self._detector.fit(available)
            self._last_train_bar = bar_index

        return self._detector.predict_regime(available)


class VIXRegimeProvider:
    """Regime provider using VIX candles for US market regime detection.

    Looks up VIX close at matching timestamp. If no VIX candle is available
    for a given bar, uses the latest available (forward-fill).

    VIX momentum: if vix_current - vix_5day_sma > 5, upgrade regime to
    at least ELEVATED.
    """

    def __init__(
        self,
        vix_candles: list[Candle],
        sma200_candles: list[Candle] | None = None,
    ) -> None:
        """Initialize with VIX candle data and optional SPY candles for SMA200.

        Args:
            vix_candles: VIX candle history, ordered chronologically.
            sma200_candles: Optional SPY candle history for SMA200 trend filter.
        """
        # Index VIX candles by timestamp for fast lookup
        self._vix_by_ts: dict[datetime, Candle] = {c.timestamp: c for c in vix_candles}
        # Keep ordered list for forward-fill
        self._vix_ordered = sorted(vix_candles, key=lambda c: c.timestamp)

        self._sma200_by_ts: dict[datetime, Candle] = {}
        self._sma200_ordered: list[Candle] = []
        if sma200_candles:
            self._sma200_by_ts = {c.timestamp: c for c in sma200_candles}
            self._sma200_ordered = sorted(sma200_candles, key=lambda c: c.timestamp)

    def _find_vix_close(self, ts: datetime) -> Decimal | None:
        """Find VIX close for the given timestamp, or forward-fill."""
        if ts in self._vix_by_ts:
            return self._vix_by_ts[ts].close

        # Forward-fill: find latest VIX candle at or before ts
        best: Candle | None = None
        for candle in self._vix_ordered:
            if candle.timestamp <= ts:
                best = candle
            else:
                break
        return best.close if best else None

    def _compute_vix_5day_sma(self, ts: datetime) -> Decimal | None:
        """Compute the 5-day SMA of VIX closes up to (not including) ts."""
        # Collect VIX closes before current timestamp
        prior_closes: list[Decimal] = []
        for candle in reversed(self._vix_ordered):
            if candle.timestamp < ts:
                prior_closes.append(candle.close)
                if len(prior_closes) >= _VIX_MOMENTUM_WINDOW:
                    break

        if len(prior_closes) < _VIX_MOMENTUM_WINDOW:
            return None

        return sum(prior_closes) / Decimal(_VIX_MOMENTUM_WINDOW)

    def _is_sma200_above(self, ts: datetime) -> bool:
        """Check if SPY price is above its 200-day SMA at the given ts."""
        if not self._sma200_ordered:
            return True  # Default to above if no SPY data

        sma_window = 200
        # Find SPY candles up to ts
        relevant: list[Candle] = [c for c in self._sma200_ordered if c.timestamp <= ts]
        if len(relevant) < sma_window:
            return True  # Insufficient data, default optimistic

        recent = relevant[-sma_window:]
        sma = sum(c.close for c in recent) / Decimal(sma_window)
        current_price = relevant[-1].close
        return current_price >= sma

    def get_regime(
        self,
        candles: list[Candle],
        bar_index: int,
    ) -> RegimeState:
        """Determine market regime from VIX at the current bar's timestamp.

        Args:
            candles: Asset candle history.
            bar_index: Index of the current bar in candles.

        Returns:
            RegimeState based on VIX level and momentum.
        """
        ts = candles[bar_index].timestamp
        vix_close = self._find_vix_close(ts)

        if vix_close is None:
            return RegimeState.normal()

        sma200_above = self._is_sma200_above(ts)

        # Compute base regime from VIX level
        state = compute_regime_state(vix_close, sma200_above=sma200_above)

        # VIX momentum upgrade: if vix_current - vix_5day_sma > 5,
        # upgrade to at least ELEVATED
        vix_sma = self._compute_vix_5day_sma(ts)
        if vix_sma is not None:
            momentum = vix_close - vix_sma
            if momentum > _VIX_MOMENTUM_THRESHOLD and state.regime in (
                MarketRegime.LOW_VOL,
                MarketRegime.NORMAL,
            ):
                state = RegimeState(
                    regime=MarketRegime.ELEVATED,
                    allow_new_longs=sma200_above,
                    position_scale=_SCALE_ELEVATED,
                    vix_value=vix_close,
                )

        return state
