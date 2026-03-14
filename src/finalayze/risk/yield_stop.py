"""Yield-based stop-loss for bond positions (Layer 4).

Bond stops are based on yield movement, not price movement.
If YTM rises by more than the stop threshold above entry YTM,
the position is stopped out.

This is more appropriate than ATR-based stops for bonds because
bond price moves are driven by yield changes, and yield changes
are more predictable (CBR meetings, macro data).

Regime-adaptive stops widen or tighten the threshold based on the
CBR monetary policy regime to reduce whipsaw in hiking environments.
"""

from __future__ import annotations

from decimal import Decimal

# Regime indices matching CBRRegime IntEnum (0=DOVISH, 1=NEUTRAL, 2=HAWKISH)
_DOVISH = 0
_NEUTRAL = 1
_HAWKISH = 2

# Regime multipliers applied to base threshold_bps:
#   DOVISH  0.8x — tighter stops to protect gains in easing cycle
#   NEUTRAL 1.0x — default, no adjustment
#   HAWKISH 2.5x — much wider stops to reduce whipsaw in hiking cycle
REGIME_MULTIPLIERS: dict[int, Decimal] = {
    _DOVISH: Decimal("0.8"),
    _NEUTRAL: Decimal("1.0"),
    _HAWKISH: Decimal("2.5"),
}

_BPS_PER_PCT = Decimal(100)


class YieldStop:
    """Yield-based stop-loss for bond positions.

    Triggers when current YTM exceeds entry YTM by more than threshold.

    For Strategic layer: threshold = 50bps above entry YTM
    For Tactical layer: threshold = 30bps above entry YTM

    Args:
        threshold_bps: Stop threshold in basis points. Position is stopped
            when yield rises strictly above this number of bps from entry.
    """

    def __init__(self, threshold_bps: int = 50) -> None:
        self._threshold_bps = Decimal(threshold_bps)

    @property
    def threshold_bps(self) -> Decimal:
        """Stop threshold in basis points."""
        return self._threshold_bps

    def is_stopped(
        self,
        entry_ytm_pct: Decimal,
        current_ytm_pct: Decimal,
    ) -> bool:
        """Check if position should be stopped out.

        Args:
            entry_ytm_pct: YTM at entry as percent (e.g. 14.50).
            current_ytm_pct: Current YTM as percent (e.g. 15.20).

        Returns:
            True if yield has risen strictly above the threshold
            (meaning price has dropped enough to trigger the stop).
        """
        # threshold_bps=0 means no yield stop (used by Core carry layer)
        if self._threshold_bps <= 0:
            return False
        # YTM rising = price falling = bad for long bond position
        yield_change_bps = (current_ytm_pct - entry_ytm_pct) * _BPS_PER_PCT
        return yield_change_bps > self._threshold_bps

    def stop_distance_bps(
        self,
        entry_ytm_pct: Decimal,
        current_ytm_pct: Decimal,
    ) -> Decimal:
        """How many bps away from stop trigger.

        Args:
            entry_ytm_pct: YTM at entry as percent (e.g. 14.50).
            current_ytm_pct: Current YTM as percent (e.g. 15.20).

        Returns:
            Positive value means still safe, negative means already triggered.
        """
        yield_change_bps = (current_ytm_pct - entry_ytm_pct) * _BPS_PER_PCT
        return self._threshold_bps - yield_change_bps

    def is_stopped_with_regime(
        self,
        entry_ytm_pct: Decimal,
        current_ytm_pct: Decimal,
        regime: int = _NEUTRAL,
    ) -> bool:
        """Check if position should be stopped, adjusting for CBR regime.

        Applies a regime-dependent multiplier to the base threshold to
        reduce whipsaw in hawkish (hiking) environments and tighten
        stops in dovish (easing) environments.

        Args:
            entry_ytm_pct: YTM at entry as percent (e.g. 14.50).
            current_ytm_pct: Current YTM as percent (e.g. 15.20).
            regime: CBR regime as int (0=DOVISH, 1=NEUTRAL, 2=HAWKISH).

        Returns:
            True if yield has risen strictly above the regime-adjusted threshold.
        """
        adjusted_threshold = self._regime_adjusted_threshold(regime)
        if adjusted_threshold <= 0:
            return False
        yield_change_bps = (current_ytm_pct - entry_ytm_pct) * _BPS_PER_PCT
        return yield_change_bps > adjusted_threshold

    def stop_distance_with_regime(
        self,
        entry_ytm_pct: Decimal,
        current_ytm_pct: Decimal,
        regime: int = _NEUTRAL,
    ) -> Decimal:
        """How many bps away from the regime-adjusted stop trigger.

        Args:
            entry_ytm_pct: YTM at entry as percent (e.g. 14.50).
            current_ytm_pct: Current YTM as percent (e.g. 15.20).
            regime: CBR regime as int (0=DOVISH, 1=NEUTRAL, 2=HAWKISH).

        Returns:
            Positive value means still safe, negative means already triggered.
        """
        adjusted_threshold = self._regime_adjusted_threshold(regime)
        yield_change_bps = (current_ytm_pct - entry_ytm_pct) * _BPS_PER_PCT
        return adjusted_threshold - yield_change_bps

    def _regime_adjusted_threshold(self, regime: int) -> Decimal:
        """Apply regime multiplier to base threshold.

        Falls back to NEUTRAL (1.0x) for unknown regime values.
        """
        multiplier = REGIME_MULTIPLIERS.get(regime, REGIME_MULTIPLIERS[_NEUTRAL])
        return self._threshold_bps * multiplier


# Preset stop-loss instances per layer
STRATEGIC_YIELD_STOP = YieldStop(threshold_bps=50)
TACTICAL_YIELD_STOP = YieldStop(threshold_bps=30)
