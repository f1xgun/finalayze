"""Yield-based stop-loss for bond positions (Layer 4).

Bond stops are based on yield movement, not price movement.
If YTM rises by more than the stop threshold above entry YTM,
the position is stopped out.

This is more appropriate than ATR-based stops for bonds because
bond price moves are driven by yield changes, and yield changes
are more predictable (CBR meetings, macro data).
"""

from __future__ import annotations

from decimal import Decimal


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
        yield_change_bps = (current_ytm_pct - entry_ytm_pct) * Decimal(100)
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
        yield_change_bps = (current_ytm_pct - entry_ytm_pct) * Decimal(100)
        return self._threshold_bps - yield_change_bps


# Preset stop-loss instances per layer
STRATEGIC_YIELD_STOP = YieldStop(threshold_bps=50)
TACTICAL_YIELD_STOP = YieldStop(threshold_bps=30)
