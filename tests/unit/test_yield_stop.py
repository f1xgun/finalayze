"""Unit tests for yield-based bond stop-loss."""

from __future__ import annotations

from decimal import Decimal

from finalayze.risk.yield_stop import (
    STRATEGIC_YIELD_STOP,
    TACTICAL_YIELD_STOP,
    YieldStop,
)


class TestYieldStop:
    """Tests for YieldStop.is_stopped and stop_distance_bps."""

    def test_no_stop_when_yield_unchanged(self) -> None:
        """Yield unchanged from entry -> not stopped."""
        stop = YieldStop(threshold_bps=50)
        assert (
            stop.is_stopped(
                entry_ytm_pct=Decimal("14.50"),
                current_ytm_pct=Decimal("14.50"),
            )
            is False
        )

    def test_stop_triggered_when_yield_rises_above_threshold(self) -> None:
        """Yield rises by 60bps (above 50bps threshold) -> stopped.

        yield_change_bps = (15.10 - 14.50) * 100 = 60
        60 > 50 -> True
        """
        stop = YieldStop(threshold_bps=50)
        assert (
            stop.is_stopped(
                entry_ytm_pct=Decimal("14.50"),
                current_ytm_pct=Decimal("15.10"),
            )
            is True
        )

    def test_no_stop_when_yield_drops(self) -> None:
        """Yield drops (price rises, good for longs) -> not stopped.

        yield_change_bps = (13.00 - 14.50) * 100 = -150
        -150 > 50 -> False
        """
        stop = YieldStop(threshold_bps=50)
        assert (
            stop.is_stopped(
                entry_ytm_pct=Decimal("14.50"),
                current_ytm_pct=Decimal("13.00"),
            )
            is False
        )

    def test_exact_threshold_not_triggered(self) -> None:
        """At exactly the threshold, stop is NOT triggered (strictly greater).

        yield_change_bps = (15.00 - 14.50) * 100 = 50
        50 > 50 -> False (not strictly greater)
        """
        stop = YieldStop(threshold_bps=50)
        assert (
            stop.is_stopped(
                entry_ytm_pct=Decimal("14.50"),
                current_ytm_pct=Decimal("15.00"),
            )
            is False
        )

    def test_stop_distance_positive_when_safe(self) -> None:
        """20bps move with 50bps threshold -> 30bps remaining.

        yield_change_bps = (14.70 - 14.50) * 100 = 20
        distance = 50 - 20 = 30
        """
        stop = YieldStop(threshold_bps=50)
        distance = stop.stop_distance_bps(
            entry_ytm_pct=Decimal("14.50"),
            current_ytm_pct=Decimal("14.70"),
        )
        assert distance == Decimal(30)

    def test_stop_distance_negative_when_triggered(self) -> None:
        """70bps move with 50bps threshold -> -20bps (already past).

        yield_change_bps = (15.20 - 14.50) * 100 = 70
        distance = 50 - 70 = -20
        """
        stop = YieldStop(threshold_bps=50)
        distance = stop.stop_distance_bps(
            entry_ytm_pct=Decimal("14.50"),
            current_ytm_pct=Decimal("15.20"),
        )
        assert distance == Decimal(-20)

    def test_stop_distance_zero_at_threshold(self) -> None:
        """Exactly at threshold -> distance is 0.

        yield_change_bps = (15.00 - 14.50) * 100 = 50
        distance = 50 - 50 = 0
        """
        stop = YieldStop(threshold_bps=50)
        distance = stop.stop_distance_bps(
            entry_ytm_pct=Decimal("14.50"),
            current_ytm_pct=Decimal("15.00"),
        )
        assert distance == Decimal(0)

    def test_strategic_preset_threshold(self) -> None:
        """Strategic layer uses 50bps threshold.

        49bps rise -> not stopped, 51bps rise -> stopped.
        """
        # 49bps: (14.99 - 14.50) * 100 = 49 -> not stopped
        assert (
            STRATEGIC_YIELD_STOP.is_stopped(
                entry_ytm_pct=Decimal("14.50"),
                current_ytm_pct=Decimal("14.99"),
            )
            is False
        )

        # 51bps: (15.01 - 14.50) * 100 = 51 -> stopped
        assert (
            STRATEGIC_YIELD_STOP.is_stopped(
                entry_ytm_pct=Decimal("14.50"),
                current_ytm_pct=Decimal("15.01"),
            )
            is True
        )

    def test_tactical_preset_threshold(self) -> None:
        """Tactical layer uses 30bps threshold (tighter).

        29bps rise -> not stopped, 31bps rise -> stopped.
        """
        # 29bps: (14.79 - 14.50) * 100 = 29 -> not stopped
        assert (
            TACTICAL_YIELD_STOP.is_stopped(
                entry_ytm_pct=Decimal("14.50"),
                current_ytm_pct=Decimal("14.79"),
            )
            is False
        )

        # 31bps: (14.81 - 14.50) * 100 = 31 -> stopped
        assert (
            TACTICAL_YIELD_STOP.is_stopped(
                entry_ytm_pct=Decimal("14.50"),
                current_ytm_pct=Decimal("14.81"),
            )
            is True
        )
