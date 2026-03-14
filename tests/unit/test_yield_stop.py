"""Unit tests for yield-based bond stop-loss."""

from __future__ import annotations

from decimal import Decimal

from finalayze.risk.yield_stop import (
    REGIME_MULTIPLIERS,
    STRATEGIC_YIELD_STOP,
    TACTICAL_YIELD_STOP,
    YieldStop,
)

# Regime constants matching CBRRegime IntEnum values
DOVISH = 0
NEUTRAL = 1
HAWKISH = 2


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


class TestRegimeMultipliers:
    """Tests for REGIME_MULTIPLIERS constant."""

    def test_dovish_multiplier(self) -> None:
        """DOVISH regime has 0.8x multiplier (tighter stops)."""
        assert REGIME_MULTIPLIERS[DOVISH] == Decimal("0.8")

    def test_neutral_multiplier(self) -> None:
        """NEUTRAL regime has 1.0x multiplier (default)."""
        assert REGIME_MULTIPLIERS[NEUTRAL] == Decimal("1.0")

    def test_hawkish_multiplier(self) -> None:
        """HAWKISH regime has 2.5x multiplier (wider stops)."""
        assert REGIME_MULTIPLIERS[HAWKISH] == Decimal("2.5")

    def test_all_three_regimes_present(self) -> None:
        """All three regime levels are defined."""
        expected_regimes = 3
        assert len(REGIME_MULTIPLIERS) == expected_regimes


class TestIsStoppedWithRegime:
    """Tests for YieldStop.is_stopped_with_regime."""

    def test_neutral_regime_matches_base_behavior(self) -> None:
        """NEUTRAL regime (1.0x) should behave identically to is_stopped.

        50bps threshold, 60bps move -> stopped in both methods.
        """
        stop = YieldStop(threshold_bps=50)
        entry = Decimal("14.50")
        current = Decimal("15.10")  # 60bps rise

        assert stop.is_stopped(entry, current) is True
        assert stop.is_stopped_with_regime(entry, current, regime=NEUTRAL) is True

    def test_hawkish_prevents_premature_stop(self) -> None:
        """HAWKISH (2.5x) widens 50bps threshold to 125bps.

        60bps move would trigger base stop, but not hawkish-adjusted.
        yield_change_bps = (15.10 - 14.50) * 100 = 60
        adjusted_threshold = 50 * 2.5 = 125
        60 > 125 -> False (not stopped)
        """
        stop = YieldStop(threshold_bps=50)
        entry = Decimal("14.50")
        current = Decimal("15.10")  # 60bps rise

        # Base stop would trigger
        assert stop.is_stopped(entry, current) is True
        # Hawkish-adjusted does NOT trigger
        assert stop.is_stopped_with_regime(entry, current, regime=HAWKISH) is False

    def test_hawkish_triggers_on_large_move(self) -> None:
        """HAWKISH stops still trigger on very large yield moves.

        adjusted_threshold = 50 * 2.5 = 125
        yield_change_bps = (15.80 - 14.50) * 100 = 130
        130 > 125 -> True
        """
        stop = YieldStop(threshold_bps=50)
        entry = Decimal("14.50")
        current = Decimal("15.80")  # 130bps rise

        assert stop.is_stopped_with_regime(entry, current, regime=HAWKISH) is True

    def test_dovish_tightens_stop(self) -> None:
        """DOVISH (0.8x) tightens 50bps threshold to 40bps.

        yield_change_bps = (14.95 - 14.50) * 100 = 45
        adjusted_threshold = 50 * 0.8 = 40
        45 > 40 -> True (stopped with dovish)
        45 > 50 -> False (NOT stopped with base)
        """
        stop = YieldStop(threshold_bps=50)
        entry = Decimal("14.50")
        current = Decimal("14.95")  # 45bps rise

        # Base stop does NOT trigger
        assert stop.is_stopped(entry, current) is False
        # Dovish-adjusted DOES trigger
        assert stop.is_stopped_with_regime(entry, current, regime=DOVISH) is True

    def test_dovish_no_stop_below_tightened_threshold(self) -> None:
        """DOVISH with small move: no stop.

        yield_change_bps = (14.85 - 14.50) * 100 = 35
        adjusted_threshold = 50 * 0.8 = 40
        35 > 40 -> False
        """
        stop = YieldStop(threshold_bps=50)
        entry = Decimal("14.50")
        current = Decimal("14.85")  # 35bps rise

        assert stop.is_stopped_with_regime(entry, current, regime=DOVISH) is False

    def test_default_regime_is_neutral(self) -> None:
        """Calling without regime argument defaults to NEUTRAL (1.0x)."""
        stop = YieldStop(threshold_bps=50)
        entry = Decimal("14.50")
        current = Decimal("15.10")  # 60bps rise

        assert stop.is_stopped_with_regime(entry, current) is True

    def test_zero_threshold_never_stops(self) -> None:
        """Zero threshold with any regime never stops (Core carry layer)."""
        stop = YieldStop(threshold_bps=0)
        entry = Decimal("14.50")
        current = Decimal("20.00")  # massive move

        assert stop.is_stopped_with_regime(entry, current, regime=HAWKISH) is False

    def test_tactical_with_hawkish_regime(self) -> None:
        """Tactical 30bps with HAWKISH -> 75bps effective threshold.

        yield_change_bps = (15.20 - 14.50) * 100 = 70
        adjusted_threshold = 30 * 2.5 = 75
        70 > 75 -> False (not stopped)
        """
        stop = YieldStop(threshold_bps=30)
        entry = Decimal("14.50")
        current = Decimal("15.20")  # 70bps rise

        # Base 30bps stop would trigger
        assert stop.is_stopped(entry, current) is True
        # Hawkish-adjusted does NOT trigger
        assert stop.is_stopped_with_regime(entry, current, regime=HAWKISH) is False


class TestStopDistanceWithRegime:
    """Tests for YieldStop.stop_distance_with_regime."""

    def test_neutral_matches_base(self) -> None:
        """NEUTRAL regime distance should match stop_distance_bps.

        yield_change_bps = (14.70 - 14.50) * 100 = 20
        distance = 50 - 20 = 30
        """
        stop = YieldStop(threshold_bps=50)
        entry = Decimal("14.50")
        current = Decimal("14.70")  # 20bps rise

        base_distance = stop.stop_distance_bps(entry, current)
        regime_distance = stop.stop_distance_with_regime(entry, current, regime=NEUTRAL)
        assert base_distance == regime_distance

    def test_hawkish_increases_distance(self) -> None:
        """HAWKISH widens threshold, so distance is larger.

        yield_change_bps = (14.70 - 14.50) * 100 = 20
        adjusted_threshold = 50 * 2.5 = 125
        distance = 125 - 20 = 105
        """
        stop = YieldStop(threshold_bps=50)
        entry = Decimal("14.50")
        current = Decimal("14.70")  # 20bps rise

        distance = stop.stop_distance_with_regime(entry, current, regime=HAWKISH)
        expected = Decimal(105)
        assert distance == expected

    def test_dovish_decreases_distance(self) -> None:
        """DOVISH tightens threshold, so distance is smaller.

        yield_change_bps = (14.70 - 14.50) * 100 = 20
        adjusted_threshold = 50 * 0.8 = 40
        distance = 40 - 20 = 20
        """
        stop = YieldStop(threshold_bps=50)
        entry = Decimal("14.50")
        current = Decimal("14.70")  # 20bps rise

        distance = stop.stop_distance_with_regime(entry, current, regime=DOVISH)
        expected = Decimal(20)
        assert distance == expected

    def test_negative_distance_when_triggered(self) -> None:
        """Distance is negative when stop has been triggered.

        yield_change_bps = (15.10 - 14.50) * 100 = 60
        adjusted_threshold = 50 * 0.8 = 40
        distance = 40 - 60 = -20
        """
        stop = YieldStop(threshold_bps=50)
        entry = Decimal("14.50")
        current = Decimal("15.10")  # 60bps rise

        distance = stop.stop_distance_with_regime(entry, current, regime=DOVISH)
        expected = Decimal(-20)
        assert distance == expected

    def test_default_regime_is_neutral(self) -> None:
        """Calling without regime argument defaults to NEUTRAL."""
        stop = YieldStop(threshold_bps=50)
        entry = Decimal("14.50")
        current = Decimal("14.70")

        default_distance = stop.stop_distance_with_regime(entry, current)
        neutral_distance = stop.stop_distance_with_regime(entry, current, regime=NEUTRAL)
        assert default_distance == neutral_distance


class TestOriginalMethodsUnchanged:
    """Verify backward compatibility: original methods still work as before."""

    def test_is_stopped_unchanged(self) -> None:
        """Original is_stopped still uses base threshold, not regime-adjusted."""
        stop = YieldStop(threshold_bps=50)
        # 60bps move triggers base stop
        assert (
            stop.is_stopped(
                entry_ytm_pct=Decimal("14.50"),
                current_ytm_pct=Decimal("15.10"),
            )
            is True
        )
        # 40bps move does not trigger base stop
        assert (
            stop.is_stopped(
                entry_ytm_pct=Decimal("14.50"),
                current_ytm_pct=Decimal("14.90"),
            )
            is False
        )

    def test_stop_distance_bps_unchanged(self) -> None:
        """Original stop_distance_bps still uses base threshold."""
        stop = YieldStop(threshold_bps=50)
        distance = stop.stop_distance_bps(
            entry_ytm_pct=Decimal("14.50"),
            current_ytm_pct=Decimal("14.70"),
        )
        expected = Decimal(30)
        assert distance == expected

    def test_threshold_bps_property_unchanged(self) -> None:
        """threshold_bps property returns the base (non-adjusted) value."""
        stop = YieldStop(threshold_bps=50)
        expected = Decimal(50)
        assert stop.threshold_bps == expected

    def test_preset_instances_exist(self) -> None:
        """Module-level preset instances are still available."""
        expected_strategic = Decimal(50)
        expected_tactical = Decimal(30)
        assert STRATEGIC_YIELD_STOP.threshold_bps == expected_strategic
        assert TACTICAL_YIELD_STOP.threshold_bps == expected_tactical
