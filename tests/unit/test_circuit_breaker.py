"""Unit tests for per-market and cross-market circuit breakers."""

from __future__ import annotations

from decimal import Decimal

from finalayze.risk.circuit_breaker import CircuitBreaker, CircuitLevel, CrossMarketCircuitBreaker

# ── Constants (ruff PLR2004: no magic numbers) ──────────────────────────────
MARKET_US = "us"
MARKET_MOEX = "moex"

BASELINE = Decimal(100000)
L1_THRESHOLD = Decimal("0.05")
L2_THRESHOLD = Decimal("0.10")
L3_THRESHOLD = Decimal("0.15")

# Equity values triggering each level
# NORMAL: drawdown < 5%
EQUITY_NORMAL = Decimal(97000)  # 3% drawdown
# CAUTION: 5% <= drawdown < 10%
EQUITY_CAUTION = Decimal(94000)  # 6% drawdown
EQUITY_CAUTION_EXACT = Decimal(95000)  # exactly 5% drawdown
# HALTED: 10% <= drawdown < 15%
EQUITY_HALTED = Decimal(89000)  # 11% drawdown
EQUITY_HALTED_EXACT = Decimal(90000)  # exactly 10% drawdown
# LIQUIDATE: drawdown >= 15%
EQUITY_LIQUIDATE = Decimal(84000)  # 16% drawdown
EQUITY_LIQUIDATE_EXACT = Decimal(85000)  # exactly 15% drawdown

# For cross-market tests
CROSS_THRESHOLD = Decimal("0.10")
US_BASELINE = Decimal(50000)
MOEX_BASELINE = Decimal(50000)
COMBINED_BASELINE = US_BASELINE + MOEX_BASELINE  # 100000
# 11% combined drawdown: combined current = 89000
US_CURRENT_OK = Decimal(48000)  # 4% down
MOEX_CURRENT_TRIP = Decimal(41000)  # combined = 89000, 11% total drawdown
US_CURRENT_SAFE = Decimal(49000)
MOEX_CURRENT_SAFE = Decimal(49000)  # combined = 98000, 2% drawdown


class TestCircuitLevel:
    def test_level_values(self) -> None:
        assert CircuitLevel.NORMAL == "normal"
        assert CircuitLevel.CAUTION == "caution"
        assert CircuitLevel.HALTED == "halted"
        assert CircuitLevel.LIQUIDATE == "liquidate"


class TestCircuitBreaker:
    def _make_breaker(self) -> CircuitBreaker:
        return CircuitBreaker(
            market_id=MARKET_US,
            l1_threshold=float(L1_THRESHOLD),
            l2_threshold=float(L2_THRESHOLD),
            l3_threshold=float(L3_THRESHOLD),
        )

    def test_initial_level_is_normal(self) -> None:
        cb = self._make_breaker()
        assert cb.level == CircuitLevel.NORMAL

    def test_market_id_property(self) -> None:
        cb = self._make_breaker()
        assert cb.market_id == MARKET_US

    def test_check_normal(self) -> None:
        cb = self._make_breaker()
        level = cb.check(current_equity=EQUITY_NORMAL, baseline_equity=BASELINE)
        assert level == CircuitLevel.NORMAL

    def test_check_caution_at_l1_boundary(self) -> None:
        cb = self._make_breaker()
        level = cb.check(current_equity=EQUITY_CAUTION_EXACT, baseline_equity=BASELINE)
        assert level == CircuitLevel.CAUTION

    def test_check_caution_above_l1(self) -> None:
        cb = self._make_breaker()
        level = cb.check(current_equity=EQUITY_CAUTION, baseline_equity=BASELINE)
        assert level == CircuitLevel.CAUTION

    def test_check_halted_at_l2_boundary(self) -> None:
        cb = self._make_breaker()
        level = cb.check(current_equity=EQUITY_HALTED_EXACT, baseline_equity=BASELINE)
        assert level == CircuitLevel.HALTED

    def test_check_halted_above_l2(self) -> None:
        cb = self._make_breaker()
        level = cb.check(current_equity=EQUITY_HALTED, baseline_equity=BASELINE)
        assert level == CircuitLevel.HALTED

    def test_check_liquidate_at_l3_boundary(self) -> None:
        cb = self._make_breaker()
        level = cb.check(current_equity=EQUITY_LIQUIDATE_EXACT, baseline_equity=BASELINE)
        assert level == CircuitLevel.LIQUIDATE

    def test_check_liquidate_above_l3(self) -> None:
        cb = self._make_breaker()
        level = cb.check(current_equity=EQUITY_LIQUIDATE, baseline_equity=BASELINE)
        assert level == CircuitLevel.LIQUIDATE

    def test_check_updates_internal_level(self) -> None:
        cb = self._make_breaker()
        cb.check(current_equity=EQUITY_CAUTION, baseline_equity=BASELINE)
        assert cb.level == CircuitLevel.CAUTION

    def test_level_escalates_on_subsequent_checks(self) -> None:
        cb = self._make_breaker()
        cb.check(current_equity=EQUITY_CAUTION, baseline_equity=BASELINE)
        assert cb.level == CircuitLevel.CAUTION
        cb.check(current_equity=EQUITY_HALTED, baseline_equity=BASELINE)
        assert cb.level == CircuitLevel.HALTED
        cb.check(current_equity=EQUITY_LIQUIDATE, baseline_equity=BASELINE)
        assert cb.level == CircuitLevel.LIQUIDATE

    def test_reset_daily_clears_caution(self) -> None:
        cb = self._make_breaker()
        cb.check(current_equity=EQUITY_CAUTION, baseline_equity=BASELINE)
        assert cb.level == CircuitLevel.CAUTION
        cb.reset_daily(new_baseline=EQUITY_CAUTION)
        assert cb.level == CircuitLevel.NORMAL

    def test_reset_daily_halted_requires_two_profitable_days(self) -> None:
        """HALTED is NOT cleared by a single reset_daily -- requires 2 profitable days."""
        cb = self._make_breaker()
        cb.check(current_equity=EQUITY_HALTED, baseline_equity=BASELINE)
        assert cb.level == CircuitLevel.HALTED
        # Single profitable day (new baseline > prev baseline of 100000): not enough
        cb.reset_daily(new_baseline=BASELINE + Decimal(1000))
        assert cb.level == CircuitLevel.HALTED

    def test_reset_daily_does_not_clear_liquidate(self) -> None:
        cb = self._make_breaker()
        cb.check(current_equity=EQUITY_LIQUIDATE, baseline_equity=BASELINE)
        assert cb.level == CircuitLevel.LIQUIDATE
        cb.reset_daily(new_baseline=EQUITY_LIQUIDATE)
        assert cb.level == CircuitLevel.LIQUIDATE  # must stay LIQUIDATE

    def test_reset_manual_clears_liquidate(self) -> None:
        cb = self._make_breaker()
        cb.check(current_equity=EQUITY_LIQUIDATE, baseline_equity=BASELINE)
        assert cb.level == CircuitLevel.LIQUIDATE
        cb.reset_manual()
        assert cb.level == CircuitLevel.NORMAL

    def test_reset_manual_from_normal_stays_normal(self) -> None:
        cb = self._make_breaker()
        cb.reset_manual()
        assert cb.level == CircuitLevel.NORMAL

    def test_reset_daily_updates_baseline(self) -> None:
        """After reset_daily with a new baseline, checks use the new baseline."""
        cb = self._make_breaker()
        new_baseline = Decimal(90000)
        cb.reset_daily(new_baseline=new_baseline)
        # With 90000 baseline, EQUITY_NORMAL (97000) is above baseline -> NORMAL
        level = cb.check(current_equity=EQUITY_NORMAL, baseline_equity=new_baseline)
        assert level == CircuitLevel.NORMAL

    def test_zero_equity_is_liquidate(self) -> None:
        cb = self._make_breaker()
        level = cb.check(current_equity=Decimal(0), baseline_equity=BASELINE)
        assert level == CircuitLevel.LIQUIDATE

    def test_equity_above_baseline_is_normal(self) -> None:
        """Positive returns should not trigger any circuit breaker level."""
        cb = self._make_breaker()
        equity_above_baseline = Decimal(110000)
        level = cb.check(current_equity=equity_above_baseline, baseline_equity=BASELINE)
        assert level == CircuitLevel.NORMAL

    # ── 6A.5: L2 sticky reset (profitable-days requirement) ────────────
    def test_halted_not_cleared_by_single_profitable_day(self) -> None:
        cb = self._make_breaker()
        cb.check(current_equity=EQUITY_HALTED, baseline_equity=BASELINE)
        assert cb.level == CircuitLevel.HALTED
        # Day 1: profitable (new baseline > prev baseline of 100000)
        cb.reset_daily(new_baseline=BASELINE + Decimal(1000))
        assert cb.level == CircuitLevel.HALTED

    def test_halted_cleared_after_two_profitable_days(self) -> None:
        cb = self._make_breaker()
        cb.check(current_equity=EQUITY_HALTED, baseline_equity=BASELINE)
        assert cb.level == CircuitLevel.HALTED
        # Day 1: profitable (new baseline > prev baseline of 100000)
        first_equity = BASELINE + Decimal(1000)
        cb.reset_daily(new_baseline=first_equity)
        assert cb.level == CircuitLevel.HALTED
        # Day 2: profitable again
        second_equity = first_equity + Decimal(1000)
        cb.reset_daily(new_baseline=second_equity)
        assert cb.level == CircuitLevel.NORMAL

    def test_halted_resets_counter_on_loss_day(self) -> None:
        cb = self._make_breaker()
        cb.check(current_equity=EQUITY_HALTED, baseline_equity=BASELINE)
        assert cb.level == CircuitLevel.HALTED
        # Day 1: profitable (new baseline > prev baseline of 100000)
        first_equity = BASELINE + Decimal(1000)
        cb.reset_daily(new_baseline=first_equity)
        assert cb.level == CircuitLevel.HALTED
        # Day 2: loss (equity drops below first_equity)
        loss_equity = first_equity - Decimal(500)
        cb.reset_daily(new_baseline=loss_equity)
        assert cb.level == CircuitLevel.HALTED
        assert cb.consecutive_profitable_days == 0

    def test_caution_still_clears_immediately(self) -> None:
        cb = self._make_breaker()
        cb.check(current_equity=EQUITY_CAUTION, baseline_equity=BASELINE)
        assert cb.level == CircuitLevel.CAUTION
        cb.reset_daily(new_baseline=EQUITY_CAUTION)
        assert cb.level == CircuitLevel.NORMAL

    def test_profitable_days_property(self) -> None:
        cb = self._make_breaker()
        assert cb.consecutive_profitable_days == 0
        cb.check(current_equity=EQUITY_HALTED, baseline_equity=BASELINE)
        # New baseline must exceed previous (100000)
        first = BASELINE + Decimal(1000)
        cb.reset_daily(new_baseline=first)
        assert cb.consecutive_profitable_days == 1

    # ── 6A.6: No intraday de-escalation ────────────────────────────────
    def test_level_does_not_deescalate_intraday(self) -> None:
        cb = self._make_breaker()
        # Hit CAUTION
        cb.check(current_equity=EQUITY_CAUTION, baseline_equity=BASELINE)
        assert cb.level == CircuitLevel.CAUTION
        # Equity recovers above baseline -- should stay CAUTION (sticky)
        cb.check(current_equity=Decimal(110000), baseline_equity=BASELINE)
        assert cb.level == CircuitLevel.CAUTION

    def test_level_escalates_from_caution_to_halted(self) -> None:
        cb = self._make_breaker()
        cb.check(current_equity=EQUITY_CAUTION, baseline_equity=BASELINE)
        assert cb.level == CircuitLevel.CAUTION
        cb.check(current_equity=EQUITY_HALTED, baseline_equity=BASELINE)
        assert cb.level == CircuitLevel.HALTED

    def test_reset_daily_allows_fresh_level(self) -> None:
        cb = self._make_breaker()
        # Hit CAUTION
        cb.check(current_equity=EQUITY_CAUTION, baseline_equity=BASELINE)
        assert cb.level == CircuitLevel.CAUTION
        # Daily reset clears CAUTION
        new_base = EQUITY_CAUTION
        cb.reset_daily(new_baseline=new_base)
        assert cb.level == CircuitLevel.NORMAL
        # Now check with equity above new baseline -- stays NORMAL
        cb.check(current_equity=new_base + Decimal(1000), baseline_equity=new_base)
        assert cb.level == CircuitLevel.NORMAL


class TestCrossMarketCircuitBreaker:
    def test_no_trip_when_within_threshold(self) -> None:
        cmcb = CrossMarketCircuitBreaker(halt_threshold=float(CROSS_THRESHOLD))
        tripped = cmcb.check(
            market_equities={MARKET_US: US_CURRENT_SAFE, MARKET_MOEX: MOEX_CURRENT_SAFE},
            baseline_equities={MARKET_US: US_BASELINE, MARKET_MOEX: MOEX_BASELINE},
        )
        assert tripped is False

    def test_trips_when_combined_drawdown_exceeds_threshold(self) -> None:
        cmcb = CrossMarketCircuitBreaker(halt_threshold=float(CROSS_THRESHOLD))
        tripped = cmcb.check(
            market_equities={MARKET_US: US_CURRENT_OK, MARKET_MOEX: MOEX_CURRENT_TRIP},
            baseline_equities={MARKET_US: US_BASELINE, MARKET_MOEX: MOEX_BASELINE},
        )
        assert tripped is True

    def test_zero_baseline_returns_false(self) -> None:
        """Zero combined baseline should not raise -- return False (no data = no trip)."""
        cmcb = CrossMarketCircuitBreaker(halt_threshold=float(CROSS_THRESHOLD))
        tripped = cmcb.check(
            market_equities={MARKET_US: Decimal(0)},
            baseline_equities={MARKET_US: Decimal(0)},
        )
        assert tripped is False

    def test_empty_markets_returns_false(self) -> None:
        cmcb = CrossMarketCircuitBreaker(halt_threshold=float(CROSS_THRESHOLD))
        tripped = cmcb.check(market_equities={}, baseline_equities={})
        assert tripped is False

    def test_reset_daily_updates_baselines(self) -> None:
        """After reset_daily, a previously tripped cross-market check can recover."""
        cmcb = CrossMarketCircuitBreaker(halt_threshold=float(CROSS_THRESHOLD))
        # First: trip it
        assert (
            cmcb.check(
                market_equities={MARKET_US: US_CURRENT_OK, MARKET_MOEX: MOEX_CURRENT_TRIP},
                baseline_equities={MARKET_US: US_BASELINE, MARKET_MOEX: MOEX_BASELINE},
            )
            is True
        )
        # Reset with new baselines matching the current equities
        cmcb.reset_daily(new_baselines={MARKET_US: US_CURRENT_OK, MARKET_MOEX: MOEX_CURRENT_TRIP})
        # Now check with those same values -> 0% drawdown -> no trip
        assert (
            cmcb.check(
                market_equities={MARKET_US: US_CURRENT_OK, MARKET_MOEX: MOEX_CURRENT_TRIP},
                baseline_equities={MARKET_US: US_CURRENT_OK, MARKET_MOEX: MOEX_CURRENT_TRIP},
            )
            is False
        )

    def test_single_market_trip(self) -> None:
        """A single market with 15% drawdown alone should trip the cross-market breaker."""
        cmcb = CrossMarketCircuitBreaker(halt_threshold=float(CROSS_THRESHOLD))
        tripped = cmcb.check(
            market_equities={MARKET_US: Decimal(80000)},
            baseline_equities={MARKET_US: Decimal(100000)},
        )
        assert tripped is True
