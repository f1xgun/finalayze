"""Integration tests for circuit breaker escalation, cross-market trip, and resets."""

from __future__ import annotations

from decimal import Decimal

import pytest

from finalayze.risk.circuit_breaker import CircuitBreaker, CircuitLevel, CrossMarketCircuitBreaker

# ── Constants ──────────────────────────────────────────────────────────────
MARKET_US = "us"
MARKET_MOEX = "moex"
BASELINE = Decimal(100000)
L1_THRESHOLD = 0.05
L2_THRESHOLD = 0.10
L3_THRESHOLD = 0.15
CROSS_THRESHOLD = 0.10

# Equity values by drawdown
EQUITY_AT_3PCT = Decimal(97000)
EQUITY_AT_6PCT = Decimal(94000)
EQUITY_AT_11PCT = Decimal(89000)
EQUITY_AT_16PCT = Decimal(84000)

US_BASELINE = Decimal(50000)
MOEX_BASELINE = Decimal(50000)
US_SAFE = Decimal(48000)  # 4% down
MOEX_TRIPPING = Decimal(41000)  # total 89000 from 100000 -> 11% combined


@pytest.mark.integration
class TestCircuitBreakerEscalation:
    def _make_cb(self) -> CircuitBreaker:
        return CircuitBreaker(
            market_id=MARKET_US,
            l1_threshold=L1_THRESHOLD,
            l2_threshold=L2_THRESHOLD,
            l3_threshold=L3_THRESHOLD,
        )

    def test_l1_to_l2_to_l3_escalation(self) -> None:
        cb = self._make_cb()
        assert cb.check(EQUITY_AT_6PCT, BASELINE) == CircuitLevel.CAUTION
        assert cb.check(EQUITY_AT_11PCT, BASELINE) == CircuitLevel.HALTED
        assert cb.check(EQUITY_AT_16PCT, BASELINE) == CircuitLevel.LIQUIDATE

    def test_auto_daily_reset_clears_caution_not_liquidate(self) -> None:
        cb = self._make_cb()
        cb.check(EQUITY_AT_6PCT, BASELINE)
        assert cb.level == CircuitLevel.CAUTION
        cb.reset_daily(new_baseline=EQUITY_AT_6PCT)
        assert cb.level == CircuitLevel.NORMAL

    def test_auto_daily_reset_clears_halted(self) -> None:
        cb = self._make_cb()
        cb.check(EQUITY_AT_11PCT, BASELINE)
        assert cb.level == CircuitLevel.HALTED
        cb.reset_daily(new_baseline=EQUITY_AT_11PCT)
        assert cb.level == CircuitLevel.NORMAL

    def test_liquidate_not_cleared_by_daily_reset(self) -> None:
        cb = self._make_cb()
        cb.check(EQUITY_AT_16PCT, BASELINE)
        assert cb.level == CircuitLevel.LIQUIDATE
        cb.reset_daily(new_baseline=EQUITY_AT_16PCT)
        assert cb.level == CircuitLevel.LIQUIDATE  # still locked

    def test_manual_reset_unblocks_liquidate(self) -> None:
        cb = self._make_cb()
        cb.check(EQUITY_AT_16PCT, BASELINE)
        assert cb.level == CircuitLevel.LIQUIDATE
        cb.reset_daily(new_baseline=EQUITY_AT_16PCT)
        assert cb.level == CircuitLevel.LIQUIDATE  # daily reset has no effect
        cb.reset_manual()
        assert cb.level == CircuitLevel.NORMAL  # manual clears it

    def test_full_escalation_then_manual_reset(self) -> None:
        """Simulate a bad day: escalate all three levels, manually reset at end."""
        cb = self._make_cb()

        cb.check(EQUITY_AT_6PCT, BASELINE)
        assert cb.level == CircuitLevel.CAUTION

        cb.check(EQUITY_AT_11PCT, BASELINE)
        assert cb.level == CircuitLevel.HALTED

        cb.check(EQUITY_AT_16PCT, BASELINE)
        assert cb.level == CircuitLevel.LIQUIDATE

        # Daily reset does NOT clear LIQUIDATE
        cb.reset_daily(new_baseline=EQUITY_AT_16PCT)
        assert cb.level == CircuitLevel.LIQUIDATE

        # Manual reset clears it
        cb.reset_manual()
        assert cb.level == CircuitLevel.NORMAL

    def test_recovery_same_day_still_reports_current_level(self) -> None:
        """If equity recovers within the day, check still reflects current status."""
        cb = self._make_cb()
        cb.check(EQUITY_AT_11PCT, BASELINE)
        assert cb.level == CircuitLevel.HALTED
        # Equity recovers to only 3% down
        cb.check(EQUITY_AT_3PCT, BASELINE)
        assert cb.level == CircuitLevel.NORMAL


@pytest.mark.integration
class TestCrossMarketIntegration:
    def test_combined_drawdown_trips_halt(self) -> None:
        cmcb = CrossMarketCircuitBreaker(halt_threshold=CROSS_THRESHOLD)
        assert (
            cmcb.check(
                market_equities={MARKET_US: US_SAFE, MARKET_MOEX: MOEX_TRIPPING},
                baseline_equities={MARKET_US: US_BASELINE, MARKET_MOEX: MOEX_BASELINE},
            )
            is True
        )

    def test_within_threshold_no_trip(self) -> None:
        cmcb = CrossMarketCircuitBreaker(halt_threshold=CROSS_THRESHOLD)
        assert (
            cmcb.check(
                market_equities={MARKET_US: Decimal(49000), MARKET_MOEX: Decimal(49000)},
                baseline_equities={MARKET_US: US_BASELINE, MARKET_MOEX: MOEX_BASELINE},
            )
            is False
        )

    def test_daily_reset_then_check_at_zero_drawdown(self) -> None:
        cmcb = CrossMarketCircuitBreaker(halt_threshold=CROSS_THRESHOLD)
        # Trip it
        assert (
            cmcb.check(
                market_equities={MARKET_US: US_SAFE, MARKET_MOEX: MOEX_TRIPPING},
                baseline_equities={MARKET_US: US_BASELINE, MARKET_MOEX: MOEX_BASELINE},
            )
            is True
        )
        # Reset baselines to current values
        cmcb.reset_daily({MARKET_US: US_SAFE, MARKET_MOEX: MOEX_TRIPPING})
        # Same equities vs new baselines -> 0% drawdown -> no trip
        assert (
            cmcb.check(
                market_equities={MARKET_US: US_SAFE, MARKET_MOEX: MOEX_TRIPPING},
                baseline_equities={MARKET_US: US_SAFE, MARKET_MOEX: MOEX_TRIPPING},
            )
            is False
        )
