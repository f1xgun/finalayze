"""Unit tests for per-layer circuit breakers (multi-asset portfolio)."""

from __future__ import annotations

from decimal import Decimal

import pytest

from finalayze.risk.layer_circuit_breaker import (
    LAYER_CIRCUIT_CONFIGS,
    PORTFOLIO_L3_THRESHOLD,
    CircuitLevel,
    LayerCircuitBreaker,
    LayerCircuitConfig,
    PortfolioCircuitBreaker,
)

# ── Constants (ruff PLR2004: no magic numbers) ──────────────────────────────
INITIAL_EQUITY = Decimal(100000)

# Strategic thresholds
STRATEGIC_L2_PCT = Decimal("0.03")
STRATEGIC_L3_PCT = Decimal("0.05")

# Tactical thresholds
TACTICAL_L1_PCT = Decimal("0.02")
TACTICAL_L2_PCT = Decimal("0.03")

# Short thresholds
SHORT_L1_PCT = Decimal("0.015")
SHORT_L2_PCT = Decimal("0.03")

# Portfolio threshold
PORTFOLIO_THRESHOLD_PCT = Decimal("0.10")

# Sizing multipliers
FULL_SIZE = Decimal(1)
HALF_SIZE = Decimal("0.5")
ZERO_SIZE = Decimal(0)

# Number of expected layer configs
EXPECTED_LAYER_COUNT = 4


def _equity_at_dd(peak: Decimal, dd_pct: Decimal) -> Decimal:
    """Compute equity after a given drawdown percentage from peak."""
    return peak * (Decimal(1) - dd_pct)


class TestCircuitLevel:
    """Test CircuitLevel enum values and ordering."""

    def test_normal_is_zero(self) -> None:
        assert CircuitLevel.NORMAL == 0

    def test_caution_is_one(self) -> None:
        assert CircuitLevel.CAUTION == 1

    def test_halt_is_two(self) -> None:
        assert CircuitLevel.HALT == 2

    def test_liquidate_is_three(self) -> None:
        assert CircuitLevel.LIQUIDATE == 3

    def test_ordering(self) -> None:
        assert CircuitLevel.NORMAL < CircuitLevel.CAUTION < CircuitLevel.HALT < CircuitLevel.LIQUIDATE


class TestLayerCircuitBreakerNormal:
    """Test that breaker stays NORMAL when drawdown is below thresholds."""

    def test_initial_level_is_normal(self) -> None:
        config = LAYER_CIRCUIT_CONFIGS["tactical"]
        breaker = LayerCircuitBreaker(config)
        assert breaker.level == CircuitLevel.NORMAL

    def test_no_drawdown_stays_normal(self) -> None:
        config = LAYER_CIRCUIT_CONFIGS["tactical"]
        breaker = LayerCircuitBreaker(config)
        level = breaker.update(INITIAL_EQUITY)
        assert level == CircuitLevel.NORMAL
        # Equity goes up
        level = breaker.update(INITIAL_EQUITY + Decimal(5000))
        assert level == CircuitLevel.NORMAL

    def test_small_drawdown_stays_normal(self) -> None:
        config = LAYER_CIRCUIT_CONFIGS["tactical"]
        breaker = LayerCircuitBreaker(config)
        breaker.update(INITIAL_EQUITY)
        # 1% DD is below tactical L1 of 2%
        level = breaker.update(_equity_at_dd(INITIAL_EQUITY, Decimal("0.01")))
        assert level == CircuitLevel.NORMAL


class TestLayerCircuitBreakerCaution:
    """Test L1 (CAUTION) triggers."""

    def test_tactical_l1_at_threshold(self) -> None:
        config = LAYER_CIRCUIT_CONFIGS["tactical"]
        breaker = LayerCircuitBreaker(config)
        breaker.update(INITIAL_EQUITY)
        level = breaker.update(_equity_at_dd(INITIAL_EQUITY, TACTICAL_L1_PCT))
        assert level == CircuitLevel.CAUTION

    def test_short_l1_at_threshold(self) -> None:
        config = LAYER_CIRCUIT_CONFIGS["short"]
        breaker = LayerCircuitBreaker(config)
        breaker.update(INITIAL_EQUITY)
        level = breaker.update(_equity_at_dd(INITIAL_EQUITY, SHORT_L1_PCT))
        assert level == CircuitLevel.CAUTION

    def test_strategic_has_no_l1(self) -> None:
        """Strategic layer has no L1 threshold -- stays NORMAL below L2."""
        config = LAYER_CIRCUIT_CONFIGS["strategic"]
        breaker = LayerCircuitBreaker(config)
        breaker.update(INITIAL_EQUITY)
        # 2% DD -- would be CAUTION if L1 existed, but strategic has no L1
        level = breaker.update(_equity_at_dd(INITIAL_EQUITY, Decimal("0.02")))
        assert level == CircuitLevel.NORMAL


class TestLayerCircuitBreakerHalt:
    """Test L2 (HALT) triggers."""

    def test_tactical_l2_at_threshold(self) -> None:
        config = LAYER_CIRCUIT_CONFIGS["tactical"]
        breaker = LayerCircuitBreaker(config)
        breaker.update(INITIAL_EQUITY)
        level = breaker.update(_equity_at_dd(INITIAL_EQUITY, TACTICAL_L2_PCT))
        assert level == CircuitLevel.HALT

    def test_strategic_l2_at_threshold(self) -> None:
        config = LAYER_CIRCUIT_CONFIGS["strategic"]
        breaker = LayerCircuitBreaker(config)
        breaker.update(INITIAL_EQUITY)
        level = breaker.update(_equity_at_dd(INITIAL_EQUITY, STRATEGIC_L2_PCT))
        assert level == CircuitLevel.HALT

    def test_short_l2_at_threshold(self) -> None:
        config = LAYER_CIRCUIT_CONFIGS["short"]
        breaker = LayerCircuitBreaker(config)
        breaker.update(INITIAL_EQUITY)
        level = breaker.update(_equity_at_dd(INITIAL_EQUITY, SHORT_L2_PCT))
        assert level == CircuitLevel.HALT


class TestLayerCircuitBreakerLiquidate:
    """Test L3 (LIQUIDATE) triggers."""

    def test_strategic_l3_at_threshold(self) -> None:
        config = LAYER_CIRCUIT_CONFIGS["strategic"]
        breaker = LayerCircuitBreaker(config)
        breaker.update(INITIAL_EQUITY)
        level = breaker.update(_equity_at_dd(INITIAL_EQUITY, STRATEGIC_L3_PCT))
        assert level == CircuitLevel.LIQUIDATE

    def test_strategic_l3_above_threshold(self) -> None:
        config = LAYER_CIRCUIT_CONFIGS["strategic"]
        breaker = LayerCircuitBreaker(config)
        breaker.update(INITIAL_EQUITY)
        level = breaker.update(_equity_at_dd(INITIAL_EQUITY, Decimal("0.07")))
        assert level == CircuitLevel.LIQUIDATE


class TestLayerCircuitBreakerRecovery:
    """Test return to NORMAL on equity recovery."""

    def test_returns_to_normal_after_recovery(self) -> None:
        """Unlike the per-market breaker, layer breakers are NOT sticky -- they recover."""
        config = LAYER_CIRCUIT_CONFIGS["tactical"]
        breaker = LayerCircuitBreaker(config)
        breaker.update(INITIAL_EQUITY)
        # Drop to CAUTION
        level = breaker.update(_equity_at_dd(INITIAL_EQUITY, TACTICAL_L1_PCT))
        assert level == CircuitLevel.CAUTION
        # Recover back to peak
        level = breaker.update(INITIAL_EQUITY)
        assert level == CircuitLevel.NORMAL

    def test_new_peak_resets_dd_calculation(self) -> None:
        config = LAYER_CIRCUIT_CONFIGS["tactical"]
        breaker = LayerCircuitBreaker(config)
        # Start at 100k, rise to 200k
        breaker.update(INITIAL_EQUITY)
        new_peak = Decimal(200000)
        breaker.update(new_peak)
        # 2% of 200k = 4000. So 196000 hits L1
        level = breaker.update(_equity_at_dd(new_peak, TACTICAL_L1_PCT))
        assert level == CircuitLevel.CAUTION


class TestLayerCircuitBreakerCoreAlwaysNormal:
    """Test that core layer has no thresholds and stays NORMAL."""

    def test_core_always_normal_no_drawdown(self) -> None:
        config = LAYER_CIRCUIT_CONFIGS["core"]
        breaker = LayerCircuitBreaker(config)
        breaker.update(INITIAL_EQUITY)
        level = breaker.update(INITIAL_EQUITY)
        assert level == CircuitLevel.NORMAL

    def test_core_always_normal_large_drawdown(self) -> None:
        config = LAYER_CIRCUIT_CONFIGS["core"]
        breaker = LayerCircuitBreaker(config)
        breaker.update(INITIAL_EQUITY)
        # Even 50% drawdown: core has no thresholds
        level = breaker.update(_equity_at_dd(INITIAL_EQUITY, Decimal("0.50")))
        assert level == CircuitLevel.NORMAL

    def test_core_config_all_none(self) -> None:
        config = LAYER_CIRCUIT_CONFIGS["core"]
        assert config.l1_threshold_pct is None
        assert config.l2_threshold_pct is None
        assert config.l3_threshold_pct is None


class TestLayerCircuitBreakerSizingMultiplier:
    """Test sizing_multiplier at each level."""

    def test_normal_full_sizing(self) -> None:
        config = LAYER_CIRCUIT_CONFIGS["tactical"]
        breaker = LayerCircuitBreaker(config)
        breaker.update(INITIAL_EQUITY)
        assert breaker.sizing_multiplier() == FULL_SIZE

    def test_caution_half_sizing(self) -> None:
        config = LAYER_CIRCUIT_CONFIGS["tactical"]
        breaker = LayerCircuitBreaker(config)
        breaker.update(INITIAL_EQUITY)
        breaker.update(_equity_at_dd(INITIAL_EQUITY, TACTICAL_L1_PCT))
        assert breaker.sizing_multiplier() == HALF_SIZE

    def test_halt_zero_sizing(self) -> None:
        config = LAYER_CIRCUIT_CONFIGS["tactical"]
        breaker = LayerCircuitBreaker(config)
        breaker.update(INITIAL_EQUITY)
        breaker.update(_equity_at_dd(INITIAL_EQUITY, TACTICAL_L2_PCT))
        assert breaker.sizing_multiplier() == ZERO_SIZE

    def test_liquidate_zero_sizing(self) -> None:
        config = LAYER_CIRCUIT_CONFIGS["strategic"]
        breaker = LayerCircuitBreaker(config)
        breaker.update(INITIAL_EQUITY)
        breaker.update(_equity_at_dd(INITIAL_EQUITY, STRATEGIC_L3_PCT))
        assert breaker.sizing_multiplier() == ZERO_SIZE


class TestLayerCircuitBreakerReset:
    """Test reset clears peak and level."""

    def test_reset_clears_peak_and_level(self) -> None:
        config = LAYER_CIRCUIT_CONFIGS["tactical"]
        breaker = LayerCircuitBreaker(config)
        breaker.update(INITIAL_EQUITY)
        breaker.update(_equity_at_dd(INITIAL_EQUITY, TACTICAL_L2_PCT))
        assert breaker.level == CircuitLevel.HALT
        breaker.reset()
        assert breaker.level == CircuitLevel.NORMAL

    def test_reset_allows_new_peak_tracking(self) -> None:
        config = LAYER_CIRCUIT_CONFIGS["tactical"]
        breaker = LayerCircuitBreaker(config)
        breaker.update(INITIAL_EQUITY)
        breaker.reset()
        # After reset, new equity becomes the peak
        new_equity = Decimal(50000)
        breaker.update(new_equity)
        # 2% of 50k = 1000
        level = breaker.update(_equity_at_dd(new_equity, TACTICAL_L1_PCT))
        assert level == CircuitLevel.CAUTION


class TestLayerCircuitBreakerDrawdownPct:
    """Test the drawdown_pct property."""

    def test_drawdown_pct_zero_when_no_updates(self) -> None:
        config = LAYER_CIRCUIT_CONFIGS["tactical"]
        breaker = LayerCircuitBreaker(config)
        assert breaker.drawdown_pct == Decimal(0)

    def test_drawdown_pct_reflects_current_dd(self) -> None:
        config = LAYER_CIRCUIT_CONFIGS["tactical"]
        breaker = LayerCircuitBreaker(config)
        breaker.update(INITIAL_EQUITY)
        breaker.update(_equity_at_dd(INITIAL_EQUITY, Decimal("0.02")))
        assert breaker.drawdown_pct == Decimal("0.02")

    def test_drawdown_pct_zero_at_peak(self) -> None:
        config = LAYER_CIRCUIT_CONFIGS["tactical"]
        breaker = LayerCircuitBreaker(config)
        breaker.update(INITIAL_EQUITY)
        assert breaker.drawdown_pct == Decimal(0)


class TestLayerCircuitBreakerZeroEquity:
    """Test edge case: zero or negative peak equity."""

    def test_zero_peak_stays_normal(self) -> None:
        config = LAYER_CIRCUIT_CONFIGS["tactical"]
        breaker = LayerCircuitBreaker(config)
        # Never update with positive equity
        level = breaker.update(Decimal(0))
        assert level == CircuitLevel.NORMAL


class TestPortfolioCircuitBreakerNotTriggered:
    """Test portfolio breaker below threshold."""

    def test_not_triggered_initially(self) -> None:
        breaker = PortfolioCircuitBreaker()
        assert breaker.is_triggered is False

    def test_not_triggered_below_threshold(self) -> None:
        breaker = PortfolioCircuitBreaker()
        breaker.update(INITIAL_EQUITY)
        # 9% DD is below 10% threshold
        triggered = breaker.update(_equity_at_dd(INITIAL_EQUITY, Decimal("0.09")))
        assert triggered is False
        assert breaker.is_triggered is False

    def test_not_triggered_at_zero_equity(self) -> None:
        breaker = PortfolioCircuitBreaker()
        triggered = breaker.update(Decimal(0))
        assert triggered is False


class TestPortfolioCircuitBreakerTriggered:
    """Test portfolio breaker at and above threshold."""

    def test_triggered_at_threshold(self) -> None:
        breaker = PortfolioCircuitBreaker()
        breaker.update(INITIAL_EQUITY)
        triggered = breaker.update(_equity_at_dd(INITIAL_EQUITY, PORTFOLIO_THRESHOLD_PCT))
        assert triggered is True
        assert breaker.is_triggered is True

    def test_triggered_above_threshold(self) -> None:
        breaker = PortfolioCircuitBreaker()
        breaker.update(INITIAL_EQUITY)
        triggered = breaker.update(_equity_at_dd(INITIAL_EQUITY, Decimal("0.15")))
        assert triggered is True

    def test_stays_triggered_on_recovery(self) -> None:
        """Once triggered, portfolio breaker stays triggered until reset."""
        breaker = PortfolioCircuitBreaker()
        breaker.update(INITIAL_EQUITY)
        breaker.update(_equity_at_dd(INITIAL_EQUITY, PORTFOLIO_THRESHOLD_PCT))
        assert breaker.is_triggered is True
        # Equity recovers
        triggered = breaker.update(INITIAL_EQUITY)
        assert triggered is True  # still triggered


class TestPortfolioCircuitBreakerLayersToLiquidate:
    """Test layers_to_liquidate property."""

    def test_no_layers_when_not_triggered(self) -> None:
        breaker = PortfolioCircuitBreaker()
        assert breaker.layers_to_liquidate == []

    def test_correct_layers_when_triggered(self) -> None:
        breaker = PortfolioCircuitBreaker()
        breaker.update(INITIAL_EQUITY)
        breaker.update(_equity_at_dd(INITIAL_EQUITY, PORTFOLIO_THRESHOLD_PCT))
        layers = breaker.layers_to_liquidate
        assert "strategic" in layers
        assert "tactical" in layers
        assert "short" in layers

    def test_core_not_in_liquidation_list(self) -> None:
        breaker = PortfolioCircuitBreaker()
        breaker.update(INITIAL_EQUITY)
        breaker.update(_equity_at_dd(INITIAL_EQUITY, PORTFOLIO_THRESHOLD_PCT))
        assert "core" not in breaker.layers_to_liquidate

    def test_exactly_three_layers_liquidated(self) -> None:
        breaker = PortfolioCircuitBreaker()
        breaker.update(INITIAL_EQUITY)
        breaker.update(_equity_at_dd(INITIAL_EQUITY, PORTFOLIO_THRESHOLD_PCT))
        expected_count = 3
        assert len(breaker.layers_to_liquidate) == expected_count


class TestPortfolioCircuitBreakerReset:
    """Test portfolio breaker reset."""

    def test_reset_clears_trigger(self) -> None:
        breaker = PortfolioCircuitBreaker()
        breaker.update(INITIAL_EQUITY)
        breaker.update(_equity_at_dd(INITIAL_EQUITY, PORTFOLIO_THRESHOLD_PCT))
        assert breaker.is_triggered is True
        breaker.reset()
        assert breaker.is_triggered is False
        assert breaker.layers_to_liquidate == []

    def test_reset_allows_re_triggering(self) -> None:
        breaker = PortfolioCircuitBreaker()
        breaker.update(INITIAL_EQUITY)
        breaker.update(_equity_at_dd(INITIAL_EQUITY, PORTFOLIO_THRESHOLD_PCT))
        breaker.reset()
        # Start fresh tracking
        new_equity = Decimal(80000)
        breaker.update(new_equity)
        triggered = breaker.update(_equity_at_dd(new_equity, PORTFOLIO_THRESHOLD_PCT))
        assert triggered is True


class TestDefaultLayerConfigs:
    """Test LAYER_CIRCUIT_CONFIGS defaults."""

    def test_four_layer_configs_defined(self) -> None:
        assert len(LAYER_CIRCUIT_CONFIGS) == EXPECTED_LAYER_COUNT

    def test_core_config(self) -> None:
        cfg = LAYER_CIRCUIT_CONFIGS["core"]
        assert cfg.layer_id == "core"
        assert cfg.l1_threshold_pct is None
        assert cfg.l2_threshold_pct is None
        assert cfg.l3_threshold_pct is None

    def test_strategic_config(self) -> None:
        cfg = LAYER_CIRCUIT_CONFIGS["strategic"]
        assert cfg.layer_id == "strategic"
        assert cfg.l1_threshold_pct is None
        assert cfg.l2_threshold_pct == STRATEGIC_L2_PCT
        assert cfg.l3_threshold_pct == STRATEGIC_L3_PCT

    def test_tactical_config(self) -> None:
        cfg = LAYER_CIRCUIT_CONFIGS["tactical"]
        assert cfg.layer_id == "tactical"
        assert cfg.l1_threshold_pct == TACTICAL_L1_PCT
        assert cfg.l2_threshold_pct == TACTICAL_L2_PCT
        assert cfg.l3_threshold_pct is None

    def test_short_config(self) -> None:
        cfg = LAYER_CIRCUIT_CONFIGS["short"]
        assert cfg.layer_id == "short"
        assert cfg.l1_threshold_pct == SHORT_L1_PCT
        assert cfg.l2_threshold_pct == SHORT_L2_PCT
        assert cfg.l3_threshold_pct is None

    def test_portfolio_threshold(self) -> None:
        assert PORTFOLIO_L3_THRESHOLD == PORTFOLIO_THRESHOLD_PCT
