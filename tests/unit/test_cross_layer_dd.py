"""Cross-layer drawdown cascading tests.

Validates that DrawdownMonitor + LayerCircuitBreaker + PortfolioCircuitBreaker
work correctly together in realistic multi-layer scenarios.

Covers:
- Single-layer DD containment (no cross-contamination)
- Multi-layer stress without portfolio breach
- Portfolio-level cascade (sticky breach, core preservation)
- Realistic scenario simulations (easing, moderate hawkish, severe crisis)
- Edge cases (zero equity, unequal layers, reset after breach)
"""

from __future__ import annotations

from decimal import Decimal

from finalayze.risk.drawdown_monitor import DrawdownStatus, LayeredDrawdownMonitor
from finalayze.risk.layer_circuit_breaker import CircuitLevel

# -- Constants (ruff PLR2004: no magic numbers) --------------------------------

# Initial layer equity allocations (RUB)
CORE_EQUITY = Decimal(4_000_000)
STRATEGIC_EQUITY = Decimal(3_000_000)
TACTICAL_EQUITY = Decimal(2_000_000)
SHORT_EQUITY = Decimal(1_000_000)
TOTAL_EQUITY = CORE_EQUITY + STRATEGIC_EQUITY + TACTICAL_EQUITY + SHORT_EQUITY

# Sizing multiplier constants
FULL_SIZE = Decimal(1)
HALF_SIZE = Decimal("0.5")
ZERO_SIZE = Decimal(0)

# Layer threshold constants (matching LAYER_CIRCUIT_CONFIGS)
STRATEGIC_L2_PCT = Decimal("0.03")
STRATEGIC_L3_PCT = Decimal("0.05")
TACTICAL_L1_PCT = Decimal("0.02")
TACTICAL_L2_PCT = Decimal("0.03")
SHORT_L1_PCT = Decimal("0.015")
SHORT_L2_PCT = Decimal("0.03")
PORTFOLIO_THRESHOLD_PCT = Decimal("0.10")

# Non-core layer count
NON_CORE_LAYER_COUNT = 3
LAYER_COUNT = 4

# Drawdown step constants for scenarios
DD_ONE_PCT = Decimal("0.01")
DD_TWO_PCT = Decimal("0.02")
DD_THREE_PCT = Decimal("0.03")
DD_FIVE_PCT = Decimal("0.05")
DD_NINE_PCT = Decimal("0.09")
DD_TEN_PCT = Decimal("0.10")
DD_TWELVE_PCT = Decimal("0.12")
DD_FIFTEEN_PCT = Decimal("0.15")
DD_FIFTY_PCT = Decimal("0.50")

# Scenario-specific equity drops
MODERATE_STRATEGIC_DROP = Decimal("0.05")
MODERATE_TACTICAL_DROP = Decimal("0.03")
MODERATE_SHORT_DROP = Decimal("0.03")

SEVERE_CORE_DROP = Decimal("0.08")
SEVERE_STRATEGIC_DROP = Decimal("0.18")
SEVERE_TACTICAL_DROP = Decimal("0.15")
SEVERE_SHORT_DROP = Decimal("0.20")


def _make_equities(
    core: Decimal = CORE_EQUITY,
    strategic: Decimal = STRATEGIC_EQUITY,
    tactical: Decimal = TACTICAL_EQUITY,
    short: Decimal = SHORT_EQUITY,
) -> dict[str, Decimal]:
    """Build a layer_equities dict with default values."""
    return {
        "core": core,
        "strategic": strategic,
        "tactical": tactical,
        "short": short,
    }


def _equity_at_dd(peak: Decimal, dd_pct: Decimal) -> Decimal:
    """Compute equity after a given drawdown percentage from peak."""
    return peak * (Decimal(1) - dd_pct)


# =============================================================================
# 1. Single-layer DD stays contained (6 tests)
# =============================================================================


class TestSingleLayerDDContained:
    """Drawdown in one layer must not affect other layers."""

    def test_core_3pct_dd_no_effect_on_others(self) -> None:
        """Core has 3% DD -- no thresholds defined, so no effect on any layer."""
        monitor = LayeredDrawdownMonitor()
        monitor.update(_make_equities())

        status = monitor.update(_make_equities(core=_equity_at_dd(CORE_EQUITY, DD_THREE_PCT)))

        # Core stays NORMAL (no thresholds)
        assert status.layer_levels["core"] == CircuitLevel.NORMAL
        assert status.sizing_multipliers["core"] == FULL_SIZE
        # All other layers unaffected
        assert status.layer_levels["strategic"] == CircuitLevel.NORMAL
        assert status.layer_levels["tactical"] == CircuitLevel.NORMAL
        assert status.layer_levels["short"] == CircuitLevel.NORMAL
        assert status.sizing_multipliers["strategic"] == FULL_SIZE
        assert status.sizing_multipliers["tactical"] == FULL_SIZE
        assert status.sizing_multipliers["short"] == FULL_SIZE

    def test_strategic_l2_halt_others_unaffected(self) -> None:
        """Strategic hits L2 at -3% -- HALT for strategic only."""
        monitor = LayeredDrawdownMonitor()
        monitor.update(_make_equities())

        status = monitor.update(
            _make_equities(strategic=_equity_at_dd(STRATEGIC_EQUITY, STRATEGIC_L2_PCT))
        )

        assert status.layer_levels["strategic"] == CircuitLevel.HALTED
        assert status.sizing_multipliers["strategic"] == ZERO_SIZE
        # Others remain NORMAL
        assert status.layer_levels["core"] == CircuitLevel.NORMAL
        assert status.layer_levels["tactical"] == CircuitLevel.NORMAL
        assert status.layer_levels["short"] == CircuitLevel.NORMAL
        assert status.sizing_multipliers["core"] == FULL_SIZE
        assert status.sizing_multipliers["tactical"] == FULL_SIZE
        assert status.sizing_multipliers["short"] == FULL_SIZE

    def test_strategic_l3_liquidate_others_unaffected(self) -> None:
        """Strategic hits L3 at -5% -- LIQUIDATE for strategic only."""
        monitor = LayeredDrawdownMonitor()
        monitor.update(_make_equities())

        status = monitor.update(
            _make_equities(strategic=_equity_at_dd(STRATEGIC_EQUITY, STRATEGIC_L3_PCT))
        )

        assert status.layer_levels["strategic"] == CircuitLevel.LIQUIDATE
        assert status.sizing_multipliers["strategic"] == ZERO_SIZE
        # Others remain NORMAL
        assert status.layer_levels["core"] == CircuitLevel.NORMAL
        assert status.layer_levels["tactical"] == CircuitLevel.NORMAL
        assert status.layer_levels["short"] == CircuitLevel.NORMAL

    def test_tactical_l1_caution_others_unaffected(self) -> None:
        """Tactical hits L1 at -2% -- CAUTION (0.5) for tactical only."""
        monitor = LayeredDrawdownMonitor()
        monitor.update(_make_equities())

        status = monitor.update(
            _make_equities(tactical=_equity_at_dd(TACTICAL_EQUITY, TACTICAL_L1_PCT))
        )

        assert status.layer_levels["tactical"] == CircuitLevel.CAUTION
        assert status.sizing_multipliers["tactical"] == HALF_SIZE
        # Others remain NORMAL
        assert status.layer_levels["core"] == CircuitLevel.NORMAL
        assert status.layer_levels["strategic"] == CircuitLevel.NORMAL
        assert status.layer_levels["short"] == CircuitLevel.NORMAL
        assert status.sizing_multipliers["core"] == FULL_SIZE
        assert status.sizing_multipliers["strategic"] == FULL_SIZE
        assert status.sizing_multipliers["short"] == FULL_SIZE

    def test_short_l1_caution_others_unaffected(self) -> None:
        """Short hits L1 at -1.5% -- CAUTION for short only."""
        monitor = LayeredDrawdownMonitor()
        monitor.update(_make_equities())

        status = monitor.update(_make_equities(short=_equity_at_dd(SHORT_EQUITY, SHORT_L1_PCT)))

        assert status.layer_levels["short"] == CircuitLevel.CAUTION
        assert status.sizing_multipliers["short"] == HALF_SIZE
        # Others remain NORMAL
        assert status.layer_levels["core"] == CircuitLevel.NORMAL
        assert status.layer_levels["strategic"] == CircuitLevel.NORMAL
        assert status.layer_levels["tactical"] == CircuitLevel.NORMAL

    def test_core_no_thresholds_always_normal(self) -> None:
        """Core has no L1/L2/L3 thresholds -- stays NORMAL even at 50% DD."""
        monitor = LayeredDrawdownMonitor()
        monitor.update(_make_equities())

        status = monitor.update(_make_equities(core=_equity_at_dd(CORE_EQUITY, DD_FIFTY_PCT)))

        assert status.layer_levels["core"] == CircuitLevel.NORMAL
        assert status.sizing_multipliers["core"] == FULL_SIZE


# =============================================================================
# 2. Multi-layer stress without portfolio breach (3 tests)
# =============================================================================


class TestMultiLayerStressNoPortfolioBreach:
    """Multiple layers under stress but total DD stays below 10% portfolio threshold."""

    def test_strategic_l2_tactical_l1_short_l1_independent(self) -> None:
        """Strategic at L2 (-3%) + Tactical at L1 (-2%) + Short at L1 (-1.5%).

        Each layer has its own independent circuit level.
        Portfolio DD is weighted by equity allocation, not summed from layer DDs.
        """
        monitor = LayeredDrawdownMonitor()
        monitor.update(_make_equities())

        status = monitor.update(
            _make_equities(
                strategic=_equity_at_dd(STRATEGIC_EQUITY, STRATEGIC_L2_PCT),
                tactical=_equity_at_dd(TACTICAL_EQUITY, TACTICAL_L1_PCT),
                short=_equity_at_dd(SHORT_EQUITY, SHORT_L1_PCT),
            )
        )

        # Each layer at its expected level
        assert status.layer_levels["strategic"] == CircuitLevel.HALTED
        assert status.layer_levels["tactical"] == CircuitLevel.CAUTION
        assert status.layer_levels["short"] == CircuitLevel.CAUTION
        assert status.layer_levels["core"] == CircuitLevel.NORMAL

        # Sizing multipliers reflect layer levels
        assert status.sizing_multipliers["strategic"] == ZERO_SIZE
        assert status.sizing_multipliers["tactical"] == HALF_SIZE
        assert status.sizing_multipliers["short"] == HALF_SIZE
        assert status.sizing_multipliers["core"] == FULL_SIZE

        # No portfolio breach -- total DD is well below 10%
        # Strategic lost 90k, tactical lost 40k, short lost 15k => 145k of 10M = 1.45%
        assert status.portfolio_breach is False
        assert status.layers_to_liquidate == []

    def test_two_layers_halt_portfolio_below_10pct(self) -> None:
        """Strategic and tactical both at HALT, but portfolio still below 10% DD."""
        monitor = LayeredDrawdownMonitor()
        monitor.update(_make_equities())

        status = monitor.update(
            _make_equities(
                strategic=_equity_at_dd(STRATEGIC_EQUITY, STRATEGIC_L2_PCT),
                tactical=_equity_at_dd(TACTICAL_EQUITY, TACTICAL_L2_PCT),
            )
        )

        assert status.layer_levels["strategic"] == CircuitLevel.HALTED
        assert status.layer_levels["tactical"] == CircuitLevel.HALTED
        assert status.portfolio_breach is False

        # Portfolio DD = (90k + 60k) / 10M = 1.5%
        expected_loss = STRATEGIC_EQUITY * STRATEGIC_L2_PCT + TACTICAL_EQUITY * TACTICAL_L2_PCT
        expected_portfolio_dd = expected_loss / TOTAL_EQUITY
        assert status.portfolio_dd == expected_portfolio_dd
        assert expected_portfolio_dd < PORTFOLIO_THRESHOLD_PCT

    def test_recovery_layer_equity_rises_above_peak(self) -> None:
        """Layer equity rises above previous peak -- DD resets toward 0."""
        monitor = LayeredDrawdownMonitor()
        monitor.update(_make_equities())

        # Drop tactical to CAUTION
        monitor.update(_make_equities(tactical=_equity_at_dd(TACTICAL_EQUITY, TACTICAL_L1_PCT)))

        # Now tactical rises to a new high above original peak
        new_tactical = TACTICAL_EQUITY + Decimal(100_000)
        status = monitor.update(_make_equities(tactical=new_tactical))

        assert status.layer_levels["tactical"] == CircuitLevel.NORMAL
        assert status.layer_drawdowns["tactical"] == ZERO_SIZE
        assert status.sizing_multipliers["tactical"] == FULL_SIZE


# =============================================================================
# 3. Portfolio-level cascade (5 tests)
# =============================================================================


class TestPortfolioLevelCascade:
    """When total equity drops >= 10%, portfolio breach triggers liquidation."""

    def test_10pct_total_drop_triggers_portfolio_breach(self) -> None:
        """Total equity drops 10% -- all non-core layers get multiplier 0.0."""
        monitor = LayeredDrawdownMonitor()
        monitor.update(_make_equities())

        factor = Decimal(1) - PORTFOLIO_THRESHOLD_PCT
        status = monitor.update(
            _make_equities(
                core=CORE_EQUITY * factor,
                strategic=STRATEGIC_EQUITY * factor,
                tactical=TACTICAL_EQUITY * factor,
                short=SHORT_EQUITY * factor,
            )
        )

        assert status.portfolio_breach is True
        assert status.portfolio_dd >= PORTFOLIO_THRESHOLD_PCT
        assert status.sizing_multipliers["strategic"] == ZERO_SIZE
        assert status.sizing_multipliers["tactical"] == ZERO_SIZE
        assert status.sizing_multipliers["short"] == ZERO_SIZE
        assert len(status.layers_to_liquidate) == NON_CORE_LAYER_COUNT

    def test_core_preserved_during_portfolio_breach(self) -> None:
        """Core keeps its layer-level multiplier even during portfolio breach.

        Core has no layer-level thresholds, so its multiplier stays at 1.0.
        """
        monitor = LayeredDrawdownMonitor()
        monitor.update(_make_equities())

        factor = Decimal(1) - PORTFOLIO_THRESHOLD_PCT
        status = monitor.update(
            _make_equities(
                core=CORE_EQUITY * factor,
                strategic=STRATEGIC_EQUITY * factor,
                tactical=TACTICAL_EQUITY * factor,
                short=SHORT_EQUITY * factor,
            )
        )

        assert status.portfolio_breach is True
        # Core has no thresholds => layer multiplier is 1.0, preserved through breach
        assert status.sizing_multipliers["core"] == FULL_SIZE
        assert "core" not in status.layers_to_liquidate

    def test_portfolio_breach_is_sticky(self) -> None:
        """Once portfolio breach triggers, it stays even if equity recovers."""
        monitor = LayeredDrawdownMonitor()
        monitor.update(_make_equities())

        # Trigger breach
        factor = Decimal(1) - PORTFOLIO_THRESHOLD_PCT
        monitor.update(
            _make_equities(
                core=CORE_EQUITY * factor,
                strategic=STRATEGIC_EQUITY * factor,
                tactical=TACTICAL_EQUITY * factor,
                short=SHORT_EQUITY * factor,
            )
        )
        assert monitor.is_portfolio_breached is True

        # Full recovery to original levels
        status = monitor.update(_make_equities())

        # Still breached -- sticky until reset
        assert status.portfolio_breach is True
        assert monitor.is_portfolio_breached is True
        assert status.sizing_multipliers["strategic"] == ZERO_SIZE
        assert status.sizing_multipliers["tactical"] == ZERO_SIZE
        assert status.sizing_multipliers["short"] == ZERO_SIZE

    def test_layer_l2_first_then_portfolio_l3_adds_on_top(self) -> None:
        """Sequential: layer-level L2 triggers first, then portfolio L3 overrides."""
        monitor = LayeredDrawdownMonitor()
        monitor.update(_make_equities())

        # Step 1: Strategic hits L2 only (small portfolio DD)
        status_step1 = monitor.update(
            _make_equities(strategic=_equity_at_dd(STRATEGIC_EQUITY, STRATEGIC_L2_PCT))
        )
        assert status_step1.layer_levels["strategic"] == CircuitLevel.HALTED
        assert status_step1.portfolio_breach is False
        # Strategic is at HALT from its own layer breaker
        assert status_step1.sizing_multipliers["strategic"] == ZERO_SIZE
        # But tactical and short are still normal
        assert status_step1.sizing_multipliers["tactical"] == FULL_SIZE
        assert status_step1.sizing_multipliers["short"] == FULL_SIZE

        # Step 2: Market crash -- portfolio hits 10% DD
        # We need total equity to drop by 10% from peak.
        # Peak was set at the initial _make_equities() call.
        # The peak total equity is TOTAL_EQUITY = 10M.
        # For 10% DD: total needs to be 9M.
        # Current: core=4M, strategic already at 2.91M (from step 1).
        # We need core + strategic + tactical + short = 9M.
        # Let everything drop sharply.
        factor = Decimal(1) - DD_TWELVE_PCT
        status_step2 = monitor.update(
            _make_equities(
                core=CORE_EQUITY * factor,
                strategic=STRATEGIC_EQUITY * factor,
                tactical=TACTICAL_EQUITY * factor,
                short=SHORT_EQUITY * factor,
            )
        )

        assert status_step2.portfolio_breach is True
        # Now all non-core get zeroed by portfolio breach, regardless of layer level
        assert status_step2.sizing_multipliers["strategic"] == ZERO_SIZE
        assert status_step2.sizing_multipliers["tactical"] == ZERO_SIZE
        assert status_step2.sizing_multipliers["short"] == ZERO_SIZE
        # Core preserved
        assert status_step2.sizing_multipliers["core"] == FULL_SIZE

    def test_portfolio_breach_with_core_at_different_equity_levels(self) -> None:
        """Portfolio breach when core has risen but non-core has crashed."""
        monitor = LayeredDrawdownMonitor()
        monitor.update(_make_equities())

        # Core increases but non-core crashes hard.
        # Peak total = 10M. For 10% DD, need total <= 9M.
        # Core rises to 4.5M (+500k), but strategic/tactical/short drop heavily.
        # strategic: 3M -> 1.5M, tactical: 2M -> 1.5M, short: 1M -> 0.5M
        # Total = 4.5M + 1.5M + 1.5M + 0.5M = 8.0M. DD = 2M/10M = 20%.
        core_up = CORE_EQUITY + Decimal(500_000)
        status = monitor.update(
            _make_equities(
                core=core_up,
                strategic=Decimal(1_500_000),
                tactical=Decimal(1_500_000),
                short=Decimal(500_000),
            )
        )

        assert status.portfolio_breach is True
        # Core had a new high so its layer DD is 0% => NORMAL, multiplier preserved
        assert status.layer_levels["core"] == CircuitLevel.NORMAL
        assert status.sizing_multipliers["core"] == FULL_SIZE
        # Non-core zeroed
        assert status.sizing_multipliers["strategic"] == ZERO_SIZE
        assert status.sizing_multipliers["tactical"] == ZERO_SIZE
        assert status.sizing_multipliers["short"] == ZERO_SIZE


# =============================================================================
# 4. Realistic scenario simulations (3 tests)
# =============================================================================


class TestRealisticScenarios:
    """Simulate real-world macro scenarios with multi-bar equity updates."""

    def test_easing_scenario_gradual_rise(self) -> None:
        """Equity gradually rises across all layers -- all stay NORMAL, no breach."""
        monitor = LayeredDrawdownMonitor()

        # Bar 1: initial
        status = monitor.update(_make_equities())
        assert status.portfolio_breach is False

        # Bar 2: +1%
        growth = Decimal("1.01")
        status = monitor.update(
            _make_equities(
                core=CORE_EQUITY * growth,
                strategic=STRATEGIC_EQUITY * growth,
                tactical=TACTICAL_EQUITY * growth,
                short=SHORT_EQUITY * growth,
            )
        )
        for level in status.layer_levels.values():
            assert level == CircuitLevel.NORMAL
        assert status.portfolio_breach is False

        # Bar 3: +2% from original
        growth = Decimal("1.02")
        status = monitor.update(
            _make_equities(
                core=CORE_EQUITY * growth,
                strategic=STRATEGIC_EQUITY * growth,
                tactical=TACTICAL_EQUITY * growth,
                short=SHORT_EQUITY * growth,
            )
        )
        for level in status.layer_levels.values():
            assert level == CircuitLevel.NORMAL
        assert status.portfolio_breach is False

        # Bar 4: +5% from original
        growth = Decimal("1.05")
        status = monitor.update(
            _make_equities(
                core=CORE_EQUITY * growth,
                strategic=STRATEGIC_EQUITY * growth,
                tactical=TACTICAL_EQUITY * growth,
                short=SHORT_EQUITY * growth,
            )
        )
        for level in status.layer_levels.values():
            assert level == CircuitLevel.NORMAL
        assert status.portfolio_breach is False
        assert status.portfolio_dd == ZERO_SIZE

        # All multipliers remain at full
        for mult in status.sizing_multipliers.values():
            assert mult == FULL_SIZE

    def test_moderate_hawkish_scenario(self) -> None:
        """Moderate stress: strategic drops 5%, tactical drops 3%, short drops 3%.

        Strategic hits L3 (LIQUIDATE), tactical hits L2 (HALT).
        But portfolio DD is relatively small (~4%) -- no portfolio breach.
        """
        monitor = LayeredDrawdownMonitor()
        monitor.update(_make_equities())

        status = monitor.update(
            _make_equities(
                strategic=_equity_at_dd(STRATEGIC_EQUITY, MODERATE_STRATEGIC_DROP),
                tactical=_equity_at_dd(TACTICAL_EQUITY, MODERATE_TACTICAL_DROP),
                short=_equity_at_dd(SHORT_EQUITY, MODERATE_SHORT_DROP),
            )
        )

        # Layer levels
        assert status.layer_levels["core"] == CircuitLevel.NORMAL
        assert status.layer_levels["strategic"] == CircuitLevel.LIQUIDATE
        assert status.layer_levels["tactical"] == CircuitLevel.HALTED
        assert status.layer_levels["short"] == CircuitLevel.HALTED

        # Sizing: layer-level only (no portfolio breach)
        assert status.sizing_multipliers["core"] == FULL_SIZE
        assert status.sizing_multipliers["strategic"] == ZERO_SIZE
        assert status.sizing_multipliers["tactical"] == ZERO_SIZE
        assert status.sizing_multipliers["short"] == ZERO_SIZE

        # Portfolio DD: (150k + 60k + 30k) / 10M = 2.4%
        expected_loss = (
            STRATEGIC_EQUITY * MODERATE_STRATEGIC_DROP
            + TACTICAL_EQUITY * MODERATE_TACTICAL_DROP
            + SHORT_EQUITY * MODERATE_SHORT_DROP
        )
        expected_portfolio_dd = expected_loss / TOTAL_EQUITY
        assert status.portfolio_dd == expected_portfolio_dd
        assert status.portfolio_breach is False

    def test_severe_crisis_2022_replay(self) -> None:
        """Severe crisis: all layers drop simultaneously.

        Core: -8%, Strategic: -18%, Tactical: -15%, Short: -20%.
        Portfolio DD should exceed 10% -- portfolio breach triggered.
        Strategic/tactical/short get liquidated, core preserved.
        """
        monitor = LayeredDrawdownMonitor()
        monitor.update(_make_equities())

        status = monitor.update(
            _make_equities(
                core=_equity_at_dd(CORE_EQUITY, SEVERE_CORE_DROP),
                strategic=_equity_at_dd(STRATEGIC_EQUITY, SEVERE_STRATEGIC_DROP),
                tactical=_equity_at_dd(TACTICAL_EQUITY, SEVERE_TACTICAL_DROP),
                short=_equity_at_dd(SHORT_EQUITY, SEVERE_SHORT_DROP),
            )
        )

        # Compute expected portfolio DD
        total_loss = (
            CORE_EQUITY * SEVERE_CORE_DROP
            + STRATEGIC_EQUITY * SEVERE_STRATEGIC_DROP
            + TACTICAL_EQUITY * SEVERE_TACTICAL_DROP
            + SHORT_EQUITY * SEVERE_SHORT_DROP
        )
        expected_portfolio_dd = total_loss / TOTAL_EQUITY
        # 320k + 540k + 300k + 200k = 1_360k / 10M = 13.6%
        assert expected_portfolio_dd > PORTFOLIO_THRESHOLD_PCT

        # Portfolio breach triggered
        assert status.portfolio_breach is True
        assert status.portfolio_dd == expected_portfolio_dd
        assert len(status.layers_to_liquidate) == NON_CORE_LAYER_COUNT

        # Core preserved (no layer-level thresholds, so stays NORMAL)
        assert status.layer_levels["core"] == CircuitLevel.NORMAL
        assert status.sizing_multipliers["core"] == FULL_SIZE

        # Non-core zeroed by portfolio breach
        assert status.sizing_multipliers["strategic"] == ZERO_SIZE
        assert status.sizing_multipliers["tactical"] == ZERO_SIZE
        assert status.sizing_multipliers["short"] == ZERO_SIZE

        # Verify layer-level circuit levels also triggered independently
        # Strategic: -18% exceeds L3 (5%) => LIQUIDATE
        assert status.layer_levels["strategic"] == CircuitLevel.LIQUIDATE
        # Tactical: -15% exceeds L2 (3%) but tactical has no L3 => HALT
        assert status.layer_levels["tactical"] == CircuitLevel.HALTED
        # Short: -20% exceeds L2 (3%) but short has no L3 => HALT
        assert status.layer_levels["short"] == CircuitLevel.HALTED


# =============================================================================
# 5. Edge cases (3 tests)
# =============================================================================


class TestEdgeCases:
    """Edge cases: zero equity, unequal layer sizes, reset after breach."""

    def test_zero_equity_in_layer_no_division_by_zero(self) -> None:
        """A layer with zero equity should not cause division by zero."""
        monitor = LayeredDrawdownMonitor()

        # Start with zero in short layer
        status = monitor.update(_make_equities(short=Decimal(0)))

        # Should not crash
        assert status.layer_levels["short"] == CircuitLevel.NORMAL
        assert status.layer_drawdowns["short"] == Decimal(0)
        assert status.sizing_multipliers["short"] == FULL_SIZE

        # Update again with zero
        status = monitor.update(_make_equities(short=Decimal(0)))
        assert status.layer_levels["short"] == CircuitLevel.NORMAL

    def test_unequal_layer_sizes_portfolio_dd_weighted_correctly(self) -> None:
        """Portfolio DD is computed from total equity, not averaged across layers.

        Only strategic drops 10% (300k loss from 3M). Other layers unchanged.
        Portfolio DD = 300k / 10M = 3%, not 10%.
        """
        monitor = LayeredDrawdownMonitor()
        monitor.update(_make_equities())

        status = monitor.update(
            _make_equities(
                strategic=_equity_at_dd(STRATEGIC_EQUITY, DD_TEN_PCT),
            )
        )

        # Portfolio DD is weighted by equity share, not raw layer DD
        expected_loss = STRATEGIC_EQUITY * DD_TEN_PCT
        expected_portfolio_dd = expected_loss / TOTAL_EQUITY
        assert status.portfolio_dd == expected_portfolio_dd
        # 300k / 10M = 3% -- well below 10% portfolio threshold
        assert status.portfolio_breach is False

        # Strategic layer at LIQUIDATE (10% > 5% L3 threshold)
        assert status.layer_levels["strategic"] == CircuitLevel.LIQUIDATE

    def test_reset_after_portfolio_breach_restores_normal(self) -> None:
        """After reset(), everything returns to NORMAL for fresh tracking."""
        monitor = LayeredDrawdownMonitor()
        monitor.update(_make_equities())

        # Trigger portfolio breach
        factor = Decimal(1) - DD_TWELVE_PCT
        monitor.update(
            _make_equities(
                core=CORE_EQUITY * factor,
                strategic=STRATEGIC_EQUITY * factor,
                tactical=TACTICAL_EQUITY * factor,
                short=SHORT_EQUITY * factor,
            )
        )
        assert monitor.is_portfolio_breached is True

        # Reset
        monitor.reset()
        assert monitor.is_portfolio_breached is False

        # Fresh update after reset -- all NORMAL
        status = monitor.update(_make_equities())

        for level in status.layer_levels.values():
            assert level == CircuitLevel.NORMAL
        for mult in status.sizing_multipliers.values():
            assert mult == FULL_SIZE
        assert status.portfolio_breach is False
        assert status.portfolio_dd == Decimal(0)
        assert status.layers_to_liquidate == []
