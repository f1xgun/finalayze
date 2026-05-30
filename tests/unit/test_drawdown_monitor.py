"""Unit tests for DrawdownMonitor (portfolio-level drawdown coordinator)."""

from __future__ import annotations

from decimal import Decimal

import pytest

from finalayze.risk.drawdown_monitor import DrawdownStatus, LayeredDrawdownMonitor
from finalayze.risk.layer_circuit_breaker import (
    CircuitLevel,
    LayerCircuitConfig,
)

# ── Constants (ruff PLR2004: no magic numbers) ──────────────────────────────
INITIAL_EQUITY = Decimal(100_000)
LAYER_COUNT = 4
CORE_EQUITY = Decimal(40_000)
STRATEGIC_EQUITY = Decimal(30_000)
TACTICAL_EQUITY = Decimal(20_000)
SHORT_EQUITY = Decimal(10_000)

# Sizing multipliers
FULL_SIZE = Decimal(1)
HALF_SIZE = Decimal("0.5")
ZERO_SIZE = Decimal(0)

# Thresholds (matching LAYER_CIRCUIT_CONFIGS)
TACTICAL_L1_PCT = Decimal("0.02")
TACTICAL_L2_PCT = Decimal("0.03")
STRATEGIC_L2_PCT = Decimal("0.03")
STRATEGIC_L3_PCT = Decimal("0.05")
SHORT_L1_PCT = Decimal("0.015")
PORTFOLIO_THRESHOLD_PCT = Decimal("0.10")

# Number of non-core layers
NON_CORE_LAYER_COUNT = 3


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


class TestInitialStateAllNormal:
    """Fresh monitor returns all NORMAL, no breach."""

    def test_all_layers_normal(self) -> None:
        monitor = LayeredDrawdownMonitor()
        equities = _make_equities()
        status = monitor.update(equities)
        for level in status.layer_levels.values():
            assert level == CircuitLevel.NORMAL

    def test_no_portfolio_breach(self) -> None:
        monitor = LayeredDrawdownMonitor()
        equities = _make_equities()
        status = monitor.update(equities)
        assert status.portfolio_breach is False

    def test_no_layers_to_liquidate(self) -> None:
        monitor = LayeredDrawdownMonitor()
        equities = _make_equities()
        status = monitor.update(equities)
        assert status.layers_to_liquidate == []

    def test_all_multipliers_full(self) -> None:
        monitor = LayeredDrawdownMonitor()
        equities = _make_equities()
        status = monitor.update(equities)
        for mult in status.sizing_multipliers.values():
            assert mult == FULL_SIZE

    def test_drawdowns_zero(self) -> None:
        monitor = LayeredDrawdownMonitor()
        equities = _make_equities()
        status = monitor.update(equities)
        for dd in status.layer_drawdowns.values():
            assert dd == Decimal(0)
        assert status.portfolio_dd == Decimal(0)


class TestLayerCautionReducesSizing:
    """Tactical DD > 2% triggers CAUTION with multiplier 0.5."""

    def test_tactical_caution(self) -> None:
        monitor = LayeredDrawdownMonitor()
        # Set peak
        monitor.update(_make_equities())
        # Drop tactical by 2%
        status = monitor.update(
            _make_equities(tactical=_equity_at_dd(TACTICAL_EQUITY, TACTICAL_L1_PCT))
        )
        assert status.layer_levels["tactical"] == CircuitLevel.CAUTION
        assert status.sizing_multipliers["tactical"] == HALF_SIZE

    def test_short_caution(self) -> None:
        monitor = LayeredDrawdownMonitor()
        monitor.update(_make_equities())
        status = monitor.update(_make_equities(short=_equity_at_dd(SHORT_EQUITY, SHORT_L1_PCT)))
        assert status.layer_levels["short"] == CircuitLevel.CAUTION
        assert status.sizing_multipliers["short"] == HALF_SIZE


class TestLayerHaltBlocksNewPositions:
    """Strategic DD > 3% triggers HALT with multiplier 0.0."""

    def test_strategic_halt(self) -> None:
        monitor = LayeredDrawdownMonitor()
        monitor.update(_make_equities())
        status = monitor.update(
            _make_equities(strategic=_equity_at_dd(STRATEGIC_EQUITY, STRATEGIC_L2_PCT))
        )
        assert status.layer_levels["strategic"] == CircuitLevel.HALTED
        assert status.sizing_multipliers["strategic"] == ZERO_SIZE

    def test_tactical_halt(self) -> None:
        monitor = LayeredDrawdownMonitor()
        monitor.update(_make_equities())
        status = monitor.update(
            _make_equities(tactical=_equity_at_dd(TACTICAL_EQUITY, TACTICAL_L2_PCT))
        )
        assert status.layer_levels["tactical"] == CircuitLevel.HALTED
        assert status.sizing_multipliers["tactical"] == ZERO_SIZE


class TestPortfolioBreachLiquidatesNonCore:
    """10% portfolio DD sets strategic/tactical/short multipliers to 0.0."""

    def test_non_core_multipliers_zero(self) -> None:
        monitor = LayeredDrawdownMonitor()
        equities = _make_equities()
        monitor.update(equities)
        # Drop all layers proportionally to hit 10% portfolio DD
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
        assert status.sizing_multipliers["strategic"] == ZERO_SIZE
        assert status.sizing_multipliers["tactical"] == ZERO_SIZE
        assert status.sizing_multipliers["short"] == ZERO_SIZE

    def test_layers_to_liquidate_populated(self) -> None:
        monitor = LayeredDrawdownMonitor()
        equities = _make_equities()
        monitor.update(equities)
        factor = Decimal(1) - PORTFOLIO_THRESHOLD_PCT
        status = monitor.update(
            _make_equities(
                core=CORE_EQUITY * factor,
                strategic=STRATEGIC_EQUITY * factor,
                tactical=TACTICAL_EQUITY * factor,
                short=SHORT_EQUITY * factor,
            )
        )
        assert len(status.layers_to_liquidate) == NON_CORE_LAYER_COUNT
        assert "strategic" in status.layers_to_liquidate
        assert "tactical" in status.layers_to_liquidate
        assert "short" in status.layers_to_liquidate


class TestPortfolioBreachPreservesCore:
    """Core keeps its layer-level multiplier even during portfolio breach."""

    def test_core_multiplier_preserved(self) -> None:
        monitor = LayeredDrawdownMonitor()
        equities = _make_equities()
        monitor.update(equities)
        # Drop all by 10% portfolio-wide
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
        # Core has no layer-level thresholds, so its layer multiplier is 1.0
        assert status.sizing_multipliers["core"] == FULL_SIZE


class TestMultipleLayersIndependent:
    """One layer breached doesn't affect others (before portfolio breach)."""

    def test_tactical_caution_others_normal(self) -> None:
        monitor = LayeredDrawdownMonitor()
        monitor.update(_make_equities())
        status = monitor.update(
            _make_equities(tactical=_equity_at_dd(TACTICAL_EQUITY, TACTICAL_L1_PCT))
        )
        assert status.layer_levels["tactical"] == CircuitLevel.CAUTION
        assert status.layer_levels["core"] == CircuitLevel.NORMAL
        assert status.layer_levels["strategic"] == CircuitLevel.NORMAL
        assert status.layer_levels["short"] == CircuitLevel.NORMAL
        # Sizing: only tactical affected
        assert status.sizing_multipliers["tactical"] == HALF_SIZE
        assert status.sizing_multipliers["core"] == FULL_SIZE
        assert status.sizing_multipliers["strategic"] == FULL_SIZE
        assert status.sizing_multipliers["short"] == FULL_SIZE


class TestRecoveryAfterPeakUpdate:
    """Equity recovery resets DD toward 0 (layer breakers are not sticky)."""

    def test_recovery_returns_to_normal(self) -> None:
        monitor = LayeredDrawdownMonitor()
        equities = _make_equities()
        monitor.update(equities)
        # Drop tactical to CAUTION
        monitor.update(_make_equities(tactical=_equity_at_dd(TACTICAL_EQUITY, TACTICAL_L1_PCT)))
        # Recover: new equity matches or exceeds old peak
        status = monitor.update(equities)
        assert status.layer_levels["tactical"] == CircuitLevel.NORMAL
        assert status.sizing_multipliers["tactical"] == FULL_SIZE

    def test_new_peak_updates_dd_baseline(self) -> None:
        monitor = LayeredDrawdownMonitor()
        equities = _make_equities()
        monitor.update(equities)
        # Increase tactical equity to new peak
        higher_tactical = TACTICAL_EQUITY + Decimal(5_000)
        monitor.update(_make_equities(tactical=higher_tactical))
        # Drop 2% from the new peak -- should be CAUTION
        status = monitor.update(
            _make_equities(tactical=_equity_at_dd(higher_tactical, TACTICAL_L1_PCT))
        )
        assert status.layer_levels["tactical"] == CircuitLevel.CAUTION


class TestPortfolioBreachIsSticky:
    """Once portfolio breach triggered, stays triggered until reset."""

    def test_breach_persists_on_recovery(self) -> None:
        monitor = LayeredDrawdownMonitor()
        equities = _make_equities()
        monitor.update(equities)
        # Trigger portfolio breach
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
        # Full recovery
        status = monitor.update(equities)
        assert status.portfolio_breach is True
        assert monitor.is_portfolio_breached is True


class TestResetClearsAll:
    """reset() brings everything back to normal."""

    def test_reset_clears_portfolio_breach(self) -> None:
        monitor = LayeredDrawdownMonitor()
        equities = _make_equities()
        monitor.update(equities)
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
        monitor.reset()
        assert monitor.is_portfolio_breached is False

    def test_reset_returns_normal_status(self) -> None:
        monitor = LayeredDrawdownMonitor()
        equities = _make_equities()
        monitor.update(equities)
        # Trigger tactical CAUTION
        monitor.update(_make_equities(tactical=_equity_at_dd(TACTICAL_EQUITY, TACTICAL_L1_PCT)))
        monitor.reset()
        # After reset, fresh update should be all NORMAL
        status = monitor.update(equities)
        for level in status.layer_levels.values():
            assert level == CircuitLevel.NORMAL
        assert status.portfolio_breach is False


class TestUnknownLayerIgnored:
    """Passing unknown layer_id doesn't crash."""

    def test_unknown_layer_not_in_status(self) -> None:
        monitor = LayeredDrawdownMonitor()
        equities = _make_equities()
        equities["alien_layer"] = Decimal(50_000)
        status = monitor.update(equities)
        # alien_layer should not appear in layer_levels or sizing_multipliers
        assert "alien_layer" not in status.layer_levels
        assert "alien_layer" not in status.sizing_multipliers

    def test_known_layers_unaffected_by_unknown(self) -> None:
        monitor = LayeredDrawdownMonitor()
        equities = _make_equities()
        equities["alien_layer"] = Decimal(50_000)
        status = monitor.update(equities)
        # All known layers still present and NORMAL
        assert len(status.layer_levels) == LAYER_COUNT
        for level in status.layer_levels.values():
            assert level == CircuitLevel.NORMAL


class TestSizingMultipliersCorrect:
    """Verify exact multiplier values at each level."""

    def test_normal_is_one(self) -> None:
        monitor = LayeredDrawdownMonitor()
        status = monitor.update(_make_equities())
        assert status.sizing_multipliers["core"] == FULL_SIZE

    def test_caution_is_half(self) -> None:
        monitor = LayeredDrawdownMonitor()
        monitor.update(_make_equities())
        status = monitor.update(
            _make_equities(tactical=_equity_at_dd(TACTICAL_EQUITY, TACTICAL_L1_PCT))
        )
        assert status.sizing_multipliers["tactical"] == HALF_SIZE

    def test_halt_is_zero(self) -> None:
        monitor = LayeredDrawdownMonitor()
        monitor.update(_make_equities())
        status = monitor.update(
            _make_equities(strategic=_equity_at_dd(STRATEGIC_EQUITY, STRATEGIC_L2_PCT))
        )
        assert status.sizing_multipliers["strategic"] == ZERO_SIZE

    def test_portfolio_breach_non_core_is_zero(self) -> None:
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
        for layer_id in ("strategic", "tactical", "short"):
            assert status.sizing_multipliers[layer_id] == ZERO_SIZE

    def test_min_of_layer_and_portfolio_effect(self) -> None:
        """If layer is CAUTION (0.5) but portfolio breach forces 0.0, result is 0.0."""
        monitor = LayeredDrawdownMonitor()
        monitor.update(_make_equities())
        # Trigger portfolio breach while tactical is in CAUTION range
        factor = Decimal(1) - PORTFOLIO_THRESHOLD_PCT
        status = monitor.update(
            _make_equities(
                core=CORE_EQUITY * factor,
                strategic=STRATEGIC_EQUITY * factor,
                tactical=TACTICAL_EQUITY * factor,
                short=SHORT_EQUITY * factor,
            )
        )
        # Even though tactical may have layer-level multiplier, portfolio override wins
        assert status.sizing_multipliers["tactical"] == ZERO_SIZE


class TestDrawdownStatusFields:
    """Verify all fields populated correctly."""

    def test_status_is_frozen_dataclass(self) -> None:
        monitor = LayeredDrawdownMonitor()
        status = monitor.update(_make_equities())
        with pytest.raises(AttributeError):
            status.portfolio_breach = True  # type: ignore[misc]

    def test_all_layers_present_in_levels(self) -> None:
        monitor = LayeredDrawdownMonitor()
        status = monitor.update(_make_equities())
        assert set(status.layer_levels.keys()) == {"core", "strategic", "tactical", "short"}

    def test_all_layers_present_in_drawdowns(self) -> None:
        monitor = LayeredDrawdownMonitor()
        status = monitor.update(_make_equities())
        assert set(status.layer_drawdowns.keys()) == {"core", "strategic", "tactical", "short"}

    def test_all_layers_present_in_multipliers(self) -> None:
        monitor = LayeredDrawdownMonitor()
        status = monitor.update(_make_equities())
        assert set(status.sizing_multipliers.keys()) == {"core", "strategic", "tactical", "short"}

    def test_portfolio_dd_is_decimal(self) -> None:
        monitor = LayeredDrawdownMonitor()
        status = monitor.update(_make_equities())
        assert isinstance(status.portfolio_dd, Decimal)

    def test_portfolio_dd_reflects_drawdown(self) -> None:
        monitor = LayeredDrawdownMonitor()
        monitor.update(_make_equities())
        # Drop all by 5%
        factor = Decimal("0.95")
        status = monitor.update(
            _make_equities(
                core=CORE_EQUITY * factor,
                strategic=STRATEGIC_EQUITY * factor,
                tactical=TACTICAL_EQUITY * factor,
                short=SHORT_EQUITY * factor,
            )
        )
        assert status.portfolio_dd == Decimal("0.05")


class TestEmptyEquities:
    """Empty dict input handled gracefully."""

    def test_empty_dict_no_crash(self) -> None:
        monitor = LayeredDrawdownMonitor()
        status = monitor.update({})
        assert status.portfolio_breach is False
        assert status.layers_to_liquidate == []

    def test_empty_dict_all_layers_still_reported(self) -> None:
        """All configured layers appear with their current state even if not updated."""
        monitor = LayeredDrawdownMonitor()
        status = monitor.update({})
        # All configured layers should appear in layer_levels
        assert set(status.layer_levels.keys()) == {"core", "strategic", "tactical", "short"}

    def test_empty_dict_multipliers_full(self) -> None:
        monitor = LayeredDrawdownMonitor()
        status = monitor.update({})
        for mult in status.sizing_multipliers.values():
            assert mult == FULL_SIZE


class TestCustomConfig:
    """Test with custom layer configs."""

    def test_custom_threshold(self) -> None:
        custom_configs = {
            "alpha": LayerCircuitConfig(
                layer_id="alpha",
                l1_threshold_pct=Decimal("0.01"),
                l2_threshold_pct=Decimal("0.02"),
                l3_threshold_pct=None,
            ),
        }
        monitor = LayeredDrawdownMonitor(layer_configs=custom_configs)
        alpha_equity = Decimal(50_000)
        monitor.update({"alpha": alpha_equity})
        status = monitor.update({"alpha": _equity_at_dd(alpha_equity, Decimal("0.01"))})
        assert status.layer_levels["alpha"] == CircuitLevel.CAUTION
        assert status.sizing_multipliers["alpha"] == HALF_SIZE

    def test_custom_portfolio_threshold(self) -> None:
        custom_threshold = Decimal("0.05")
        monitor = LayeredDrawdownMonitor(portfolio_threshold=custom_threshold)
        equities = _make_equities()
        monitor.update(equities)
        # 5% portfolio drop should trigger with custom threshold
        factor = Decimal("0.95")
        status = monitor.update(
            _make_equities(
                core=CORE_EQUITY * factor,
                strategic=STRATEGIC_EQUITY * factor,
                tactical=TACTICAL_EQUITY * factor,
                short=SHORT_EQUITY * factor,
            )
        )
        assert status.portfolio_breach is True
