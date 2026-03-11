"""Unit tests for bond portfolio stress testing utilities.

Simulates CBR rate shocks of various magnitudes and validates
that the system's drawdown limits hold. Uses realistic OFZ portfolio
parameters from the multi-asset plan.

Portfolio layout (1,500,000 RUB total):
- Core (45%):      OFZ-PK floaters, duration ~0.5Y (low rate sensitivity)
- Strategic (27.5%): OFZ-PD fixed-coupon, average duration ~4.0Y at 15% yield
- Tactical (17.5%):  OFZ-PD fixed-coupon, average duration ~3.0Y
- Short (10%):       Equities (no bond exposure in these tests)
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from finalayze.backtest.stress_test import (
    STRESS_SCENARIOS,
    StressResult,
    StressScenario,
    estimate_bond_pnl_from_rate_change,
    run_portfolio_stress,
)

# ── Portfolio constants (no magic numbers, ruff PLR2004) ──────────────

TOTAL_CAPITAL = Decimal(1500000)  # 1.5M RUB

# Layer allocations
CORE_ALLOCATION_PCT = Decimal("0.45")  # 45%
STRATEGIC_ALLOCATION_PCT = Decimal("0.275")  # 27.5%
TACTICAL_ALLOCATION_PCT = Decimal("0.175")  # 17.5%
SHORT_ALLOCATION_PCT = Decimal("0.10")  # 10%

CORE_VALUE = TOTAL_CAPITAL * CORE_ALLOCATION_PCT  # 675,000
STRATEGIC_VALUE = TOTAL_CAPITAL * STRATEGIC_ALLOCATION_PCT  # 412,500
TACTICAL_VALUE = TOTAL_CAPITAL * TACTICAL_ALLOCATION_PCT  # 262,500
SHORT_VALUE = TOTAL_CAPITAL * SHORT_ALLOCATION_PCT  # 150,000

# Bond characteristics per layer
FACE_VALUE = Decimal(1000)  # OFZ face value

CORE_DURATION = 0.5  # Floaters: very low rate sensitivity
CORE_CONVEXITY = 0.5  # Minimal convexity for short duration
CORE_QUANTITY = 675  # 675 bonds x 1000 = 675,000

STRATEGIC_DURATION = 4.0  # ~4Y average duration
STRATEGIC_CONVEXITY = 20.0  # Typical for 4Y bond at 15% yield
STRATEGIC_QUANTITY = 412  # ~412 bonds x 1000 = 412,000 (rest is cash)

TACTICAL_DURATION = 3.0  # ~3Y average duration
TACTICAL_CONVEXITY = 12.0  # Typical for 3Y bond
TACTICAL_QUANTITY = 262  # ~262 bonds x 1000 = 262,000 (rest is cash)

# Scenario rate changes in bps
RATE_CHANGE_ZERO = 0
RATE_CHANGE_100BPS = 100
RATE_CHANGE_NEG_100BPS = -100
RATE_CHANGE_300BPS = 300
RATE_CHANGE_500BPS = 500
RATE_CHANGE_1000BPS = 1000
RATE_CHANGE_NEG_300BPS = -300

# Duration for generic tests
DURATION_5Y = 5.0
CONVEXITY_30 = 30.0
QUANTITY_100 = 100

# Tolerances
PNL_ZERO = Decimal(0)
PORTFOLIO_DD_LIMIT = 0.10  # 10%
STRATEGIC_LOSS_300BPS = -0.12  # ~-12% loss on strategic layer
STRATEGIC_LOSS_500BPS = -0.20  # ~-20% loss on strategic layer

# Number of standard scenarios
NUM_STANDARD_SCENARIOS = 4


# ── Helper: build standard portfolio positions ────────────────────────


def _build_portfolio() -> tuple[dict[str, list[dict]], dict[str, Decimal]]:
    """Build the standard test portfolio with 4 layers.

    Returns (layer_positions, layer_cash).
    """
    layer_positions: dict[str, list[dict]] = {
        "core": [
            {
                "face_value": FACE_VALUE,
                "quantity": CORE_QUANTITY,
                "mod_duration": CORE_DURATION,
                "convexity": CORE_CONVEXITY,
            },
        ],
        "strategic": [
            {
                "face_value": FACE_VALUE,
                "quantity": STRATEGIC_QUANTITY,
                "mod_duration": STRATEGIC_DURATION,
                "convexity": STRATEGIC_CONVEXITY,
            },
        ],
        "tactical": [
            {
                "face_value": FACE_VALUE,
                "quantity": TACTICAL_QUANTITY,
                "mod_duration": TACTICAL_DURATION,
                "convexity": TACTICAL_CONVEXITY,
            },
        ],
        "short": [],  # Equities, no bond exposure
    }
    # Cash = allocation - bonds at face
    layer_cash: dict[str, Decimal] = {
        "core": CORE_VALUE - FACE_VALUE * CORE_QUANTITY,
        "strategic": STRATEGIC_VALUE - FACE_VALUE * STRATEGIC_QUANTITY,
        "tactical": TACTICAL_VALUE - FACE_VALUE * TACTICAL_QUANTITY,
        "short": SHORT_VALUE,  # All cash / equity (no bonds)
    }
    return layer_positions, layer_cash


# ══════════════════════════════════════════════════════════════════════
# 1. TestEstimateBondPnl
# ══════════════════════════════════════════════════════════════════════


class TestEstimateBondPnl:
    """Tests for the single-position PnL estimation function."""

    def test_zero_rate_change_gives_zero_pnl(self) -> None:
        """Zero rate change should produce zero PnL."""
        result = estimate_bond_pnl_from_rate_change(
            face_value=FACE_VALUE,
            quantity=QUANTITY_100,
            mod_duration=DURATION_5Y,
            convexity_val=CONVEXITY_30,
            rate_change_bps=RATE_CHANGE_ZERO,
        )
        assert result == PNL_ZERO

    def test_100bps_hike_5y_duration_approx_neg5pct(self) -> None:
        """+100bps on 5Y duration should produce ~-5% loss."""
        position_value = FACE_VALUE * QUANTITY_100
        result = estimate_bond_pnl_from_rate_change(
            face_value=FACE_VALUE,
            quantity=QUANTITY_100,
            mod_duration=DURATION_5Y,
            convexity_val=CONVEXITY_30,
            rate_change_bps=RATE_CHANGE_100BPS,
        )
        pct = float(result / position_value)
        assert -0.06 < pct < -0.04, f"Expected ~-5%, got {pct:.4%}"

    def test_100bps_cut_5y_duration_approx_pos5pct(self) -> None:
        """-100bps on 5Y duration should produce ~+5% gain."""
        position_value = FACE_VALUE * QUANTITY_100
        result = estimate_bond_pnl_from_rate_change(
            face_value=FACE_VALUE,
            quantity=QUANTITY_100,
            mod_duration=DURATION_5Y,
            convexity_val=CONVEXITY_30,
            rate_change_bps=RATE_CHANGE_NEG_100BPS,
        )
        pct = float(result / position_value)
        assert 0.04 < pct < 0.06, f"Expected ~+5%, got {pct:.4%}"

    def test_convexity_helps_in_large_moves(self) -> None:
        """Convexity should make large-move losses smaller than linear estimate."""
        # Linear (duration-only) loss for +500bps, 5Y duration = -25%
        # With convexity, actual loss should be less severe
        result_with_cx = estimate_bond_pnl_from_rate_change(
            face_value=FACE_VALUE,
            quantity=QUANTITY_100,
            mod_duration=DURATION_5Y,
            convexity_val=CONVEXITY_30,
            rate_change_bps=RATE_CHANGE_500BPS,
        )
        result_no_cx = estimate_bond_pnl_from_rate_change(
            face_value=FACE_VALUE,
            quantity=QUANTITY_100,
            mod_duration=DURATION_5Y,
            convexity_val=0.0,
            rate_change_bps=RATE_CHANGE_500BPS,
        )
        # Both are negative, but with convexity the loss is less severe
        assert result_with_cx > result_no_cx, (
            f"With convexity ({result_with_cx}) should be less negative "
            f"than without ({result_no_cx})"
        )

    def test_short_duration_floater_minimal_impact(self) -> None:
        """Very short duration (0.5Y floater) should have minimal rate impact."""
        position_value = FACE_VALUE * QUANTITY_100
        result = estimate_bond_pnl_from_rate_change(
            face_value=FACE_VALUE,
            quantity=QUANTITY_100,
            mod_duration=CORE_DURATION,
            convexity_val=CORE_CONVEXITY,
            rate_change_bps=RATE_CHANGE_300BPS,
        )
        pct = float(abs(result) / position_value)
        # 0.5Y duration x 3% change = ~1.5% -- very small
        assert pct < 0.02, f"Floater impact {pct:.4%} should be < 2%"


# ══════════════════════════════════════════════════════════════════════
# 2. TestModerateHike300bps
# ══════════════════════════════════════════════════════════════════════


class TestModerateHike300bps:
    """Tests for +300bps scenario (CBR hikes from e.g. 15.5% to 18.5%)."""

    @pytest.fixture
    def stress_result(self) -> StressResult:
        """Run the +300bps scenario on the standard portfolio."""
        positions, cash = _build_portfolio()
        scenario = next(s for s in STRESS_SCENARIOS if s.name == "moderate_hike")
        return run_portfolio_stress(positions, cash, scenario)

    def test_core_layer_impact_small(self, stress_result: StressResult) -> None:
        """Core layer (floaters) should have < 2% impact from +300bps."""
        core_pct = abs(stress_result.layer_pnl_pct["core"])
        assert core_pct < 0.02, f"Core layer loss {core_pct:.4%} should be < 2%"

    def test_strategic_layer_approx_12pct_loss(self, stress_result: StressResult) -> None:
        """Strategic layer should lose roughly 12% at +300bps with 4Y duration."""
        strategic_pct = stress_result.layer_pnl_pct["strategic"]
        # Duration 4Y x 3% = ~12% loss, convexity softens slightly
        assert -0.15 < strategic_pct < -0.08, (
            f"Strategic layer PnL {strategic_pct:.4%} outside expected -8% to -15%"
        )

    def test_portfolio_dd_within_10pct(self, stress_result: StressResult) -> None:
        """Total portfolio DD should be < 10% at +300bps (moderate hike)."""
        assert stress_result.dd_pct < PORTFOLIO_DD_LIMIT, (
            f"Portfolio DD {stress_result.dd_pct:.4%} should be < 10%"
        )
        assert not stress_result.breaches_portfolio_limit


# ══════════════════════════════════════════════════════════════════════
# 3. TestSevereHike500bps
# ══════════════════════════════════════════════════════════════════════


class TestSevereHike500bps:
    """Tests for +500bps scenario (CBR hikes from e.g. 15.5% to 20.5%)."""

    @pytest.fixture
    def stress_result(self) -> StressResult:
        """Run the +500bps scenario on the standard portfolio."""
        positions, cash = _build_portfolio()
        scenario = next(s for s in STRESS_SCENARIOS if s.name == "severe_hike")
        return run_portfolio_stress(positions, cash, scenario)

    def test_strategic_layer_approx_20pct_loss(self, stress_result: StressResult) -> None:
        """Strategic layer should lose ~20% at +500bps with 4Y duration."""
        strategic_pct = stress_result.layer_pnl_pct["strategic"]
        # Duration 4Y x 5% = ~20%, convexity helps somewhat
        assert -0.25 < strategic_pct < -0.14, (
            f"Strategic layer PnL {strategic_pct:.4%} outside expected -14% to -25%"
        )

    def test_portfolio_dd_approaches_limit(self, stress_result: StressResult) -> None:
        """Portfolio DD should approach but may not breach 10% at +500bps."""
        # With floater core and cash cushion, total portfolio DD is dampened
        # 27.5% * ~18% + 17.5% * ~13.5% + 45% * ~1.5% = ~8%
        assert stress_result.dd_pct < 0.15, (
            f"Portfolio DD {stress_result.dd_pct:.4%} seems too high for +500bps"
        )

    def test_non_core_layers_significant_losses(self, stress_result: StressResult) -> None:
        """Non-core layers (strategic + tactical) should have significant losses."""
        strategic_loss = abs(stress_result.layer_pnl_pct.get("strategic", 0.0))
        tactical_loss = abs(stress_result.layer_pnl_pct.get("tactical", 0.0))
        # Both should be meaningful
        assert strategic_loss > 0.10, f"Strategic loss {strategic_loss:.4%} should be > 10%"
        assert tactical_loss > 0.08, f"Tactical loss {tactical_loss:.4%} should be > 8%"


# ══════════════════════════════════════════════════════════════════════
# 4. TestExtremeHike1000bps
# ══════════════════════════════════════════════════════════════════════


class TestExtremeHike1000bps:
    """Tests for +1000bps scenario (2022-style: CBR 15.5% to 25.5%)."""

    @pytest.fixture
    def stress_result(self) -> StressResult:
        """Run the +1000bps scenario on the standard portfolio."""
        positions, cash = _build_portfolio()
        scenario = next(s for s in STRESS_SCENARIOS if s.name == "extreme_hike")
        return run_portfolio_stress(positions, cash, scenario)

    def test_portfolio_dd_breaches_limit(self, stress_result: StressResult) -> None:
        """Extreme +1000bps should breach the 10% portfolio DD limit."""
        assert stress_result.breaches_portfolio_limit, (
            f"Portfolio DD {stress_result.dd_pct:.4%} should breach 10% at +1000bps"
        )
        assert stress_result.dd_pct > PORTFOLIO_DD_LIMIT

    def test_core_layer_preserved(self, stress_result: StressResult) -> None:
        """Core layer (floaters, 0.5Y duration) should still be < 5% loss."""
        core_pct = abs(stress_result.layer_pnl_pct["core"])
        assert core_pct < 0.05, f"Core layer loss {core_pct:.4%} should be < 5%"

    def test_strategic_layer_massive_loss(self, stress_result: StressResult) -> None:
        """Strategic layer (4Y duration) should suffer heavy losses at +1000bps."""
        strategic_pct = stress_result.layer_pnl_pct["strategic"]
        # 4Y x 10% = ~40% linear, convexity helps -> ~35%
        assert strategic_pct < -0.25, (
            f"Strategic layer PnL {strategic_pct:.4%} should be worse than -25%"
        )


# ══════════════════════════════════════════════════════════════════════
# 5. TestModerateEasingCut300bps
# ══════════════════════════════════════════════════════════════════════


class TestModerateEasingCut300bps:
    """Tests for -300bps scenario (CBR easing from e.g. 15.5% to 12.5%)."""

    @pytest.fixture
    def stress_result(self) -> StressResult:
        """Run the -300bps scenario on the standard portfolio."""
        positions, cash = _build_portfolio()
        scenario = next(s for s in STRESS_SCENARIOS if s.name == "moderate_cut")
        return run_portfolio_stress(positions, cash, scenario)

    def test_positive_pnl_for_bond_holders(self, stress_result: StressResult) -> None:
        """Rate cut should produce positive PnL for the bond portfolio."""
        assert stress_result.pnl > PNL_ZERO, (
            f"PnL {stress_result.pnl} should be positive for rate cut"
        )

    def test_strategic_layer_gains_most(self, stress_result: StressResult) -> None:
        """Strategic layer (highest duration) should gain the most."""
        strategic_gain_pct = stress_result.layer_pnl_pct["strategic"]
        core_gain_pct = stress_result.layer_pnl_pct["core"]
        tactical_gain_pct = stress_result.layer_pnl_pct["tactical"]
        assert strategic_gain_pct > core_gain_pct, (
            f"Strategic gain {strategic_gain_pct:.4%} should exceed core {core_gain_pct:.4%}"
        )
        assert strategic_gain_pct > tactical_gain_pct, (
            f"Strategic gain {strategic_gain_pct:.4%} should exceed "
            f"tactical {tactical_gain_pct:.4%}"
        )


# ══════════════════════════════════════════════════════════════════════
# 6. TestPortfolioStressIntegration
# ══════════════════════════════════════════════════════════════════════


class TestPortfolioStressIntegration:
    """Integration tests running all scenarios and checking cross-scenario invariants."""

    @pytest.fixture
    def all_results(self) -> list[StressResult]:
        """Run all 4 standard scenarios."""
        positions, cash = _build_portfolio()
        return [run_portfolio_stress(positions, cash, scenario) for scenario in STRESS_SCENARIOS]

    def test_all_scenarios_produce_results(self, all_results: list[StressResult]) -> None:
        """All 4 standard scenarios should produce valid results."""
        assert len(all_results) == NUM_STANDARD_SCENARIOS
        for result in all_results:
            assert isinstance(result, StressResult)
            assert result.portfolio_value_before > PNL_ZERO

    def test_pnl_monotonic_in_rate_direction(self, all_results: list[StressResult]) -> None:
        """For fixed long-only bond positions, higher rate hikes = worse PnL."""
        # Sort by rate change (ascending: -300, +300, +500, +1000)
        sorted_results = sorted(all_results, key=lambda r: r.scenario.rate_change_bps)
        # For long-only bonds: lower rate change (more negative / less positive) = better PnL
        for i in range(len(sorted_results) - 1):
            left = sorted_results[i]
            right = sorted_results[i + 1]
            assert left.pnl > right.pnl, (
                f"PnL at {left.scenario.rate_change_bps}bps ({left.pnl}) "
                f"should be > PnL at {right.scenario.rate_change_bps}bps ({right.pnl})"
            )

    def test_convexity_makes_large_moves_better_than_linear(
        self, all_results: list[StressResult]
    ) -> None:
        """Convexity should make actual PnL slightly better than pure duration estimate."""
        positions, _ = _build_portfolio()
        # Check the extreme hike scenario
        extreme = next(r for r in all_results if r.scenario.name == "extreme_hike")
        # Compute linear-only PnL (no convexity) for comparison
        linear_pnl = Decimal(0)
        for layer_bonds in positions.values():
            for bond in layer_bonds:
                dy = Decimal(extreme.scenario.rate_change_bps) / Decimal(10000)
                pos_value = bond["face_value"] * bond["quantity"]
                linear_effect = -Decimal(str(bond["mod_duration"])) * dy * pos_value
                linear_pnl += linear_effect
        # Convexity adds a positive term, so actual PnL > linear PnL
        assert extreme.pnl > linear_pnl, (
            f"Actual PnL ({extreme.pnl}) should be better than "
            f"linear estimate ({linear_pnl}) due to convexity"
        )

    def test_breach_flag_correctly_set(self, all_results: list[StressResult]) -> None:
        """Portfolio DD breach flag should match the computed drawdown."""
        for result in all_results:
            if result.dd_pct > PORTFOLIO_DD_LIMIT:
                assert result.breaches_portfolio_limit, (
                    f"DD {result.dd_pct:.4%} > 10% but breach flag is False"
                )
            else:
                assert not result.breaches_portfolio_limit, (
                    f"DD {result.dd_pct:.4%} <= 10% but breach flag is True"
                )
