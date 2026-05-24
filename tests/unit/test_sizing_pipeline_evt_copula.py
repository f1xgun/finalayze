"""Unit tests for EVTStep, CopulaStep, and SizingContext extensions.

Tests cover:
- SizingContext default and custom returns_history
- CopulaStep scaling behaviour
- EVTStep no-op guards and active scaling
- Full pipeline integration with EVT + Copula steps
"""

from __future__ import annotations

import random
from decimal import Decimal

import pytest

from finalayze.risk.position_sizing_pipeline import (
    BrentGateStep,
    CopulaStep,
    EVTStep,
    HardCapsStep,
    PositionSizingPipeline,
    RubOilRegimeStep,
    SizingContext,
    VolTargetStep,
)
from finalayze.risk.regime import MarketRegime, RegimeState

# ---------------------------------------------------------------------------
# Constants (no magic numbers in assertions)
# ---------------------------------------------------------------------------

EQUITY = Decimal(100000)
BASE_POSITION = Decimal(10000)
MAX_POSITION_PCT = Decimal("0.20")
MIN_POSITION_SIZE = Decimal(500)
ASSET_VOL = Decimal("0.25")
TARGET_VOL = Decimal("0.15")
REGIME_SCALE = Decimal("1.0")
CORRELATION_SCALE_HALF = Decimal("0.5")
CORRELATION_SCALE_FULL = Decimal("1.0")

EVT_SCALE_FACTOR = Decimal("0.5")
EVT_MIN_HISTORY = 100
EVT_RECENT_WINDOW = 60

# Tolerance for Decimal comparisons with quantize
FOUR_DP = Decimal("0.0001")


# ---------------------------------------------------------------------------
# Helper factory
# ---------------------------------------------------------------------------


def _make_context(
    *,
    equity: Decimal = EQUITY,
    base_position: Decimal = BASE_POSITION,
    max_position_pct: Decimal = MAX_POSITION_PCT,
    min_position_size: Decimal = MIN_POSITION_SIZE,
    asset_vol: Decimal = ASSET_VOL,
    target_vol: Decimal = TARGET_VOL,
    regime_scale: Decimal = REGIME_SCALE,
    correlation_scale: Decimal = CORRELATION_SCALE_FULL,
    returns_history: tuple[float, ...] = (),
) -> SizingContext:
    """Build a SizingContext with sensible defaults for testing."""
    return SizingContext(
        equity=equity,
        base_position=base_position,
        max_position_pct=max_position_pct,
        min_position_size=min_position_size,
        asset_vol=asset_vol,
        target_vol=target_vol,
        regime_scale=regime_scale,
        correlation_scale=correlation_scale,
        returns_history=returns_history,
    )


# ---------------------------------------------------------------------------
# SizingContext tests
# ---------------------------------------------------------------------------


class TestSizingContext:
    def test_sizing_context_default_returns_history(self) -> None:
        """returns_history defaults to an empty tuple."""
        ctx = _make_context()
        assert ctx.returns_history == ()

    def test_sizing_context_with_returns_history(self) -> None:
        """returns_history accepts a non-empty tuple of floats."""
        history: tuple[float, ...] = (0.01, -0.02, 0.005, -0.03)
        ctx = _make_context(returns_history=history)
        assert ctx.returns_history == history
        assert len(ctx.returns_history) == len(history)


# ---------------------------------------------------------------------------
# CopulaStep tests
# ---------------------------------------------------------------------------


class TestCopulaStep:
    def test_copula_step_scales_by_correlation(self) -> None:
        """correlation_scale=0.5 should halve the position size."""
        step = CopulaStep()
        ctx = _make_context(correlation_scale=CORRELATION_SCALE_HALF)

        result = step.adjust(BASE_POSITION, ctx)

        expected = (BASE_POSITION * CORRELATION_SCALE_HALF).quantize(FOUR_DP)
        assert result == expected

    def test_copula_step_full_scale(self) -> None:
        """correlation_scale=1.0 should leave the position size unchanged."""
        step = CopulaStep()
        ctx = _make_context(correlation_scale=CORRELATION_SCALE_FULL)

        result = step.adjust(BASE_POSITION, ctx)

        expected = (BASE_POSITION * CORRELATION_SCALE_FULL).quantize(FOUR_DP)
        assert result == expected
        assert result == BASE_POSITION.quantize(FOUR_DP)


# ---------------------------------------------------------------------------
# EVTStep tests
# ---------------------------------------------------------------------------


def _normal_returns(
    n: int,
    *,
    mean: float = 0.001,
    std: float = 0.01,
    seed: int = 42,
) -> tuple[float, ...]:
    """Generate n normally distributed returns with a fixed seed."""
    rng = random.Random(seed)  # noqa: S311
    return tuple(rng.gauss(mean, std) for _ in range(n))


class TestEVTStep:
    def test_evt_step_no_history(self) -> None:
        """EVTStep returns size unchanged when returns_history is empty."""
        step = EVTStep()
        ctx = _make_context(returns_history=())

        result = step.adjust(BASE_POSITION, ctx)

        assert result == BASE_POSITION

    def test_evt_step_short_history(self) -> None:
        """EVTStep returns size unchanged when history has fewer than 100 returns."""
        step = EVTStep()
        short_history = _normal_returns(EVT_MIN_HISTORY - 1)
        ctx = _make_context(returns_history=short_history)

        result = step.adjust(BASE_POSITION, ctx)

        assert result == BASE_POSITION

    def test_evt_step_normal_market(self) -> None:
        """EVTStep should NOT scale down for mild, normally distributed returns.

        We build 200 returns (mean=0.001, std=0.01) with a seeded RNG so the test
        is deterministic, then replace the last 60 with mild negative returns
        (-0.005 to -0.001) that stay well within the historical loss distribution.
        The current_loss derived from the recent window will be small compared to
        the 99%-VaR estimated from the full history, so no scale-down occurs.
        """
        HISTORY_LEN = 200
        MILD_LOSS_RECENT = -0.003  # mild; far from tail threshold

        rng = random.Random(42)  # noqa: S311
        history_list = [rng.gauss(0.001, 0.01) for _ in range(HISTORY_LEN - EVT_RECENT_WINDOW)]
        # Recent window: mild losses, well inside typical return range
        recent = [MILD_LOSS_RECENT for _ in range(EVT_RECENT_WINDOW)]
        history: tuple[float, ...] = tuple(history_list + recent)

        step = EVTStep()
        ctx = _make_context(returns_history=history)

        result = step.adjust(BASE_POSITION, ctx)

        # Should be unchanged (no EVT scale-down applied)
        assert result == BASE_POSITION

    def test_evt_step_extreme_tail(self) -> None:
        """EVTStep should scale by 0.5x when recent returns include extreme losses.

        Construction rationale:
        - EVT.fit() uses the 95th-percentile of ALL losses as the GPD threshold.
          For the recent extremes to produce >= 30 exceedances, the extremes must
          represent < 5% of total losses so the threshold is set by the bulk, not
          by the extremes themselves.
        - With a base of 1200 N(0, 0.01) returns (~600 negative) and 60 recent
          extreme returns (-0.50 to -0.559), the extremes are ~9% of total losses
          (60/660) but their magnitude (0.50+) is so far above the bulk 95th-pctile
          (~0.025) that at least 30 of them are exceedances, satisfying the GPD
          fit requirement.
        - current_loss = abs(min(recent)) = 0.559 > fitted 99%-VaR => elevated.
        """
        N_BASE = 1200  # large enough so extremes < 5% of total losses
        EXTREME_LOSS_START = -0.50  # base extreme; clearly far past N(0,0.01) tails
        EXTREME_LOSS_STEP = -0.001  # slight variation across the 60 recent values

        rng = random.Random(42)  # noqa: S311
        base_history = [rng.gauss(0.0, 0.01) for _ in range(N_BASE)]
        # 60 recent losses slightly varied so the GPD threshold (95th pctile of all
        # losses) falls below at least 30 of them, enabling a valid GPD fit.
        recent = [EXTREME_LOSS_START + i * EXTREME_LOSS_STEP for i in range(EVT_RECENT_WINDOW)]
        history: tuple[float, ...] = tuple(base_history + recent)

        step = EVTStep()
        ctx = _make_context(returns_history=history)

        result = step.adjust(BASE_POSITION, ctx)

        expected = (BASE_POSITION * EVT_SCALE_FACTOR).quantize(FOUR_DP)
        assert result == expected


# ---------------------------------------------------------------------------
# Pipeline integration test
# ---------------------------------------------------------------------------


class TestPipelineWithEVTAndCopula:
    def test_pipeline_with_evt_and_copula_steps(self) -> None:
        """Pipeline with [VolTargetStep, CopulaStep, EVTStep, HardCapsStep]
        should run without error and return a non-negative Decimal."""
        steps = [
            VolTargetStep(),
            CopulaStep(),
            EVTStep(),
            HardCapsStep(),
        ]
        pipeline = PositionSizingPipeline(steps=steps)

        # Provide a modest returns history (more than EVT_MIN_HISTORY entries)
        rng = random.Random(7)  # noqa: S311
        history: tuple[float, ...] = tuple(rng.gauss(0.001, 0.01) for _ in range(150))

        ctx = _make_context(
            correlation_scale=Decimal("0.8"),
            returns_history=history,
        )

        result = pipeline.compute(ctx)

        assert isinstance(result, Decimal)
        assert result >= Decimal(0)


# ---------------------------------------------------------------------------
# Mock RubOilRegimeSignal for testing
# ---------------------------------------------------------------------------


class _MockRubOilRegimeSignal:
    """Mock that returns a configurable RegimeState."""

    def __init__(self, regime_state: RegimeState) -> None:
        self._state = regime_state

    def get_regime(self, candles: list[object], bar_index: int) -> RegimeState:  # noqa: ARG002
        return self._state


# ---------------------------------------------------------------------------
# RubOilRegimeStep tests
# ---------------------------------------------------------------------------

SCALE_NORMAL = Decimal("1.0")
SCALE_ELEVATED = Decimal("0.5")
SCALE_CRISIS = Decimal("0.25")


class TestRubOilRegimeStep:
    def test_normal_regime_unchanged(self) -> None:
        """NORMAL regime (scale=1.0) should leave size unchanged for ru_* segment."""
        state = RegimeState(
            regime=MarketRegime.NORMAL,
            allow_new_longs=True,
            position_scale=SCALE_NORMAL,
        )
        step = RubOilRegimeStep(_MockRubOilRegimeSignal(state), segment_id="ru_blue_chips")
        ctx = _make_context()

        result = step.adjust(BASE_POSITION, ctx)

        expected = (BASE_POSITION * SCALE_NORMAL).quantize(FOUR_DP)
        assert result == expected

    def test_elevated_regime_halves(self) -> None:
        """ELEVATED regime (scale=0.5) should halve position for ru_* segment."""
        state = RegimeState(
            regime=MarketRegime.ELEVATED,
            allow_new_longs=True,
            position_scale=SCALE_ELEVATED,
        )
        step = RubOilRegimeStep(_MockRubOilRegimeSignal(state), segment_id="ru_energy")
        ctx = _make_context()

        result = step.adjust(BASE_POSITION, ctx)

        expected = (BASE_POSITION * SCALE_ELEVATED).quantize(FOUR_DP)
        assert result == expected

    def test_crisis_regime_quarters(self) -> None:
        """CRISIS regime (scale=0.25) should quarter position for ru_* segment."""
        state = RegimeState(
            regime=MarketRegime.CRISIS,
            allow_new_longs=False,
            position_scale=SCALE_CRISIS,
        )
        step = RubOilRegimeStep(_MockRubOilRegimeSignal(state), segment_id="ru_blue_chips")
        ctx = _make_context()

        result = step.adjust(BASE_POSITION, ctx)

        expected = (BASE_POSITION * SCALE_CRISIS).quantize(FOUR_DP)
        assert result == expected

    def test_non_ru_segment_passthrough(self) -> None:
        """Non-ru_* segment should pass through unchanged regardless of regime."""
        state = RegimeState(
            regime=MarketRegime.CRISIS,
            allow_new_longs=False,
            position_scale=SCALE_CRISIS,
        )
        step = RubOilRegimeStep(_MockRubOilRegimeSignal(state), segment_id="us_tech")
        ctx = _make_context()

        result = step.adjust(BASE_POSITION, ctx)

        assert result == BASE_POSITION


# ---------------------------------------------------------------------------
# BrentGateStep tests
# ---------------------------------------------------------------------------

BRENT_RUB_THRESHOLD = 5000.0
BRENT_GATE_SCALE = Decimal("0.5")


class TestBrentGateStep:
    def test_above_threshold_unchanged(self) -> None:
        """Brent-in-RUB >= 5000 should leave size unchanged for ru_energy."""
        step = BrentGateStep(brent_rub_price=6000.0, segment_id="ru_energy")
        ctx = _make_context()

        result = step.adjust(BASE_POSITION, ctx)

        assert result == BASE_POSITION

    def test_below_threshold_halves(self) -> None:
        """Brent-in-RUB < 5000 should halve position for ru_energy."""
        step = BrentGateStep(brent_rub_price=4500.0, segment_id="ru_energy")
        ctx = _make_context()

        result = step.adjust(BASE_POSITION, ctx)

        expected = (BASE_POSITION * BRENT_GATE_SCALE).quantize(FOUR_DP)
        assert result == expected

    def test_non_energy_segment_unchanged(self) -> None:
        """Non-ru_energy segment should pass through unchanged."""
        step = BrentGateStep(brent_rub_price=3000.0, segment_id="ru_blue_chips")
        ctx = _make_context()

        result = step.adjust(BASE_POSITION, ctx)

        assert result == BASE_POSITION

    def test_missing_data_graceful(self) -> None:
        """brent_rub=0.0 (missing data) should pass through unchanged."""
        step = BrentGateStep(brent_rub_price=0.0, segment_id="ru_energy")
        ctx = _make_context()

        result = step.adjust(BASE_POSITION, ctx)

        assert result == BASE_POSITION

    def test_us_segment_unchanged(self) -> None:
        """US segment should pass through unchanged even with low Brent."""
        step = BrentGateStep(brent_rub_price=2000.0, segment_id="us_tech")
        ctx = _make_context()

        result = step.adjust(BASE_POSITION, ctx)

        assert result == BASE_POSITION
