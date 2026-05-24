"""Tests for Thompson sampling exploration in AdaptiveStrategyCombiner."""

from __future__ import annotations

from decimal import Decimal

from finalayze.core.schemas import Candle, Signal, SignalDirection
from finalayze.strategies.adaptive_combiner import AdaptiveStrategyCombiner
from finalayze.strategies.base import BaseStrategy

# ---------------------------------------------------------------------------
# Constants (no magic numbers -- ruff PLR2004)
# ---------------------------------------------------------------------------
_DEFAULT_EPSILON = 0.1
_CUSTOM_EPSILON = 0.5
_ZERO_EPSILON = 0.0
_FULL_EPSILON = 1.0
_NUM_OUTCOMES = 10
_WEIGHT_SUM_TOLERANCE = 1e-9
_WEIGHT_LOWER_BOUND = 0.0
_WEIGHT_UPPER_BOUND = 1.0
_SAMPLING_ITERATIONS = 50
_INITIAL_ALPHA = 1
_INITIAL_BETA = 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
class _StubStrategy(BaseStrategy):
    """Minimal strategy stub for testing the combiner."""

    def __init__(self, name: str) -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def supported_segments(self) -> list[str]:
        return ["us_tech"]

    def generate_signal(
        self,
        symbol: str,
        candles: list[Candle],
        segment_id: str,
        sentiment_score: float = 0.0,
        has_open_position: bool = False,
    ) -> Signal | None:
        return Signal(
            strategy_name=self._name,
            symbol=symbol,
            market_id="us",
            segment_id=segment_id,
            direction=SignalDirection.BUY,
            confidence=0.8,
            strategy_payload={},
            reasoning="stub",
        )

    def get_parameters(self, segment_id: str) -> dict[str, object]:
        return {}


def _make_combiner(
    strategies: list[BaseStrategy] | None = None,
    exploration_mode: str = "deterministic",
    epsilon: float = _DEFAULT_EPSILON,
) -> AdaptiveStrategyCombiner:
    if strategies is None:
        strategies = [
            _StubStrategy("momentum"),
            _StubStrategy("mean_reversion"),
        ]
    combiner = AdaptiveStrategyCombiner(
        strategies,
        segment_id="us_tech",
        exploration_mode=exploration_mode,
        epsilon=epsilon,
    )
    # Avoid YAML loading
    combiner._presets_dir = combiner._presets_dir / "__nonexistent__"
    return combiner


# ===========================================================================
# Tests
# ===========================================================================


class TestThompsonSamplingInit:
    """Tests for initialization parameters."""

    def test_default_mode_is_deterministic(self) -> None:
        combiner = _make_combiner()
        assert combiner._exploration_mode == "deterministic"

    def test_thompson_mode_accepted(self) -> None:
        combiner = _make_combiner(exploration_mode="thompson")
        assert combiner._exploration_mode == "thompson"

    def test_default_epsilon(self) -> None:
        combiner = _make_combiner()
        assert combiner._epsilon == _DEFAULT_EPSILON

    def test_custom_epsilon(self) -> None:
        combiner = _make_combiner(epsilon=_CUSTOM_EPSILON)
        assert combiner._epsilon == _CUSTOM_EPSILON

    def test_initial_outcome_counts_empty(self) -> None:
        combiner = _make_combiner(exploration_mode="thompson")
        assert combiner._outcome_counts == {}


class TestRecordOutcome:
    """Tests for recording success/failure outcomes."""

    def test_record_success_increments_alpha(self) -> None:
        combiner = _make_combiner(exploration_mode="thompson")
        combiner.record_outcome("momentum", profitable=True)
        alpha, beta = combiner._outcome_counts["momentum"]
        assert alpha == _INITIAL_ALPHA + 1
        assert beta == _INITIAL_BETA

    def test_record_failure_increments_beta(self) -> None:
        combiner = _make_combiner(exploration_mode="thompson")
        combiner.record_outcome("momentum", profitable=False)
        alpha, beta = combiner._outcome_counts["momentum"]
        assert alpha == _INITIAL_ALPHA
        assert beta == _INITIAL_BETA + 1

    def test_multiple_outcomes_accumulate(self) -> None:
        combiner = _make_combiner(exploration_mode="thompson")
        for _ in range(_NUM_OUTCOMES):
            combiner.record_outcome("momentum", profitable=True)
        for _ in range(_NUM_OUTCOMES):
            combiner.record_outcome("momentum", profitable=False)
        alpha, beta = combiner._outcome_counts["momentum"]
        assert alpha == _INITIAL_ALPHA + _NUM_OUTCOMES
        assert beta == _INITIAL_BETA + _NUM_OUTCOMES

    def test_record_outcome_works_in_deterministic_mode(self) -> None:
        """record_outcome should work even in deterministic mode (just stores data)."""
        combiner = _make_combiner(exploration_mode="deterministic")
        combiner.record_outcome("momentum", profitable=True)
        assert "momentum" in combiner._outcome_counts


class TestThompsonWeights:
    """Tests for Thompson sampling weight computation."""

    def test_thompson_weights_sum_to_one(self) -> None:
        combiner = _make_combiner(exploration_mode="thompson")
        combiner.record_outcome("momentum", profitable=True)
        combiner.record_outcome("mean_reversion", profitable=True)
        weights = combiner._compute_thompson_weights()
        total = sum(weights.values())
        assert abs(total - _WEIGHT_UPPER_BOUND) < _WEIGHT_SUM_TOLERANCE

    def test_thompson_weights_in_valid_range(self) -> None:
        combiner = _make_combiner(exploration_mode="thompson")
        combiner.record_outcome("momentum", profitable=True)
        combiner.record_outcome("mean_reversion", profitable=False)
        weights = combiner._compute_thompson_weights()
        for w in weights.values():
            assert _WEIGHT_LOWER_BOUND <= w <= _WEIGHT_UPPER_BOUND

    def test_thompson_weights_only_for_known_strategies(self) -> None:
        combiner = _make_combiner(exploration_mode="thompson")
        # Record for strategies that are in the combiner
        combiner.record_outcome("momentum", profitable=True)
        combiner.record_outcome("mean_reversion", profitable=True)
        weights = combiner._compute_thompson_weights()
        assert set(weights.keys()) == {"momentum", "mean_reversion"}

    def test_thompson_weights_use_prior_when_no_outcomes(self) -> None:
        """With no outcomes recorded, all strategies get uniform prior."""
        combiner = _make_combiner(exploration_mode="thompson")
        weights = combiner._compute_thompson_weights()
        # With uniform Beta(1,1) prior, weights should exist for all strategies
        assert len(weights) == 2  # noqa: PLR2004

    def test_exploration_produces_varied_weights(self) -> None:
        """Thompson sampling should produce different weights across samples."""
        combiner = _make_combiner(exploration_mode="thompson")
        # Give one strategy much better outcomes
        for _ in range(_NUM_OUTCOMES):
            combiner.record_outcome("momentum", profitable=True)
            combiner.record_outcome("mean_reversion", profitable=False)

        weight_sets: list[dict[str, float]] = []
        for _ in range(_SAMPLING_ITERATIONS):
            weights = combiner._compute_thompson_weights()
            weight_sets.append(weights)

        # At least some variation in weights across samples
        momentum_weights = [ws["momentum"] for ws in weight_sets]
        assert max(momentum_weights) != min(momentum_weights)


class TestEpsilonBlending:
    """Tests for epsilon-greedy blending of exploration and exploitation."""

    def test_zero_epsilon_gives_pure_exploitation(self) -> None:
        """With epsilon=0, Thompson weights should not affect the result."""
        combiner = _make_combiner(
            exploration_mode="thompson",
            epsilon=_ZERO_EPSILON,
        )
        combiner.record_outcome("momentum", profitable=True)
        exploit = {"momentum": Decimal("0.7"), "mean_reversion": Decimal("0.3")}
        blended = combiner._blend_weights(exploit)
        for name, value in exploit.items():
            assert blended[name] == value

    def test_full_epsilon_gives_pure_exploration(self) -> None:
        """With epsilon=1, result should be entirely from Thompson sampling."""
        combiner = _make_combiner(
            exploration_mode="thompson",
            epsilon=_FULL_EPSILON,
        )
        for _ in range(_NUM_OUTCOMES):
            combiner.record_outcome("momentum", profitable=True)
            combiner.record_outcome("mean_reversion", profitable=False)

        exploit = {"momentum": Decimal("0.5"), "mean_reversion": Decimal("0.5")}
        blended = combiner._blend_weights(exploit)
        # With epsilon=1.0, result is purely Thompson weights (converted to Decimal)
        total = sum(blended.values())
        assert abs(total - Decimal(1)) < Decimal(str(_WEIGHT_SUM_TOLERANCE))

    def test_deterministic_mode_blend_returns_unchanged(self) -> None:
        """In deterministic mode, _blend_weights should return exploit weights unchanged."""
        combiner = _make_combiner(exploration_mode="deterministic")
        exploit = {"momentum": Decimal("0.6"), "mean_reversion": Decimal("0.4")}
        blended = combiner._blend_weights(exploit)
        assert blended == exploit

    def test_blended_weights_sum_to_one(self) -> None:
        combiner = _make_combiner(
            exploration_mode="thompson",
            epsilon=_CUSTOM_EPSILON,
        )
        combiner.record_outcome("momentum", profitable=True)
        combiner.record_outcome("mean_reversion", profitable=True)
        exploit = {"momentum": Decimal("0.7"), "mean_reversion": Decimal("0.3")}
        blended = combiner._blend_weights(exploit)
        total = sum(blended.values())
        assert abs(total - Decimal(1)) < Decimal(str(_WEIGHT_SUM_TOLERANCE))
