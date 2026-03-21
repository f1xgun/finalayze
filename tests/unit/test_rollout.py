"""Tests for rollout phase configuration (ROLL-01)."""

from __future__ import annotations

from decimal import Decimal

from finalayze.core.modes import RolloutPhase
from finalayze.risk.rollout import ROLLOUT_LIMITS, RolloutLimits


class TestRolloutPhaseEnum:
    def test_rollout_phase_is_strenum(self) -> None:
        assert issubclass(RolloutPhase, str)
        assert RolloutPhase.MINIMAL == "minimal"
        assert RolloutPhase.STANDARD == "standard"
        assert RolloutPhase.FULL == "full"

    def test_rollout_phase_has_exactly_3_members(self) -> None:
        assert len(RolloutPhase) == 3
        assert set(RolloutPhase) == {
            RolloutPhase.MINIMAL,
            RolloutPhase.STANDARD,
            RolloutPhase.FULL,
        }


class TestRolloutLimitsFrozen:
    def test_rollout_limits_is_frozen_dataclass(self) -> None:
        limits = ROLLOUT_LIMITS[RolloutPhase.FULL]
        import pytest

        with pytest.raises(AttributeError):
            limits.max_position_pct = Decimal("0.99")  # type: ignore[misc]

    def test_rollout_limits_fields(self) -> None:
        limits = ROLLOUT_LIMITS[RolloutPhase.FULL]
        assert isinstance(limits.max_position_pct, Decimal)
        assert isinstance(limits.max_positions_per_market, int)
        assert isinstance(limits.daily_loss_limit_pct, float)
        assert isinstance(limits.circuit_breaker_l1, float)
        assert isinstance(limits.circuit_breaker_l2, float)
        assert isinstance(limits.circuit_breaker_l3, float)
        assert isinstance(limits.max_sector_concentration_pct, Decimal)
        assert isinstance(limits.min_cash_reserve_pct, Decimal)


class TestRolloutLimitsMapping:
    def test_rollout_limits_has_exactly_3_keys(self) -> None:
        assert len(ROLLOUT_LIMITS) == 3
        assert set(ROLLOUT_LIMITS.keys()) == set(RolloutPhase)

    def test_minimal_limits(self) -> None:
        limits = ROLLOUT_LIMITS[RolloutPhase.MINIMAL]
        assert limits.max_position_pct == Decimal("0.03")
        assert limits.max_positions_per_market == 5
        assert limits.daily_loss_limit_pct == 0.01
        assert limits.circuit_breaker_l1 == 0.01
        assert limits.circuit_breaker_l2 == 0.02
        assert limits.circuit_breaker_l3 == 0.03
        assert limits.max_sector_concentration_pct == Decimal("0.20")
        assert limits.min_cash_reserve_pct == Decimal("0.40")

    def test_standard_limits(self) -> None:
        limits = ROLLOUT_LIMITS[RolloutPhase.STANDARD]
        assert limits.max_position_pct == Decimal("0.10")
        assert limits.max_positions_per_market == 8
        assert limits.daily_loss_limit_pct == 0.03
        assert limits.circuit_breaker_l1 == 0.03
        assert limits.circuit_breaker_l2 == 0.05
        assert limits.circuit_breaker_l3 == 0.10
        assert limits.max_sector_concentration_pct == Decimal("0.30")
        assert limits.min_cash_reserve_pct == Decimal("0.30")

    def test_full_matches_defaults(self) -> None:
        """FULL phase limits must match current Settings defaults for backward compat."""
        limits = ROLLOUT_LIMITS[RolloutPhase.FULL]
        assert limits.max_position_pct == Decimal("0.20")
        assert limits.max_positions_per_market == 10
        assert limits.daily_loss_limit_pct == 0.02
        assert limits.circuit_breaker_l1 == 0.05
        assert limits.circuit_breaker_l2 == 0.10
        assert limits.circuit_breaker_l3 == 0.15
        assert limits.max_sector_concentration_pct == Decimal("0.40")
        assert limits.min_cash_reserve_pct == Decimal("0.20")
