"""Tests for rollout phase configuration (ROLL-01, ROLL-02, ROLL-03)."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest

from finalayze.core.modes import RolloutPhase
from finalayze.risk.circuit_breaker import CircuitBreaker, CircuitLevel, CrossMarketCircuitBreaker
from finalayze.risk.loss_limits import LossLimitTracker
from finalayze.risk.pre_trade_check import PreTradeChecker
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


class TestSettingsRolloutIntegration:
    """Tests for Settings.rollout_phase and effective_risk_limits()."""

    def test_settings_rollout_phase_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Default rollout_phase is FULL (backward compatible)."""
        monkeypatch.delenv("FINALAYZE_ROLLOUT_PHASE", raising=False)
        from config.settings import Settings

        s = Settings()
        assert s.rollout_phase == RolloutPhase.FULL

    def test_settings_rollout_phase_env_override(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("FINALAYZE_ROLLOUT_PHASE", "minimal")
        from config.settings import Settings

        s = Settings()
        assert s.rollout_phase == RolloutPhase.MINIMAL

    def test_effective_risk_limits_full(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("FINALAYZE_ROLLOUT_PHASE", raising=False)
        from config.settings import Settings

        s = Settings()
        limits = s.effective_risk_limits()
        assert limits == ROLLOUT_LIMITS[RolloutPhase.FULL]

    def test_effective_risk_limits_minimal(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("FINALAYZE_ROLLOUT_PHASE", "minimal")
        from config.settings import Settings

        s = Settings()
        limits = s.effective_risk_limits()
        assert limits.max_position_pct == Decimal("0.03")

    def test_effective_risk_limits_standard(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("FINALAYZE_ROLLOUT_PHASE", "standard")
        from config.settings import Settings

        s = Settings()
        limits = s.effective_risk_limits()
        assert limits.circuit_breaker_l2 == 0.05


class TestRolloutWiring:
    """Tests for wiring rollout limits into risk components (ROLL-02)."""

    # MOEX market hours: 07:00-15:45 UTC
    _MOEX_OPEN = datetime(2026, 3, 23, 10, 0, 0, tzinfo=UTC)  # Monday 10:00 UTC

    def test_pretrade_minimal_position_cap(self) -> None:
        """MINIMAL limits (3% max position) should reject 4% position."""
        checker = PreTradeChecker(
            max_position_pct=Decimal("0.03"),
            max_positions_per_market=5,
        )
        result = checker.check(
            order_value=Decimal("4000"),
            portfolio_equity=Decimal("100000"),
            available_cash=Decimal("100000"),
            open_position_count=0,
            market_id="moex",
            dt=self._MOEX_OPEN,
        )
        assert not result.passed
        assert any("position" in v.lower() or "exposure" in v.lower() for v in result.violations)

    def test_pretrade_minimal_position_pass(self) -> None:
        """MINIMAL limits (3% max position) should allow 2.5% position."""
        checker = PreTradeChecker(
            max_position_pct=Decimal("0.03"),
            max_positions_per_market=5,
        )
        result = checker.check(
            order_value=Decimal("2500"),
            portfolio_equity=Decimal("100000"),
            available_cash=Decimal("100000"),
            open_position_count=0,
            market_id="moex",
            dt=self._MOEX_OPEN,
        )
        assert result.passed

    def test_circuit_breaker_minimal_dd(self) -> None:
        """MINIMAL CB l2=0.02 should HALT at 2.1% drawdown."""
        cb = CircuitBreaker(
            market_id="moex",
            l1_threshold=0.01,
            l2_threshold=0.02,
            l3_threshold=0.03,
        )
        baseline = Decimal("100000")
        current = Decimal("97900")  # 2.1% drawdown
        level = cb.check(current, baseline)
        assert level == CircuitLevel.HALTED

    def test_circuit_breaker_full_dd(self) -> None:
        """FULL CB l2=0.10 should stay NORMAL at 2.1% drawdown."""
        cb = CircuitBreaker(
            market_id="moex",
            l1_threshold=0.05,
            l2_threshold=0.10,
            l3_threshold=0.15,
        )
        baseline = Decimal("100000")
        current = Decimal("97900")  # 2.1% drawdown
        level = cb.check(current, baseline)
        assert level == CircuitLevel.NORMAL

    def test_loss_limit_minimal(self) -> None:
        """LossLimitTracker with 1% daily limit should halt at 1% loss."""
        tracker = LossLimitTracker(daily_loss_limit_pct=1.0)  # 1% in percent form
        now = datetime(2026, 3, 21, 12, 0, 0, tzinfo=UTC)
        tracker.reset_day(now, Decimal("100000"))
        # 1.1% loss should trigger halt
        assert tracker.is_halted(now, Decimal("98900"))

    def test_cross_market_breaker_default(self) -> None:
        """CrossMarketCircuitBreaker default halt_threshold should be 0.10, not 0.80."""
        breaker = CrossMarketCircuitBreaker()
        # Internal threshold should be 0.10 (10% drawdown)
        assert breaker._threshold == Decimal("0.10")
