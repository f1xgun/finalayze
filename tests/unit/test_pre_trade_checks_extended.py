"""Tests for extended pre-trade checks 12-14 (C.3)."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

from finalayze.risk.pre_trade_check import PreTradeChecker
from finalayze.risk.regime import MarketRegime, RegimeState

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MARKET_OPEN_DT = datetime(2026, 3, 2, 15, 0, tzinfo=UTC)  # Monday 15:00 UTC = US open


def _base_check_kwargs() -> dict:
    """Base kwargs that pass all 11 original checks."""
    return {
        "order_value": Decimal(5000),
        "portfolio_equity": Decimal(100000),
        "available_cash": Decimal(50000),
        "open_position_count": 3,
        "market_id": "us",
        "dt": _MARKET_OPEN_DT,
        "stop_loss_price": Decimal(95),
        "require_stop_loss": True,
        "has_pending_order": False,
        "symbol": "AAPL",
    }


# ---------------------------------------------------------------------------
# Check 12: Regime gate
# ---------------------------------------------------------------------------


class TestCheck12RegimeGate:
    def test_regime_crisis_blocks_longs(self) -> None:
        """CRISIS regime should block new longs."""
        checker = PreTradeChecker()
        result = checker.check(
            **_base_check_kwargs(),
            regime_state=RegimeState.crisis(),
        )
        assert not result.passed
        violations_str = " ".join(result.violations)
        assert "check 12 fail" in violations_str.lower()
        assert "blocks new longs" in violations_str.lower()

    def test_regime_elevated_below_sma_blocks(self) -> None:
        """ELEVATED regime with allow_new_longs=False should block."""
        checker = PreTradeChecker()
        state = RegimeState(
            regime=MarketRegime.ELEVATED,
            allow_new_longs=False,
            position_scale=Decimal("0.5"),
        )
        result = checker.check(
            **_base_check_kwargs(),
            regime_state=state,
        )
        assert not result.passed
        violations_str = " ".join(result.violations)
        assert "check 12 fail" in violations_str.lower()
        assert "blocks new longs" in violations_str.lower()

    def test_regime_normal_allows(self) -> None:
        """NORMAL regime should pass check 12."""
        checker = PreTradeChecker()
        result = checker.check(
            **_base_check_kwargs(),
            regime_state=RegimeState.normal(),
        )
        assert result.passed

    def test_regime_none_skips(self) -> None:
        """When regime_state is None, check 12 is skipped."""
        checker = PreTradeChecker()
        result = checker.check(**_base_check_kwargs())
        assert result.passed


# ---------------------------------------------------------------------------
# Check 13: Parameter freshness
# ---------------------------------------------------------------------------


class TestCheck13ParamFreshness:
    def test_stale_ou_params_fail(self) -> None:
        """OU strategy with param_age_bars > 5 should fail."""
        checker = PreTradeChecker()
        result = checker.check(
            **_base_check_kwargs(),
            strategy_name="ou_mean_reversion",
            param_age_bars=10,
        )
        assert not result.passed
        violations_str = " ".join(result.violations)
        assert "params stale" in violations_str.lower()

    def test_fresh_ou_params_pass(self) -> None:
        """OU strategy with param_age_bars <= 5 should pass."""
        checker = PreTradeChecker()
        result = checker.check(
            **_base_check_kwargs(),
            strategy_name="ou_mean_reversion",
            param_age_bars=3,
        )
        assert result.passed

    def test_stale_pairs_params_fail(self) -> None:
        """Pairs strategy with param_age_bars > 5 should fail."""
        checker = PreTradeChecker()
        result = checker.check(
            **_base_check_kwargs(),
            strategy_name="pairs",
            param_age_bars=6,
        )
        assert not result.passed
        violations_str = " ".join(result.violations)
        assert "params stale" in violations_str.lower()

    def test_non_ou_strategy_skips(self) -> None:
        """Momentum strategy should skip check 13 even with stale params."""
        checker = PreTradeChecker()
        result = checker.check(
            **_base_check_kwargs(),
            strategy_name="momentum",
            param_age_bars=100,
        )
        assert result.passed

    def test_none_strategy_skips(self) -> None:
        """None strategy_name should skip check 13."""
        checker = PreTradeChecker()
        result = checker.check(
            **_base_check_kwargs(),
            param_age_bars=100,
        )
        assert result.passed

    def test_none_param_age_skips(self) -> None:
        """None param_age_bars should skip check 13 even for OU."""
        checker = PreTradeChecker()
        result = checker.check(
            **_base_check_kwargs(),
            strategy_name="ou_mean_reversion",
        )
        assert result.passed

    def test_boundary_5_passes(self) -> None:
        """param_age_bars == 5 should pass (> 5 is the threshold)."""
        checker = PreTradeChecker()
        result = checker.check(
            **_base_check_kwargs(),
            strategy_name="ou_mean_reversion",
            param_age_bars=5,
        )
        assert result.passed


# ---------------------------------------------------------------------------
# Check 14: Correlation position limit
# ---------------------------------------------------------------------------


class TestCheck14CorrelationLimit:
    def test_too_many_correlated_fails(self) -> None:
        """3+ correlated positions should fail."""
        correlations = {
            ("AAPL", "MSFT"): 0.8,
            ("AAPL", "GOOG"): 0.9,
            ("AAPL", "AMZN"): 0.75,
        }
        checker = PreTradeChecker()
        result = checker.check(
            **_base_check_kwargs(),
            open_positions=["MSFT", "GOOG", "AMZN"],
            correlations=correlations,
        )
        assert not result.passed
        violations_str = " ".join(result.violations)
        assert "correlated positions" in violations_str.lower()

    def test_few_correlated_passes(self) -> None:
        """1 correlated position should pass."""
        correlations = {
            ("AAPL", "MSFT"): 0.8,
            ("AAPL", "GOOG"): 0.3,
        }
        checker = PreTradeChecker()
        result = checker.check(
            **_base_check_kwargs(),
            open_positions=["MSFT", "GOOG"],
            correlations=correlations,
        )
        assert result.passed

    def test_exactly_two_correlated_passes(self) -> None:
        """2 correlated positions (below max 3) should pass."""
        correlations = {
            ("AAPL", "MSFT"): 0.8,
            ("AAPL", "GOOG"): 0.9,
            ("AAPL", "AMZN"): 0.3,
        }
        checker = PreTradeChecker()
        result = checker.check(
            **_base_check_kwargs(),
            open_positions=["MSFT", "GOOG", "AMZN"],
            correlations=correlations,
        )
        assert result.passed

    def test_no_correlations_skips(self) -> None:
        """When correlations is None, check 14 is skipped."""
        checker = PreTradeChecker()
        result = checker.check(
            **_base_check_kwargs(),
            open_positions=["MSFT", "GOOG"],
        )
        assert result.passed

    def test_no_open_positions_skips(self) -> None:
        """When open_positions is None, check 14 is skipped."""
        checker = PreTradeChecker()
        result = checker.check(
            **_base_check_kwargs(),
            correlations={("AAPL", "MSFT"): 0.9},
        )
        assert result.passed
