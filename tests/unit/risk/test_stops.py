"""Unit tests for risk/stops.py — unified ATR stop multiplier resolution (S1.4)."""

from __future__ import annotations

from decimal import Decimal

import pytest

from finalayze.risk.stops import (
    DEFAULT_STRATEGY_STOP_ATR,
    resolve_stop_atr_multiplier,
)


class TestResolveStopAtrMultiplier:
    """Single function consumed by backtest engine + live signal_executor + retro stop path."""

    def test_known_strategy_returns_table_value(self) -> None:
        result = resolve_stop_atr_multiplier("momentum")
        assert result == Decimal(str(DEFAULT_STRATEGY_STOP_ATR["momentum"]))

    def test_unknown_strategy_falls_back(self) -> None:
        result = resolve_stop_atr_multiplier("does_not_exist")
        assert result == Decimal("3.0")  # _DEFAULT_STOP_ATR_FALLBACK

    def test_segment_id_moex_applies_uplift(self) -> None:
        us_mult = resolve_stop_atr_multiplier("momentum", segment_id="us_tech")
        ru_mult = resolve_stop_atr_multiplier("momentum", segment_id="ru_blue_chips")
        assert ru_mult == us_mult * Decimal("1.2")

    def test_market_id_moex_applies_uplift(self) -> None:
        # S1.4: live path passes market_id (no segment knowledge at fill time).
        us_mult = resolve_stop_atr_multiplier("momentum", market_id="us")
        ru_mult = resolve_stop_atr_multiplier("momentum", market_id="moex")
        assert ru_mult == us_mult * Decimal("1.2")

    def test_market_id_and_segment_id_consistent(self) -> None:
        seg = resolve_stop_atr_multiplier("momentum", segment_id="ru_blue_chips")
        mkt = resolve_stop_atr_multiplier("momentum", market_id="moex")
        assert seg == mkt

    def test_either_path_triggers_moex_uplift(self) -> None:
        # If either identifier indicates MOEX, uplift is applied (defensive OR).
        mkt_only = resolve_stop_atr_multiplier("momentum", market_id="moex")
        seg_only = resolve_stop_atr_multiplier("momentum", segment_id="ru_blue_chips")
        both = resolve_stop_atr_multiplier("momentum", segment_id="ru_blue_chips", market_id="moex")
        assert mkt_only == seg_only == both

    def test_neither_identifier_returns_base(self) -> None:
        base = resolve_stop_atr_multiplier("momentum")
        explicit = resolve_stop_atr_multiplier("momentum", segment_id="", market_id="")
        assert base == explicit
        assert base == Decimal(str(DEFAULT_STRATEGY_STOP_ATR["momentum"]))


class TestBackwardCompatRexport:
    """backtest/config.py must re-export the function (existing imports must keep working)."""

    def test_function_importable_from_backtest_config(self) -> None:
        from finalayze.backtest.config import (
            resolve_stop_atr_multiplier as bt_resolve,
        )

        assert bt_resolve is resolve_stop_atr_multiplier

    def test_strategy_table_importable_from_backtest_config(self) -> None:
        from finalayze.backtest import config as bt_config

        assert bt_config.DEFAULT_STRATEGY_STOP_ATR is DEFAULT_STRATEGY_STOP_ATR


# --------------------------------------------------------------------------- #
# Parametrised: actual numbers for the major (strategy, market) combinations  #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("strategy", "market", "expected"),
    [
        ("momentum", "us", "2.5"),
        ("momentum", "moex", "3.0"),
        ("mean_reversion", "us", "3.5"),
        ("mean_reversion", "moex", "4.2"),
        ("ml_ensemble", "us", "2.0"),
        ("ml_ensemble", "moex", "2.4"),
    ],
)
def test_strategy_market_matrix(strategy: str, market: str, expected: str) -> None:
    """Concrete sanity grid so future param drift is caught."""
    result = resolve_stop_atr_multiplier(strategy, market_id=market)
    assert result == Decimal(expected)
