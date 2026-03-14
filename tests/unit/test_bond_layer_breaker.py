"""Unit tests for BondLayerBreaker and AggregateBondBreaker."""

from __future__ import annotations

from decimal import Decimal

import pytest

from finalayze.core.layer_ledger import LayerLedger
from finalayze.core.schemas import DEFAULT_LAYER_CONFIGS, LayerConfig, PortfolioLayer
from finalayze.risk.layer_circuit_breaker import AggregateBondBreaker, BondLayerBreaker

INITIAL_CASH = Decimal(100_000)
DRAWDOWN_THRESHOLD = Decimal("0.05")


@pytest.fixture
def tactical_config() -> LayerConfig:
    return DEFAULT_LAYER_CONFIGS[PortfolioLayer.TACTICAL]


@pytest.fixture
def core_config() -> LayerConfig:
    return DEFAULT_LAYER_CONFIGS[PortfolioLayer.CORE]


@pytest.fixture
def ledger() -> LayerLedger:
    return LayerLedger(layer_id="tactical", cash=INITIAL_CASH)


# ── BondLayerBreaker ──────────────────────────────────────────────────


def test_allows_trading_when_no_drawdown(tactical_config: LayerConfig, ledger: LayerLedger) -> None:
    cb = BondLayerBreaker(tactical_config, ledger)
    assert cb.check() is True


def test_halts_when_drawdown_exceeds_threshold(
    tactical_config: LayerConfig, ledger: LayerLedger
) -> None:
    cb = BondLayerBreaker(tactical_config, ledger)
    # Simulate 6% drawdown (threshold is 5%)
    ledger.update_equity(INITIAL_CASH * Decimal("0.94"))
    assert cb.check() is False


def test_sticky_halt_does_not_auto_recover(
    tactical_config: LayerConfig, ledger: LayerLedger
) -> None:
    cb = BondLayerBreaker(tactical_config, ledger)
    ledger.update_equity(INITIAL_CASH * Decimal("0.94"))
    cb.check()  # triggers halt
    ledger.update_equity(INITIAL_CASH)  # recover fully
    assert cb.check() is False  # still halted (sticky)


def test_tactical_auto_clears_after_1_ok_day(
    tactical_config: LayerConfig, ledger: LayerLedger
) -> None:
    cb = BondLayerBreaker(tactical_config, ledger)
    ledger.update_equity(INITIAL_CASH * Decimal("0.94"))
    cb.check()
    ledger.update_equity(INITIAL_CASH)
    cb.daily_reset_check()
    assert cb.check() is True


def test_core_requires_manual_reset(
    core_config: LayerConfig,
) -> None:
    ledger = LayerLedger(layer_id="core", cash=INITIAL_CASH)
    cb = BondLayerBreaker(core_config, ledger)
    ledger.update_equity(INITIAL_CASH * Decimal("0.96"))  # 4% dd > 3% threshold
    cb.check()
    ledger.update_equity(INITIAL_CASH)
    cb.daily_reset_check()
    assert cb.check() is False  # still halted
    cb.reset_manual()
    assert cb.check() is True


def test_manual_reset_clears_halt(tactical_config: LayerConfig, ledger: LayerLedger) -> None:
    cb = BondLayerBreaker(tactical_config, ledger)
    ledger.update_equity(INITIAL_CASH * Decimal("0.94"))
    cb.check()
    cb.reset_manual()
    assert cb.check() is True


# ── AggregateBondBreaker ─────────────────────────────────────────────────

AGGREGATE_THRESHOLD = Decimal("0.03")


def test_aggregate_allows_when_no_drawdown() -> None:
    ledgers = {
        PortfolioLayer.CORE: LayerLedger(layer_id="core", cash=Decimal(40_000)),
        PortfolioLayer.TACTICAL: LayerLedger(layer_id="tactical", cash=Decimal(20_000)),
    }
    ab = AggregateBondBreaker(ledgers, max_total_drawdown_pct=AGGREGATE_THRESHOLD)
    assert ab.check() is True


def test_aggregate_halts_on_combined_drawdown() -> None:
    ledgers = {
        PortfolioLayer.CORE: LayerLedger(layer_id="core", cash=Decimal(40_000)),
        PortfolioLayer.TACTICAL: LayerLedger(layer_id="tactical", cash=Decimal(20_000)),
    }
    ab = AggregateBondBreaker(ledgers, max_total_drawdown_pct=AGGREGATE_THRESHOLD)
    # 3.33% total drawdown > 3% threshold
    ledgers[PortfolioLayer.CORE].update_equity(Decimal(38_000))
    ledgers[PortfolioLayer.TACTICAL].update_equity(Decimal(20_000))
    assert ab.check() is False


def test_aggregate_requires_manual_reset() -> None:
    ledgers = {
        PortfolioLayer.CORE: LayerLedger(layer_id="core", cash=Decimal(40_000)),
    }
    ab = AggregateBondBreaker(ledgers, max_total_drawdown_pct=AGGREGATE_THRESHOLD)
    ledgers[PortfolioLayer.CORE].update_equity(Decimal(37_000))
    ab.check()
    ledgers[PortfolioLayer.CORE].update_equity(Decimal(40_000))
    assert ab.check() is False  # still halted
    ab.reset_manual()
    assert ab.check() is True


def test_aggregate_allows_when_zero_peak() -> None:
    ledgers = {
        PortfolioLayer.CORE: LayerLedger(layer_id="core", cash=Decimal(0)),
    }
    # peak_equity is set to cash in __post_init__, which is 0
    # Override: set peak to 0 explicitly
    ledgers[PortfolioLayer.CORE].peak_equity = Decimal(0)
    ab = AggregateBondBreaker(ledgers, max_total_drawdown_pct=AGGREGATE_THRESHOLD)
    assert ab.check() is True
