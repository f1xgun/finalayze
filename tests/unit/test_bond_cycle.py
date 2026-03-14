"""Unit tests for BondCycleProcessor."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from finalayze.core.bond_cycle import BondCycleProcessor, BondCycleResult, LayerResult
from finalayze.core.layer_ledger import LayerLedger
from finalayze.core.schemas import DEFAULT_LAYER_CONFIGS, PortfolioLayer
from finalayze.data.fetchers.cbr import MacroSnapshot
from finalayze.data.macro_cache import MacroCacheService
from finalayze.risk.layer_circuit_breaker import AggregateBondBreaker, BondLayerBreaker

INITIAL_CASH = Decimal(100000)


def _make_processor(
    macro_snapshot: MacroSnapshot | None = None,
    aggregate_halted: bool = False,
) -> BondCycleProcessor:
    """Create a minimal BondCycleProcessor with mocked dependencies."""
    layer_configs = DEFAULT_LAYER_CONFIGS
    layer_ledgers = {
        layer: LayerLedger(layer_id=layer.value, cash=INITIAL_CASH) for layer in PortfolioLayer
    }
    layer_breakers = {
        layer: BondLayerBreaker(cfg, layer_ledgers[layer]) for layer, cfg in layer_configs.items()
    }
    aggregate_breaker = AggregateBondBreaker(layer_ledgers)
    if aggregate_halted:
        aggregate_breaker._halted = True  # noqa: SLF001

    macro_cache = MagicMock(spec=MacroCacheService)
    macro_cache.get.return_value = macro_snapshot

    return BondCycleProcessor(
        layer_configs=layer_configs,
        layer_ledgers=layer_ledgers,
        layer_breakers=layer_breakers,
        aggregate_breaker=aggregate_breaker,
        strategies={layer: [] for layer in PortfolioLayer},
        macro_cache=macro_cache,
        dv01_sizer=MagicMock(),
        equal_weight_sizer=MagicMock(),
        yield_stops={layer: MagicMock() for layer in PortfolioLayer},
        broker_router=MagicMock(),
        instrument_registry=MagicMock(),
        fetcher=MagicMock(),
        alerter=MagicMock(),
    )


def test_skips_when_no_macro_data() -> None:
    proc = _make_processor(macro_snapshot=None)
    result = proc.run_cycle()
    assert result.skipped is True
    assert result.reason == "no macro data"


def test_skips_when_aggregate_breaker_halted() -> None:
    snapshot = MacroSnapshot(
        key_rate=Decimal("16.00"),
        ruonia_7d_avg=Decimal("15.50"),
        cpi_yoy=Decimal("9.0"),
        last_cbr_decision="hold",
    )
    proc = _make_processor(macro_snapshot=snapshot, aggregate_halted=True)
    result = proc.run_cycle()
    assert result.skipped is True
    assert "aggregate" in result.reason


def test_processes_all_layers_when_healthy() -> None:
    snapshot = MacroSnapshot(
        key_rate=Decimal("16.00"),
        ruonia_7d_avg=Decimal("15.50"),
        cpi_yoy=Decimal("9.0"),
        last_cbr_decision="hold",
    )
    proc = _make_processor(macro_snapshot=snapshot)
    result = proc.run_cycle()
    assert result.skipped is False
    assert len(result.layer_results) == len(PortfolioLayer)


def test_bond_cycle_result_to_log_dict() -> None:
    result = BondCycleResult(
        layer_results=[
            LayerResult(layer=PortfolioLayer.CORE, signals=2, executed=1),
            LayerResult(layer=PortfolioLayer.TACTICAL, halted=True),
        ],
    )
    log_dict = result.to_log_dict()
    assert "layers_processed" in log_dict
    assert "layers_halted" in log_dict


def test_layer_result_defaults() -> None:
    lr = LayerResult(layer=PortfolioLayer.CORE)
    assert lr.signals == 0
    assert lr.executed == 0
    assert lr.exits == 0
    assert lr.halted is False
    assert lr.error is False
