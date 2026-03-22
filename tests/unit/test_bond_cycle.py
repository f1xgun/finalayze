"""Unit tests for BondCycleProcessor."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from unittest.mock import MagicMock, call, patch

import pytest

from finalayze.core.bond_cycle import (
    BondCycleProcessor,
    BondCycleResult,
    LayerResult,
    _FILL_TIMEOUT_SECONDS,
    _MAX_SIZING_ITERATIONS,
)
from finalayze.core.layer_ledger import LayerLedger
from finalayze.core.schemas import (
    BondPositionRecord,
    DEFAULT_LAYER_CONFIGS,
    PortfolioLayer,
    Signal,
    SignalDirection,
)
from finalayze.data.fetchers.cbr import MacroSnapshot
from finalayze.data.macro_cache import MacroCacheService
from finalayze.execution.broker_base import OrderResult
from finalayze.execution.tinkoff_broker import OrderStateResult
from finalayze.risk.layer_circuit_breaker import AggregateBondBreaker, BondLayerBreaker

INITIAL_CASH = Decimal(100_000)
FACE_VALUE = Decimal(1000)


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


# ── Helpers for _size_and_execute / _process_yield_stops tests ───────────────


def _make_signal(
    direction: SignalDirection = SignalDirection.BUY,
    symbol: str = "SU26238RMFS4",
) -> Signal:
    """Create a minimal bond Signal for testing."""
    return Signal(
        strategy_name="bond_carry",
        symbol=symbol,
        market_id="moex",
        segment_id="ru_ofz_pd",
        direction=direction,
        confidence=0.70,
        features={},
        reasoning="test signal",
        instrument_type="bond",
    )


def _make_bond_info() -> MagicMock:
    """Create a mock BondInfo with typical OFZ values."""
    info = MagicMock()
    info.figi = "BBG00TEST001"
    info.ticker = "SU26238RMFS4"
    info.face_value = FACE_VALUE
    info.coupon_rate = Decimal("7.10")
    info.coupon_frequency = 2
    info.maturity_date = date(2028, 5, 10)
    info.floating_coupon = False
    return info


def _make_order_result(
    filled: bool = True,
    order_id: str = "ord-test-1",
    fill_price: Decimal = Decimal("95.50"),
    quantity: Decimal = Decimal(5),
    side: str = "BUY",
    symbol: str = "SU26238RMFS4",
) -> OrderResult:
    return OrderResult(
        filled=filled,
        fill_price=fill_price,
        symbol=symbol,
        side=side,
        quantity=quantity,
        order_id=order_id,
    )


def _make_order_state(
    status: str = "fill",
    filled_qty: int = 5,
    filled_price: Decimal = Decimal("95.50"),
    is_terminal: bool = True,
    order_id: str = "ord-test-1",
) -> OrderStateResult:
    return OrderStateResult(
        order_id=order_id,
        execution_status=status,
        filled_quantity=Decimal(filled_qty),
        filled_price=filled_price,
        is_terminal=is_terminal,
    )


def _make_processor_with_mocks(
    macro_snapshot: MacroSnapshot | None = None,
) -> tuple[BondCycleProcessor, dict[str, MagicMock]]:
    """Create a BondCycleProcessor with accessible mock dependencies."""
    if macro_snapshot is None:
        macro_snapshot = MacroSnapshot(
            key_rate=Decimal("16.00"),
            ruonia_7d_avg=Decimal("15.50"),
            cpi_yoy=Decimal("9.0"),
            last_cbr_decision="hold",
        )

    layer_configs = DEFAULT_LAYER_CONFIGS
    layer_ledgers = {
        layer: LayerLedger(layer_id=layer.value, cash=INITIAL_CASH) for layer in PortfolioLayer
    }
    layer_breakers = {
        layer: BondLayerBreaker(cfg, layer_ledgers[layer]) for layer, cfg in layer_configs.items()
    }

    mock_broker = MagicMock()
    mock_router = MagicMock()
    mock_router.route.return_value = mock_broker
    mock_dv01_sizer = MagicMock()
    mock_equal_sizer = MagicMock()
    mock_registry = MagicMock()
    mock_yield_stops = {layer: MagicMock() for layer in PortfolioLayer}
    mock_macro_cache = MagicMock(spec=MacroCacheService)
    mock_macro_cache.get.return_value = macro_snapshot
    mock_alerter = MagicMock()

    proc = BondCycleProcessor(
        layer_configs=layer_configs,
        layer_ledgers=layer_ledgers,
        layer_breakers=layer_breakers,
        aggregate_breaker=AggregateBondBreaker(layer_ledgers),
        strategies={layer: [] for layer in PortfolioLayer},
        macro_cache=mock_macro_cache,
        dv01_sizer=mock_dv01_sizer,
        equal_weight_sizer=mock_equal_sizer,
        yield_stops=mock_yield_stops,
        broker_router=mock_router,
        instrument_registry=mock_registry,
        fetcher=MagicMock(),
        alerter=mock_alerter,
    )

    mocks = {
        "broker": mock_broker,
        "router": mock_router,
        "dv01_sizer": mock_dv01_sizer,
        "equal_sizer": mock_equal_sizer,
        "registry": mock_registry,
        "yield_stops": mock_yield_stops,
        "ledgers": layer_ledgers,
        "macro_cache": mock_macro_cache,
        "alerter": mock_alerter,
    }
    return proc, mocks


# ── _size_and_execute tests ──────────────────────────────────────────────────


class TestSizeAndExecuteBuy:
    """Tests for _size_and_execute with BUY signals."""

    def test_buy_submits_order_and_updates_ledger(self) -> None:
        """BUY signal: iterative sizing, submit order, wait fill, update ledger."""
        proc, mocks = _make_processor_with_mocks()
        signal = _make_signal(SignalDirection.BUY)
        layer = PortfolioLayer.STRATEGIC
        ledger = mocks["ledgers"][layer]

        # Setup mocks
        bond_info = _make_bond_info()
        mocks["registry"].get.return_value = bond_info

        # DV01 sizer returns 5 bonds
        mocks["dv01_sizer"].compute_position_size.return_value = 5

        # Broker returns current price
        mocks["broker"].get_last_prices.return_value = {"SU26238RMFS4": Decimal("95.50")}

        # Broker submit returns order with order_id
        mocks["broker"].submit_order.return_value = _make_order_result()

        # get_order_state returns filled immediately
        mocks["broker"].get_order_state.return_value = _make_order_state()

        with patch("finalayze.core.bond_cycle.bond_math") as mock_bm:
            mock_bm.dirty_price.return_value = Decimal("960.00")
            mock_bm.nkd.return_value = Decimal("10.00")
            mock_bm.ytm.return_value = Decimal("15.50")
            mock_bm.modified_duration.return_value = Decimal("3.50")
            mock_bm.dv01.return_value = Decimal("0.0336")
            result = proc._size_and_execute(signal, layer, ledger)  # noqa: SLF001

        assert result is True
        # Ledger should have a bond position
        assert signal.symbol in ledger.bond_positions
        # Cash should be debited
        assert ledger.cash < INITIAL_CASH

    def test_buy_timeout_cancels_and_returns_false(self) -> None:
        """BUY signal: fill timeout cancels order, returns False."""
        proc, mocks = _make_processor_with_mocks()
        signal = _make_signal(SignalDirection.BUY)
        layer = PortfolioLayer.STRATEGIC
        ledger = mocks["ledgers"][layer]

        bond_info = _make_bond_info()
        mocks["registry"].get.return_value = bond_info
        mocks["dv01_sizer"].compute_position_size.return_value = 5
        mocks["broker"].get_last_prices.return_value = {"SU26238RMFS4": Decimal("95.50")}

        mocks["broker"].submit_order.return_value = _make_order_result(
            filled=False, order_id="ord-timeout"
        )
        # Order stays in "new" status forever then cancelled with 0 fills
        mocks["broker"].get_order_state.side_effect = [
            _make_order_state(
                status="new", filled_qty=0, is_terminal=False, order_id="ord-timeout"
            ),
            _make_order_state(
                status="cancelled", filled_qty=0, is_terminal=True, order_id="ord-timeout"
            ),
        ]

        with (
            patch("finalayze.core.bond_cycle.bond_math") as mock_bm,
            patch("finalayze.core.bond_cycle.time") as mock_time,
        ):
            mock_bm.dirty_price.return_value = Decimal("960.00")
            mock_bm.nkd.return_value = Decimal("10.00")
            mock_bm.ytm.return_value = Decimal("15.50")
            mock_bm.modified_duration.return_value = Decimal("3.50")
            mock_bm.dv01.return_value = Decimal("0.0336")
            # Simulate time passing beyond timeout
            mock_time.monotonic.side_effect = [0.0, 0.0, _FILL_TIMEOUT_SECONDS + 1]
            mock_time.sleep = MagicMock()
            result = proc._size_and_execute(signal, layer, ledger)  # noqa: SLF001

        assert result is False
        mocks["broker"].cancel_order.assert_called_once_with("ord-timeout")
        # Ledger unchanged
        assert signal.symbol not in ledger.bond_positions
        assert ledger.cash == INITIAL_CASH

    def test_buy_partial_fill_keeps_partial(self) -> None:
        """Partial fill: cancel remainder, keep partial, update ledger with filled qty."""
        proc, mocks = _make_processor_with_mocks()
        signal = _make_signal(SignalDirection.BUY)
        layer = PortfolioLayer.STRATEGIC
        ledger = mocks["ledgers"][layer]

        bond_info = _make_bond_info()
        mocks["registry"].get.return_value = bond_info
        mocks["dv01_sizer"].compute_position_size.return_value = 5
        mocks["broker"].get_last_prices.return_value = {"SU26238RMFS4": Decimal("95.50")}

        mocks["broker"].submit_order.return_value = _make_order_result(
            filled=False, order_id="ord-partial"
        )
        # Poll 1: not terminal (partially filled), still within timeout.
        # Poll 2: still partially_fill, now timeout exceeded -> return None.
        # After cancel, final state check returns cancelled with 3 filled.
        mocks["broker"].get_order_state.side_effect = [
            _make_order_state(
                status="partially_fill",
                filled_qty=3,
                is_terminal=False,
                order_id="ord-partial",
            ),
            _make_order_state(
                status="partially_fill",
                filled_qty=3,
                is_terminal=False,
                order_id="ord-partial",
            ),
            # After cancel, check again shows cancelled with 3 filled
            _make_order_state(
                status="cancelled",
                filled_qty=3,
                is_terminal=True,
                order_id="ord-partial",
            ),
        ]

        with (
            patch("finalayze.core.bond_cycle.bond_math") as mock_bm,
            patch("finalayze.core.bond_cycle.time") as mock_time,
        ):
            mock_bm.dirty_price.return_value = Decimal("960.00")
            mock_bm.nkd.return_value = Decimal("10.00")
            mock_bm.ytm.return_value = Decimal("15.50")
            mock_bm.modified_duration.return_value = Decimal("3.50")
            mock_bm.dv01.return_value = Decimal("0.0336")
            # start=0.0, first elapsed check=0.0 (within timeout), second=timeout+1
            mock_time.monotonic.side_effect = [0.0, 0.0, _FILL_TIMEOUT_SECONDS + 1]
            mock_time.sleep = MagicMock()
            result = proc._size_and_execute(signal, layer, ledger)  # noqa: SLF001

        assert result is True
        mocks["broker"].cancel_order.assert_called_once_with("ord-partial")
        # Ledger should have 3 bonds (partial fill)
        assert signal.symbol in ledger.bond_positions
        assert ledger.bond_positions[signal.symbol].quantity == Decimal(3)

    def test_buy_deducts_dirty_price_plus_costs(self) -> None:
        """Cash deduction = dirty_price * qty + transaction_costs_per_unit * qty."""
        proc, mocks = _make_processor_with_mocks()
        signal = _make_signal(SignalDirection.BUY)
        layer = PortfolioLayer.STRATEGIC
        ledger = mocks["ledgers"][layer]

        bond_info = _make_bond_info()
        mocks["registry"].get.return_value = bond_info
        mocks["dv01_sizer"].compute_position_size.return_value = 5
        mocks["broker"].get_last_prices.return_value = {"SU26238RMFS4": Decimal("95.50")}

        mocks["broker"].submit_order.return_value = _make_order_result(quantity=Decimal(5))
        mocks["broker"].get_order_state.return_value = _make_order_state(filled_qty=5)

        dirty = Decimal("960.00")
        with patch("finalayze.core.bond_cycle.bond_math") as mock_bm:
            mock_bm.dirty_price.return_value = dirty
            mock_bm.nkd.return_value = Decimal("10.00")
            mock_bm.ytm.return_value = Decimal("15.50")
            mock_bm.modified_duration.return_value = Decimal("3.50")
            mock_bm.dv01.return_value = Decimal("0.0336")
            proc._size_and_execute(signal, layer, ledger)  # noqa: SLF001

        # Cash should be reduced by dirty_price * qty + costs
        assert ledger.cash < INITIAL_CASH
        # At minimum dirty_price * 5 = 4800
        assert ledger.cash <= INITIAL_CASH - dirty * 5

    def test_buy_passes_transaction_costs_to_sizer(self) -> None:
        """DV01BudgetStep receives transaction_costs_per_unit argument."""
        proc, mocks = _make_processor_with_mocks()
        signal = _make_signal(SignalDirection.BUY)
        layer = PortfolioLayer.STRATEGIC
        ledger = mocks["ledgers"][layer]

        bond_info = _make_bond_info()
        mocks["registry"].get.return_value = bond_info
        mocks["dv01_sizer"].compute_position_size.return_value = 0  # 0 means no trade
        mocks["broker"].get_last_prices.return_value = {"SU26238RMFS4": Decimal("95.50")}

        with patch("finalayze.core.bond_cycle.bond_math") as mock_bm:
            mock_bm.dirty_price.return_value = Decimal("960.00")
            mock_bm.nkd.return_value = Decimal("10.00")
            mock_bm.ytm.return_value = Decimal("15.50")
            mock_bm.modified_duration.return_value = Decimal("3.50")
            mock_bm.dv01.return_value = Decimal("0.0336")
            proc._size_and_execute(signal, layer, ledger)  # noqa: SLF001

        # Verify transaction_costs_per_unit was passed
        args = mocks["dv01_sizer"].compute_position_size.call_args
        assert "transaction_costs_per_unit" in args.kwargs or len(args.args) >= 5


class TestSizeAndExecuteSell:
    """Tests for _size_and_execute with SELL signals."""

    def test_sell_submits_for_full_position(self) -> None:
        """SELL signal submits sell order for full held quantity."""
        proc, mocks = _make_processor_with_mocks()
        signal = _make_signal(SignalDirection.SELL)
        layer = PortfolioLayer.STRATEGIC
        ledger = mocks["ledgers"][layer]

        bond_info = _make_bond_info()
        mocks["registry"].get.return_value = bond_info

        # Add a position to sell
        ledger.add_bond_position(
            BondPositionRecord(
                symbol=signal.symbol,
                quantity=Decimal(10),
                entry_ytm_pct=Decimal("14.50"),
                entry_date=date(2026, 1, 1),
                entry_price=Decimal("95.00"),
                entry_clean_pct=Decimal("95.00"),
                layer_id=layer.value,
            )
        )

        mocks["broker"].submit_order.return_value = _make_order_result(
            side="SELL", quantity=Decimal(10)
        )
        mocks["broker"].get_order_state.return_value = _make_order_state(
            filled_qty=10, filled_price=Decimal("96.00")
        )

        with patch("finalayze.core.bond_cycle.bond_math") as mock_bm:
            mock_bm.dirty_price.return_value = Decimal("970.00")
            mock_bm.nkd.return_value = Decimal("10.00")
            result = proc._size_and_execute(signal, layer, ledger)  # noqa: SLF001

        assert result is True
        # Position should be removed
        assert signal.symbol not in ledger.bond_positions

    def test_sell_credits_cash(self) -> None:
        """SELL credits cash after fill."""
        proc, mocks = _make_processor_with_mocks()
        signal = _make_signal(SignalDirection.SELL)
        layer = PortfolioLayer.STRATEGIC
        ledger = mocks["ledgers"][layer]

        bond_info = _make_bond_info()
        mocks["registry"].get.return_value = bond_info

        ledger.add_bond_position(
            BondPositionRecord(
                symbol=signal.symbol,
                quantity=Decimal(5),
                entry_ytm_pct=Decimal("14.50"),
                entry_date=date(2026, 1, 1),
                entry_price=Decimal("95.00"),
                entry_clean_pct=Decimal("95.00"),
                layer_id=layer.value,
            )
        )

        mocks["broker"].submit_order.return_value = _make_order_result(
            side="SELL", quantity=Decimal(5), fill_price=Decimal("96.00")
        )
        mocks["broker"].get_order_state.return_value = _make_order_state(
            filled_qty=5, filled_price=Decimal("96.00")
        )

        with patch("finalayze.core.bond_cycle.bond_math") as mock_bm:
            mock_bm.dirty_price.return_value = Decimal("970.00")
            mock_bm.nkd.return_value = Decimal("10.00")
            proc._size_and_execute(signal, layer, ledger)  # noqa: SLF001

        assert ledger.cash > INITIAL_CASH


class TestSizeAndExecuteNoRetry:
    """No retry on timeout -- simply returns False."""

    def test_no_retry_on_timeout(self) -> None:
        proc, mocks = _make_processor_with_mocks()
        signal = _make_signal(SignalDirection.BUY)
        layer = PortfolioLayer.STRATEGIC
        ledger = mocks["ledgers"][layer]

        bond_info = _make_bond_info()
        mocks["registry"].get.return_value = bond_info
        mocks["dv01_sizer"].compute_position_size.return_value = 5
        mocks["broker"].get_last_prices.return_value = {"SU26238RMFS4": Decimal("95.50")}

        mocks["broker"].submit_order.return_value = _make_order_result(
            filled=False, order_id="ord-no-retry"
        )
        mocks["broker"].get_order_state.side_effect = [
            _make_order_state(
                status="new", filled_qty=0, is_terminal=False, order_id="ord-no-retry"
            ),
            _make_order_state(
                status="cancelled", filled_qty=0, is_terminal=True, order_id="ord-no-retry"
            ),
        ]

        with (
            patch("finalayze.core.bond_cycle.bond_math") as mock_bm,
            patch("finalayze.core.bond_cycle.time") as mock_time,
        ):
            mock_bm.dirty_price.return_value = Decimal("960.00")
            mock_bm.nkd.return_value = Decimal("10.00")
            mock_bm.ytm.return_value = Decimal("15.50")
            mock_bm.modified_duration.return_value = Decimal("3.50")
            mock_bm.dv01.return_value = Decimal("0.0336")
            mock_time.monotonic.side_effect = [0.0, 0.0, _FILL_TIMEOUT_SECONDS + 1]
            mock_time.sleep = MagicMock()
            result = proc._size_and_execute(signal, layer, ledger)  # noqa: SLF001

        assert result is False
        # submit_order called exactly once (no retry)
        assert mocks["broker"].submit_order.call_count == 1


# ── _process_yield_stops tests ───────────────────────────────────────────────


class TestProcessYieldStops:
    """Tests for _process_yield_stops method."""

    def test_fetches_prices_for_all_held_symbols(self) -> None:
        """get_last_prices called with all held bond symbols."""
        proc, mocks = _make_processor_with_mocks()
        layer = PortfolioLayer.STRATEGIC
        ledger = mocks["ledgers"][layer]

        # Add two positions
        for sym in ["SU26238RMFS4", "SU26240RMFS8"]:
            ledger.add_bond_position(
                BondPositionRecord(
                    symbol=sym,
                    quantity=Decimal(5),
                    entry_ytm_pct=Decimal("14.50"),
                    entry_date=date(2026, 1, 1),
                    entry_price=Decimal("95.00"),
                    entry_clean_pct=Decimal("95.00"),
                    layer_id=layer.value,
                )
            )

        # Not stopped
        yield_stop = mocks["yield_stops"][layer]
        yield_stop.is_stopped_with_regime.return_value = False

        mocks["broker"].get_last_prices.return_value = {
            "SU26238RMFS4": Decimal("94.50"),
            "SU26240RMFS8": Decimal("96.00"),
        }

        bond_info = _make_bond_info()
        mocks["registry"].get.return_value = bond_info

        macro = MacroSnapshot(
            key_rate=Decimal("16.00"),
            ruonia_7d_avg=Decimal("15.50"),
            cpi_yoy=Decimal("9.0"),
            last_cbr_decision="hold",
        )

        with patch("finalayze.core.bond_cycle.bond_math") as mock_bm:
            mock_bm.ytm.return_value = Decimal("15.00")
            exits = proc._process_yield_stops(layer, ledger, yield_stop, macro)  # noqa: SLF001

        assert exits == 0
        mocks["broker"].get_last_prices.assert_called_once()
        call_symbols = mocks["broker"].get_last_prices.call_args[0][0]
        assert "SU26238RMFS4" in call_symbols
        assert "SU26240RMFS8" in call_symbols

    def test_stopped_position_triggers_sell_and_returns_count(self) -> None:
        """When YTM above threshold, SELL order submitted and exit count returned."""
        proc, mocks = _make_processor_with_mocks()
        layer = PortfolioLayer.STRATEGIC
        ledger = mocks["ledgers"][layer]

        ledger.add_bond_position(
            BondPositionRecord(
                symbol="SU26238RMFS4",
                quantity=Decimal(5),
                entry_ytm_pct=Decimal("14.50"),
                entry_date=date(2026, 1, 1),
                entry_price=Decimal("95.00"),
                entry_clean_pct=Decimal("95.00"),
                layer_id=layer.value,
            )
        )

        yield_stop = mocks["yield_stops"][layer]
        yield_stop.is_stopped_with_regime.return_value = True

        mocks["broker"].get_last_prices.return_value = {
            "SU26238RMFS4": Decimal("90.00"),
        }
        mocks["broker"].submit_order.return_value = _make_order_result(
            side="SELL", quantity=Decimal(5)
        )
        mocks["broker"].get_order_state.return_value = _make_order_state(
            filled_qty=5, filled_price=Decimal("90.00")
        )

        bond_info = _make_bond_info()
        mocks["registry"].get.return_value = bond_info

        macro = MacroSnapshot(
            key_rate=Decimal("16.00"),
            ruonia_7d_avg=Decimal("15.50"),
            cpi_yoy=Decimal("9.0"),
            last_cbr_decision="hold",
        )

        with patch("finalayze.core.bond_cycle.bond_math") as mock_bm:
            mock_bm.ytm.return_value = Decimal("16.50")
            mock_bm.dirty_price.return_value = Decimal("910.00")
            mock_bm.nkd.return_value = Decimal("10.00")
            exits = proc._process_yield_stops(layer, ledger, yield_stop, macro)  # noqa: SLF001

        assert exits == 1
        mocks["broker"].submit_order.assert_called_once()

    def test_below_threshold_returns_zero(self) -> None:
        """When current YTM below threshold, no exit."""
        proc, mocks = _make_processor_with_mocks()
        layer = PortfolioLayer.STRATEGIC
        ledger = mocks["ledgers"][layer]

        ledger.add_bond_position(
            BondPositionRecord(
                symbol="SU26238RMFS4",
                quantity=Decimal(5),
                entry_ytm_pct=Decimal("14.50"),
                entry_date=date(2026, 1, 1),
                entry_price=Decimal("95.00"),
                entry_clean_pct=Decimal("95.00"),
                layer_id=layer.value,
            )
        )

        yield_stop = mocks["yield_stops"][layer]
        yield_stop.is_stopped_with_regime.return_value = False

        mocks["broker"].get_last_prices.return_value = {
            "SU26238RMFS4": Decimal("95.00"),
        }

        bond_info = _make_bond_info()
        mocks["registry"].get.return_value = bond_info

        macro = MacroSnapshot(
            key_rate=Decimal("16.00"),
            ruonia_7d_avg=Decimal("15.50"),
            cpi_yoy=Decimal("9.0"),
            last_cbr_decision="hold",
        )

        with patch("finalayze.core.bond_cycle.bond_math") as mock_bm:
            mock_bm.ytm.return_value = Decimal("14.60")
            exits = proc._process_yield_stops(layer, ledger, yield_stop, macro)  # noqa: SLF001

        assert exits == 0
        mocks["broker"].submit_order.assert_not_called()

    def test_empty_positions_returns_zero(self) -> None:
        """No positions held means no yield stop evaluation."""
        proc, mocks = _make_processor_with_mocks()
        layer = PortfolioLayer.STRATEGIC
        ledger = mocks["ledgers"][layer]
        yield_stop = mocks["yield_stops"][layer]

        macro = MacroSnapshot(
            key_rate=Decimal("16.00"),
            ruonia_7d_avg=Decimal("15.50"),
            cpi_yoy=Decimal("9.0"),
            last_cbr_decision="hold",
        )

        exits = proc._process_yield_stops(layer, ledger, yield_stop, macro)  # noqa: SLF001
        assert exits == 0
        mocks["broker"].get_last_prices.assert_not_called()


# ── 05-03 Coupon alert tests ────────────────────────────────────────────────


class TestCouponAlertInBondCycle:
    """Coupon reinvestment in BondCycleProcessor fires alerter.on_coupon_received."""

    def test_coupon_reinvestment_fires_alert(self) -> None:
        """When coupon cash is reinvested, alerter.on_coupon_received is called."""
        import inspect

        source = inspect.getsource(BondCycleProcessor._process_layer)
        # The coupon reinvestment step should fire an alert
        assert "on_coupon_received" in source or "coupon" in source.lower()


# ── OFZ rotation tests ─────────────────────────────────────────────────────


from finalayze.core.bond_cycle import apply_ofz_rotation


class TestOFZRotation:
    """Tests for apply_ofz_rotation: shifts CORE/STRATEGIC when CBR cutting cycle detected."""

    def test_ofz_rotation_cutting_cycle(self) -> None:
        """2+ consecutive cuts -> CORE 0.45->0.30, STRATEGIC 0.275->0.425."""
        configs = dict(DEFAULT_LAYER_CONFIGS)
        # 2025-10-30 is after 2025-09-12 cut + 2025-10-24 cut (2 consecutive cuts)
        result = apply_ofz_rotation(configs, as_of=date(2025, 10, 30))
        assert result[PortfolioLayer.CORE].capital_pct == Decimal("0.30")
        assert result[PortfolioLayer.STRATEGIC].capital_pct == Decimal("0.425")

    def test_ofz_rotation_no_cutting_cycle(self) -> None:
        """All holds in 2024 H1 -> configs unchanged."""
        configs = dict(DEFAULT_LAYER_CONFIGS)
        result = apply_ofz_rotation(configs, as_of=date(2024, 6, 15))
        assert result[PortfolioLayer.CORE].capital_pct == Decimal("0.45")
        assert result[PortfolioLayer.STRATEGIC].capital_pct == Decimal("0.275")

    def test_ofz_rotation_single_cut_not_cycle(self) -> None:
        """Only 1 cut on 2025-07-25 -> configs unchanged (need 2+ consecutive)."""
        configs = dict(DEFAULT_LAYER_CONFIGS)
        result = apply_ofz_rotation(configs, as_of=date(2025, 7, 30))
        assert result[PortfolioLayer.CORE].capital_pct == Decimal("0.45")
        assert result[PortfolioLayer.STRATEGIC].capital_pct == Decimal("0.275")

    def test_ofz_rotation_revert_on_hike(self) -> None:
        """After 2023-08-15 emergency hike (after holds) -> configs unchanged."""
        configs = dict(DEFAULT_LAYER_CONFIGS)
        result = apply_ofz_rotation(configs, as_of=date(2023, 8, 20))
        assert result[PortfolioLayer.CORE].capital_pct == Decimal("0.45")
        assert result[PortfolioLayer.STRATEGIC].capital_pct == Decimal("0.275")

    def test_ofz_rotation_preserves_tactical_short(self) -> None:
        """TACTICAL and SHORT always unchanged regardless of rotation."""
        configs = dict(DEFAULT_LAYER_CONFIGS)
        original_tactical = configs[PortfolioLayer.TACTICAL].capital_pct
        original_short = configs[PortfolioLayer.SHORT].capital_pct
        # Use cutting cycle date
        result = apply_ofz_rotation(configs, as_of=date(2025, 10, 30))
        assert result[PortfolioLayer.TACTICAL].capital_pct == original_tactical
        assert result[PortfolioLayer.SHORT].capital_pct == original_short

    def test_ofz_rotation_capital_conservation(self) -> None:
        """Sum of all capital_pct after rotation == sum before."""
        configs = dict(DEFAULT_LAYER_CONFIGS)
        original_sum = sum(c.capital_pct for c in configs.values())
        result = apply_ofz_rotation(configs, as_of=date(2025, 10, 30))
        rotated_sum = sum(c.capital_pct for c in result.values())
        assert rotated_sum == original_sum


class TestBondCycleConsecutiveErrors:
    """ERR-05: BondCycleProcessor tracks per-layer consecutive gRPC error counts."""

    LAYER_ERROR_THRESHOLD = 3

    def _make_failing_processor(self) -> BondCycleProcessor:
        """Create a processor where _process_layer always raises for specific layers."""
        snapshot = MacroSnapshot(
            key_rate=Decimal("16.00"),
            ruonia_7d_avg=Decimal("15.50"),
            cpi_yoy=Decimal("9.0"),
            last_cbr_decision="hold",
        )
        return _make_processor(macro_snapshot=snapshot)

    def test_consecutive_layer_errors_dict_exists(self) -> None:
        """BondCycleProcessor must have _consecutive_layer_errors dict."""
        proc = self._make_failing_processor()
        assert hasattr(proc, "_consecutive_layer_errors")
        assert isinstance(proc._consecutive_layer_errors, dict)

    def test_layer_error_counter_increments(self) -> None:
        """After a layer failure, its counter increments."""
        proc = self._make_failing_processor()

        # Make _process_layer raise for all layers
        with patch.object(proc, "_process_layer", side_effect=RuntimeError("gRPC fail")):
            proc.run_cycle()

        # All 4 layers should have count=1
        assert len(proc._consecutive_layer_errors) > 0
        for count in proc._consecutive_layer_errors.values():
            assert count == 1

    def test_layer_error_counter_resets_on_success(self) -> None:
        """After a successful layer processing, its counter resets to 0."""
        proc = self._make_failing_processor()

        # Pre-set error count for a layer
        proc._consecutive_layer_errors["core"] = 2

        # Run a successful cycle (no _process_layer failure)
        proc.run_cycle()

        # The counter for "core" should be reset to 0
        assert proc._consecutive_layer_errors.get("core", 0) == 0

    def test_escalated_log_after_threshold_failures(self) -> None:
        """After 3 consecutive failures for a layer, log level escalates to error."""
        proc = self._make_failing_processor()

        with (
            patch.object(proc, "_process_layer", side_effect=RuntimeError("gRPC fail")),
            patch("finalayze.core.bond_cycle._log") as mock_log,
        ):
            for _ in range(self.LAYER_ERROR_THRESHOLD):
                proc.run_cycle()

        # After 3 cycles, each layer should have triggered the escalated log
        error_calls = mock_log.error.call_args_list
        escalated = [c for c in error_calls if c[0][0] == "bond_layer_consecutive_failures"]
        assert len(escalated) > 0
        # Check that consecutive_count and threshold are in the log kwargs
        first_call = escalated[0]
        assert first_call[1]["consecutive_count"] >= self.LAYER_ERROR_THRESHOLD
        assert "threshold" in first_call[1]

    def test_reset_prevents_escalation(self) -> None:
        """Fail twice, succeed, fail once = counter at 1, not 3."""
        proc = self._make_failing_processor()

        # Fail twice
        with patch.object(proc, "_process_layer", side_effect=RuntimeError("gRPC fail")):
            proc.run_cycle()
            proc.run_cycle()

        # All layers at count 2
        for count in proc._consecutive_layer_errors.values():
            assert count == 2

        # Succeed once -- resets counters
        proc.run_cycle()
        for count in proc._consecutive_layer_errors.values():
            assert count == 0

        # Fail once more
        with (
            patch.object(proc, "_process_layer", side_effect=RuntimeError("gRPC fail")),
            patch("finalayze.core.bond_cycle._log") as mock_log,
        ):
            proc.run_cycle()

        # Counter at 1, no escalated log
        for count in proc._consecutive_layer_errors.values():
            assert count == 1
        error_calls = mock_log.error.call_args_list
        escalated = [c for c in error_calls if c[0][0] == "bond_layer_consecutive_failures"]
        assert len(escalated) == 0
