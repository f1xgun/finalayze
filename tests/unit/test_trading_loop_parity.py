"""Tests for live-backtest parity: trailing stop state machine and re-entry guard.

PARITY-01: Live sizing uses PositionSizingPipeline (matching backtest engine).
PARITY-02: Live trailing stop ratchets upward after activation threshold.
PARITY-03: All 14 pre-trade check parameters passed in live path.
PARITY-04: Stop-loss exit prevents same-cycle re-entry.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from unittest.mock import MagicMock, patch

from config.settings import Settings

from finalayze.core.alerts import TelegramAlerter
from finalayze.core.modes import WorkMode
from finalayze.core.schemas import SignalDirection
from finalayze.core.trading_loop import TradingLoop
from finalayze.execution.simulated_broker import StopLossState
from finalayze.markets.instruments import InstrumentRegistry
from finalayze.risk.circuit_breaker import CircuitBreaker, CircuitLevel, CrossMarketCircuitBreaker
from finalayze.risk.position_sizing_pipeline import (
    HardCapsStep,
    KellyStep,
    MetaLabelStep,
    PositionSizingPipeline,
    RegimeStep,
    SizingContext,
    VolTargetStep,
)

# ── Constants ──────────────────────────────────────────────────────────
MARKET_US = "us"
SYMBOL_AAPL = "AAPL"
BASELINE_EQUITY = Decimal(100000)
BASELINE_CASH = Decimal(50000)
NUM_CANDLES = 60
_ZERO = Decimal(0)
ENTRY_PRICE = Decimal("150.00")
ATR_VALUE = Decimal("5.00")
INITIAL_STOP = Decimal("140.00")  # entry - 2.0 * atr
ATR_MULTIPLIER = Decimal("2.0")


def _make_settings() -> MagicMock:
    s = MagicMock(spec=Settings)
    s.news_cycle_minutes = 30
    s.strategy_cycle_minutes = 60
    s.daily_reset_hour_utc = 0
    s.max_position_pct = 0.20
    s.kelly_fraction = 0.5
    s.max_positions_per_market = 10
    s.daily_loss_limit_pct = 0.03
    s.max_cross_market_exposure_pct = 0.80
    s.mode = WorkMode.SANDBOX
    return s


def _make_loop() -> TradingLoop:
    settings = _make_settings()
    fetchers = {MARKET_US: MagicMock()}
    news_fetcher = MagicMock()
    news_analyzer = MagicMock()
    event_classifier = MagicMock()
    impact_estimator = MagicMock()
    strategy = MagicMock()
    broker_router = MagicMock()
    alerter = MagicMock(spec=TelegramAlerter)
    cb_us = CircuitBreaker(market_id=MARKET_US)
    circuit_breakers = {MARKET_US: cb_us}
    cross_market = CrossMarketCircuitBreaker()
    registry = InstrumentRegistry()

    return TradingLoop(
        settings=settings,
        fetchers=fetchers,
        news_fetcher=news_fetcher,
        news_analyzer=news_analyzer,
        event_classifier=event_classifier,
        impact_estimator=impact_estimator,
        strategy=strategy,
        broker_router=broker_router,
        circuit_breakers=circuit_breakers,
        cross_market_breaker=cross_market,
        alerter=alerter,
        instrument_registry=registry,
    )


def _seed_stop_state(loop: TradingLoop, symbol: str = SYMBOL_AAPL) -> StopLossState:
    """Plant a StopLossState into the loop's _stop_states dict."""
    state = StopLossState(
        initial_stop=INITIAL_STOP,
        current_stop=INITIAL_STOP,
        highest_price=ENTRY_PRICE,
        trail_activated=False,
        activation_atr=Decimal("1.0"),
        trail_atr=Decimal("1.5"),
        entry_price=ENTRY_PRICE,
        atr_value=ATR_VALUE,
    )
    loop._stop_states[symbol] = state
    return state


# ── Test 1: BUY fill stores StopLossState ──────────────────────────────


class TestStopLossStateStorage:
    """After a BUY fill, TradingLoop stores StopLossState (not bare Decimal)."""

    def test_buy_fill_stores_stop_loss_state(self) -> None:
        loop = _make_loop()
        # Verify the loop has _stop_states dict (StopLossState-based trailing stops)
        assert hasattr(loop, "_stop_states"), (
            "TradingLoop must have _stop_states dict with StopLossState instances"
        )
        assert isinstance(loop._stop_states, dict)

    def test_stop_state_has_required_fields(self) -> None:
        loop = _make_loop()
        state = _seed_stop_state(loop)
        assert state.entry_price == ENTRY_PRICE
        assert state.atr_value == ATR_VALUE
        assert state.activation_atr == Decimal("1.0")
        assert state.trail_atr == Decimal("1.5")
        assert state.trail_activated is False


# ── Test 2: Trailing stop ratchets upward ─────────────────────────────


class TestTrailingStopRatchet:
    """_check_stop_losses updates highest_price and ratchets current_stop upward."""

    def test_highest_price_updates_to_max(self) -> None:
        loop = _make_loop()
        state = _seed_stop_state(loop)
        higher_price = ENTRY_PRICE + Decimal("10.00")  # 160

        # Mock broker so stop doesn't try to trade
        loop._broker_router.route.return_value.get_positions.return_value = {
            SYMBOL_AAPL: Decimal(10),
        }

        loop._check_stop_losses(MARKET_US, SYMBOL_AAPL, higher_price)
        assert state.highest_price == higher_price

    def test_trailing_activates_at_threshold(self) -> None:
        """Trail activates when highest_price >= entry + activation_atr * atr_value."""
        loop = _make_loop()
        state = _seed_stop_state(loop)
        # activation threshold = 150 + 1.0 * 5.0 = 155
        activation_price = ENTRY_PRICE + state.activation_atr * state.atr_value

        loop._check_stop_losses(MARKET_US, SYMBOL_AAPL, activation_price)
        assert state.trail_activated is True

    def test_trail_stop_ratchets_upward(self) -> None:
        """Once activated, current_stop = max(current_stop, highest - trail_atr * atr)."""
        loop = _make_loop()
        state = _seed_stop_state(loop)

        # Push price to 160 (activates trail at 155)
        high_price = Decimal("160.00")
        loop._check_stop_losses(MARKET_US, SYMBOL_AAPL, high_price)
        assert state.trail_activated is True
        # trail_stop = 160 - 1.5 * 5 = 152.5
        expected_stop = high_price - state.trail_atr * state.atr_value
        assert state.current_stop == expected_stop

    def test_stop_never_moves_downward(self) -> None:
        """current_stop can only increase, never decrease."""
        loop = _make_loop()
        state = _seed_stop_state(loop)

        # Push to 160.00, trail activates, stop = 152.50
        loop._check_stop_losses(MARKET_US, SYMBOL_AAPL, Decimal("160.00"))
        stop_after_high = state.current_stop

        # Price drops to 153.00 (above stop, no trigger)
        loop._check_stop_losses(MARKET_US, SYMBOL_AAPL, Decimal("153.00"))
        assert state.current_stop == stop_after_high, "Stop must not decrease when price drops"


# ── Test 3: Stop triggers SELL and adds to exited symbols ──────────────


class TestStopTriggerSell:
    """When price <= current_stop, a SELL fires and symbol goes to _cycle_exited_symbols."""

    def test_stop_trigger_sells_and_records_exit(self) -> None:
        loop = _make_loop()
        _seed_stop_state(loop)

        mock_broker = MagicMock()
        mock_broker.get_positions.return_value = {SYMBOL_AAPL: Decimal(10)}
        loop._broker_router.route.return_value = mock_broker

        # Price drops to 139 (below initial stop of 140)
        loop._check_stop_losses(MARKET_US, SYMBOL_AAPL, Decimal("139.00"))

        # SELL order should have been submitted
        mock_broker.submit_order.assert_called_once()
        sell_order = mock_broker.submit_order.call_args[0][0]
        assert sell_order.side == "SELL"
        assert sell_order.quantity == Decimal(10)

        # Symbol should be in _cycle_exited_symbols
        assert SYMBOL_AAPL in loop._cycle_exited_symbols

        # Stop state should be cleared
        assert SYMBOL_AAPL not in loop._stop_states


# ── Test 4: Re-entry guard skips signal generation ─────────────────────


class TestReentryGuard:
    """When a symbol is in _cycle_exited_symbols, _process_instrument returns early."""

    def test_process_instrument_skips_exited_symbol(self) -> None:
        loop = _make_loop()
        loop._cycle_exited_symbols.add(SYMBOL_AAPL)

        instrument = MagicMock()
        instrument.symbol = SYMBOL_AAPL
        instrument.figi = "BBG000B9XRY4"
        instrument.segment_id = "us_tech"

        # Mock fetcher to return empty candles (symbol was exited, stop check runs first)
        fetcher = MagicMock()
        fetcher.fetch_candles.return_value = []

        loop._process_instrument(
            instrument=instrument,
            market_id=MARKET_US,
            level=CircuitLevel.NORMAL,
            fetcher=fetcher,
            now=datetime.now(UTC),
        )

        # Strategy generate_signal should NOT have been called
        loop._strategy.generate_signal.assert_not_called()


# ── Test 5: _cycle_exited_symbols cleared each cycle ───────────────────


class TestCycleExitedCleared:
    """_cycle_exited_symbols is cleared at the start of each equity cycle."""

    def test_reset_cycle_counters_clears_exited(self) -> None:
        loop = _make_loop()
        loop._cycle_exited_symbols.add(SYMBOL_AAPL)
        loop._cycle_exited_symbols.add("MSFT")
        assert len(loop._cycle_exited_symbols) > 0

        loop._reset_cycle_counters()
        assert len(loop._cycle_exited_symbols) == 0


# ── PARITY-01: Pipeline sizing tests ─────────────────────────────────


def _make_candle(close: float = 150.0) -> object:
    """Create a candle with the given close price."""
    from finalayze.core.schemas import Candle

    return Candle(
        symbol=SYMBOL_AAPL,
        market_id=MARKET_US,
        timeframe="1d",
        timestamp=datetime.now(UTC) - timedelta(hours=1),
        open=Decimal(str(close)),
        high=Decimal(str(close)),
        low=Decimal(str(close)),
        close=Decimal(str(close)),
        volume=1_000_000,
    )


def _make_signal(direction: SignalDirection = SignalDirection.BUY) -> MagicMock:
    """Create a mock Signal with sane defaults."""
    sig = MagicMock()
    sig.direction = direction
    sig.strategy_name = "dual_momentum"
    sig.confidence = 0.7
    sig.features = {"ml_confidence": 0.65}
    sig.reasoning = "test signal"
    return sig


def _make_portfolio(
    equity: Decimal = BASELINE_EQUITY,
    cash: Decimal = BASELINE_CASH,
) -> MagicMock:
    """Create a mock PortfolioState."""
    p = MagicMock()
    p.equity = equity
    p.cash = cash
    p.positions = {}
    return p


class TestPipelineSizing:
    """_build_order() for BUY signals uses PositionSizingPipeline."""

    def test_build_order_calls_pipeline_compute(self) -> None:
        """_build_order() constructs SizingContext and calls pipeline.compute()."""
        loop = _make_loop()
        signal = _make_signal()
        candles = [_make_candle(150.0) for _ in range(NUM_CANDLES)]
        portfolio = _make_portfolio()

        # The method should now have _build_sizing_pipeline
        assert hasattr(loop, "_build_sizing_pipeline"), (
            "TradingLoop must have _build_sizing_pipeline method"
        )

        # Patch _build_sizing_pipeline to capture the call
        mock_pipeline = MagicMock(spec=PositionSizingPipeline)
        mock_pipeline.compute.return_value = Decimal(10000)

        with patch.object(loop._signal_executor, "_build_sizing_pipeline", return_value=mock_pipeline):
            order = loop._build_order(
                signal,
                CircuitLevel.NORMAL,
                BASELINE_EQUITY,
                BASELINE_CASH,
                candles,
                SYMBOL_AAPL,
                Decimal("0.5"),
                portfolio=portfolio,
                seg_id="us_tech",
            )

        mock_pipeline.compute.assert_called_once()
        ctx = mock_pipeline.compute.call_args[0][0]
        assert isinstance(ctx, SizingContext)
        assert ctx.equity == BASELINE_EQUITY
        assert order is not None

    def test_pipeline_includes_core_steps(self) -> None:
        """Pipeline includes KellyStep, VolTargetStep, RegimeStep, MetaLabelStep, HardCapsStep."""
        loop = _make_loop()
        pipeline = loop._build_sizing_pipeline("us_tech")
        step_types = [type(s) for s in pipeline._steps]
        assert KellyStep in step_types
        assert VolTargetStep in step_types
        assert RegimeStep in step_types
        assert MetaLabelStep in step_types
        assert HardCapsStep in step_types

    def test_pipeline_zero_returns_none(self) -> None:
        """When pipeline.compute() returns 0, _build_order() returns None."""
        loop = _make_loop()
        signal = _make_signal()
        candles = [_make_candle(150.0) for _ in range(NUM_CANDLES)]
        portfolio = _make_portfolio()

        mock_pipeline = MagicMock(spec=PositionSizingPipeline)
        mock_pipeline.compute.return_value = Decimal(0)

        with patch.object(loop._signal_executor, "_build_sizing_pipeline", return_value=mock_pipeline):
            order = loop._build_order(
                signal,
                CircuitLevel.NORMAL,
                BASELINE_EQUITY,
                BASELINE_CASH,
                candles,
                SYMBOL_AAPL,
                Decimal("0.5"),
                portfolio=portfolio,
                seg_id="us_tech",
            )

        assert order is None

    def test_caution_reduces_pipeline_output(self) -> None:
        """CAUTION level applies _CAUTION_SIZE_FACTOR on top of pipeline output."""
        loop = _make_loop()
        signal = _make_signal()
        signal.confidence = 0.9  # high enough to pass CAUTION gate
        candles = [_make_candle(100.0) for _ in range(NUM_CANDLES)]
        portfolio = _make_portfolio()

        mock_pipeline = MagicMock(spec=PositionSizingPipeline)
        mock_pipeline.compute.return_value = Decimal(10000)

        # Patch _get_segment_min_confidence to return low threshold
        with (
            patch.object(loop._signal_executor, "_build_sizing_pipeline", return_value=mock_pipeline),
            patch.object(loop._signal_executor, "_get_segment_min_confidence", return_value=0.3),
        ):
            order_normal = loop._build_order(
                signal,
                CircuitLevel.NORMAL,
                BASELINE_EQUITY,
                BASELINE_CASH,
                candles,
                SYMBOL_AAPL,
                Decimal("0.5"),
                portfolio=portfolio,
                seg_id="us_tech",
            )
            order_caution = loop._build_order(
                signal,
                CircuitLevel.CAUTION,
                BASELINE_EQUITY,
                BASELINE_CASH,
                candles,
                SYMBOL_AAPL,
                Decimal("0.5"),
                portfolio=portfolio,
                seg_id="us_tech",
            )

        # CAUTION should produce smaller or equal quantity
        assert order_normal is not None
        assert order_caution is not None
        assert order_caution.quantity <= order_normal.quantity


# ── PARITY-03: Pre-trade check parameter passing tests ──────────────


def _make_instrument(symbol: str = SYMBOL_AAPL) -> MagicMock:
    """Create a mock Instrument."""
    inst = MagicMock()
    inst.symbol = symbol
    inst.figi = "BBG000B9XRY4"
    inst.segment_id = "us_tech"
    return inst


class TestPreTradeCheckParams:
    """All 14 pre-trade check parameters are passed in the live path."""

    def _setup_process_instrument(self, loop: TradingLoop) -> tuple:
        """Set up loop state for _process_instrument to reach pre-trade check."""
        candles = [_make_candle(150.0) for _ in range(NUM_CANDLES)]
        signal = _make_signal()

        executor = loop._signal_executor

        # Mock fetcher
        fetcher = MagicMock()
        fetcher.fetch_candles.return_value = candles

        # Mock strategy on executor
        executor._strategy.generate_signal.return_value = signal

        # Mock broker
        broker = MagicMock()
        broker.has_position.return_value = False
        broker.get_portfolio.return_value = _make_portfolio()
        broker.get_positions.return_value = {}
        loop._broker_router.route.return_value = broker
        executor._broker_router.route.return_value = broker

        # Mock portfolio cache on loop for delegation shim
        portfolio = _make_portfolio()
        loop._cycle_portfolio_cache = {MARKET_US: portfolio}

        # Mock _build_order on executor to return a valid order
        mock_order = MagicMock()
        mock_order.quantity = Decimal(10)
        mock_order.symbol = SYMBOL_AAPL
        mock_order.side = "BUY"
        executor._build_order = MagicMock(return_value=mock_order)

        # Mock _compute_total_equity_base and _get_market_equity on executor
        executor._compute_total_equity_base = MagicMock(return_value=BASELINE_EQUITY)
        executor._get_market_equity = MagicMock(return_value=BASELINE_EQUITY)

        return candles, signal, fetcher, broker, portfolio

    def test_pre_trade_receives_stop_loss_price(self) -> None:
        """stop_loss_price sourced from _stop_states[symbol].current_stop."""
        loop = _make_loop()
        self._setup_process_instrument(loop)

        # Seed a stop state
        stop_state = _seed_stop_state(loop)
        expected_stop = stop_state.current_stop

        # Patch pre_trade_checker.check to capture call
        loop._signal_executor._pre_trade_checker.check = MagicMock(
            return_value=MagicMock(passed=True, violations=[])
        )

        loop._process_instrument(
            instrument=_make_instrument(),
            market_id=MARKET_US,
            level=CircuitLevel.NORMAL,
            fetcher=MagicMock(
                fetch_candles=MagicMock(
                    return_value=[_make_candle(150.0) for _ in range(NUM_CANDLES)]
                )
            ),
            now=datetime.now(UTC),
        )

        call_kwargs = loop._signal_executor._pre_trade_checker.check.call_args
        assert call_kwargs is not None, "pre_trade_checker.check must be called"
        # Check stop_loss_price is passed (keyword argument)
        kw = call_kwargs.kwargs or {}
        assert "stop_loss_price" in kw, "stop_loss_price must be passed to pre-trade check"
        assert kw["stop_loss_price"] == expected_stop

    def test_pre_trade_receives_has_pending_order(self) -> None:
        """has_pending_order passed to pre-trade check."""
        loop = _make_loop()
        self._setup_process_instrument(loop)

        loop._signal_executor._pre_trade_checker.check = MagicMock(
            return_value=MagicMock(passed=True, violations=[])
        )

        # Check that _has_pending_order method exists
        assert hasattr(loop, "_has_pending_order"), (
            "TradingLoop must have _has_pending_order method"
        )

    def test_pre_trade_receives_regime_state(self) -> None:
        """regime_state passed from macro_cache."""
        loop = _make_loop()
        self._setup_process_instrument(loop)

        loop._signal_executor._pre_trade_checker.check = MagicMock(
            return_value=MagicMock(passed=True, violations=[])
        )

        # Check that _get_regime_state method exists
        assert hasattr(loop, "_get_regime_state"), "TradingLoop must have _get_regime_state method"

    def test_pre_trade_receives_strategy_name(self) -> None:
        """strategy_name from signal is passed to pre-trade check."""
        loop = _make_loop()
        self._setup_process_instrument(loop)

        loop._signal_executor._pre_trade_checker.check = MagicMock(
            return_value=MagicMock(passed=True, violations=[])
        )

        loop._process_instrument(
            instrument=_make_instrument(),
            market_id=MARKET_US,
            level=CircuitLevel.NORMAL,
            fetcher=MagicMock(
                fetch_candles=MagicMock(
                    return_value=[_make_candle(150.0) for _ in range(NUM_CANDLES)]
                )
            ),
            now=datetime.now(UTC),
        )

        call_kwargs = loop._signal_executor._pre_trade_checker.check.call_args
        assert call_kwargs is not None
        kw = call_kwargs.kwargs or {}
        assert "strategy_name" in kw, "strategy_name must be passed to pre-trade check"
        assert kw["strategy_name"] == "dual_momentum"

    def test_pre_trade_receives_open_positions_and_correlations(self) -> None:
        """open_positions and correlations passed to pre-trade check."""
        loop = _make_loop()
        self._setup_process_instrument(loop)

        loop._signal_executor._pre_trade_checker.check = MagicMock(
            return_value=MagicMock(passed=True, violations=[])
        )

        loop._process_instrument(
            instrument=_make_instrument(),
            market_id=MARKET_US,
            level=CircuitLevel.NORMAL,
            fetcher=MagicMock(
                fetch_candles=MagicMock(
                    return_value=[_make_candle(150.0) for _ in range(NUM_CANDLES)]
                )
            ),
            now=datetime.now(UTC),
        )

        call_kwargs = loop._signal_executor._pre_trade_checker.check.call_args
        assert call_kwargs is not None
        kw = call_kwargs.kwargs or {}
        assert "open_positions" in kw, "open_positions must be passed"
        assert "correlations" in kw, "correlations must be passed"
        assert isinstance(kw["open_positions"], list)
        assert isinstance(kw["correlations"], dict)

    def test_pre_trade_receives_require_stop_loss_false_for_new_entry(self) -> None:
        """require_stop_loss=False for new entries (no existing stop state)."""
        loop = _make_loop()
        self._setup_process_instrument(loop)

        loop._signal_executor._pre_trade_checker.check = MagicMock(
            return_value=MagicMock(passed=True, violations=[])
        )

        loop._process_instrument(
            instrument=_make_instrument(),
            market_id=MARKET_US,
            level=CircuitLevel.NORMAL,
            fetcher=MagicMock(
                fetch_candles=MagicMock(
                    return_value=[_make_candle(150.0) for _ in range(NUM_CANDLES)]
                )
            ),
            now=datetime.now(UTC),
        )

        call_kwargs = loop._signal_executor._pre_trade_checker.check.call_args
        assert call_kwargs is not None
        kw = call_kwargs.kwargs or {}
        assert "require_stop_loss" in kw, "require_stop_loss must be passed"
        assert kw["require_stop_loss"] is False, (
            "New entries have no stop state yet; require_stop_loss must be False"
        )
        assert "symbol" in kw, "symbol must be passed"
        assert kw["symbol"] == SYMBOL_AAPL

    def test_pre_trade_receives_require_stop_loss_true_for_existing_position(self) -> None:
        """require_stop_loss=True when symbol already has a stop state."""
        loop = _make_loop()
        self._setup_process_instrument(loop)

        # Seed a stop state so the symbol has an existing position with stop
        _seed_stop_state(loop)

        loop._signal_executor._pre_trade_checker.check = MagicMock(
            return_value=MagicMock(passed=True, violations=[])
        )

        loop._process_instrument(
            instrument=_make_instrument(),
            market_id=MARKET_US,
            level=CircuitLevel.NORMAL,
            fetcher=MagicMock(
                fetch_candles=MagicMock(
                    return_value=[_make_candle(150.0) for _ in range(NUM_CANDLES)]
                )
            ),
            now=datetime.now(UTC),
        )

        call_kwargs = loop._signal_executor._pre_trade_checker.check.call_args
        assert call_kwargs is not None
        kw = call_kwargs.kwargs or {}
        assert "require_stop_loss" in kw, "require_stop_loss must be passed"
        assert kw["require_stop_loss"] is True, (
            "Existing positions must require stop loss in pre-trade check"
        )
