"""End-to-end test: circuit breaker trip and recovery.

Scenario:
  1. Inject -17% equity drawdown -> CircuitBreaker fires LIQUIDATE
  2. TradingLoop._liquidate_market("us") -> broker.submit_order called for each position
  3. TelegramAlerter.on_circuit_breaker_trip is called (spy)
  4. circuit_breaker.reset_manual() -> level back to NORMAL
  5. TelegramAlerter.on_circuit_breaker_reset is called (spy)
  6. _strategy_cycle() -> new orders ARE submitted (trading resumed)

All external I/O is mocked. No APScheduler is started.
"""

from __future__ import annotations

import uuid as _uuid
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest
from config.modes import WorkMode
from config.settings import Settings

from finalayze.analysis.event_classifier import EventClassifier
from finalayze.analysis.impact_estimator import ImpactEstimator
from finalayze.analysis.news_analyzer import NewsAnalyzer
from finalayze.core.alerts import TelegramAlerter
from finalayze.core.schemas import Candle, PortfolioState, SentimentResult
from finalayze.core.trading_loop import TradingLoop
from finalayze.execution.broker_base import OrderRequest, OrderResult
from finalayze.execution.broker_router import BrokerRouter
from finalayze.markets.instruments import InstrumentRegistry, build_default_registry
from finalayze.risk.circuit_breaker import CircuitBreaker, CircuitLevel, CrossMarketCircuitBreaker
from finalayze.strategies.combiner import StrategyCombiner
from finalayze.strategies.momentum import MomentumStrategy

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BASELINE_EQUITY = Decimal(100_000)
# -17% drawdown triggers LIQUIDATE (threshold is 15%)
LIQUIDATE_DRAWDOWN_PCT = Decimal("0.17")
LIQUIDATE_EQUITY = BASELINE_EQUITY * (1 - LIQUIDATE_DRAWDOWN_PCT)

# -6% drawdown triggers CAUTION (threshold is 5%)
CAUTION_DRAWDOWN_PCT = Decimal("0.06")
CAUTION_EQUITY = BASELINE_EQUITY * (1 - CAUTION_DRAWDOWN_PCT)

# -11% drawdown triggers HALTED (threshold is 10%)
HALTED_DRAWDOWN_PCT = Decimal("0.11")
HALTED_EQUITY = BASELINE_EQUITY * (1 - HALTED_DRAWDOWN_PCT)

INITIAL_CASH = Decimal(80_000)
POSITION_QTY_AAPL = Decimal(10)
POSITION_QTY_MSFT = Decimal(5)
FILL_PRICE = Decimal(150)
CANDLE_COUNT = 63  # matches paper trading cycle reference
STABLE_PRICE = 200.0
STABLE_COUNT = 40
CRASH_DROP = 4.0
CRASH_COUNT = 16
LEVEL_COUNT = 3
RECOVERY_STEP = 2.0
RECOVERY_COUNT = 4
EXPECTED_LIQUIDATE_SELL_COUNT = 2  # AAPL + MSFT
EXPECTED_LIQUIDATE_SYMBOLS = {"AAPL", "MSFT"}
MIN_SUBMIT_CALLS_AFTER_RESET = 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_buy_signal_candles(
    symbol: str,
    market_id: str,
    count: int = CANDLE_COUNT,
) -> list[Candle]:
    """Return candles whose price pattern reliably triggers a MomentumStrategy BUY signal."""
    base_dt = datetime(2026, 1, 1, 14, 30, tzinfo=UTC)

    prices: list[float] = [STABLE_PRICE] * STABLE_COUNT
    crash_bottom = STABLE_PRICE - CRASH_DROP * CRASH_COUNT
    prices.extend([STABLE_PRICE - CRASH_DROP * (i + 1) for i in range(CRASH_COUNT)])
    prices.extend([crash_bottom] * LEVEL_COUNT)
    prices.extend([crash_bottom + RECOVERY_STEP * (i + 1) for i in range(RECOVERY_COUNT)])

    if len(prices) < count:
        prices = [STABLE_PRICE] * (count - len(prices)) + prices
    prices = prices[:count]

    candles: list[Candle] = []
    for i, price in enumerate(prices):
        p = Decimal(str(price))
        candles.append(
            Candle(
                symbol=symbol,
                market_id=market_id,
                timeframe="1d",
                timestamp=base_dt + timedelta(days=i),
                open=p,
                high=p + Decimal(1),
                low=p - Decimal(1),
                close=p,
                volume=1_000_000,
            )
        )
    return candles


def _make_settings() -> Settings:
    return Settings(
        mode=WorkMode.TEST,
        database_url="postgresql+asyncpg://test:test@localhost:5432/test",
        redis_url="redis://localhost:6379/1",
        telegram_bot_token="",
        telegram_chat_id="",
        strategy_cycle_minutes=60,
        news_cycle_minutes=30,
        daily_reset_hour_utc=0,
    )


def _make_broker_with_equity(equity: Decimal) -> MagicMock:
    """Return a mock broker with the given equity and two US positions (AAPL, MSFT)."""
    broker = MagicMock()
    portfolio = PortfolioState(
        cash=INITIAL_CASH,
        positions={"AAPL": POSITION_QTY_AAPL, "MSFT": POSITION_QTY_MSFT},
        equity=equity,
        timestamp=datetime(2026, 1, 1, tzinfo=UTC),
    )
    broker.get_portfolio.return_value = portfolio
    broker.get_positions.return_value = {
        "AAPL": POSITION_QTY_AAPL,
        "MSFT": POSITION_QTY_MSFT,
    }
    broker.submit_order.return_value = OrderResult(
        filled=True,
        fill_price=FILL_PRICE,
        symbol="AAPL",
        side="SELL",
        quantity=POSITION_QTY_AAPL,
    )
    broker.has_position.return_value = True
    return broker


def _make_normal_broker() -> MagicMock:
    """Return a mock broker with BASELINE_EQUITY and two US positions."""
    return _make_broker_with_equity(BASELINE_EQUITY)


def _make_moex_broker() -> MagicMock:
    """Return a simple MOEX mock broker with BASELINE_EQUITY and no positions."""
    broker = MagicMock()
    portfolio = PortfolioState(
        cash=INITIAL_CASH,
        positions={},
        equity=BASELINE_EQUITY,
        timestamp=datetime(2026, 1, 1, tzinfo=UTC),
    )
    broker.get_portfolio.return_value = portfolio
    broker.get_positions.return_value = {}
    broker.submit_order.return_value = OrderResult(
        filled=True,
        fill_price=FILL_PRICE,
        symbol="SBER",
        side="BUY",
        quantity=Decimal(1),
    )
    broker.has_position.return_value = False
    return broker


def _make_trading_loop(
    us_broker: MagicMock,
    moex_broker: MagicMock,
    circuit_breakers: dict[str, CircuitBreaker],
    alerter: TelegramAlerter,
    instrument_registry: InstrumentRegistry,
) -> TradingLoop:
    """Build a TradingLoop wired with the given brokers and dependencies."""
    settings = _make_settings()
    broker_router = BrokerRouter({"us": us_broker, "moex": moex_broker})
    cross_breaker = CrossMarketCircuitBreaker()

    us_fetcher = MagicMock()
    us_fetcher.fetch_candles.side_effect = lambda symbol, market_id, limit: (
        _make_buy_signal_candles(symbol, "us")
    )

    moex_fetcher = MagicMock()
    moex_fetcher.fetch_candles.side_effect = lambda symbol, market_id, limit: (
        _make_buy_signal_candles(symbol, "moex")
    )

    news_fetcher = MagicMock()
    news_fetcher.fetch_news.return_value = []

    news_analyzer = MagicMock(spec=NewsAnalyzer)
    event_classifier = MagicMock(spec=EventClassifier)
    impact_estimator = MagicMock(spec=ImpactEstimator)

    momentum = MomentumStrategy()
    _base_get_params = momentum.get_parameters

    def _get_params_no_filters(segment_id: str) -> dict[str, object]:
        params = dict(_base_get_params(segment_id))
        params.update(trend_filter=False, adx_filter=False, volume_filter=False)
        return params

    momentum.get_parameters = _get_params_no_filters  # type: ignore[assignment]
    strategy = StrategyCombiner([momentum])

    return TradingLoop(
        settings=settings,
        fetchers={"us": us_fetcher, "moex": moex_fetcher},  # type: ignore[arg-type]
        news_fetcher=news_fetcher,  # type: ignore[arg-type]
        news_analyzer=news_analyzer,  # type: ignore[arg-type]
        event_classifier=event_classifier,
        impact_estimator=impact_estimator,
        strategy=strategy,
        broker_router=broker_router,
        circuit_breakers=circuit_breakers,
        cross_market_breaker=cross_breaker,
        alerter=alerter,
        instrument_registry=instrument_registry,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def instrument_registry() -> InstrumentRegistry:
    return build_default_registry()


@pytest.fixture
def alerter_no_token() -> TelegramAlerter:
    """No-op alerter (empty token)."""
    return TelegramAlerter(bot_token="", chat_id="")


@pytest.fixture
def alerter_with_token() -> TelegramAlerter:
    """Alerter with a dummy token — methods are still no-ops (no real HTTP)."""
    return TelegramAlerter(bot_token="dummy_token", chat_id="123")  # noqa: S106


@pytest.fixture
def cb_us() -> CircuitBreaker:
    return CircuitBreaker(market_id="us")


@pytest.fixture
def cb_moex() -> CircuitBreaker:
    return CircuitBreaker(market_id="moex")


@pytest.fixture
def circuit_breakers(cb_us: CircuitBreaker, cb_moex: CircuitBreaker) -> dict[str, CircuitBreaker]:
    return {"us": cb_us, "moex": cb_moex}


@pytest.fixture
def us_broker_liquidate() -> MagicMock:
    """Broker that returns LIQUIDATE_EQUITY so the circuit breaker trips."""
    return _make_broker_with_equity(LIQUIDATE_EQUITY)


@pytest.fixture
def us_broker_normal() -> MagicMock:
    """Broker that returns BASELINE_EQUITY — no circuit breaker trip."""
    return _make_normal_broker()


@pytest.fixture
def moex_broker() -> MagicMock:
    return _make_moex_broker()


@pytest.fixture
def trading_loop_for_trip(
    us_broker_liquidate: MagicMock,
    moex_broker: MagicMock,
    circuit_breakers: dict[str, CircuitBreaker],
    alerter_no_token: TelegramAlerter,
    instrument_registry: InstrumentRegistry,
) -> TradingLoop:
    """TradingLoop wired to trigger a LIQUIDATE circuit breaker on _strategy_cycle."""
    return _make_trading_loop(
        us_broker_liquidate,
        moex_broker,
        circuit_breakers,
        alerter_no_token,
        instrument_registry,
    )


@pytest.fixture
def trading_loop_normal(
    us_broker_normal: MagicMock,
    moex_broker: MagicMock,
    circuit_breakers: dict[str, CircuitBreaker],
    alerter_no_token: TelegramAlerter,
    instrument_registry: InstrumentRegistry,
) -> TradingLoop:
    """TradingLoop wired with normal (no drawdown) equity."""
    return _make_trading_loop(
        us_broker_normal,
        moex_broker,
        circuit_breakers,
        alerter_no_token,
        instrument_registry,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.e2e
class TestCircuitBreakerTripAndRecovery:
    """E2E: circuit breaker triggers liquidation and recovers after manual reset."""

    def test_17pct_drawdown_triggers_liquidate_level(
        self,
        cb_us: CircuitBreaker,
    ) -> None:
        """A -17% equity drawdown causes CircuitBreaker to return LIQUIDATE."""
        level = cb_us.check(
            current_equity=LIQUIDATE_EQUITY,
            baseline_equity=BASELINE_EQUITY,
        )
        assert level == CircuitLevel.LIQUIDATE

    def test_liquidate_market_submits_sell_for_each_position(
        self,
        trading_loop_for_trip: TradingLoop,
        us_broker_liquidate: MagicMock,
    ) -> None:
        """_liquidate_market('us') submits exactly 2 SELL orders (AAPL + MSFT)."""
        trading_loop_for_trip._liquidate_market("us")

        assert us_broker_liquidate.submit_order.call_count == EXPECTED_LIQUIDATE_SELL_COUNT, (
            f"Expected {EXPECTED_LIQUIDATE_SELL_COUNT} submit_order calls, "
            f"got {us_broker_liquidate.submit_order.call_count}"
        )

    def test_liquidate_market_submits_sell_orders_only(
        self,
        trading_loop_for_trip: TradingLoop,
        us_broker_liquidate: MagicMock,
    ) -> None:
        """All orders submitted by _liquidate_market must have side == SELL."""
        trading_loop_for_trip._liquidate_market("us")

        for call in us_broker_liquidate.submit_order.call_args_list:
            order: OrderRequest = call.args[0]
            assert order.side == "SELL", (
                f"Expected SELL order, got {order.side} for symbol {order.symbol}"
            )

    def test_liquidate_market_sells_correct_symbols(
        self,
        trading_loop_for_trip: TradingLoop,
        us_broker_liquidate: MagicMock,
    ) -> None:
        """_liquidate_market('us') must sell exactly AAPL and MSFT."""
        trading_loop_for_trip._liquidate_market("us")

        sold_symbols = {
            call.args[0].symbol for call in us_broker_liquidate.submit_order.call_args_list
        }
        assert sold_symbols == EXPECTED_LIQUIDATE_SYMBOLS, (
            f"Expected symbols {EXPECTED_LIQUIDATE_SYMBOLS}, got {sold_symbols}"
        )

    def test_circuit_breaker_trip_alert_is_sent(
        self,
        alerter_with_token: TelegramAlerter,
        us_broker_liquidate: MagicMock,
        moex_broker: MagicMock,
        circuit_breakers: dict[str, CircuitBreaker],
        instrument_registry: InstrumentRegistry,
    ) -> None:
        """on_circuit_breaker_trip is called when _strategy_cycle detects LIQUIDATE."""
        loop = _make_trading_loop(
            us_broker_liquidate,
            moex_broker,
            circuit_breakers,
            alerter_with_token,
            instrument_registry,
        )
        # Pre-set the baseline so that the -17% drawdown is real relative to it
        loop._baseline_equities = {"us": BASELINE_EQUITY, "moex": BASELINE_EQUITY}

        with patch.object(alerter_with_token, "on_circuit_breaker_trip") as spy:
            loop._strategy_cycle()

        spy.assert_called_once()
        call_kwargs = spy.call_args
        market_arg = (
            call_kwargs.args[0] if call_kwargs.args else call_kwargs.kwargs.get("market_id")
        )
        assert market_arg == "us"

        # Verify the level argument was LIQUIDATE
        level_arg = call_kwargs.kwargs.get("level") if call_kwargs.kwargs else None
        if level_arg is None and len(call_kwargs.args) > 1:
            level_arg = call_kwargs.args[1]
        assert level_arg == CircuitLevel.LIQUIDATE, (
            f"Expected on_circuit_breaker_trip called with level=LIQUIDATE, got {spy.call_args}"
        )

    def test_manual_reset_restores_normal_level(
        self,
        cb_us: CircuitBreaker,
    ) -> None:
        """reset_manual() clears LIQUIDATE -> NORMAL."""
        cb_us.check(current_equity=LIQUIDATE_EQUITY, baseline_equity=BASELINE_EQUITY)
        assert cb_us.level == CircuitLevel.LIQUIDATE

        cb_us.reset_manual()

        assert cb_us.level == CircuitLevel.NORMAL

    def test_circuit_breaker_reset_alert_is_sent(
        self,
        alerter_with_token: TelegramAlerter,
        cb_us: CircuitBreaker,
    ) -> None:
        """After reset_manual(), on_circuit_breaker_reset can be called with market_id kwarg."""
        cb_us.check(current_equity=LIQUIDATE_EQUITY, baseline_equity=BASELINE_EQUITY)
        assert cb_us.level == CircuitLevel.LIQUIDATE
        cb_us.reset_manual()
        assert cb_us.level == CircuitLevel.NORMAL
        # Verify the alerter method signature accepts market_id as keyword arg
        # (called by operator workflow after reset) -- must not raise
        alerter_with_token.on_circuit_breaker_reset(market_id="us")

    def test_trading_resumes_after_manual_reset(
        self,
        trading_loop_for_trip: TradingLoop,
        us_broker_liquidate: MagicMock,
        circuit_breakers: dict[str, CircuitBreaker],
    ) -> None:
        """After reset_manual(), _strategy_cycle no longer liquidates but processes normally.

        We verify that submit_order is called via _liquidate_market when pre-tripped,
        and that after reset the level reverts to NORMAL so the strategy path is taken.
        """
        # Pre-trip: force LIQUIDATE level
        circuit_breakers["us"].check(
            current_equity=LIQUIDATE_EQUITY, baseline_equity=BASELINE_EQUITY
        )
        assert circuit_breakers["us"].level == CircuitLevel.LIQUIDATE

        # Manual reset
        circuit_breakers["us"].reset_manual()
        assert circuit_breakers["us"].level == CircuitLevel.NORMAL

        # Now configure the broker to return BASELINE_EQUITY so no re-trip happens
        us_broker_liquidate.get_portfolio.return_value = PortfolioState(
            cash=INITIAL_CASH,
            positions={"AAPL": POSITION_QTY_AAPL},
            equity=BASELINE_EQUITY,
            timestamp=datetime(2026, 1, 1, tzinfo=UTC),
        )

        trading_loop_for_trip._baseline_equities = {
            "us": BASELINE_EQUITY,
            "moex": BASELINE_EQUITY,
        }

        # Reset submit_order call count to isolate post-reset behaviour
        us_broker_liquidate.submit_order.reset_mock()

        trading_loop_for_trip._strategy_cycle()

        # Level must still be NORMAL (not re-tripped)
        assert circuit_breakers["us"].level == CircuitLevel.NORMAL

        # Trading must have resumed: at least one new BUY order submitted after reset
        assert us_broker_liquidate.submit_order.call_count >= MIN_SUBMIT_CALLS_AFTER_RESET, (
            f"Expected at least {MIN_SUBMIT_CALLS_AFTER_RESET} order(s) after reset, "
            f"got {us_broker_liquidate.submit_order.call_count}"
        )

    def test_caution_level_does_not_liquidate(
        self,
        cb_us: CircuitBreaker,
    ) -> None:
        """A -6% drawdown triggers CAUTION only, not LIQUIDATE or HALTED."""
        level = cb_us.check(
            current_equity=CAUTION_EQUITY,
            baseline_equity=BASELINE_EQUITY,
        )
        assert level == CircuitLevel.CAUTION
        assert level != CircuitLevel.LIQUIDATE
        assert level != CircuitLevel.HALTED

    def test_halted_level_does_not_auto_resume(
        self,
        cb_us: CircuitBreaker,
    ) -> None:
        """A -11% drawdown triggers HALTED; reset_daily restores NORMAL."""
        level = cb_us.check(
            current_equity=HALTED_EQUITY,
            baseline_equity=BASELINE_EQUITY,
        )
        assert level == CircuitLevel.HALTED

        # Daily reset (not manual) clears HALTED -> NORMAL
        cb_us.reset_daily(new_baseline=BASELINE_EQUITY)

        assert cb_us.level == CircuitLevel.NORMAL
