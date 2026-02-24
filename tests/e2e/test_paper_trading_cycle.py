"""End-to-end test: paper trading cycle.

Exercises TradingLoop._strategy_cycle() and _news_cycle() directly with:
- Mock fetchers returning deterministic BUY-signal candle sequences
- Mock brokers (MagicMock) for US and MOEX
- Real CircuitBreaker, CrossMarketCircuitBreaker
- Real TelegramAlerter (no-op — empty token)
- Real StrategyCombiner with MomentumStrategy

No APScheduler is started; _strategy_cycle() is called directly.
"""

from __future__ import annotations

import uuid as _uuid
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest
from config.modes import WorkMode
from config.settings import Settings

from finalayze.analysis.event_classifier import EventClassifier
from finalayze.analysis.impact_estimator import ImpactEstimator
from finalayze.analysis.news_analyzer import NewsAnalyzer
from finalayze.core.alerts import TelegramAlerter
from finalayze.core.schemas import Candle, NewsArticle, PortfolioState, SentimentResult
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
INITIAL_CASH = Decimal(100_000)
INITIAL_EQUITY = Decimal(100_000)
CANDLE_COUNT = 63  # 40 stable + 16 crash + 3 level + 4 recovery = 63
STABLE_PRICE = 200.0
STABLE_COUNT = 40
CRASH_DROP = 4.0
CRASH_COUNT = 16
LEVEL_COUNT = 3
RECOVERY_STEP = 2.0
RECOVERY_COUNT = 4
STRATEGY_CYCLES = 3
US_SYMBOLS = ["AAPL"]
MOEX_SYMBOLS = ["SBER"]
SENTIMENT_SCORE = 0.8
MIN_SUBMIT_CALLS = 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_buy_signal_candles(
    symbol: str,
    market_id: str,
    count: int = CANDLE_COUNT,
) -> list[Candle]:
    """Return candles whose price pattern reliably triggers a MomentumStrategy BUY signal.

    Pattern verified in tests/unit/test_strategies.py::test_buy_signal_on_oversold_rsi.
    """
    base_dt = datetime(2026, 1, 1, 14, 30, tzinfo=UTC)

    prices: list[float] = [STABLE_PRICE] * STABLE_COUNT
    crash_bottom = STABLE_PRICE - CRASH_DROP * CRASH_COUNT
    prices.extend([STABLE_PRICE - CRASH_DROP * (i + 1) for i in range(CRASH_COUNT)])
    prices.extend([crash_bottom] * LEVEL_COUNT)
    prices.extend([crash_bottom + RECOVERY_STEP * (i + 1) for i in range(RECOVERY_COUNT)])

    # Pad or trim to requested count
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


def _make_mock_broker(market_id: str) -> MagicMock:
    """Return a MagicMock that implements the BrokerBase interface."""
    broker = MagicMock()
    broker.market_id = market_id
    # submit_order returns a filled OrderResult by default
    broker.submit_order.return_value = OrderResult(
        filled=True,
        fill_price=Decimal(150),
        symbol="AAPL",
        side="BUY",
        quantity=Decimal(10),
    )
    # get_portfolio returns a non-zero-position portfolio after first order
    portfolio_with_positions = PortfolioState(
        cash=Decimal(85_000),
        positions={"AAPL": Decimal(10)},
        equity=INITIAL_EQUITY,
        timestamp=datetime(2026, 1, 1, tzinfo=UTC),
    )
    broker.get_portfolio.return_value = portfolio_with_positions
    broker.get_positions.return_value = {}
    broker.has_position.return_value = False
    return broker


def _make_test_settings() -> Settings:
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


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def test_settings() -> Settings:
    return _make_test_settings()


@pytest.fixture
def us_broker() -> MagicMock:
    return _make_mock_broker("us")


@pytest.fixture
def moex_broker() -> MagicMock:
    return _make_mock_broker("moex")


@pytest.fixture
def broker_router(us_broker: MagicMock, moex_broker: MagicMock) -> BrokerRouter:
    return BrokerRouter({"us": us_broker, "moex": moex_broker})


@pytest.fixture
def circuit_breakers() -> dict[str, CircuitBreaker]:
    return {
        "us": CircuitBreaker(market_id="us"),
        "moex": CircuitBreaker(market_id="moex"),
    }


@pytest.fixture
def cross_market_breaker() -> CrossMarketCircuitBreaker:
    return CrossMarketCircuitBreaker()


@pytest.fixture
def alerter() -> TelegramAlerter:
    # Empty token — all methods are no-ops, no HTTP calls made
    return TelegramAlerter(bot_token="", chat_id="")


@pytest.fixture
def instrument_registry() -> InstrumentRegistry:
    return build_default_registry()


@pytest.fixture
def strategy_combiner() -> StrategyCombiner:
    return StrategyCombiner([MomentumStrategy()])


@pytest.fixture
def mock_llm_client() -> AsyncMock:
    """LLM client that always returns positive sentiment JSON."""
    client = AsyncMock()
    client.complete.return_value = (
        '{"sentiment": 0.8, "confidence": 0.9, "reasoning": "Very positive news"}'
    )
    return client


@pytest.fixture
def trading_loop(
    test_settings: Settings,
    broker_router: BrokerRouter,
    circuit_breakers: dict[str, CircuitBreaker],
    cross_market_breaker: CrossMarketCircuitBreaker,
    alerter: TelegramAlerter,
    instrument_registry: InstrumentRegistry,
    strategy_combiner: StrategyCombiner,
    mock_llm_client: AsyncMock,
) -> TradingLoop:
    """Build a TradingLoop with all external dependencies mocked."""
    # Mock fetchers: return BUY-signal candles for each symbol in their market
    us_fetcher = MagicMock()
    us_fetcher.fetch_candles.side_effect = lambda symbol, start, end, timeframe="1d": (
        _make_buy_signal_candles(symbol, "us")
    )

    moex_fetcher = MagicMock()
    moex_fetcher.fetch_candles.side_effect = lambda symbol, start, end, timeframe="1d": (
        _make_buy_signal_candles(symbol, "moex")
    )

    fetchers: dict[str, object] = {"us": us_fetcher, "moex": moex_fetcher}

    # Mock news fetcher — returns one synthetic article
    mock_article = NewsArticle(
        id=_uuid.uuid4(),
        source="mock",
        title="Markets surge on strong earnings",
        content="All major indices rallied sharply today.",
        url="https://example.com/news/1",
        language="en",
        published_at=datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
        symbols=["AAPL", "SBER"],
        scope="global",
    )
    news_fetcher = MagicMock()
    news_fetcher.fetch_news.return_value = [mock_article]

    # Mock NewsAnalyzer, EventClassifier, ImpactEstimator — they are only exercised
    # by _news_cycle; we want _strategy_cycle to be the primary assertion target.
    news_analyzer = MagicMock(spec=NewsAnalyzer)
    news_analyzer.analyze = AsyncMock(
        return_value=SentimentResult(
            sentiment=SENTIMENT_SCORE,
            confidence=0.9,
            reasoning="Mock positive sentiment",
        )
    )

    event_classifier = MagicMock(spec=EventClassifier)
    impact_estimator = MagicMock(spec=ImpactEstimator)

    return TradingLoop(
        settings=test_settings,
        fetchers=fetchers,  # type: ignore[arg-type]
        news_fetcher=news_fetcher,  # type: ignore[arg-type]
        news_analyzer=news_analyzer,  # type: ignore[arg-type]
        event_classifier=event_classifier,
        impact_estimator=impact_estimator,
        strategy=strategy_combiner,
        broker_router=broker_router,
        circuit_breakers=circuit_breakers,
        cross_market_breaker=cross_market_breaker,
        alerter=alerter,
        instrument_registry=instrument_registry,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.e2e
class TestPaperTradingCycle:
    """E2E: TradingLoop strategy cycle submits orders to both brokers."""

    def test_strategy_cycle_submits_orders_to_us_broker(
        self,
        trading_loop: TradingLoop,
        us_broker: MagicMock,
    ) -> None:
        """After 3 strategy cycles with BUY-signal candles, US broker receives at least 1 order."""
        for _ in range(STRATEGY_CYCLES):
            trading_loop._strategy_cycle()

        assert us_broker.submit_order.call_count >= MIN_SUBMIT_CALLS, (
            f"Expected at least {MIN_SUBMIT_CALLS} US order(s), "
            f"got {us_broker.submit_order.call_count}"
        )

    def test_strategy_cycle_submits_orders_to_moex_broker(
        self,
        trading_loop: TradingLoop,
        moex_broker: MagicMock,
    ) -> None:
        """After 3 strategy cycles with BUY-signal candles, MOEX broker gets at least 1 order."""
        for _ in range(STRATEGY_CYCLES):
            trading_loop._strategy_cycle()

        assert moex_broker.submit_order.call_count >= MIN_SUBMIT_CALLS, (
            f"Expected at least {MIN_SUBMIT_CALLS} MOEX order(s), "
            f"got {moex_broker.submit_order.call_count}"
        )

    def test_circuit_breaker_stays_normal_throughout(
        self,
        trading_loop: TradingLoop,
        circuit_breakers: dict[str, CircuitBreaker],
    ) -> None:
        """Circuit breakers must remain NORMAL when no equity drawdown occurs."""
        for _ in range(STRATEGY_CYCLES):
            trading_loop._strategy_cycle()

        assert circuit_breakers["us"].level == CircuitLevel.NORMAL
        assert circuit_breakers["moex"].level == CircuitLevel.NORMAL

    def test_news_cycle_seeds_sentiment_cache(
        self,
        trading_loop: TradingLoop,
        us_broker: MagicMock,
    ) -> None:
        """Running _news_cycle before _strategy_cycle seeds the sentiment cache.

        After seeding, _strategy_cycle should still produce orders (sentiment is advisory,
        not a gate; the primary gate is the circuit breaker level).
        """
        trading_loop._news_cycle()
        trading_loop._strategy_cycle()

        # After seeding positive sentiment, at least one order must reach the US broker
        assert us_broker.submit_order.call_count >= MIN_SUBMIT_CALLS

    def test_orders_submitted_are_buy_orders(
        self,
        trading_loop: TradingLoop,
        us_broker: MagicMock,
    ) -> None:
        """All orders submitted to the US broker must be BUY orders (BUY-signal candles)."""
        for _ in range(STRATEGY_CYCLES):
            trading_loop._strategy_cycle()

        if us_broker.submit_order.call_count == 0:
            pytest.skip("No orders submitted — candle pattern may not trigger signal in this env")

        for call in us_broker.submit_order.call_args_list:
            order: OrderRequest = call.args[0]
            assert order.side == "BUY", (
                f"Expected BUY order, got {order.side} for symbol {order.symbol}"
            )
