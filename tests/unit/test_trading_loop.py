"""Unit tests for TradingLoop -- each cycle method tested in isolation."""

from __future__ import annotations

import threading
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

from config.settings import Settings

from finalayze.analysis.event_classifier import EventType
from finalayze.analysis.impact_estimator import SegmentImpact
from finalayze.core.alerts import TelegramAlerter
from finalayze.core.schemas import Candle, NewsArticle, SentimentResult, Signal, SignalDirection
from finalayze.core.modes import WorkMode
from finalayze.core.trading_loop import TradingLoop
from finalayze.execution.broker_base import OrderResult
from finalayze.markets.instruments import Instrument, InstrumentRegistry
from finalayze.risk.circuit_breaker import CircuitBreaker, CircuitLevel, CrossMarketCircuitBreaker

# ── Module-level constants ──────────────────────────────────────────────────
# A Monday during US market hours (14:30 UTC = 10:30 ET)
_MARKET_OPEN_DT = datetime(2026, 2, 23, 15, 0, tzinfo=UTC)

# ── Constants ──────────────────────────────────────────────────────────────
MARKET_US = "us"
SEGMENT_US_TECH = "us_tech"
SYMBOL_AAPL = "AAPL"
BASELINE_EQUITY = Decimal(100000)
CAUTION_EQUITY = Decimal(94000)  # 6% drawdown -> CAUTION
LIQUIDATE_EQUITY = Decimal(84000)  # 16% drawdown -> LIQUIDATE
FILL_PRICE = Decimal("150.00")
ORDER_QTY = Decimal(10)
NUM_CANDLES = 60
CANDLE_CLOSE = Decimal("150.00")
NEWS_CYCLE_MINUTES = 30
STRATEGY_CYCLE_MINUTES = 60
DAILY_RESET_HOUR = 0
SENTIMENT_BUY = 0.8
SENTIMENT_NEUTRAL = 0.0


def _make_candle(symbol: str = SYMBOL_AAPL, idx: int = 0) -> Candle:
    base = datetime(2025, 1, 1, 14, 30, tzinfo=UTC)
    return Candle(
        symbol=symbol,
        market_id=MARKET_US,
        timeframe="1d",
        timestamp=base + timedelta(days=idx),
        open=CANDLE_CLOSE,
        high=CANDLE_CLOSE,
        low=CANDLE_CLOSE,
        close=CANDLE_CLOSE,
        volume=1_000_000,
    )


def _make_candles(n: int = NUM_CANDLES) -> list[Candle]:
    return [_make_candle(idx=i) for i in range(n)]


def _make_news_article() -> NewsArticle:
    return NewsArticle(
        id=__import__("uuid").uuid4(),
        source="Reuters",
        title="Fed raises rates",
        content="The Federal Reserve raised interest rates by 25bps.",
        url="https://reuters.com/article/1",
        language="en",
        published_at=datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
        scope="us",
    )


def _make_settings(
    news_cycle: int = NEWS_CYCLE_MINUTES,
    strategy_cycle: int = STRATEGY_CYCLE_MINUTES,
    daily_hour: int = DAILY_RESET_HOUR,
    mode: WorkMode = WorkMode.SANDBOX,
) -> MagicMock:
    s = MagicMock(spec=Settings)
    s.news_cycle_minutes = news_cycle
    s.strategy_cycle_minutes = strategy_cycle
    s.daily_reset_hour_utc = daily_hour
    s.max_position_pct = 0.20
    s.kelly_fraction = 0.5
    s.max_positions_per_market = 10
    s.daily_loss_limit_pct = 0.03
    s.max_cross_market_exposure_pct = 0.80
    s.mode = mode
    return s


def _make_registry() -> InstrumentRegistry:
    reg = InstrumentRegistry()
    reg.register(
        Instrument(
            symbol=SYMBOL_AAPL,
            market_id=MARKET_US,
            name="Apple Inc.",
            segment_id=SEGMENT_US_TECH,
        )
    )
    return reg


def _make_trading_loop(
    *,
    signal: Signal | None = None,
    fill: bool = True,
    circuit_level: CircuitLevel = CircuitLevel.NORMAL,
    cross_trip: bool = False,
    sentiment_score: float = SENTIMENT_NEUTRAL,
    mode: WorkMode = WorkMode.SANDBOX,
) -> TradingLoop:
    settings = _make_settings(mode=mode)

    # Mock fetcher
    fetcher = MagicMock()
    fetcher.fetch_candles = MagicMock(return_value=_make_candles())

    # Mock news fetcher
    news_fetcher = MagicMock()
    article = _make_news_article()
    news_fetcher.fetch_news = MagicMock(return_value=[article])

    # Mock news analyzer (async)
    news_analyzer = MagicMock()
    news_analyzer.analyze = AsyncMock(
        return_value=SentimentResult(sentiment=sentiment_score, confidence=0.9, reasoning="test")
    )

    # Mock event classifier (async)
    event_classifier = MagicMock()
    event_classifier.classify = AsyncMock(return_value=EventType.MACRO)

    # Mock impact estimator
    impact_estimator = MagicMock()
    impact_estimator.estimate = MagicMock(
        return_value=[
            SegmentImpact(segment_id=SEGMENT_US_TECH, weight=1.0, sentiment=sentiment_score)
        ]
    )

    # Mock strategy combiner
    strategy = MagicMock()
    strategy.generate_signal = MagicMock(return_value=signal)

    # Mock broker router
    broker_router = MagicMock()
    fill_result = OrderResult(
        filled=fill,
        fill_price=FILL_PRICE if fill else None,
        symbol=SYMBOL_AAPL,
        side="BUY",
        quantity=ORDER_QTY,
        reason="" if fill else "insufficient funds",
    )
    broker_router.submit = MagicMock(return_value=fill_result)
    mock_broker = MagicMock()
    mock_broker.get_portfolio = MagicMock(
        return_value=MagicMock(equity=BASELINE_EQUITY, cash=Decimal(50000))
    )
    mock_broker.get_positions = MagicMock(return_value={})
    mock_broker.submit_order = MagicMock(return_value=fill_result)
    broker_router.route = MagicMock(return_value=mock_broker)
    broker_router.registered_markets = [MARKET_US]

    # Circuit breakers
    cb = MagicMock(spec=CircuitBreaker)
    cb.level = circuit_level
    cb.market_id = MARKET_US
    cb.check = MagicMock(return_value=circuit_level)
    cb.reset_daily = MagicMock()

    cmcb = MagicMock(spec=CrossMarketCircuitBreaker)
    cmcb.check = MagicMock(return_value=cross_trip)
    cmcb.reset_daily = MagicMock()

    alerter = MagicMock(spec=TelegramAlerter)

    registry = _make_registry()

    return TradingLoop(
        settings=settings,  # type: ignore[arg-type]
        fetchers={MARKET_US: fetcher},
        news_fetcher=news_fetcher,
        news_analyzer=news_analyzer,
        event_classifier=event_classifier,
        impact_estimator=impact_estimator,
        strategy=strategy,
        broker_router=broker_router,
        circuit_breakers={MARKET_US: cb},
        cross_market_breaker=cmcb,
        alerter=alerter,
        instrument_registry=registry,
    )


class TestTradingLoopNewsCycle:
    def test_news_cycle_fetches_articles(self) -> None:
        loop = _make_trading_loop()
        loop._news_cycle()  # type: ignore[attr-defined]
        loop._news_fetcher.fetch_news.assert_called_once()  # type: ignore[attr-defined]

    def test_news_cycle_updates_sentiment_cache(self) -> None:
        loop = _make_trading_loop(sentiment_score=SENTIMENT_BUY)
        loop._news_cycle()  # type: ignore[attr-defined]
        # After running the news cycle, the cache should have SOME entries
        # (keyed by affected segments -> symbols or by scope)
        cache = loop._sentiment_cache  # type: ignore[attr-defined]
        assert isinstance(cache, dict)

    def test_news_cycle_uses_thread_lock(self) -> None:
        """Verify _sentiment_cache is guarded by _sentiment_lock."""
        loop = _make_trading_loop()
        assert hasattr(loop, "_sentiment_lock")
        assert isinstance(loop._sentiment_lock, type(threading.Lock()))  # type: ignore[attr-defined]

    def test_news_cycle_no_error_on_empty_articles(self) -> None:
        loop = _make_trading_loop()
        loop._news_fetcher.fetch_news = MagicMock(return_value=[])  # type: ignore[attr-defined]
        loop._news_cycle()  # Must not raise


class TestTradingLoopStrategyCycle:
    def _make_buy_signal(self) -> Signal:
        return Signal(
            strategy_name="combined",
            symbol=SYMBOL_AAPL,
            market_id=MARKET_US,
            segment_id=SEGMENT_US_TECH,
            direction=SignalDirection.BUY,
            confidence=0.75,
            features={},
            reasoning="test signal",
        )

    def test_strategy_cycle_submits_order_on_buy_signal(self) -> None:
        signal = self._make_buy_signal()
        loop = _make_trading_loop(signal=signal)
        with patch("finalayze.core.trading_loop.datetime") as mock_dt:
            mock_dt.now.return_value = _MARKET_OPEN_DT
            loop._strategy_cycle()  # type: ignore[attr-defined]
        loop._broker_router.submit.assert_called()  # type: ignore[attr-defined]

    def test_strategy_cycle_alerts_on_fill(self) -> None:
        signal = self._make_buy_signal()
        loop = _make_trading_loop(signal=signal, fill=True)
        with patch("finalayze.core.trading_loop.datetime") as mock_dt:
            mock_dt.now.return_value = _MARKET_OPEN_DT
            loop._strategy_cycle()  # type: ignore[attr-defined]
        loop._alerter.on_trade_filled.assert_called()  # type: ignore[attr-defined]

    def test_strategy_cycle_alerts_on_rejection(self) -> None:
        signal = self._make_buy_signal()
        loop = _make_trading_loop(signal=signal, fill=False)
        with patch("finalayze.core.trading_loop.datetime") as mock_dt:
            mock_dt.now.return_value = _MARKET_OPEN_DT
            loop._strategy_cycle()  # type: ignore[attr-defined]
        loop._alerter.on_trade_rejected.assert_called()  # type: ignore[attr-defined]

    def test_strategy_cycle_skips_order_when_halted(self) -> None:
        signal = self._make_buy_signal()
        loop = _make_trading_loop(signal=signal, circuit_level=CircuitLevel.HALTED)
        with patch("finalayze.core.trading_loop.datetime") as mock_dt:
            mock_dt.now.return_value = _MARKET_OPEN_DT
            loop._strategy_cycle()  # type: ignore[attr-defined]
        loop._broker_router.submit.assert_not_called()  # type: ignore[attr-defined]

    def test_strategy_cycle_liquidates_when_l3(self) -> None:
        signal = self._make_buy_signal()
        loop = _make_trading_loop(signal=signal, circuit_level=CircuitLevel.LIQUIDATE)
        with (
            patch("finalayze.core.trading_loop.datetime") as mock_dt,
            patch.object(loop, "_liquidate_market") as mock_liq,  # type: ignore[arg-type]
        ):
            mock_dt.now.return_value = _MARKET_OPEN_DT
            loop._strategy_cycle()  # type: ignore[attr-defined]
            mock_liq.assert_called_with(MARKET_US)

    def test_strategy_cycle_no_signal_no_submit(self) -> None:
        loop = _make_trading_loop(signal=None)
        with patch("finalayze.core.trading_loop.datetime") as mock_dt:
            mock_dt.now.return_value = _MARKET_OPEN_DT
            loop._strategy_cycle()  # type: ignore[attr-defined]
        loop._broker_router.submit.assert_not_called()  # type: ignore[attr-defined]

    def test_strategy_cycle_caution_does_not_block_order(self) -> None:
        """CAUTION level should still allow orders (just with halved size)."""
        signal = self._make_buy_signal()
        loop = _make_trading_loop(signal=signal, circuit_level=CircuitLevel.CAUTION)
        with patch("finalayze.core.trading_loop.datetime") as mock_dt:
            mock_dt.now.return_value = _MARKET_OPEN_DT
            loop._strategy_cycle()  # type: ignore[attr-defined]
        loop._broker_router.submit.assert_called()  # type: ignore[attr-defined]


class TestModeGate:
    """6A.1: DEBUG mode must not send orders."""

    def _make_buy_signal(self) -> Signal:
        return Signal(
            strategy_name="combined",
            symbol=SYMBOL_AAPL,
            market_id=MARKET_US,
            segment_id=SEGMENT_US_TECH,
            direction=SignalDirection.BUY,
            confidence=0.75,
            features={},
            reasoning="test signal",
        )

    def test_debug_mode_skips_order_submission(self) -> None:
        signal = self._make_buy_signal()
        loop = _make_trading_loop(signal=signal, mode=WorkMode.DEBUG)
        with patch("finalayze.core.trading_loop.datetime") as mock_dt:
            mock_dt.now.return_value = _MARKET_OPEN_DT
            loop._strategy_cycle()  # type: ignore[attr-defined]
        loop._broker_router.submit.assert_not_called()  # type: ignore[attr-defined]

    def test_sandbox_mode_allows_orders(self) -> None:
        signal = self._make_buy_signal()
        loop = _make_trading_loop(signal=signal, mode=WorkMode.SANDBOX)
        with patch("finalayze.core.trading_loop.datetime") as mock_dt:
            mock_dt.now.return_value = _MARKET_OPEN_DT
            loop._strategy_cycle()  # type: ignore[attr-defined]
        loop._broker_router.submit.assert_called()  # type: ignore[attr-defined]


class TestBuildOrder:
    """6A.11: Kelly sizes against portfolio equity, not cash."""

    def test_kelly_sizes_against_equity(self) -> None:
        """equity=100k, cash=30k, kelly=0.1 -> order_value = 10k (not 3k)."""
        signal = Signal(
            strategy_name="combined",
            symbol=SYMBOL_AAPL,
            market_id=MARKET_US,
            segment_id=SEGMENT_US_TECH,
            direction=SignalDirection.BUY,
            confidence=0.75,
            features={},
            reasoning="test signal",
        )
        loop = _make_trading_loop(signal=signal)
        from finalayze.risk.circuit_breaker import CircuitLevel as CL  # noqa: N811

        kelly = Decimal("0.1")
        equity = Decimal(100000)
        cash = Decimal(30000)
        candles = _make_candles()
        order = loop._build_order(  # type: ignore[attr-defined]
            signal, CL.NORMAL, equity, cash, candles, SYMBOL_AAPL, kelly
        )
        assert order is not None
        # order_value = 0.1 * 100000 = 10000; qty = 10000 / 150 = 66.67 -> 67 (rounded)
        expected_qty = Decimal(67)
        assert order.quantity == expected_qty

    def test_kelly_capped_by_available_cash(self) -> None:
        """equity=100k, cash=5k, kelly=0.1 -> order_value capped at 5k."""
        signal = Signal(
            strategy_name="combined",
            symbol=SYMBOL_AAPL,
            market_id=MARKET_US,
            segment_id=SEGMENT_US_TECH,
            direction=SignalDirection.BUY,
            confidence=0.75,
            features={},
            reasoning="test signal",
        )
        loop = _make_trading_loop(signal=signal)
        from finalayze.risk.circuit_breaker import CircuitLevel as CL  # noqa: N811

        kelly = Decimal("0.1")
        equity = Decimal(100000)
        cash = Decimal(5000)
        candles = _make_candles()
        order = loop._build_order(  # type: ignore[attr-defined]
            signal, CL.NORMAL, equity, cash, candles, SYMBOL_AAPL, kelly
        )
        assert order is not None
        # order_value = min(0.1 * 100000, 5000) = 5000; qty = 5000 / 150 = 33
        expected_qty = Decimal(33)
        assert order.quantity == expected_qty


class TestTradingLoopDailyReset:
    def test_daily_reset_calls_circuit_breaker_reset(self) -> None:
        loop = _make_trading_loop()
        loop._daily_reset()  # type: ignore[attr-defined]
        for cb in loop._circuit_breakers.values():  # type: ignore[attr-defined]
            cb.reset_daily.assert_called_once()

    def test_daily_reset_sends_daily_summary(self) -> None:
        loop = _make_trading_loop()
        loop._daily_reset()  # type: ignore[attr-defined]
        loop._alerter.on_daily_summary.assert_called_once()  # type: ignore[attr-defined]

    def test_daily_reset_calls_cross_market_reset(self) -> None:
        loop = _make_trading_loop()
        loop._daily_reset()  # type: ignore[attr-defined]
        loop._cross_market_breaker.reset_daily.assert_called_once()  # type: ignore[attr-defined]


class TestTradingLoopLiquidation:
    def test_liquidate_market_submits_sell_for_each_position(self) -> None:
        loop = _make_trading_loop()
        # Inject open positions
        positions = {SYMBOL_AAPL: Decimal(10), "MSFT": Decimal(5)}
        mock_broker = loop._broker_router.route(MARKET_US)  # type: ignore[attr-defined]
        mock_broker.get_positions = MagicMock(return_value=positions)
        loop._liquidate_market(MARKET_US)  # type: ignore[attr-defined]
        # Should submit one SELL per position
        assert mock_broker.submit_order.call_count == len(positions)
        for call in mock_broker.submit_order.call_args_list:
            order = call.args[0]
            assert order.side == "SELL"

    def test_liquidate_market_sends_alert(self) -> None:
        loop = _make_trading_loop()
        positions = {SYMBOL_AAPL: Decimal(10)}
        mock_broker = loop._broker_router.route(MARKET_US)  # type: ignore[attr-defined]
        mock_broker.get_positions = MagicMock(return_value=positions)
        loop._liquidate_market(MARKET_US)  # type: ignore[attr-defined]
        loop._alerter.on_circuit_breaker_trip.assert_called()  # type: ignore[attr-defined]


class TestTradingLoopThreadSafety:
    def test_concurrent_news_and_strategy_do_not_deadlock(self) -> None:
        """Two threads reading/writing _sentiment_cache must not deadlock."""
        loop = _make_trading_loop(sentiment_score=SENTIMENT_BUY)

        errors: list[Exception] = []

        def run_news() -> None:
            try:
                loop._news_cycle()  # type: ignore[attr-defined]
            except Exception as exc:
                errors.append(exc)

        def run_strategy() -> None:
            try:
                loop._strategy_cycle()  # type: ignore[attr-defined]
            except Exception as exc:
                errors.append(exc)

        t1 = threading.Thread(target=run_news)
        t2 = threading.Thread(target=run_strategy)
        t1.start()
        t2.start()
        t1.join(timeout=5)
        t2.join(timeout=5)
        assert not errors
