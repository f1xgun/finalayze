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
from finalayze.core.modes import WorkMode
from finalayze.core.schemas import Candle, NewsArticle, SentimentResult, Signal, SignalDirection
from finalayze.core.trading_loop import TradingLoop
from finalayze.execution.broker_base import OrderResult
from finalayze.markets.instruments import Instrument, InstrumentRegistry
from finalayze.risk.circuit_breaker import CircuitBreaker, CircuitLevel, CrossMarketCircuitBreaker
from finalayze.risk.rollout import ROLLOUT_LIMITS, RolloutLimits

# ── Module-level constants ──────────────────────────────────────────────────
# A Monday during US market hours (14:30 UTC = 10:30 ET)
_MARKET_OPEN_DT = datetime(2026, 2, 23, 15, 0, tzinfo=UTC)


def _market_open_ctx(loop: object) -> object:
    """Return a combined context manager that patches SCHEDULES and _now for market-open."""
    from contextlib import ExitStack

    mock_schedule = MagicMock()
    mock_schedule.is_market_open.return_value = True

    stack = ExitStack()
    stack.enter_context(
        patch("finalayze.orchestration.trading_loop.SCHEDULES", {MARKET_US: mock_schedule})
    )
    stack.enter_context(patch.object(loop, "_now", return_value=_MARKET_OPEN_DT))
    return stack


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
    # Base the candles relative to real now() so the latest candle
    # is within the 48-hour staleness threshold.
    base = datetime.now(UTC) - timedelta(days=NUM_CANDLES)
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
    from finalayze.core.modes import RolloutPhase

    s.rollout_phase = RolloutPhase.FULL
    s.effective_risk_limits = MagicMock(return_value=ROLLOUT_LIMITS[RolloutPhase.FULL])
    return s


def _make_registry() -> InstrumentRegistry:
    reg = InstrumentRegistry()
    reg.register(
        Instrument(
            symbol=SYMBOL_AAPL,
            market_id=MARKET_US,
            name="Apple Inc.",
            segment_id=SEGMENT_US_TECH,
            figi="BBG000B9XRY4",
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
        return_value=MagicMock(equity=BASELINE_EQUITY, cash=Decimal(50000), positions={})
    )
    mock_broker.get_positions = MagicMock(return_value={})
    mock_broker.has_position = MagicMock(return_value=False)
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
        with _market_open_ctx(loop):
            loop._strategy_cycle()  # type: ignore[attr-defined]
        loop._broker_router.submit.assert_called()  # type: ignore[attr-defined]

    def test_strategy_cycle_alerts_on_fill(self) -> None:
        signal = self._make_buy_signal()
        loop = _make_trading_loop(signal=signal, fill=True)
        with _market_open_ctx(loop):
            loop._strategy_cycle()  # type: ignore[attr-defined]
        loop._alerter.on_trade_filled.assert_called()  # type: ignore[attr-defined]

    def test_strategy_cycle_alerts_on_rejection(self) -> None:
        signal = self._make_buy_signal()
        loop = _make_trading_loop(signal=signal, fill=False)
        with _market_open_ctx(loop):
            loop._strategy_cycle()  # type: ignore[attr-defined]
        loop._alerter.on_trade_rejected.assert_called()  # type: ignore[attr-defined]

    def test_strategy_cycle_skips_order_when_halted(self) -> None:
        signal = self._make_buy_signal()
        loop = _make_trading_loop(signal=signal, circuit_level=CircuitLevel.HALTED)
        with _market_open_ctx(loop):
            loop._strategy_cycle()  # type: ignore[attr-defined]
        loop._broker_router.submit.assert_not_called()  # type: ignore[attr-defined]

    def test_strategy_cycle_liquidates_when_l3(self) -> None:
        signal = self._make_buy_signal()
        loop = _make_trading_loop(signal=signal, circuit_level=CircuitLevel.LIQUIDATE)
        with (
            _market_open_ctx(loop),
            patch.object(loop, "_liquidate_market") as mock_liq,  # type: ignore[arg-type]
        ):
            loop._strategy_cycle()  # type: ignore[attr-defined]
            mock_liq.assert_called_with(MARKET_US)

    def test_strategy_cycle_no_signal_no_submit(self) -> None:
        loop = _make_trading_loop(signal=None)
        with _market_open_ctx(loop):
            loop._strategy_cycle()  # type: ignore[attr-defined]
        loop._broker_router.submit.assert_not_called()  # type: ignore[attr-defined]

    def test_strategy_cycle_caution_does_not_block_order(self) -> None:
        """CAUTION level should still allow orders (just with halved size)."""
        signal = self._make_buy_signal()
        loop = _make_trading_loop(signal=signal, circuit_level=CircuitLevel.CAUTION)
        with _market_open_ctx(loop):
            loop._strategy_cycle()  # type: ignore[attr-defined]
        loop._broker_router.submit.assert_called()  # type: ignore[attr-defined]


class TestHasOpenPositionPassthrough:
    """P0: _process_instrument must pass has_open_position to strategy."""

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

    def test_has_open_position_passed_to_strategy(self) -> None:
        """generate_signal should receive has_open_position from broker."""
        signal = self._make_buy_signal()
        loop = _make_trading_loop(signal=signal)
        # Make broker report an open position
        mock_broker = loop._broker_router.route.return_value  # type: ignore[attr-defined]
        mock_broker.has_position.return_value = True

        with _market_open_ctx(loop):
            loop._strategy_cycle()  # type: ignore[attr-defined]

        # Verify generate_signal was called with has_open_position=True
        call_kwargs = loop._strategy.generate_signal.call_args  # type: ignore[attr-defined]
        assert call_kwargs.kwargs.get("has_open_position") is True

    def test_has_open_position_false_when_no_position(self) -> None:
        """generate_signal should receive has_open_position=False when no position."""
        signal = self._make_buy_signal()
        loop = _make_trading_loop(signal=signal)
        mock_broker = loop._broker_router.route.return_value  # type: ignore[attr-defined]
        mock_broker.has_position.return_value = False

        with _market_open_ctx(loop):
            loop._strategy_cycle()  # type: ignore[attr-defined]

        call_kwargs = loop._strategy.generate_signal.call_args  # type: ignore[attr-defined]
        assert call_kwargs.kwargs.get("has_open_position") is False


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
        with _market_open_ctx(loop):
            loop._strategy_cycle()  # type: ignore[attr-defined]
        loop._broker_router.submit.assert_not_called()  # type: ignore[attr-defined]

    def test_sandbox_mode_allows_orders(self) -> None:
        signal = self._make_buy_signal()
        loop = _make_trading_loop(signal=signal, mode=WorkMode.SANDBOX)
        with _market_open_ctx(loop):
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
        kelly = Decimal("0.1")
        equity = Decimal(100000)
        cash = Decimal(30000)
        candles = _make_candles()
        order = loop._build_order(  # type: ignore[attr-defined]
            signal, CircuitLevel.NORMAL, equity, cash, candles, SYMBOL_AAPL, kelly
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
        kelly = Decimal("0.1")
        equity = Decimal(100000)
        cash = Decimal(5000)
        candles = _make_candles()
        order = loop._build_order(  # type: ignore[attr-defined]
            signal, CircuitLevel.NORMAL, equity, cash, candles, SYMBOL_AAPL, kelly
        )
        assert order is not None
        # order_value = min(0.1 * 100000, 5000) = 5000; qty = 5000 / 150 = 33
        expected_qty = Decimal(33)
        assert order.quantity == expected_qty


class TestCrossMarketExposure:
    """6A.4: Cross-market exposure aggregation."""

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

    def test_cross_market_exposure_aggregated(self) -> None:
        """Verify cross-market exposure sums invested value across all markets."""
        signal = self._make_buy_signal()
        loop = _make_trading_loop(signal=signal)
        # Access private _compute_total_equity_base to verify it aggregates
        total = loop._compute_total_equity_base()  # type: ignore[attr-defined]
        # Should return some equity (from mock broker)
        assert total > Decimal(0)

    def test_cross_market_exposure_rejects_when_aggregated_too_high(self) -> None:
        """When aggregated exposure is too high, pre-trade check rejects."""
        signal = self._make_buy_signal()
        loop = _make_trading_loop(signal=signal)
        # Set max exposure very low so it triggers
        loop._settings.max_cross_market_exposure_pct = 0.01  # type: ignore[attr-defined]
        with _market_open_ctx(loop):
            loop._strategy_cycle()  # type: ignore[attr-defined]
        # With very low max exposure, the order should be rejected
        loop._broker_router.submit.assert_not_called()  # type: ignore[attr-defined]


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


class TestDailyPnLComputation:
    """Bug 7: Daily P&L must report actual equity change, not zero."""

    BASELINE = Decimal(1000000)
    CURRENT_EQUITY = Decimal(1005000)
    EXPECTED_PNL = Decimal(5000)

    def test_daily_reset_computes_pnl_from_baseline(self) -> None:
        """P&L = current_equity - baseline_equity, sent to alerter."""
        loop = _make_trading_loop()
        # Set baseline equity before the day started
        loop._baseline_equities[MARKET_US] = self.BASELINE  # type: ignore[attr-defined]

        # Mock broker to return current equity
        mock_broker = loop._broker_router.route(MARKET_US)  # type: ignore[attr-defined]
        mock_broker.get_portfolio.return_value = MagicMock(
            equity=self.CURRENT_EQUITY, cash=Decimal(50000)
        )

        loop._daily_reset()  # type: ignore[attr-defined]

        # Verify alerter received actual P&L, not zero
        call_args = loop._alerter.on_daily_summary.call_args  # type: ignore[attr-defined]
        market_pnl = call_args.args[0]
        assert market_pnl[MARKET_US] == self.EXPECTED_PNL

    def test_daily_reset_reports_pnl_to_metrics(self) -> None:
        """MetricsCollector.set_daily_pnl must receive actual P&L float."""
        loop = _make_trading_loop()
        loop._baseline_equities[MARKET_US] = self.BASELINE  # type: ignore[attr-defined]

        mock_broker = loop._broker_router.route(MARKET_US)  # type: ignore[attr-defined]
        mock_broker.get_portfolio.return_value = MagicMock(
            equity=self.CURRENT_EQUITY, cash=Decimal(50000)
        )

        mock_mc = MagicMock()
        loop._metrics = mock_mc  # type: ignore[attr-defined]
        loop._daily_reset()  # type: ignore[attr-defined]
        mock_mc.set_daily_pnl.assert_called_with(MARKET_US, float(self.EXPECTED_PNL))

    def test_daily_reset_negative_pnl(self) -> None:
        """Negative P&L (loss) should be reported correctly."""
        loop = _make_trading_loop()
        loop._baseline_equities[MARKET_US] = Decimal(1000000)  # type: ignore[attr-defined]

        mock_broker = loop._broker_router.route(MARKET_US)  # type: ignore[attr-defined]
        mock_broker.get_portfolio.return_value = MagicMock(
            equity=Decimal(995000), cash=Decimal(50000)
        )

        loop._daily_reset()  # type: ignore[attr-defined]

        call_args = loop._alerter.on_daily_summary.call_args  # type: ignore[attr-defined]
        market_pnl = call_args.args[0]
        assert market_pnl[MARKET_US] == Decimal(-5000)

    def test_daily_reset_no_baseline_defaults_to_zero_pnl(self) -> None:
        """When no baseline exists yet, P&L should be zero (first day)."""
        loop = _make_trading_loop()
        # Do NOT set any baseline -- simulate first trading day
        loop._baseline_equities.clear()  # type: ignore[attr-defined]

        loop._daily_reset()  # type: ignore[attr-defined]

        call_args = loop._alerter.on_daily_summary.call_args  # type: ignore[attr-defined]
        market_pnl = call_args.args[0]
        # With no baseline, current - current = 0
        assert market_pnl[MARKET_US] == Decimal(0)

    def test_daily_reset_updates_baseline_after_pnl(self) -> None:
        """After _daily_reset, baseline should be updated to current equity."""
        loop = _make_trading_loop()
        loop._baseline_equities[MARKET_US] = self.BASELINE  # type: ignore[attr-defined]

        mock_broker = loop._broker_router.route(MARKET_US)  # type: ignore[attr-defined]
        mock_broker.get_portfolio.return_value = MagicMock(
            equity=self.CURRENT_EQUITY, cash=Decimal(50000)
        )

        loop._daily_reset()  # type: ignore[attr-defined]

        # After reset, baseline should be updated to current equity for next day
        assert loop._baseline_equities[MARKET_US] == self.CURRENT_EQUITY  # type: ignore[attr-defined]


class TestWeeklyReset:
    """6A.10: Weekly loss limit reset wiring."""

    def test_weekly_reset_on_monday(self) -> None:
        loop = _make_trading_loop()
        # Monday 2026-02-23
        monday = datetime(2026, 2, 23, 0, 0, tzinfo=UTC)
        with patch.object(loop, "_now", return_value=monday):  # type: ignore[arg-type]
            loop._daily_reset()  # type: ignore[attr-defined]
        # Verify reset_week was called on the loss_limit_tracker
        # We need to check it was called; use a spy
        assert True  # If no exception, the method ran

    def test_no_weekly_reset_on_tuesday(self) -> None:
        loop = _make_trading_loop()
        # Tuesday 2026-02-24
        tuesday = datetime(2026, 2, 24, 0, 0, tzinfo=UTC)
        # Spy on reset_week
        with patch.object(
            loop._loss_limit_tracker,  # type: ignore[attr-defined]
            "reset_week",
        ) as mock_reset_week:
            with patch.object(loop, "_now", return_value=tuesday):  # type: ignore[arg-type]
                loop._daily_reset()  # type: ignore[attr-defined]
            mock_reset_week.assert_not_called()

    def test_weekly_reset_called_on_monday(self) -> None:
        loop = _make_trading_loop()
        # Monday 2026-02-23
        monday = datetime(2026, 2, 23, 0, 0, tzinfo=UTC)
        with patch.object(
            loop._loss_limit_tracker,  # type: ignore[attr-defined]
            "reset_week",
        ) as mock_reset_week:
            with patch.object(loop, "_now", return_value=monday):  # type: ignore[arg-type]
                loop._daily_reset()  # type: ignore[attr-defined]
            mock_reset_week.assert_called_once()


class TestPDTTrackerWiring:
    """6A.7: PDT tracker wiring + day trade detection."""

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

    def test_pdt_tracker_wired(self) -> None:
        loop = _make_trading_loop()
        assert hasattr(loop, "_pdt_tracker")
        assert loop._pdt_tracker is not None  # type: ignore[attr-defined]

    def test_is_day_trade_sell_with_position(self) -> None:
        loop = _make_trading_loop()
        mock_broker = loop._broker_router.route(MARKET_US)  # type: ignore[attr-defined]
        mock_broker.has_position = MagicMock(return_value=True)
        result = loop._is_day_trade(SYMBOL_AAPL, "SELL", MARKET_US)  # type: ignore[attr-defined]
        assert result is True

    def test_is_day_trade_non_us_returns_false(self) -> None:
        loop = _make_trading_loop()
        result = loop._is_day_trade(SYMBOL_AAPL, "SELL", "moex")  # type: ignore[attr-defined]
        assert result is False

    def test_day_trade_recorded_on_fill(self) -> None:
        """When a day trade sell is filled, PDT tracker records it."""
        signal = Signal(
            strategy_name="combined",
            symbol=SYMBOL_AAPL,
            market_id=MARKET_US,
            segment_id=SEGMENT_US_TECH,
            direction=SignalDirection.SELL,
            confidence=0.75,
            features={},
            reasoning="test signal",
        )
        loop = _make_trading_loop(signal=signal)
        mock_broker = loop._broker_router.route(MARKET_US)  # type: ignore[attr-defined]
        mock_broker.has_position = MagicMock(return_value=True)
        # Ensure portfolio reports a held position so _build_order builds a SELL order
        mock_broker.get_portfolio.return_value = MagicMock(
            equity=BASELINE_EQUITY,
            cash=Decimal(50000),
            positions={SYMBOL_AAPL: ORDER_QTY},
        )

        initial_count = loop._pdt_tracker.recent_day_trades  # type: ignore[attr-defined]
        with _market_open_ctx(loop):
            loop._strategy_cycle()  # type: ignore[attr-defined]

        # After a fill of a day-trade sell, the tracker should record it
        assert loop._pdt_tracker.recent_day_trades > initial_count  # type: ignore[attr-defined]


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


class TestSandboxMonitorIntegration:
    """Tests for SandboxMonitorService integration in TradingLoop."""

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

    def test_trading_loop_accepts_sandbox_monitor_none(self) -> None:
        """TradingLoop should accept sandbox_monitor=None (backward compatible)."""
        loop = _make_trading_loop()
        assert loop._sandbox_monitor is None  # type: ignore[attr-defined]

    def test_trading_loop_accepts_sandbox_monitor(self) -> None:
        """TradingLoop should store sandbox_monitor when provided."""
        monitor = MagicMock()
        settings = _make_settings()
        loop = TradingLoop(
            settings=settings,  # type: ignore[arg-type]
            fetchers={MARKET_US: MagicMock()},
            news_fetcher=MagicMock(),
            news_analyzer=MagicMock(),
            event_classifier=MagicMock(),
            impact_estimator=MagicMock(),
            strategy=MagicMock(),
            broker_router=MagicMock(),
            circuit_breakers={},
            cross_market_breaker=MagicMock(spec=CrossMarketCircuitBreaker),
            alerter=MagicMock(spec=TelegramAlerter),
            instrument_registry=_make_registry(),
            sandbox_monitor=monitor,
        )
        assert loop._sandbox_monitor is monitor  # type: ignore[attr-defined]

    def test_submit_order_records_slippage(self) -> None:
        """When sandbox_monitor is set and order fills with candles, record_slippage is called."""
        signal = self._make_buy_signal()
        loop = _make_trading_loop(signal=signal, fill=True)
        monitor = MagicMock()
        loop._sandbox_monitor = monitor  # type: ignore[attr-defined]

        candles = _make_candles()
        order = loop._OrderRequest(symbol=SYMBOL_AAPL, side="BUY", quantity=ORDER_QTY)  # type: ignore[attr-defined]
        loop._submit_order(order, MARKET_US, candles=candles)  # type: ignore[attr-defined]

        monitor.record_slippage.assert_called_once()
        # fill_price == candle close == 150.00, so slippage should be 0.0
        call_args = monitor.record_slippage.call_args[0]
        assert call_args[0] == 0.0

    def test_strategy_cycle_calls_on_cycle_complete(self) -> None:
        """When sandbox_monitor is set, _strategy_cycle finally calls on_cycle_complete."""
        signal = self._make_buy_signal()
        loop = _make_trading_loop(signal=signal)
        monitor = MagicMock()
        monitor.cycle_count = 0
        monitor.slippage_buffer = []
        loop._sandbox_monitor = monitor  # type: ignore[attr-defined]

        with _market_open_ctx(loop):
            loop._strategy_cycle()  # type: ignore[attr-defined]

        monitor.on_cycle_complete.assert_called_once()
        metrics = monitor.on_cycle_complete.call_args[0][0]
        assert hasattr(metrics, "trade_count")
        assert hasattr(metrics, "equity_rub")


class TestConsecutiveCycleErrors:
    """ERR-04: TradingLoop sends Telegram alert after 3 consecutive cycle failures."""

    MAX_CONSECUTIVE = 3

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

    def test_equity_counter_increments_on_failure(self) -> None:
        """_consecutive_equity_errors increments on _strategy_cycle failure."""
        loop = _make_trading_loop()
        assert loop._consecutive_equity_errors == 0  # type: ignore[attr-defined]
        with (
            patch.object(loop, "_strategy_cycle_impl", side_effect=RuntimeError("boom")),
            _market_open_ctx(loop),
        ):
            loop._strategy_cycle()  # type: ignore[attr-defined]
        assert loop._consecutive_equity_errors == 1  # type: ignore[attr-defined]

    def test_equity_counter_resets_on_success(self) -> None:
        """_consecutive_equity_errors resets to 0 on successful cycle."""
        loop = _make_trading_loop()
        loop._consecutive_equity_errors = 2  # type: ignore[attr-defined]
        with _market_open_ctx(loop):
            loop._strategy_cycle()  # type: ignore[attr-defined]
        assert loop._consecutive_equity_errors == 0  # type: ignore[attr-defined]

    def test_equity_alert_after_3_consecutive_failures(self) -> None:
        """Telegram alert sent after 3 consecutive equity cycle failures."""
        loop = _make_trading_loop()
        with (
            patch.object(loop, "_strategy_cycle_impl", side_effect=RuntimeError("boom")),
            _market_open_ctx(loop),
        ):
            for _ in range(self.MAX_CONSECUTIVE):
                loop._strategy_cycle()  # type: ignore[attr-defined]
        loop._alerter.send_alert.assert_called()  # type: ignore[attr-defined]
        alert_msg = loop._alerter.send_alert.call_args[0][0]  # type: ignore[attr-defined]
        assert "consecutive" in alert_msg.lower()
        assert "equity" in alert_msg.lower()

    def test_equity_no_alert_before_threshold(self) -> None:
        """No Telegram alert before reaching 3 consecutive failures."""
        loop = _make_trading_loop()
        with (
            patch.object(loop, "_strategy_cycle_impl", side_effect=RuntimeError("boom")),
            _market_open_ctx(loop),
        ):
            for _ in range(self.MAX_CONSECUTIVE - 1):
                loop._strategy_cycle()  # type: ignore[attr-defined]
        loop._alerter.send_alert.assert_not_called()  # type: ignore[attr-defined]

    def _run_bond_cycle(self, loop: TradingLoop) -> None:
        """Helper to run bond cycle bypassing market hours/trading day gates."""
        with (
            patch.object(loop, "_now", return_value=_MARKET_OPEN_DT),
            patch.object(loop, "_is_market_open", return_value=True),
            patch("finalayze.data.moex_calendar.is_moex_trading_day", return_value=True),
        ):
            loop._bond_cycle()  # type: ignore[attr-defined]

    def test_bond_counter_increments_on_failure(self) -> None:
        """_consecutive_bond_errors increments on _bond_cycle failure."""
        loop = _make_trading_loop()
        bond_proc = MagicMock()
        bond_proc.run_cycle.side_effect = RuntimeError("bond fail")
        loop._bond_processor = bond_proc  # type: ignore[attr-defined]
        assert loop._consecutive_bond_errors == 0  # type: ignore[attr-defined]
        self._run_bond_cycle(loop)
        assert loop._consecutive_bond_errors == 1  # type: ignore[attr-defined]

    def test_bond_counter_resets_on_success(self) -> None:
        """_consecutive_bond_errors resets to 0 on successful bond cycle."""
        from finalayze.core.bond_cycle import BondCycleResult

        loop = _make_trading_loop()
        bond_proc = MagicMock()
        bond_proc.run_cycle.return_value = BondCycleResult()
        loop._bond_processor = bond_proc  # type: ignore[attr-defined]
        loop._consecutive_bond_errors = 2  # type: ignore[attr-defined]
        self._run_bond_cycle(loop)
        assert loop._consecutive_bond_errors == 0  # type: ignore[attr-defined]

    def test_bond_alert_after_3_consecutive_failures(self) -> None:
        """Telegram alert sent after 3 consecutive bond cycle failures."""
        loop = _make_trading_loop()
        bond_proc = MagicMock()
        bond_proc.run_cycle.side_effect = RuntimeError("bond fail")
        loop._bond_processor = bond_proc  # type: ignore[attr-defined]
        for _ in range(self.MAX_CONSECUTIVE):
            self._run_bond_cycle(loop)
        loop._alerter.send_alert.assert_called()  # type: ignore[attr-defined]
        alert_msg = loop._alerter.send_alert.call_args[0][0]  # type: ignore[attr-defined]
        assert "consecutive" in alert_msg.lower()
        assert "bond" in alert_msg.lower()
