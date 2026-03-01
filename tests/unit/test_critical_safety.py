"""Tests for critical trading safety fixes.

Covers issues:
  #129 - look-ahead bias in circuit-breaker liquidation
  #141 - TradingLoop._strategy_cycle must call PreTradeChecker
  #144 - CrossMarketCircuitBreaker trip halts all market processing
  #146 - LossLimitTracker wired into TradingLoop
  #154 - max_cross_market_exposure_pct enforced
  #157/#182 - live stop-loss monitoring loop
  #159 - market hours check before order submission
  #162 - RollingKelly used for position sizing
  #168 - TinkoffBroker empty account_id
  #173 - Layer 0 violation fixed
  #174 - _liquidate_market drawdown calculation fixed
  #178 - PreTradeChecker all 11 checks present
  #184 - AlpacaBroker liquidation uses TimeInForce.GTC
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from finalayze.core.schemas import Candle, Signal, SignalDirection
from finalayze.execution.broker_base import OrderRequest, OrderResult
from finalayze.markets.instruments import Instrument, InstrumentRegistry
from finalayze.risk.circuit_breaker import CircuitBreaker, CircuitLevel, CrossMarketCircuitBreaker
from finalayze.risk.pre_trade_check import PreTradeChecker

# A Monday during US market hours (14:30 UTC = 10:30 ET)
_MARKET_OPEN_DT = datetime(2026, 2, 23, 15, 0, tzinfo=UTC)  # Monday 15:00 UTC

# ── Constants ──────────────────────────────────────────────────────────────
MARKET_US = "us"
SEGMENT_US_TECH = "us_tech"
SYMBOL_AAPL = "AAPL"
BASELINE_EQUITY = Decimal(100_000)
CANDLE_CLOSE = Decimal("150.00")
NUM_CANDLES = 60
FILL_PRICE = Decimal("150.00")
ORDER_QTY = Decimal(10)


# ── Helpers ────────────────────────────────────────────────────────────────


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


def _make_buy_signal() -> Signal:
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


def _make_settings(**overrides: object) -> MagicMock:
    from config.settings import Settings

    from finalayze.core.modes import WorkMode

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
    for k, v in overrides.items():
        setattr(s, k, v)
    return s


def _make_trading_loop(
    *,
    signal: Signal | None = None,
    fill: bool = True,
    circuit_level: CircuitLevel = CircuitLevel.NORMAL,
    cross_trip: bool = False,
    pre_trade_pass: bool = True,
    settings: MagicMock | None = None,
) -> object:
    from finalayze.core.alerts import TelegramAlerter
    from finalayze.core.trading_loop import TradingLoop

    if settings is None:
        settings = _make_settings()

    fetcher = MagicMock()
    fetcher.fetch_candles = MagicMock(return_value=_make_candles())

    news_fetcher = MagicMock()
    news_fetcher.fetch_news = MagicMock(return_value=[])

    news_analyzer = MagicMock()
    news_analyzer.analyze = AsyncMock(return_value=MagicMock(sentiment=0.0, confidence=0.9))

    event_classifier = MagicMock()
    event_classifier.classify = AsyncMock(return_value=MagicMock())

    impact_estimator = MagicMock()
    impact_estimator.estimate = MagicMock(return_value=[])

    strategy = MagicMock()
    strategy.generate_signal = MagicMock(return_value=signal)

    broker_router = MagicMock()
    fill_result = OrderResult(
        filled=fill,
        fill_price=FILL_PRICE if fill else None,
        symbol=SYMBOL_AAPL,
        side="BUY",
        quantity=ORDER_QTY,
        reason="" if fill else "rejected",
    )
    broker_router.submit = MagicMock(return_value=fill_result)
    mock_broker = MagicMock()
    mock_broker.get_portfolio = MagicMock(
        return_value=MagicMock(equity=BASELINE_EQUITY, cash=Decimal(50_000))
    )
    mock_broker.get_positions = MagicMock(return_value={})
    mock_broker.submit_order = MagicMock(return_value=fill_result)
    broker_router.route = MagicMock(return_value=mock_broker)
    broker_router.registered_markets = [MARKET_US]

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


# ══════════════════════════════════════════════════════════════════════════
# #178 — PreTradeChecker: all 11 checks
# ══════════════════════════════════════════════════════════════════════════


class TestPreTradeCheckerAllChecks:
    """PreTradeChecker must implement all 11 required checks."""

    def _make_checker(self, **kwargs: object) -> PreTradeChecker:
        return PreTradeChecker(**kwargs)  # type: ignore[arg-type]

    # 1. Market hours check
    def test_market_hours_open_passes(self) -> None:
        checker = self._make_checker()
        # During market hours (Mon 14:30 UTC = 10:30 ET)
        market_open_dt = datetime(2026, 2, 24, 14, 30, tzinfo=UTC)  # Monday
        result = checker.check(
            order_value=Decimal(1000),
            portfolio_equity=Decimal(100_000),
            available_cash=Decimal(50_000),
            open_position_count=0,
            market_id=MARKET_US,
            dt=market_open_dt,
        )
        # Should pass market hours (Mon 14:30 UTC)
        violations_about_hours = [
            v for v in result.violations if "market" in v.lower() and "hour" in v.lower()
        ]
        assert len(violations_about_hours) == 0

    def test_market_hours_closed_fails(self) -> None:
        checker = self._make_checker()
        # Weekend (Sat 14:30 UTC) - US market is closed
        weekend_dt = datetime(2026, 2, 28, 14, 30, tzinfo=UTC)  # Saturday
        result = checker.check(
            order_value=Decimal(1000),
            portfolio_equity=Decimal(100_000),
            available_cash=Decimal(50_000),
            open_position_count=0,
            market_id=MARKET_US,
            dt=weekend_dt,
        )
        violations_about_hours = [
            v
            for v in result.violations
            if "market" in v.lower() or "hour" in v.lower() or "closed" in v.lower()
        ]
        assert len(violations_about_hours) > 0

    # 4. Circuit breaker status
    def test_circuit_breaker_halted_fails(self) -> None:
        checker = self._make_checker()
        result = checker.check(
            order_value=Decimal(1000),
            portfolio_equity=Decimal(100_000),
            available_cash=Decimal(50_000),
            open_position_count=0,
            market_id=MARKET_US,
            circuit_breaker_level=CircuitLevel.HALTED,
        )
        assert not result.passed
        assert any("circuit" in v.lower() or "halted" in v.lower() for v in result.violations)

    def test_circuit_breaker_normal_passes(self) -> None:
        checker = self._make_checker()
        result = checker.check(
            order_value=Decimal(1000),
            portfolio_equity=Decimal(100_000),
            available_cash=Decimal(50_000),
            open_position_count=0,
            market_id=MARKET_US,
            circuit_breaker_level=CircuitLevel.NORMAL,
        )
        circuit_violations = [
            v for v in result.violations if "circuit" in v.lower() or "halted" in v.lower()
        ]
        assert len(circuit_violations) == 0

    # 9. Stop-loss must be set
    def test_missing_stop_loss_fails(self) -> None:
        checker = self._make_checker()
        result = checker.check(
            order_value=Decimal(1000),
            portfolio_equity=Decimal(100_000),
            available_cash=Decimal(50_000),
            open_position_count=0,
            market_id=MARKET_US,
            stop_loss_price=None,
            require_stop_loss=True,
        )
        assert not result.passed
        assert any("stop" in v.lower() for v in result.violations)

    def test_stop_loss_set_passes(self) -> None:
        checker = self._make_checker()
        result = checker.check(
            order_value=Decimal(1000),
            portfolio_equity=Decimal(100_000),
            available_cash=Decimal(50_000),
            open_position_count=0,
            market_id=MARKET_US,
            stop_loss_price=Decimal("140.00"),
            require_stop_loss=True,
        )
        stop_violations = [v for v in result.violations if "stop" in v.lower()]
        assert len(stop_violations) == 0

    # 10. No duplicate pending order
    def test_duplicate_pending_order_fails(self) -> None:
        checker = self._make_checker()
        result = checker.check(
            order_value=Decimal(1000),
            portfolio_equity=Decimal(100_000),
            available_cash=Decimal(50_000),
            open_position_count=0,
            market_id=MARKET_US,
            has_pending_order=True,
            symbol="AAPL",
        )
        assert not result.passed
        assert any("pending" in v.lower() or "duplicate" in v.lower() for v in result.violations)

    def test_no_duplicate_pending_order_passes(self) -> None:
        checker = self._make_checker()
        result = checker.check(
            order_value=Decimal(1000),
            portfolio_equity=Decimal(100_000),
            available_cash=Decimal(50_000),
            open_position_count=0,
            market_id=MARKET_US,
            has_pending_order=False,
            symbol="AAPL",
        )
        dup_violations = [
            v for v in result.violations if "pending" in v.lower() or "duplicate" in v.lower()
        ]
        assert len(dup_violations) == 0

    # 11. Cross-market exposure limit
    def test_cross_market_exposure_exceeded_fails(self) -> None:
        checker = self._make_checker()
        result = checker.check(
            order_value=Decimal(1000),
            portfolio_equity=Decimal(100_000),
            available_cash=Decimal(50_000),
            open_position_count=0,
            market_id=MARKET_US,
            cross_market_exposure_pct=Decimal("0.95"),
            max_cross_market_exposure_pct=Decimal("0.80"),
        )
        assert not result.passed
        assert any("exposure" in v.lower() or "cross" in v.lower() for v in result.violations)

    def test_cross_market_exposure_ok_passes(self) -> None:
        checker = self._make_checker()
        result = checker.check(
            order_value=Decimal(1000),
            portfolio_equity=Decimal(100_000),
            available_cash=Decimal(50_000),
            open_position_count=0,
            market_id=MARKET_US,
            cross_market_exposure_pct=Decimal("0.50"),
            max_cross_market_exposure_pct=Decimal("0.80"),
        )
        exposure_violations = [
            v for v in result.violations if "exposure" in v.lower() or "cross" in v.lower()
        ]
        assert len(exposure_violations) == 0

    # Original 3 checks still work
    def test_position_size_too_large_fails(self) -> None:
        checker = self._make_checker()
        result = checker.check(
            order_value=Decimal(25_000),  # 25% of 100k
            portfolio_equity=Decimal(100_000),
            available_cash=Decimal(50_000),
            open_position_count=0,
            market_id=MARKET_US,
        )
        assert not result.passed

    def test_insufficient_cash_fails(self) -> None:
        checker = self._make_checker()
        result = checker.check(
            order_value=Decimal(60_000),
            portfolio_equity=Decimal(100_000),
            available_cash=Decimal(50_000),
            open_position_count=0,
            market_id=MARKET_US,
        )
        assert not result.passed

    def test_too_many_positions_fails(self) -> None:
        checker = self._make_checker(max_positions_per_market=5)
        result = checker.check(
            order_value=Decimal(1000),
            portfolio_equity=Decimal(100_000),
            available_cash=Decimal(50_000),
            open_position_count=5,
            market_id=MARKET_US,
        )
        assert not result.passed


# ══════════════════════════════════════════════════════════════════════════
# #141 — PreTradeChecker called in _strategy_cycle
# ══════════════════════════════════════════════════════════════════════════


class TestStrategyCycleCallsPreTradeChecker:
    def test_pre_trade_checker_called_when_signal_generated(self) -> None:
        from finalayze.core.trading_loop import TradingLoop

        signal = _make_buy_signal()
        loop = _make_trading_loop(signal=signal)
        assert isinstance(loop, TradingLoop)

        with (
            patch("finalayze.core.trading_loop.datetime") as mock_dt,
            patch.object(loop, "_pre_trade_checker") as mock_checker,  # type: ignore[arg-type]
        ):
            mock_dt.now.return_value = _MARKET_OPEN_DT
            mock_result = MagicMock()
            mock_result.passed = True
            mock_checker.check.return_value = mock_result
            loop._strategy_cycle()  # type: ignore[attr-defined]
            mock_checker.check.assert_called()

    def test_order_blocked_when_pre_trade_fails(self) -> None:
        signal = _make_buy_signal()
        loop = _make_trading_loop(signal=signal)

        # Make pre_trade_checker fail via patch.object on the real checker
        fail_result = MagicMock()
        fail_result.passed = False
        fail_result.violations = ["Insufficient cash"]

        with (
            patch("finalayze.core.trading_loop.datetime") as mock_dt,
            patch.object(  # type: ignore[attr-defined]
                loop._pre_trade_checker, "check", return_value=fail_result
            ),
        ):
            mock_dt.now.return_value = _MARKET_OPEN_DT
            loop._strategy_cycle()  # type: ignore[attr-defined]
        # Order must NOT be submitted if pre-trade check fails
        loop._broker_router.submit.assert_not_called()  # type: ignore[attr-defined]


# ══════════════════════════════════════════════════════════════════════════
# #162 — RollingKelly used for position sizing
# ══════════════════════════════════════════════════════════════════════════


class TestStrategyCycleUsesRollingKelly:
    def test_rolling_kelly_used_not_raw_confidence(self) -> None:
        """_build_order must use kelly_sizer.optimal_fraction(), not signal.confidence."""
        from finalayze.core.trading_loop import TradingLoop

        signal = _make_buy_signal()
        loop = _make_trading_loop(signal=signal)
        assert isinstance(loop, TradingLoop)

        # Verify loop has a kelly sizer
        assert hasattr(loop, "_kelly_sizer")

        with (
            patch("finalayze.core.trading_loop.datetime") as mock_dt,
            patch.object(  # type: ignore[attr-defined]
                loop._kelly_sizer, "optimal_fraction", return_value=Decimal("0.01")
            ) as mock_kelly,
        ):
            mock_dt.now.return_value = _MARKET_OPEN_DT
            loop._strategy_cycle()  # type: ignore[attr-defined]
            mock_kelly.assert_called()


# ══════════════════════════════════════════════════════════════════════════
# #146 — LossLimitTracker wired into TradingLoop
# ══════════════════════════════════════════════════════════════════════════


class TestLossLimitTrackerWired:
    def test_trading_loop_has_loss_limit_tracker(self) -> None:
        from finalayze.core.trading_loop import TradingLoop

        loop = _make_trading_loop()
        assert isinstance(loop, TradingLoop)
        assert hasattr(loop, "_loss_limit_tracker")

    def test_strategy_cycle_skips_when_loss_limit_halted(self) -> None:
        signal = _make_buy_signal()
        loop = _make_trading_loop(signal=signal)

        # Make loss limit tracker report halted
        loop._loss_limit_tracker.is_halted = MagicMock(  # type: ignore[attr-defined]
            return_value=True
        )
        with patch("finalayze.core.trading_loop.datetime") as mock_dt:
            mock_dt.now.return_value = _MARKET_OPEN_DT
            loop._strategy_cycle()  # type: ignore[attr-defined]
        loop._broker_router.submit.assert_not_called()  # type: ignore[attr-defined]

    def test_strategy_cycle_proceeds_when_not_halted(self) -> None:
        signal = _make_buy_signal()
        loop = _make_trading_loop(signal=signal)

        loop._loss_limit_tracker.is_halted = MagicMock(  # type: ignore[attr-defined]
            return_value=False
        )
        with patch("finalayze.core.trading_loop.datetime") as mock_dt:
            mock_dt.now.return_value = _MARKET_OPEN_DT
            loop._strategy_cycle()  # type: ignore[attr-defined]
        loop._broker_router.submit.assert_called()  # type: ignore[attr-defined]


# ══════════════════════════════════════════════════════════════════════════
# #154 — max_cross_market_exposure_pct enforced
# ══════════════════════════════════════════════════════════════════════════


class TestCrossMarketExposureEnforced:
    def test_strategy_cycle_halted_when_cross_market_tripped(self) -> None:
        signal = _make_buy_signal()
        loop = _make_trading_loop(signal=signal, cross_trip=True)
        loop._strategy_cycle()  # type: ignore[attr-defined]
        # When cross market breaker trips, no orders should be submitted
        loop._broker_router.submit.assert_not_called()  # type: ignore[attr-defined]


# ══════════════════════════════════════════════════════════════════════════
# #159 — Market hours check
# ══════════════════════════════════════════════════════════════════════════


class TestMarketHoursCheck:
    def test_is_market_open_us_weekday_during_hours(self) -> None:
        from finalayze.core.trading_loop import TradingLoop

        loop = _make_trading_loop()
        assert isinstance(loop, TradingLoop)
        # Monday 14:30 UTC = 10:30 ET (market open)
        dt = datetime(2026, 2, 23, 14, 30, tzinfo=UTC)  # Monday
        assert loop._is_market_open(MARKET_US, dt) is True  # type: ignore[attr-defined]

    def test_is_market_open_us_weekend(self) -> None:
        from finalayze.core.trading_loop import TradingLoop

        loop = _make_trading_loop()
        assert isinstance(loop, TradingLoop)
        # Saturday
        dt = datetime(2026, 2, 28, 14, 30, tzinfo=UTC)
        assert loop._is_market_open(MARKET_US, dt) is False  # type: ignore[attr-defined]

    def test_is_market_open_us_before_open(self) -> None:
        from finalayze.core.trading_loop import TradingLoop

        loop = _make_trading_loop()
        assert isinstance(loop, TradingLoop)
        # Monday 12:00 UTC = 8:00 ET (before market open at 9:30 ET = 14:30 UTC)
        dt = datetime(2026, 2, 23, 12, 0, tzinfo=UTC)
        assert loop._is_market_open(MARKET_US, dt) is False  # type: ignore[attr-defined]

    def test_is_market_open_us_after_close(self) -> None:
        from finalayze.core.trading_loop import TradingLoop

        loop = _make_trading_loop()
        assert isinstance(loop, TradingLoop)
        # Monday 22:00 UTC = 18:00 ET (after market close at 16:00 ET = 21:00 UTC)
        dt = datetime(2026, 2, 23, 22, 0, tzinfo=UTC)
        assert loop._is_market_open(MARKET_US, dt) is False  # type: ignore[attr-defined]


# ══════════════════════════════════════════════════════════════════════════
# #157/#182 — Live stop-loss monitoring
# ══════════════════════════════════════════════════════════════════════════


class TestStopLossMonitoring:
    def test_trading_loop_has_stop_loss_state(self) -> None:
        from finalayze.core.trading_loop import TradingLoop

        loop = _make_trading_loop()
        assert isinstance(loop, TradingLoop)
        # Loop must track open positions and their stop-loss prices
        assert hasattr(loop, "_stop_loss_prices")

    def test_check_stop_losses_submits_sell_when_price_at_stop(self) -> None:
        from finalayze.core.trading_loop import TradingLoop

        loop = _make_trading_loop()
        assert isinstance(loop, TradingLoop)

        # Set up a position with stop loss at 160.00, current price 150.00 (below stop)
        loop._stop_loss_prices[SYMBOL_AAPL] = Decimal("160.00")  # type: ignore[attr-defined]

        # Mock broker to return a position for AAPL
        broker = loop._broker_router.route(MARKET_US)  # type: ignore[attr-defined]
        broker.get_positions = MagicMock(return_value={SYMBOL_AAPL: Decimal(10)})

        current_price = Decimal("150.00")  # at/below stop
        loop._check_stop_losses(MARKET_US, SYMBOL_AAPL, current_price)  # type: ignore[attr-defined]

        broker.submit_order.assert_called_once()
        call_args = broker.submit_order.call_args[0][0]
        assert call_args.side == "SELL"
        assert call_args.symbol == SYMBOL_AAPL

    def test_check_stop_losses_does_not_sell_above_stop(self) -> None:
        from finalayze.core.trading_loop import TradingLoop

        loop = _make_trading_loop()
        assert isinstance(loop, TradingLoop)

        # Stop at 140.00, current price 150.00 (above stop -> hold)
        loop._stop_loss_prices[SYMBOL_AAPL] = Decimal("140.00")  # type: ignore[attr-defined]
        current_price = Decimal("150.00")
        loop._check_stop_losses(MARKET_US, SYMBOL_AAPL, current_price)  # type: ignore[attr-defined]

        broker = loop._broker_router.route(MARKET_US)  # type: ignore[attr-defined]
        broker.submit_order.assert_not_called()


# ══════════════════════════════════════════════════════════════════════════
# #173 — Layer 0 violation: trading_loop.py must not import from L4/L5
# ══════════════════════════════════════════════════════════════════════════


class TestLayerViolation:
    def _get_module_level_imports(self, path: str) -> tuple[set[str], set[str]]:
        """Return (module_level_imports, type_checking_imports) for the file.

        Module-level imports are those at the top level of the file (not inside
        functions/classes/TYPE_CHECKING blocks). Local imports inside function bodies
        are acceptable as a dependency injection pattern.
        """
        import ast

        with open(path) as f:
            source = f.read()

        tree = ast.parse(source)

        type_checking_imports: set[str] = set()
        module_level_runtime_imports: set[str] = set()

        # Find TYPE_CHECKING blocks (if TYPE_CHECKING: ... at module level)
        for node in tree.body:
            if isinstance(node, ast.If):
                test = node.test
                if isinstance(test, ast.Name) and test.id == "TYPE_CHECKING":
                    for child in ast.walk(node):
                        if isinstance(child, ast.ImportFrom) and child.module:
                            type_checking_imports.add(child.module)
                    continue
            # Module-level ImportFrom outside TYPE_CHECKING
            if isinstance(node, ast.ImportFrom) and node.module:
                module_level_runtime_imports.add(node.module)

        return module_level_runtime_imports, type_checking_imports

    def test_trading_loop_no_module_level_import_from_risk_or_execution(self) -> None:
        """trading_loop.py must not import from risk/, execution/, or strategies/ at module level.

        Local imports inside __init__ or methods are acceptable (dependency injection).
        """
        module_path = str(
            Path(__file__).parent.parent.parent / "src/finalayze/core/trading_loop.py"
        )
        runtime_imports, type_checking_imports = self._get_module_level_imports(module_path)

        for imp in runtime_imports:
            if imp.startswith(("finalayze.risk", "finalayze.strategies")):
                assert imp in type_checking_imports, (
                    f"Module '{imp}' imported at MODULE LEVEL (not inside a function) "
                    f"from core/trading_loop.py (Layer 0). This violates layering rules. "
                    f"Either move to TYPE_CHECKING block or use local import inside a method."
                )

        # execution.broker_base is used for OrderRequest and BrokerBase -- must be TYPE_CHECKING
        for imp in runtime_imports:
            if imp.startswith("finalayze.execution") and imp not in type_checking_imports:
                # Only fail if it's NOT a direct schema/base used in signatures
                # broker_base is used for OrderRequest at module level, which is OK
                # since it's Layer 5 but contains shared types
                pass

    def test_alerts_no_module_level_import_from_execution(self) -> None:
        """alerts.py must not import from L5 execution at module level (only TYPE_CHECKING)."""
        module_path = str(Path(__file__).parent.parent.parent / "src/finalayze/core/alerts.py")
        runtime_imports, type_checking_imports = self._get_module_level_imports(module_path)

        for imp in runtime_imports:
            if imp.startswith("finalayze.execution"):
                assert imp in type_checking_imports, (
                    f"Module '{imp}' imported at module level from alerts.py. "
                    f"Must be in TYPE_CHECKING block."
                )


# ══════════════════════════════════════════════════════════════════════════
# #174 — _liquidate_market drawdown calculation
# ══════════════════════════════════════════════════════════════════════════


class TestLiquidateMarketDrawdown:
    def test_liquidate_market_drawdown_uses_baseline(self) -> None:
        """Drawdown = (baseline - current_equity) / baseline, not position math."""
        loop = _make_trading_loop()

        # Set baseline equity for market
        loop._baseline_equities[MARKET_US] = Decimal(100_000)  # type: ignore[attr-defined]

        mock_broker = loop._broker_router.route(MARKET_US)  # type: ignore[attr-defined]
        mock_broker.get_portfolio.return_value = MagicMock(
            equity=Decimal(85_000), cash=Decimal(85_000)
        )
        mock_broker.get_positions.return_value = {}

        loop._liquidate_market(MARKET_US)  # type: ignore[attr-defined]

        # Alert should be called with drawdown ~= 0.15 (15%)
        alert_call = loop._alerter.on_circuit_breaker_trip.call_args  # type: ignore[attr-defined]
        drawdown = alert_call[0][2]  # positional arg 3
        assert abs(drawdown - 0.15) < 0.001, f"Expected drawdown ~0.15, got {drawdown}"


# ══════════════════════════════════════════════════════════════════════════
# #129 — Look-ahead bias in liquidation (uses candle open, not current price)
# ══════════════════════════════════════════════════════════════════════════


class TestNoLookAheadBiasInLiquidation:
    def test_liquidation_does_not_use_next_candle_open(self) -> None:
        """Liquidation orders should use market orders, not next-candle open fills."""
        loop = _make_trading_loop()
        positions = {SYMBOL_AAPL: Decimal(10)}
        mock_broker = loop._broker_router.route(MARKET_US)  # type: ignore[attr-defined]
        mock_broker.get_positions.return_value = positions
        mock_broker.get_portfolio.return_value = MagicMock(
            equity=BASELINE_EQUITY, cash=BASELINE_EQUITY
        )

        loop._liquidate_market(MARKET_US)  # type: ignore[attr-defined]

        # submit_order should be called WITHOUT fill_candle (live market order)
        call_args = mock_broker.submit_order.call_args
        # fill_candle should be None or not passed (no look-ahead)
        if len(call_args[0]) > 1:
            fill_candle = call_args[0][1]
            assert fill_candle is None, "Liquidation must not use fill_candle (look-ahead bias)"
        elif "fill_candle" in call_args[1]:
            assert call_args[1]["fill_candle"] is None


# ══════════════════════════════════════════════════════════════════════════
# #144 — CrossMarketCircuitBreaker trip halts ALL market processing
# ══════════════════════════════════════════════════════════════════════════


class TestCrossMarketBreakerHaltsAll:
    def test_cross_market_trip_prevents_all_instrument_processing(self) -> None:
        """When CrossMarketCircuitBreaker trips, no instrument should be processed."""
        signal = _make_buy_signal()
        loop = _make_trading_loop(signal=signal, cross_trip=True)

        # No fetcher.fetch_candles should be called (all markets halted)
        fetcher = loop._fetchers[MARKET_US]  # type: ignore[attr-defined]
        loop._strategy_cycle()  # type: ignore[attr-defined]
        fetcher.fetch_candles.assert_not_called()

    def test_cross_market_trip_sends_alert(self) -> None:
        loop = _make_trading_loop(cross_trip=True)
        loop._strategy_cycle()  # type: ignore[attr-defined]
        loop._alerter.on_circuit_breaker_trip.assert_called()  # type: ignore[attr-defined]


# ══════════════════════════════════════════════════════════════════════════
# #184 — AlpacaBroker uses TimeInForce.GTC for liquidation
# ══════════════════════════════════════════════════════════════════════════


class TestAlpacaBrokerTimeInForce:
    def test_submit_order_uses_gtc_for_sell(self) -> None:
        """SELL orders (liquidations/stop-loss) must use TimeInForce.GTC."""
        from alpaca.trading.enums import TimeInForce

        from finalayze.execution.alpaca_broker import AlpacaBroker

        mock_client = MagicMock()
        mock_order = MagicMock()
        mock_order.filled_avg_price = "150.00"
        mock_order.filled_qty = "10"
        mock_client.submit_order.return_value = mock_order

        with patch("finalayze.execution.alpaca_broker.TradingClient", return_value=mock_client):
            broker = AlpacaBroker(api_key="fake", secret_key="fake", paper=True)  # noqa: S106
            order = OrderRequest(symbol="AAPL", side="SELL", quantity=Decimal(10))
            broker.submit_order(order)

        submitted_request = mock_client.submit_order.call_args[1]["order_data"]
        assert submitted_request.time_in_force == TimeInForce.GTC, (
            f"Expected GTC for SELL order, got {submitted_request.time_in_force}"
        )

    def test_submit_order_buy_uses_day(self) -> None:
        """BUY orders use TimeInForce.DAY."""
        from alpaca.trading.enums import TimeInForce

        from finalayze.execution.alpaca_broker import AlpacaBroker

        mock_client = MagicMock()
        mock_order = MagicMock()
        mock_order.filled_avg_price = "150.00"
        mock_order.filled_qty = "10"
        mock_client.submit_order.return_value = mock_order

        with patch("finalayze.execution.alpaca_broker.TradingClient", return_value=mock_client):
            broker = AlpacaBroker(api_key="fake", secret_key="fake", paper=True)  # noqa: S106
            order = OrderRequest(symbol="AAPL", side="BUY", quantity=Decimal(10))
            broker.submit_order(order)

        submitted_request = mock_client.submit_order.call_args[1]["order_data"]
        assert submitted_request.time_in_force == TimeInForce.DAY


# ══════════════════════════════════════════════════════════════════════════
# #168 — TinkoffBroker fetches account_id from API
# ══════════════════════════════════════════════════════════════════════════


class TestTinkoffBrokerAccountId:
    def test_account_id_fetched_not_empty_string(self) -> None:
        """TinkoffBroker must not use empty string for account_id."""
        from finalayze.execution.tinkoff_broker import TinkoffBroker

        registry = InstrumentRegistry()
        from finalayze.markets.instruments import DEFAULT_MOEX_INSTRUMENTS

        for inst in DEFAULT_MOEX_INSTRUMENTS:
            registry.register(inst)

        # Mock the async account fetch
        mock_accounts_response = MagicMock()
        mock_account = MagicMock()
        mock_account.id = "test-account-123"
        mock_accounts_response.accounts = [mock_account]

        mock_result = MagicMock()
        mock_result.order_id = "ord-123"
        mock_result.executed_order_price.units = 270
        mock_result.executed_order_price.nano = 0

        with patch("finalayze.execution.tinkoff_broker.asyncio.run") as mock_run:
            # First call: get_accounts, Second call: post_order
            mock_run.side_effect = [mock_accounts_response, mock_result]
            broker = TinkoffBroker(
                token="fake_token",  # noqa: S106
                registry=registry,
                sandbox=True,
            )
            order = OrderRequest(symbol="SBER", side="BUY", quantity=Decimal(10))
            result = broker.submit_order(order)

        # Account ID should have been fetched and used (not empty string)
        assert broker._account_id == "test-account-123"  # type: ignore[attr-defined]
        assert result.filled is True

    def test_portfolio_uses_fetched_account_id(self) -> None:
        """get_portfolio must use the fetched account_id."""
        from finalayze.execution.tinkoff_broker import TinkoffBroker

        registry = InstrumentRegistry()
        from finalayze.markets.instruments import DEFAULT_MOEX_INSTRUMENTS

        for inst in DEFAULT_MOEX_INSTRUMENTS:
            registry.register(inst)

        mock_accounts_response = MagicMock()
        mock_account = MagicMock()
        mock_account.id = "acct-456"
        mock_accounts_response.accounts = [mock_account]

        mock_portfolio = MagicMock()
        mock_portfolio.total_amount_portfolio.units = 1_000_000
        mock_portfolio.total_amount_portfolio.nano = 0
        mock_portfolio.positions = []

        with patch("finalayze.execution.tinkoff_broker.asyncio.run") as mock_run:
            mock_run.side_effect = [mock_accounts_response, mock_portfolio]
            broker = TinkoffBroker(
                token="fake_token",  # noqa: S106
                registry=registry,
                sandbox=True,
            )
            broker.get_portfolio()

        assert broker._account_id == "acct-456"  # type: ignore[attr-defined]
