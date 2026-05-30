"""Test segment position cap enforcement in BacktestEngine."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import MagicMock

from finalayze.backtest.config import BacktestConfig
from finalayze.backtest.engine import BacktestEngine
from finalayze.core.schemas import Candle, Signal, SignalDirection
from finalayze.risk.pre_trade_check import PreTradeResult


def _make_candle(
    ts: datetime | None = None,
    open_: float = 100.0,
    high: float = 105.0,
    low: float = 95.0,
    close: float = 102.0,
    volume: int = 1000,
) -> Candle:
    return Candle(
        symbol="TEST",
        market_id="us",
        timeframe="1d",
        timestamp=ts or datetime(2024, 6, 3, 14, 30, tzinfo=UTC),
        open=Decimal(str(open_)),
        high=Decimal(str(high)),
        low=Decimal(str(low)),
        close=Decimal(str(close)),
        volume=volume,
    )


def _make_signal() -> Signal:
    return Signal(
        direction=SignalDirection.BUY,
        confidence=0.8,
        strategy_name="momentum",
        symbol="TEST",
        market_id="us",
        segment_id="us_tech",
        strategy_payload={},
        reasoning="test",
    )


class TestSegmentPositionCap:
    """Verify engine enforces max_positions_per_segment."""

    def test_segment_cap_blocks_when_at_limit(self) -> None:
        """Engine skips BUY when segment position count >= max_positions_per_segment."""
        config = BacktestConfig(
            initial_cash=Decimal(100000),
            max_positions_per_segment=2,
        )
        strategy = MagicMock()
        engine = BacktestEngine(strategy=strategy, config=config)
        engine._sizing_pipeline = MagicMock()
        engine._sizing_pipeline.compute.return_value = Decimal(5000)

        broker = MagicMock()
        broker.has_position.return_value = False
        portfolio = MagicMock()
        portfolio.equity = Decimal(100000)
        portfolio.cash = Decimal(50000)
        portfolio.positions = {}
        broker.get_portfolio.return_value = portfolio
        # 2 existing positions = at cap
        broker.get_positions.return_value = {"AAPL": Decimal(10), "MSFT": Decimal(10)}

        checker = MagicMock()
        checker.check.return_value = MagicMock(passed=True, violations=[])

        engine._handle_buy(
            broker=broker,
            checker=checker,
            fill_candle=_make_candle(),
            symbol="GOOG",
            history=[_make_candle() for _ in range(20)],
            entry_prices={},
            segment_id="us_tech",
            signal=_make_signal(),
            entry_bars={},
            bar_index=5,
        )

        # Checker should NOT have been called -- cap blocks before pre-trade check
        checker.check.assert_not_called()

    def test_segment_cap_allows_when_below_limit(self) -> None:
        """Engine proceeds with BUY when segment position count < max_positions_per_segment."""
        config = BacktestConfig(
            initial_cash=Decimal(100000),
            max_positions_per_segment=3,
        )
        strategy = MagicMock()
        engine = BacktestEngine(strategy=strategy, config=config)
        engine._sizing_pipeline = MagicMock()
        engine._sizing_pipeline.compute.return_value = Decimal(5000)

        broker = MagicMock()
        broker.has_position.return_value = False
        portfolio = MagicMock()
        portfolio.equity = Decimal(100000)
        portfolio.cash = Decimal(50000)
        portfolio.positions = {}
        broker.get_portfolio.return_value = portfolio
        # 2 existing positions, cap is 3
        broker.get_positions.return_value = {"AAPL": Decimal(10), "MSFT": Decimal(10)}

        checker = MagicMock(spec=PreTradeResult)
        checker = MagicMock()
        result_mock = MagicMock(spec=PreTradeResult)
        result_mock.passed = True
        result_mock.violations = []
        checker.check.return_value = result_mock

        engine._handle_buy(
            broker=broker,
            checker=checker,
            fill_candle=_make_candle(),
            symbol="GOOG",
            history=[_make_candle() for _ in range(20)],
            entry_prices={},
            segment_id="us_tech",
            signal=_make_signal(),
            entry_bars={},
            bar_index=5,
        )

        # Checker SHOULD have been called since we're below cap
        checker.check.assert_called_once()

    def test_default_segment_cap_is_eight(self) -> None:
        """Default max_positions_per_segment is 8."""
        config = BacktestConfig()
        assert config.max_positions_per_segment == 8


class TestMoexMinPositionSize:
    """Verify MOEX segments use USD-denominated min_pos ($100), not RUB 5000."""

    def _run_handle_buy_and_get_min_pos(self, segment_id: str) -> Decimal:
        """Run _handle_buy and capture the min_position_size passed to sizing pipeline."""
        config = BacktestConfig(initial_cash=Decimal(100000))
        strategy = MagicMock()
        engine = BacktestEngine(strategy=strategy, config=config)

        captured_context = {}

        def capture_compute(ctx: object) -> Decimal:
            captured_context["min_position_size"] = ctx.min_position_size  # type: ignore[union-attr]
            return Decimal(5000)

        engine._sizing_pipeline = MagicMock()
        engine._sizing_pipeline.compute.side_effect = capture_compute

        broker = MagicMock()
        broker.has_position.return_value = False
        portfolio = MagicMock()
        portfolio.equity = Decimal(100000)
        portfolio.cash = Decimal(50000)
        portfolio.positions = {}
        broker.get_portfolio.return_value = portfolio
        broker.get_positions.return_value = {}

        checker = MagicMock()
        result_mock = MagicMock(spec=PreTradeResult)
        result_mock.passed = True
        result_mock.violations = []
        checker.check.return_value = result_mock

        signal = Signal(
            direction=SignalDirection.BUY,
            confidence=0.8,
            strategy_name="momentum",
            symbol="TEST",
            market_id="moex" if segment_id.startswith("ru_") else "us",
            segment_id=segment_id,
            strategy_payload={},
            reasoning="test",
        )

        engine._handle_buy(
            broker=broker,
            checker=checker,
            fill_candle=_make_candle(),
            symbol="TEST",
            history=[_make_candle() for _ in range(20)],
            entry_prices={},
            segment_id=segment_id,
            signal=signal,
            entry_bars={},
            bar_index=5,
        )

        return captured_context["min_position_size"]

    def test_moex_segment_scales_min_pos_with_equity(self) -> None:
        """MOEX dust floor: min(5000, max(1000, equity * 0.001)).

        Recalibrated (audit #16): the coefficient was 0.02, which at a 1M-RUB
        book collided with the Kelly sizing band and zeroed thin positions
        (position_value_zero). 0.1% keeps the floor below Kelly.
        """
        min_pos = self._run_handle_buy_and_get_min_pos("ru_blue_chips")
        # With 100K equity: min(5000, max(1000, 100)) = 1000 (absolute minimum).
        expected = Decimal(1000)
        assert min_pos == expected

    def test_us_segment_uses_500_usd_min_pos(self) -> None:
        """US segments should use $500 min_pos (capped)."""
        min_pos = self._run_handle_buy_and_get_min_pos("us_tech")
        assert min_pos == Decimal(500)
