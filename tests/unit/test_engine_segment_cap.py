"""Test segment position cap enforcement in BacktestEngine."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from unittest.mock import MagicMock

from finalayze.backtest.config import BacktestConfig
from finalayze.backtest.engine import BacktestEngine
from finalayze.core.schemas import Candle, Signal, SignalDirection
from finalayze.markets.liquidity import eligible_universe_as_of
from finalayze.risk.pre_trade_check import PreTradeResult
from finalayze.strategies.base import BaseStrategy


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


# ── Cross-symbol concurrent-position cap in run_portfolio (D-09 / LIQ-07) ───────
_CAP_SECTOR = "oil_gas"
_CAP_SEGMENT = "ru_energy"
_CAP_BARS = 120
_CAP_VOLUME = 5_000_000
_CAP_START = datetime(2024, 1, 1, 14, 30, tzinfo=UTC)


class _AlwaysBuyStrategy(BaseStrategy):
    """Emits BUY for every symbol with no open position -- maximises entry pressure."""

    @property
    def name(self) -> str:
        return "always_buy"

    def supported_segments(self) -> list[str]:
        return [_CAP_SEGMENT]

    def generate_signal(  # type: ignore[override]
        self,
        symbol: str,
        candles: list[Candle],
        segment_id: str,
        **kwargs: object,
    ) -> Signal | None:
        if kwargs.get("has_open_position"):
            return None
        return Signal(
            strategy_name=self.name,
            symbol=symbol,
            market_id="moex",
            segment_id=segment_id,
            direction=SignalDirection.BUY,
            confidence=0.9,
            strategy_payload={"momentum": 1.0},
            reasoning="cap test buy",
        )

    def get_parameters(self, segment_id: str) -> dict[str, object]:
        return {}


def _cap_series(symbol: str, close: int) -> list[Candle]:
    return [
        Candle(
            symbol=symbol,
            market_id="moex",
            timeframe="1d",
            timestamp=_CAP_START + timedelta(days=i),
            open=Decimal(close),
            high=Decimal(close) + Decimal(2),
            low=Decimal(close) - Decimal(2),
            close=Decimal(close),
            volume=_CAP_VOLUME,
        )
        for i in range(_CAP_BARS)
    ]


class TestSharedBrokerConcurrentCap:
    """The per-segment concurrent-position cap holds ACROSS symbols in shared-broker mode.

    In the per-symbol ``run`` path each symbol owns its own broker, so the cap is silently
    ineffective (PATTERNS Pitfall 4). ``run_portfolio`` shares one broker + one ``PreTradeChecker``,
    so ``max_concurrent_positions`` is the real portfolio-wide cap. This constructs a scenario that
    WOULD open more than the cap (every symbol fires BUY) and asserts the simultaneous open count
    never exceeds the cap.
    """

    def test_cap_holds_across_symbols_in_run_portfolio(self) -> None:
        cap = 2
        n_symbols = 6
        candles_by_symbol = {f"SYM{i}": _cap_series(f"SYM{i}", 100 + i) for i in range(n_symbols)}
        engine = BacktestEngine(
            strategy=_AlwaysBuyStrategy(),
            config=BacktestConfig(
                initial_cash=Decimal(1_000_000),
                max_concurrent_positions=cap,
                force_close_at_end=True,
            ),
        )
        sector_map = dict.fromkeys(candles_by_symbol, _CAP_SECTOR)

        def eligible_at(ts: datetime) -> set[str]:
            # Top-N >= n_symbols so the eligible set never limits below the cap -- the cap is the
            # only thing that can bound the concurrent open count.
            return eligible_universe_as_of(candles_by_symbol, ts, sector_map, n_symbols)

        _trades, snaps = engine.run_portfolio(
            list(candles_by_symbol),
            _CAP_SEGMENT,
            candles_by_symbol,
            eligible_at=eligible_at,
        )

        max_open = max(
            (sum(1 for q in s.positions.values() if q > 0) for s in snaps),
            default=0,
        )
        assert max_open <= cap

    def test_cap_none_preserves_global_max_positions(self) -> None:
        """When max_concurrent_positions is None, the portfolio-wide max_positions still applies."""
        n_symbols = 6
        candles_by_symbol = {f"SYM{i}": _cap_series(f"SYM{i}", 100 + i) for i in range(n_symbols)}
        engine = BacktestEngine(
            strategy=_AlwaysBuyStrategy(),
            config=BacktestConfig(
                initial_cash=Decimal(1_000_000),
                max_positions=3,
                max_concurrent_positions=None,
                force_close_at_end=True,
            ),
        )
        _trades, snaps = engine.run_portfolio(
            list(candles_by_symbol),
            _CAP_SEGMENT,
            candles_by_symbol,
        )
        max_open = max(
            (sum(1 for q in s.positions.values() if q > 0) for s in snaps),
            default=0,
        )
        assert max_open <= 3
