"""Unit tests for core Pydantic schemas (Layer 0)."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from uuid import uuid4

import pytest
from pydantic import ValidationError

from finalayze.core.schemas import (
    BacktestResult,
    Candle,
    PortfolioState,
    Signal,
    SignalDirection,
    TradeResult,
)

# ── Constants (ruff PLR2004: no magic numbers) ──────────────────────────

EXPECTED_DIRECTION_COUNT = 3

CANDLE_OPEN = Decimal("150.00")
CANDLE_HIGH = Decimal("155.50")
CANDLE_LOW = Decimal("149.00")
CANDLE_CLOSE = Decimal("153.25")
CANDLE_VOLUME = 1_000_000

SIGNAL_CONFIDENCE = 0.85

TRADE_QUANTITY = Decimal(10)
TRADE_ENTRY = Decimal("150.00")
TRADE_EXIT = Decimal("155.00")
TRADE_PNL = Decimal("50.00")
TRADE_PNL_PCT = Decimal("3.33")

PORTFOLIO_CASH = Decimal("10000.00")
PORTFOLIO_POSITION_QTY = Decimal(100)
PORTFOLIO_EQUITY = Decimal("25000.00")

BACKTEST_SHARPE = Decimal("1.45")
BACKTEST_MAX_DD = Decimal("0.12")
BACKTEST_WIN_RATE = Decimal("0.55")
BACKTEST_PROFIT_FACTOR = Decimal("1.80")
BACKTEST_TOTAL_RETURN = Decimal("0.25")
BACKTEST_TOTAL_TRADES = 150


# ── SignalDirection ──────────────────────────────────────────────────────


class TestSignalDirection:
    def test_values_exist(self) -> None:
        assert SignalDirection.BUY == "BUY"
        assert SignalDirection.SELL == "SELL"
        assert SignalDirection.HOLD == "HOLD"

    def test_is_str_enum(self) -> None:
        assert isinstance(SignalDirection.BUY, str)

    def test_member_count(self) -> None:
        assert len(SignalDirection) == EXPECTED_DIRECTION_COUNT


# ── Candle ───────────────────────────────────────────────────────────────


class TestCandle:
    @pytest.fixture
    def candle(self) -> Candle:
        return Candle(
            symbol="AAPL",
            market_id="us",
            timeframe="1d",
            timestamp=datetime(2024, 1, 15, 14, 30, tzinfo=UTC),
            open=CANDLE_OPEN,
            high=CANDLE_HIGH,
            low=CANDLE_LOW,
            close=CANDLE_CLOSE,
            volume=CANDLE_VOLUME,
        )

    def test_creation(self, candle: Candle) -> None:
        assert candle.symbol == "AAPL"
        assert candle.market_id == "us"
        assert candle.timeframe == "1d"
        assert candle.open == CANDLE_OPEN
        assert candle.high == CANDLE_HIGH
        assert candle.low == CANDLE_LOW
        assert candle.close == CANDLE_CLOSE
        assert candle.volume == CANDLE_VOLUME

    def test_timestamp_is_utc(self, candle: Candle) -> None:
        assert candle.timestamp.tzinfo is not None

    def test_naive_timestamp_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Candle(
                symbol="AAPL",
                market_id="us",
                timeframe="1d",
                timestamp=datetime(2024, 1, 15, 14, 30),  # noqa: DTZ001  # intentionally naive
                open=CANDLE_OPEN,
                high=CANDLE_HIGH,
                low=CANDLE_LOW,
                close=CANDLE_CLOSE,
                volume=CANDLE_VOLUME,
            )

    def test_frozen(self, candle: Candle) -> None:
        with pytest.raises(ValidationError):
            candle.symbol = "GOOG"  # type: ignore[misc]

    def test_decimal_fields(self, candle: Candle) -> None:
        assert isinstance(candle.open, Decimal)
        assert isinstance(candle.high, Decimal)
        assert isinstance(candle.low, Decimal)
        assert isinstance(candle.close, Decimal)


# ── Signal ───────────────────────────────────────────────────────────────


class TestSignal:
    @pytest.fixture
    def signal(self) -> Signal:
        return Signal(
            strategy_name="momentum_v1",
            symbol="AAPL",
            market_id="us",
            segment_id="large_cap",
            direction=SignalDirection.BUY,
            confidence=SIGNAL_CONFIDENCE,
            features={"rsi": 65.0, "macd": 1.2},
            reasoning="Strong upward momentum with RSI above 60",
        )

    def test_creation(self, signal: Signal) -> None:
        assert signal.strategy_name == "momentum_v1"
        assert signal.symbol == "AAPL"
        assert signal.market_id == "us"
        assert signal.segment_id == "large_cap"
        assert signal.direction == SignalDirection.BUY
        assert signal.confidence == SIGNAL_CONFIDENCE
        assert signal.reasoning == "Strong upward momentum with RSI above 60"

    def test_features_dict(self, signal: Signal) -> None:
        assert "rsi" in signal.features
        assert "macd" in signal.features

    def test_frozen(self, signal: Signal) -> None:
        with pytest.raises(ValidationError):
            signal.direction = SignalDirection.SELL  # type: ignore[misc]

    def test_direction_type(self, signal: Signal) -> None:
        assert isinstance(signal.direction, SignalDirection)

    def test_confidence_out_of_range_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Signal(
                strategy_name="momentum_v1",
                symbol="AAPL",
                market_id="us",
                segment_id="large_cap",
                direction=SignalDirection.BUY,
                confidence=1.5,  # out of range
                features={},
                reasoning="test",
            )

    def test_confidence_negative_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Signal(
                strategy_name="momentum_v1",
                symbol="AAPL",
                market_id="us",
                segment_id="large_cap",
                direction=SignalDirection.BUY,
                confidence=-0.1,  # negative
                features={},
                reasoning="test",
            )


# ── TradeResult ──────────────────────────────────────────────────────────


class TestTradeResult:
    @pytest.fixture
    def trade(self) -> TradeResult:
        return TradeResult(
            signal_id=uuid4(),
            symbol="AAPL",
            side="buy",
            quantity=TRADE_QUANTITY,
            entry_price=TRADE_ENTRY,
            exit_price=TRADE_EXIT,
            pnl=TRADE_PNL,
            pnl_pct=TRADE_PNL_PCT,
        )

    def test_creation(self, trade: TradeResult) -> None:
        assert trade.symbol == "AAPL"
        assert trade.side == "buy"
        assert trade.quantity == TRADE_QUANTITY
        assert trade.entry_price == TRADE_ENTRY
        assert trade.exit_price == TRADE_EXIT
        assert trade.pnl == TRADE_PNL
        assert trade.pnl_pct == TRADE_PNL_PCT

    def test_signal_id_is_uuid(self, trade: TradeResult) -> None:
        assert trade.signal_id is not None

    def test_frozen(self, trade: TradeResult) -> None:
        with pytest.raises(ValidationError):
            trade.pnl = Decimal(999)  # type: ignore[misc]

    def test_decimal_fields(self, trade: TradeResult) -> None:
        assert isinstance(trade.quantity, Decimal)
        assert isinstance(trade.entry_price, Decimal)
        assert isinstance(trade.exit_price, Decimal)
        assert isinstance(trade.pnl, Decimal)
        assert isinstance(trade.pnl_pct, Decimal)


# ── PortfolioState ───────────────────────────────────────────────────────


class TestPortfolioState:
    @pytest.fixture
    def portfolio(self) -> PortfolioState:
        return PortfolioState(
            cash=PORTFOLIO_CASH,
            positions={"AAPL": PORTFOLIO_POSITION_QTY},
            equity=PORTFOLIO_EQUITY,
            timestamp=datetime(2024, 1, 15, 16, 0, tzinfo=UTC),
        )

    def test_creation(self, portfolio: PortfolioState) -> None:
        assert portfolio.cash == PORTFOLIO_CASH
        assert portfolio.positions["AAPL"] == PORTFOLIO_POSITION_QTY
        assert portfolio.equity == PORTFOLIO_EQUITY

    def test_timestamp_is_utc(self, portfolio: PortfolioState) -> None:
        assert portfolio.timestamp.tzinfo is not None

    def test_naive_timestamp_rejected(self) -> None:
        with pytest.raises(ValidationError):
            PortfolioState(
                cash=PORTFOLIO_CASH,
                positions={"AAPL": PORTFOLIO_POSITION_QTY},
                equity=PORTFOLIO_EQUITY,
                timestamp=datetime(2024, 1, 15, 16, 0),  # noqa: DTZ001  # intentionally naive
            )

    def test_frozen(self, portfolio: PortfolioState) -> None:
        with pytest.raises(ValidationError):
            portfolio.cash = Decimal(0)  # type: ignore[misc]

    def test_decimal_fields(self, portfolio: PortfolioState) -> None:
        assert isinstance(portfolio.cash, Decimal)
        assert isinstance(portfolio.equity, Decimal)


# ── BacktestResult ───────────────────────────────────────────────────────


class TestBacktestResult:
    @pytest.fixture
    def result(self) -> BacktestResult:
        return BacktestResult(
            sharpe=BACKTEST_SHARPE,
            max_drawdown=BACKTEST_MAX_DD,
            win_rate=BACKTEST_WIN_RATE,
            profit_factor=BACKTEST_PROFIT_FACTOR,
            total_return=BACKTEST_TOTAL_RETURN,
            total_trades=BACKTEST_TOTAL_TRADES,
        )

    def test_creation(self, result: BacktestResult) -> None:
        assert result.sharpe == BACKTEST_SHARPE
        assert result.max_drawdown == BACKTEST_MAX_DD
        assert result.win_rate == BACKTEST_WIN_RATE
        assert result.profit_factor == BACKTEST_PROFIT_FACTOR
        assert result.total_return == BACKTEST_TOTAL_RETURN
        assert result.total_trades == BACKTEST_TOTAL_TRADES

    def test_frozen(self, result: BacktestResult) -> None:
        with pytest.raises(ValidationError):
            result.sharpe = Decimal(0)  # type: ignore[misc]

    def test_decimal_fields(self, result: BacktestResult) -> None:
        assert isinstance(result.sharpe, Decimal)
        assert isinstance(result.max_drawdown, Decimal)
        assert isinstance(result.win_rate, Decimal)
        assert isinstance(result.profit_factor, Decimal)
        assert isinstance(result.total_return, Decimal)
