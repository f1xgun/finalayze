"""Shared Pydantic schemas (Layer 0).

See docs/architecture/DEPENDENCY_LAYERS.md for layering rules.
"""

from __future__ import annotations

from datetime import datetime  # noqa: TC003
from decimal import Decimal  # noqa: TC003
from enum import StrEnum
from uuid import UUID  # noqa: TC003

from pydantic import BaseModel, ConfigDict


class SignalDirection(StrEnum):
    """Direction of a trading signal."""

    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class Candle(BaseModel):
    """OHLCV candle for a single timeframe bar."""

    model_config = ConfigDict(frozen=True)

    symbol: str
    market_id: str
    timeframe: str
    timestamp: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: int


class Signal(BaseModel):
    """Trading signal produced by a strategy."""

    model_config = ConfigDict(frozen=True)

    strategy_name: str
    symbol: str
    market_id: str
    segment_id: str
    direction: SignalDirection
    confidence: float
    features: dict[str, float]
    reasoning: str


class TradeResult(BaseModel):
    """Result of an executed trade."""

    model_config = ConfigDict(frozen=True)

    signal_id: UUID
    symbol: str
    side: str
    quantity: Decimal
    entry_price: Decimal
    exit_price: Decimal
    pnl: Decimal
    pnl_pct: Decimal


class PortfolioState(BaseModel):
    """Snapshot of portfolio at a point in time."""

    model_config = ConfigDict(frozen=True)

    cash: Decimal
    positions: dict[str, Decimal]
    equity: Decimal
    timestamp: datetime


class BacktestResult(BaseModel):
    """Aggregate metrics from a backtest run."""

    model_config = ConfigDict(frozen=True)

    sharpe: Decimal
    max_drawdown: Decimal
    win_rate: Decimal
    profit_factor: Decimal
    total_return: Decimal
    total_trades: int
