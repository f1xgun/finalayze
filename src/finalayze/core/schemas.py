"""Shared Pydantic schemas (Layer 0).

See docs/architecture/DEPENDENCY_LAYERS.md for layering rules.
"""

from __future__ import annotations

from datetime import datetime  # noqa: TC003
from decimal import Decimal  # noqa: TC003
from enum import StrEnum
from uuid import UUID  # noqa: TC003

from pydantic import BaseModel, ConfigDict, Field, field_validator


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
    volume: int = Field(ge=0)
    source: str | None = None

    @field_validator("timestamp")
    @classmethod
    def must_be_utc_aware(cls, v: datetime) -> datetime:
        """Reject naive datetimes; all timestamps must be UTC-aware."""
        if v.tzinfo is None:
            msg = "timestamp must be timezone-aware (UTC)"
            raise ValueError(msg)
        return v


class Signal(BaseModel):
    """Trading signal produced by a strategy.

    Notes:
        ``confidence`` is typed as ``float`` (not ``Decimal``) because it
        represents a probability/ratio in [0.0, 1.0], not a monetary value.
        The "Decimal for money fields" rule does not apply here.
    """

    model_config = ConfigDict(frozen=True)

    strategy_name: str
    symbol: str
    market_id: str
    segment_id: str
    direction: SignalDirection
    confidence: float
    features: dict[str, float]
    reasoning: str

    @field_validator("confidence")
    @classmethod
    def confidence_must_be_probability(cls, v: float) -> float:
        """Validate that confidence is a probability in [0.0, 1.0]."""
        if not (0.0 <= v <= 1.0):
            msg = f"confidence must be in [0.0, 1.0], got {v}"
            raise ValueError(msg)
        return v


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

    @field_validator("timestamp")
    @classmethod
    def must_be_utc_aware(cls, v: datetime) -> datetime:
        """Reject naive datetimes; all timestamps must be UTC-aware."""
        if v.tzinfo is None:
            msg = "timestamp must be timezone-aware (UTC)"
            raise ValueError(msg)
        return v


class BacktestResult(BaseModel):
    """Aggregate metrics from a backtest run."""

    model_config = ConfigDict(frozen=True)

    sharpe: Decimal
    max_drawdown: Decimal
    win_rate: Decimal
    profit_factor: Decimal
    total_return: Decimal
    total_trades: int


class NewsArticle(BaseModel):
    """A news article fetched from an external source."""

    model_config = ConfigDict(frozen=True)

    id: UUID
    source: str
    title: str
    content: str
    url: str
    language: str  # "en" | "ru"
    published_at: datetime
    symbols: list[str] = []
    affected_segments: list[str] = []
    scope: str | None = None  # "global" | "us" | "russia" | "sector"
    raw_sentiment: float | None = None
    credibility_score: float | None = None

    @field_validator("published_at")
    @classmethod
    def must_be_utc_aware(cls, v: datetime) -> datetime:
        """Reject naive datetimes."""
        if v.tzinfo is None:
            msg = "published_at must be timezone-aware (UTC)"
            raise ValueError(msg)
        return v

    @field_validator("raw_sentiment")
    @classmethod
    def sentiment_in_range(cls, v: float | None) -> float | None:
        """Validate sentiment is in [-1.0, 1.0] when provided."""
        if v is not None and not (-1.0 <= v <= 1.0):
            msg = f"raw_sentiment must be in [-1.0, 1.0], got {v}"
            raise ValueError(msg)
        return v


class SentimentResult(BaseModel):
    """Result of LLM sentiment analysis on a news article."""

    model_config = ConfigDict(frozen=True)

    sentiment: float  # -1.0 to +1.0
    confidence: float  # 0.0 to 1.0
    reasoning: str

    @field_validator("sentiment")
    @classmethod
    def sentiment_in_range(cls, v: float) -> float:
        """Validate sentiment is in [-1.0, 1.0]."""
        if not (-1.0 <= v <= 1.0):
            msg = f"sentiment must be in [-1.0, 1.0], got {v}"
            raise ValueError(msg)
        return v

    @field_validator("confidence")
    @classmethod
    def confidence_in_range(cls, v: float) -> float:
        """Validate confidence is in [0.0, 1.0]."""
        if not (0.0 <= v <= 1.0):
            msg = f"confidence must be in [0.0, 1.0], got {v}"
            raise ValueError(msg)
        return v
