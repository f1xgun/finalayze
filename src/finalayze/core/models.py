"""SQLAlchemy ORM models for all database tables.

See docs/architecture/OVERVIEW.md for database schema.
"""

from __future__ import annotations

import uuid
from datetime import datetime, time  # noqa: TC003
from decimal import Decimal

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    ForeignKey,
    Integer,
    Numeric,
    String,
    Text,
    Time,
)
from sqlalchemy.dialects import postgresql
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""


class MarketModel(Base):
    """Supported trading markets (e.g. US, MOEX)."""

    __tablename__ = "markets"

    id: Mapped[str] = mapped_column(String(10), primary_key=True)
    name: Mapped[str] = mapped_column(String(50), nullable=False)
    currency: Mapped[str] = mapped_column(String(3), nullable=False)
    timezone: Mapped[str] = mapped_column(String(30), nullable=False)
    open_time: Mapped[time] = mapped_column(Time, nullable=False)
    close_time: Mapped[time] = mapped_column(Time, nullable=False)

    segments: Mapped[list[SegmentModel]] = relationship(back_populates="market")


class SegmentModel(Base):
    """Market segments with strategy configuration."""

    __tablename__ = "segments"

    id: Mapped[str] = mapped_column(String(30), primary_key=True)
    market_id: Mapped[str] = mapped_column(
        String(10),
        ForeignKey("markets.id"),
        nullable=False,
    )
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    active_strategies: Mapped[list[str] | None] = mapped_column(
        ARRAY(Text),
        nullable=True,
    )
    strategy_params: Mapped[dict[str, object] | None] = mapped_column(
        JSONB,
        nullable=True,
    )
    ml_model_id: Mapped[str | None] = mapped_column(String(50), nullable=True)
    max_allocation_pct: Mapped[Decimal] = mapped_column(
        Numeric(5, 4),
        nullable=False,
        default=Decimal("0.30"),
    )
    news_languages: Mapped[list[str] | None] = mapped_column(
        ARRAY(Text),
        nullable=True,
    )

    market: Mapped[MarketModel] = relationship(back_populates="segments")


class InstrumentModel(Base):
    """Tradeable financial instruments."""

    __tablename__ = "instruments"

    symbol: Mapped[str] = mapped_column(String(20), primary_key=True)
    market_id: Mapped[str] = mapped_column(String(10), primary_key=True)
    segment_id: Mapped[str | None] = mapped_column(
        String(30),
        ForeignKey("segments.id"),
        nullable=True,
    )
    name: Mapped[str | None] = mapped_column(String(200), nullable=True)
    figi: Mapped[str | None] = mapped_column(String(20), nullable=True)
    instrument_type: Mapped[str | None] = mapped_column(String(20), nullable=True)
    currency: Mapped[str | None] = mapped_column(String(3), nullable=True)
    lot_size: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)


class CandleModel(Base):
    """OHLCV price candles."""

    __tablename__ = "candles"

    symbol: Mapped[str] = mapped_column(String(20), primary_key=True)
    market_id: Mapped[str] = mapped_column(String(10), primary_key=True)
    timeframe: Mapped[str] = mapped_column(String(5), primary_key=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        primary_key=True,
    )
    open: Mapped[Decimal] = mapped_column(Numeric(12, 4), nullable=False)
    high: Mapped[Decimal] = mapped_column(Numeric(12, 4), nullable=False)
    low: Mapped[Decimal] = mapped_column(Numeric(12, 4), nullable=False)
    close: Mapped[Decimal] = mapped_column(Numeric(12, 4), nullable=False)
    volume: Mapped[int] = mapped_column(BigInteger, nullable=False)
    source: Mapped[str | None] = mapped_column(String(20), nullable=True)


class SignalModel(Base):
    """Trading signals produced by strategies."""

    __tablename__ = "signals"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    strategy_name: Mapped[str] = mapped_column(String(50), nullable=False)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    market_id: Mapped[str] = mapped_column(String(10), nullable=False)
    segment_id: Mapped[str] = mapped_column(String(30), nullable=False)
    direction: Mapped[str] = mapped_column(String(4), nullable=False)
    confidence: Mapped[Decimal] = mapped_column(Numeric(5, 4), nullable=False)
    features: Mapped[dict[str, object] | None] = mapped_column(
        JSONB,
        nullable=True,
    )
    reasoning: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )
    mode: Mapped[str | None] = mapped_column(String(10), nullable=True)

    orders: Mapped[list[OrderModel]] = relationship(back_populates="signal")


class OrderModel(Base):
    """Broker orders linked to signals."""

    __tablename__ = "orders"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    signal_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("signals.id"),
        nullable=True,
    )
    broker: Mapped[str] = mapped_column(String(20), nullable=False)
    broker_order_id: Mapped[str | None] = mapped_column(String(100), nullable=True)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    market_id: Mapped[str] = mapped_column(String(10), nullable=False)
    side: Mapped[str] = mapped_column(String(4), nullable=False)
    order_type: Mapped[str] = mapped_column(String(20), nullable=False)
    quantity: Mapped[Decimal] = mapped_column(Numeric(12, 4), nullable=False)
    limit_price: Mapped[Decimal | None] = mapped_column(Numeric(12, 4), nullable=True)
    stop_price: Mapped[Decimal | None] = mapped_column(Numeric(12, 4), nullable=True)
    currency: Mapped[str | None] = mapped_column(String(3), nullable=True)
    status: Mapped[str] = mapped_column(String(20), nullable=False)
    filled_quantity: Mapped[Decimal] = mapped_column(
        Numeric(12, 4),
        nullable=False,
        default=Decimal(0),
    )
    filled_avg_price: Mapped[Decimal | None] = mapped_column(
        Numeric(12, 4),
        nullable=True,
    )
    submitted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    filled_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    risk_checks: Mapped[dict[str, object] | None] = mapped_column(
        JSONB,
        nullable=True,
    )
    mode: Mapped[str | None] = mapped_column(String(10), nullable=True)

    signal: Mapped[SignalModel | None] = relationship(back_populates="orders")


class NewsArticleModel(Base):
    """ORM model for news articles."""

    __tablename__ = "news_articles"

    id: Mapped[uuid.UUID] = mapped_column(
        postgresql.UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    source: Mapped[str] = mapped_column(String(50), nullable=False)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    content: Mapped[str | None] = mapped_column(Text, nullable=True)
    url: Mapped[str | None] = mapped_column(Text, nullable=True)
    language: Mapped[str] = mapped_column(String(5), nullable=False, server_default="en")
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    symbols: Mapped[list[str]] = mapped_column(
        postgresql.ARRAY(String(20)), nullable=False, server_default="{}"
    )
    affected_segments: Mapped[list[str]] = mapped_column(
        postgresql.ARRAY(String(30)), nullable=False, server_default="{}"
    )
    scope: Mapped[str | None] = mapped_column(String(20), nullable=True)
    category: Mapped[str | None] = mapped_column(String(30), nullable=True)
    raw_sentiment: Mapped[Decimal | None] = mapped_column(Numeric(5, 4), nullable=True)
    credibility_score: Mapped[Decimal | None] = mapped_column(Numeric(5, 4), nullable=True)
    llm_analysis: Mapped[dict[str, object] | None] = mapped_column(postgresql.JSONB, nullable=True)
    is_processed: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="false")


class SentimentScoreModel(Base):
    """ORM model for sentiment scores (TimescaleDB hypertable on timestamp)."""

    __tablename__ = "sentiment_scores"

    symbol: Mapped[str] = mapped_column(String(20), nullable=False, primary_key=True)
    market_id: Mapped[str] = mapped_column(String(10), nullable=False, primary_key=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, primary_key=True
    )
    news_sentiment: Mapped[Decimal | None] = mapped_column(Numeric(5, 4), nullable=True)
    social_sentiment: Mapped[Decimal | None] = mapped_column(Numeric(5, 4), nullable=True)
    composite_sentiment: Mapped[Decimal | None] = mapped_column(Numeric(5, 4), nullable=True)
    confidence: Mapped[Decimal | None] = mapped_column(Numeric(5, 4), nullable=True)
