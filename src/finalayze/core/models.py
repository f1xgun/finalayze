"""SQLAlchemy ORM models for all database tables.

See docs/architecture/OVERVIEW.md for database schema.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, time
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
    credibility: Mapped[Decimal | None] = mapped_column(Numeric(5, 4), nullable=True)


class MacroSnapshotModel(Base):
    """Macro snapshot persisted to TimescaleDB on each cache refresh."""

    __tablename__ = "macro_snapshots"

    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), primary_key=True)
    key_rate: Mapped[Decimal | None] = mapped_column(Numeric(5, 2))
    ruonia_7d_avg: Mapped[Decimal | None] = mapped_column(Numeric(5, 2))
    cpi_yoy: Mapped[Decimal | None] = mapped_column(Numeric(5, 2))
    last_cbr_decision: Mapped[str | None] = mapped_column(String(10))
    breakeven_inflation: Mapped[Decimal | None] = mapped_column(Numeric(5, 2))
    yield_curve: Mapped[dict[str, str] | None] = mapped_column(JSONB)
    usdrub: Mapped[Decimal | None] = mapped_column(Numeric(8, 4))
    ofzin_indexation_coefficient: Mapped[Decimal | None] = mapped_column(Numeric(8, 6))


class BondCandleModel(Base):
    """Bond OHLCV candle cache for TimescaleDB."""

    __tablename__ = "bond_candles"

    bond_figi: Mapped[str] = mapped_column(String(20), primary_key=True)
    date: Mapped[datetime] = mapped_column(DateTime(timezone=True), primary_key=True)
    open: Mapped[Decimal] = mapped_column(Numeric(12, 6), nullable=False)
    high: Mapped[Decimal] = mapped_column(Numeric(12, 6), nullable=False)
    low: Mapped[Decimal] = mapped_column(Numeric(12, 6), nullable=False)
    close: Mapped[Decimal] = mapped_column(Numeric(12, 6), nullable=False)
    volume: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)


class CouponScheduleModel(Base):
    """Coupon schedule cache for bond coupon payments."""

    __tablename__ = "coupon_schedules"

    bond_figi: Mapped[str] = mapped_column(String(20), primary_key=True)
    coupon_number: Mapped[int] = mapped_column(Integer, primary_key=True)
    coupon_date: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    record_date: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    amount_per_bond: Mapped[Decimal] = mapped_column(Numeric(12, 4), nullable=False)
    is_floating: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)


class AmortizationEventModel(Base):
    """Amortization event schedule for amortizing bonds."""

    __tablename__ = "amortization_events"

    bond_figi: Mapped[str] = mapped_column(String(20), primary_key=True)
    event_number: Mapped[int] = mapped_column(Integer, primary_key=True)
    event_date: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    remaining_nominal_pct: Mapped[Decimal] = mapped_column(
        Numeric(8, 4), nullable=False, default=Decimal("100.0")
    )


class LayerLedgerModel(Base):
    """Persisted layer ledger state for bond positions.

    Composite primary key (layer_id, symbol) stores one row per bond position
    per portfolio layer.
    """

    __tablename__ = "layer_ledger"

    layer_id: Mapped[str] = mapped_column(String(20), primary_key=True)
    symbol: Mapped[str] = mapped_column(String(30), primary_key=True)
    quantity: Mapped[Decimal] = mapped_column(Numeric(14, 4), nullable=False)
    entry_ytm_pct: Mapped[Decimal] = mapped_column(Numeric(8, 4), nullable=False)
    entry_price: Mapped[Decimal] = mapped_column(Numeric(14, 4), nullable=False)
    entry_clean_pct: Mapped[Decimal] = mapped_column(Numeric(8, 4), nullable=False)
    entry_date: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class DailyEquitySnapshot(Base):
    """Start-of-day equity snapshot per market, persisted to TimescaleDB."""

    __tablename__ = "daily_equity_snapshots"

    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), primary_key=True)
    market_id: Mapped[str] = mapped_column(String(20), primary_key=True)
    equity: Mapped[Decimal] = mapped_column(Numeric(14, 4), nullable=False)
    currency: Mapped[str] = mapped_column(String(3), nullable=False, default="USD")


class StopLossEventModel(Base):
    """Append-only stop-loss state event log, persisted to TimescaleDB hypertable.

    One row per (symbol, timestamp) from per-cycle snapshots plus key events
    (entry, activation, trigger, exit). Used for STOP-03 history chart and
    post-mortem analysis (ALRT-01 follow-up).
    """

    __tablename__ = "stop_loss_events"

    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), primary_key=True)
    symbol: Mapped[str] = mapped_column(String(30), primary_key=True)
    market_id: Mapped[str] = mapped_column(String(20), primary_key=True)
    event_type: Mapped[str] = mapped_column(String(20), nullable=False)
    entry_price: Mapped[Decimal | None] = mapped_column(Numeric(14, 4))
    current_stop: Mapped[Decimal | None] = mapped_column(Numeric(14, 4))
    highest_price: Mapped[Decimal | None] = mapped_column(Numeric(14, 4))
    atr_value: Mapped[Decimal | None] = mapped_column(Numeric(14, 4))
    activation_atr: Mapped[Decimal | None] = mapped_column(Numeric(6, 4))
    trail_atr: Mapped[Decimal | None] = mapped_column(Numeric(6, 4))
    trail_activated: Mapped[bool | None] = mapped_column(Boolean)
    current_price: Mapped[Decimal | None] = mapped_column(Numeric(14, 4))


class AlertModel(Base):
    """Alert emission log (raw + LLM follow-up threaded via parent_id).

    ALRT-03 (Phase 57). Two-row schema for anomaly pair: raw row with
    alert_type='anomaly_raw' + parent_id=NULL; LLM follow-up with
    alert_type='anomaly_llm' + parent_id=<raw.id>.
    """

    __tablename__ = "alerts"

    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), primary_key=True,
    )
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4,
    )
    alert_type: Mapped[str] = mapped_column(String(30), nullable=False)
    priority: Mapped[str] = mapped_column(String(10), nullable=False)
    symbol: Mapped[str | None] = mapped_column(String(30), nullable=True)
    market_id: Mapped[str | None] = mapped_column(String(20), nullable=True)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    # parent_id is a plain nullable UUID without a database FK — see migration
    # 009 docstring. TimescaleDB hypertables forbid UNIQUE constraints that
    # exclude the partition column, so the self-FK is impractical. parent_id
    # integrity is managed at the application layer (raw alert persists first;
    # child alert threads parent_id only after the raw write succeeds).
    parent_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        nullable=True,
    )
    delivery_status: Mapped[str] = mapped_column(
        String(10), nullable=False, default="queued",
    )
    alert_metadata: Mapped[dict[str, object] | None] = mapped_column(
        "metadata",  # column name; Python attr renamed to avoid SQLAlchemy reserved word
        JSONB,
        nullable=True,
    )

    def __init__(self, **kwargs: object) -> None:
        # SQLAlchemy 2.0 `default=` only applies at flush time, not at __init__.
        # Apply the Python-side default for delivery_status so callers can rely
        # on AlertModel().delivery_status == "queued" without an explicit pass.
        kwargs.setdefault("delivery_status", "queued")
        super().__init__(**kwargs)


class MetaAgentDecisionModel(Base):
    """Meta-agent decision log — one row per cycle (Phase 58-01, META-03).

    Mirrors AlertModel ergonomics line-for-line:
      - composite primary key (timestamp, id) for TimescaleDB hypertable
      - parent_decision_id is a plain nullable UUID without a database FK
        (hypertables forbid the UNIQUE (id) constraint a self-FK would
        require — same constraint as alerts.parent_id; integrity is managed
        at the application layer)
      - decision_metadata Python attribute maps to the bare DB column
        ``metadata`` (SQLAlchemy DeclarativeBase reserves ``metadata``)
      - ``__init__`` override applies Python-side defaults that
        ``mapped_column(default=...)`` only fires at flush time
    """

    __tablename__ = "agent_decisions"

    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), primary_key=True,
    )
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4,
    )
    severity: Mapped[str] = mapped_column(String(15), nullable=False)
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    rationale: Mapped[str] = mapped_column(Text, nullable=False)
    actions: Mapped[list[dict[str, object]]] = mapped_column(
        JSONB, nullable=False, default=list,
    )
    outcome: Mapped[str | None] = mapped_column(Text, nullable=True)
    dry_run: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    decision_metadata: Mapped[dict[str, object] | None] = mapped_column(
        "metadata",  # column name; Python attr renamed to avoid SQLAlchemy reserved word
        JSONB,
        nullable=True,
    )
    # parent_decision_id is a plain nullable UUID without a database FK — see
    # migration 010 docstring. TimescaleDB hypertables forbid UNIQUE
    # constraints that exclude the partition column, so the self-FK is
    # impractical. Integrity is managed at the application layer (the runner
    # threads parent_decision_id only after the parent row is persisted).
    parent_decision_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        nullable=True,
    )
    status: Mapped[str] = mapped_column(
        String(15), nullable=False, default="queued",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
    )

    def __init__(self, **kwargs: object) -> None:
        # SQLAlchemy 2.0 `default=` only applies at flush time, not at __init__.
        # Apply Python-side defaults so callers can construct
        # MetaAgentDecisionModel(severity=..., summary=..., rationale=...)
        # and read .actions, .dry_run, .status without explicit passes.
        kwargs.setdefault("actions", [])
        kwargs.setdefault("dry_run", True)
        kwargs.setdefault("status", "queued")
        super().__init__(**kwargs)


class PortfolioSnapshot(Base):
    """Portfolio equity snapshot written after each strategy cycle."""

    __tablename__ = "portfolio_snapshots"

    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), primary_key=True)
    market_id: Mapped[str] = mapped_column(String(10), primary_key=True)
    equity: Mapped[Decimal | None] = mapped_column(Numeric(14, 4))
    cash: Mapped[Decimal | None] = mapped_column(Numeric(14, 4))
    daily_pnl: Mapped[Decimal | None] = mapped_column(Numeric(14, 4))
    drawdown_pct: Mapped[Decimal | None] = mapped_column(Numeric(7, 4))
    mode: Mapped[str | None] = mapped_column(String(10))


class SandboxMetricRow(Base):
    """Sandbox monitoring metrics persisted per cycle (TimescaleDB hypertable)."""

    __tablename__ = "sandbox_metrics"

    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), primary_key=True)
    market_id: Mapped[str] = mapped_column(String(10), primary_key=True)
    trade_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    pnl_rub: Mapped[Decimal | None] = mapped_column(Numeric(14, 4))
    equity_rub: Mapped[Decimal] = mapped_column(Numeric(14, 4), nullable=False)
    fill_rate: Mapped[Decimal | None] = mapped_column(Numeric(5, 4))
    uptime_cycles: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    signals_generated: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    errors_caught: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    max_slippage_bps: Mapped[Decimal | None] = mapped_column(Numeric(8, 2))
    avg_slippage_bps: Mapped[Decimal | None] = mapped_column(Numeric(8, 2))
    drawdown_pct: Mapped[Decimal | None] = mapped_column(Numeric(7, 4))
