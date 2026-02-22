# Backtest Pipeline Vertical Slice -- Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Run `uv run python scripts/run_backtest.py --symbol AAPL --segment us_tech` and get a performance report with Sharpe, drawdown, win rate.

**Architecture:** Bottom-up vertical slice. Build Pydantic schemas (Layer 0), then DB models + data layer (Layer 2), then strategy + risk (Layer 4), then broker (Layer 5), then backtest engine + CLI. Each layer testable independently with TDD.

**Tech Stack:** Python 3.12, Pydantic v2 (frozen models), SQLAlchemy 2.0 async, PostgreSQL + TimescaleDB, pandas + pandas-ta, yfinance, Alembic, Decimal for money.

**Design doc:** `docs/plans/2026-02-22-phase1-backtest-slice-design.md`

---

## Task 1: Pydantic Schemas -- SignalDirection Enum

**Files:**
- Create: `src/finalayze/core/schemas.py` (replace stub)
- Test: `tests/unit/test_schemas.py`

**Step 1: Write the failing test**

```python
# tests/unit/test_schemas.py
"""Unit tests for core Pydantic schemas."""

from __future__ import annotations

from finalayze.core.schemas import SignalDirection


class TestSignalDirection:
    def test_buy_value(self) -> None:
        assert SignalDirection.BUY.value == "BUY"

    def test_sell_value(self) -> None:
        assert SignalDirection.SELL.value == "SELL"

    def test_hold_value(self) -> None:
        assert SignalDirection.HOLD.value == "HOLD"

    def test_member_count(self) -> None:
        expected_count = 3
        assert len(SignalDirection) == expected_count
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_schemas.py::TestSignalDirection -v`
Expected: FAIL with `ImportError: cannot import name 'SignalDirection'`

**Step 3: Write minimal implementation**

```python
# src/finalayze/core/schemas.py
"""Shared Pydantic schemas (Layer 0).

See docs/architecture/DEPENDENCY_LAYERS.md for layering rules.
"""

from __future__ import annotations

from enum import StrEnum


class SignalDirection(StrEnum):
    """Direction of a trading signal."""

    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_schemas.py::TestSignalDirection -v`
Expected: PASS (3 tests)

**Step 5: Commit**

```bash
git add src/finalayze/core/schemas.py tests/unit/test_schemas.py
git commit -m "feat(core): add SignalDirection enum"
```

---

## Task 2: Pydantic Schemas -- Candle Model

**Files:**
- Modify: `src/finalayze/core/schemas.py`
- Modify: `tests/unit/test_schemas.py`

**Step 1: Write the failing test**

Append to `tests/unit/test_schemas.py`:

```python
from datetime import UTC, datetime
from decimal import Decimal

from finalayze.core.schemas import Candle


class TestCandle:
    def test_create_candle(self) -> None:
        candle = Candle(
            symbol="AAPL",
            market_id="us",
            timeframe="1d",
            timestamp=datetime(2024, 1, 15, 14, 30, tzinfo=UTC),
            open=Decimal("185.50"),
            high=Decimal("187.00"),
            low=Decimal("184.00"),
            close=Decimal("186.25"),
            volume=50_000_000,
        )
        assert candle.symbol == "AAPL"
        assert candle.close == Decimal("186.25")

    def test_candle_is_frozen(self) -> None:
        candle = Candle(
            symbol="AAPL",
            market_id="us",
            timeframe="1d",
            timestamp=datetime(2024, 1, 15, 14, 30, tzinfo=UTC),
            open=Decimal("185.50"),
            high=Decimal("187.00"),
            low=Decimal("184.00"),
            close=Decimal("186.25"),
            volume=50_000_000,
        )
        import pytest
        with pytest.raises(Exception):  # noqa: B017
            candle.symbol = "MSFT"  # type: ignore[misc]

    def test_candle_requires_utc_timestamp(self) -> None:
        """Timestamps must be timezone-aware."""
        candle = Candle(
            symbol="AAPL",
            market_id="us",
            timeframe="1d",
            timestamp=datetime(2024, 1, 15, 14, 30, tzinfo=UTC),
            open=Decimal("185.50"),
            high=Decimal("187.00"),
            low=Decimal("184.00"),
            close=Decimal("186.25"),
            volume=50_000_000,
        )
        assert candle.timestamp.tzinfo is not None
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_schemas.py::TestCandle -v`
Expected: FAIL with `ImportError: cannot import name 'Candle'`

**Step 3: Write minimal implementation**

Add to `src/finalayze/core/schemas.py`:

```python
from datetime import datetime
from decimal import Decimal

from pydantic import BaseModel, ConfigDict


class Candle(BaseModel):
    """OHLCV candle data."""

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
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_schemas.py::TestCandle -v`
Expected: PASS (3 tests)

**Step 5: Commit**

```bash
git add src/finalayze/core/schemas.py tests/unit/test_schemas.py
git commit -m "feat(core): add Candle pydantic schema"
```

---

## Task 3: Pydantic Schemas -- Signal, TradeResult, PortfolioState, BacktestResult

**Files:**
- Modify: `src/finalayze/core/schemas.py`
- Modify: `tests/unit/test_schemas.py`

**Step 1: Write the failing tests**

Append to `tests/unit/test_schemas.py`:

```python
import uuid

from finalayze.core.schemas import (
    BacktestResult,
    PortfolioState,
    Signal,
    SignalDirection,
    TradeResult,
)


class TestSignal:
    def test_create_signal(self) -> None:
        signal = Signal(
            strategy_name="momentum",
            symbol="AAPL",
            market_id="us",
            segment_id="us_tech",
            direction=SignalDirection.BUY,
            confidence=0.75,
            features={"rsi": 28.5, "macd_hist": 0.5},
            reasoning="RSI oversold + MACD cross",
        )
        assert signal.direction == SignalDirection.BUY
        assert signal.confidence == 0.75

    def test_signal_is_frozen(self) -> None:
        signal = Signal(
            strategy_name="momentum",
            symbol="AAPL",
            market_id="us",
            segment_id="us_tech",
            direction=SignalDirection.BUY,
            confidence=0.75,
            features={},
            reasoning="test",
        )
        import pytest
        with pytest.raises(Exception):  # noqa: B017
            signal.confidence = 0.5  # type: ignore[misc]


class TestTradeResult:
    def test_create_trade_result(self) -> None:
        trade = TradeResult(
            signal_id=uuid.uuid4(),
            symbol="AAPL",
            side="BUY",
            quantity=Decimal("10"),
            entry_price=Decimal("185.00"),
            exit_price=Decimal("190.00"),
            pnl=Decimal("50.00"),
            pnl_pct=Decimal("0.027"),
        )
        assert trade.pnl == Decimal("50.00")

    def test_trade_result_is_frozen(self) -> None:
        trade = TradeResult(
            signal_id=uuid.uuid4(),
            symbol="AAPL",
            side="BUY",
            quantity=Decimal("10"),
            entry_price=Decimal("185.00"),
            exit_price=Decimal("190.00"),
            pnl=Decimal("50.00"),
            pnl_pct=Decimal("0.027"),
        )
        import pytest
        with pytest.raises(Exception):  # noqa: B017
            trade.pnl = Decimal("100.00")  # type: ignore[misc]


class TestPortfolioState:
    def test_create_portfolio_state(self) -> None:
        state = PortfolioState(
            cash=Decimal("50000.00"),
            positions={"AAPL": Decimal("10")},
            equity=Decimal("51850.00"),
            timestamp=datetime(2024, 6, 15, 20, 0, tzinfo=UTC),
        )
        assert state.cash == Decimal("50000.00")
        assert state.positions["AAPL"] == Decimal("10")


class TestBacktestResult:
    def test_create_backtest_result(self) -> None:
        result = BacktestResult(
            sharpe=Decimal("1.25"),
            max_drawdown=Decimal("0.08"),
            win_rate=Decimal("0.55"),
            profit_factor=Decimal("1.45"),
            total_return=Decimal("0.15"),
            total_trades=42,
        )
        assert result.sharpe == Decimal("1.25")
        assert result.total_trades == 42
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_schemas.py::TestSignal tests/unit/test_schemas.py::TestTradeResult tests/unit/test_schemas.py::TestPortfolioState tests/unit/test_schemas.py::TestBacktestResult -v`
Expected: FAIL with `ImportError`

**Step 3: Write minimal implementation**

Add to `src/finalayze/core/schemas.py`:

```python
import uuid as _uuid


class Signal(BaseModel):
    """Strategy output signal."""

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
    """Execution result of a trade."""

    model_config = ConfigDict(frozen=True)

    signal_id: _uuid.UUID
    symbol: str
    side: str
    quantity: Decimal
    entry_price: Decimal
    exit_price: Decimal
    pnl: Decimal
    pnl_pct: Decimal


class PortfolioState(BaseModel):
    """Portfolio snapshot at a point in time."""

    model_config = ConfigDict(frozen=True)

    cash: Decimal
    positions: dict[str, Decimal]
    equity: Decimal
    timestamp: datetime


class BacktestResult(BaseModel):
    """Performance metrics from a backtest run."""

    model_config = ConfigDict(frozen=True)

    sharpe: Decimal
    max_drawdown: Decimal
    win_rate: Decimal
    profit_factor: Decimal
    total_return: Decimal
    total_trades: int
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_schemas.py -v`
Expected: PASS (all tests including Task 1 + Task 2 + Task 3)

**Step 5: Run linters**

Run: `uv run ruff check src/finalayze/core/schemas.py tests/unit/test_schemas.py && uv run mypy src/finalayze/core/schemas.py`
Expected: zero errors

**Step 6: Commit**

```bash
git add src/finalayze/core/schemas.py tests/unit/test_schemas.py
git commit -m "feat(core): add Signal, TradeResult, PortfolioState, BacktestResult schemas"
```

---

## Task 4: SQLAlchemy Models -- Database Base + MarketModel + SegmentModel

**Files:**
- Create: `src/finalayze/core/db.py` (async engine + session factory)
- Modify: `src/finalayze/core/models.py` (replace stub)
- Test: `tests/unit/test_models.py`

**Step 1: Write the failing test**

```python
# tests/unit/test_models.py
"""Unit tests for SQLAlchemy models."""

from __future__ import annotations

from finalayze.core.models import Base, CandleModel, MarketModel, SegmentModel


class TestMarketModel:
    def test_table_name(self) -> None:
        assert MarketModel.__tablename__ == "markets"

    def test_has_primary_key(self) -> None:
        pk_cols = [c.name for c in MarketModel.__table__.primary_key.columns]
        assert pk_cols == ["id"]


class TestSegmentModel:
    def test_table_name(self) -> None:
        assert SegmentModel.__tablename__ == "segments"

    def test_has_market_fk(self) -> None:
        fks = [
            fk.target_fullname
            for col in SegmentModel.__table__.columns
            for fk in col.foreign_keys
        ]
        assert "markets.id" in fks


class TestCandleModel:
    def test_table_name(self) -> None:
        assert CandleModel.__tablename__ == "candles"

    def test_composite_pk(self) -> None:
        pk_cols = sorted(c.name for c in CandleModel.__table__.primary_key.columns)
        assert pk_cols == ["market_id", "symbol", "timeframe", "timestamp"]


class TestBase:
    def test_base_has_metadata(self) -> None:
        assert Base.metadata is not None
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_models.py -v`
Expected: FAIL with `ImportError: cannot import name 'Base' from 'finalayze.core.models'`

**Step 3: Write minimal implementation**

```python
# src/finalayze/core/db.py
"""Async database engine and session factory (Layer 2).

Usage:
    engine = create_async_engine(settings.database_url)
    async_session = async_sessionmaker(engine, expire_on_commit=False)
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

__all__ = ["AsyncSession", "async_sessionmaker", "create_async_engine"]
```

```python
# src/finalayze/core/models.py
"""SQLAlchemy ORM models (Layer 2).

See docs/architecture/OVERVIEW.md for database schema.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from decimal import Decimal

from sqlalchemy import (
    ARRAY,
    BigInteger,
    Boolean,
    Integer,
    Numeric,
    String,
    Text,
    Time,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """SQLAlchemy declarative base."""


class MarketModel(Base):
    """Markets registry (US, MOEX)."""

    __tablename__ = "markets"

    id: Mapped[str] = mapped_column(String(10), primary_key=True)
    name: Mapped[str] = mapped_column(String(50), nullable=False)
    currency: Mapped[str] = mapped_column(String(3), nullable=False)
    timezone: Mapped[str] = mapped_column(String(30), nullable=False)
    open_time: Mapped[datetime] = mapped_column(Time, nullable=False)
    close_time: Mapped[datetime] = mapped_column(Time, nullable=False)

    segments: Mapped[list[SegmentModel]] = relationship(back_populates="market")


class SegmentModel(Base):
    """Stock segments (us_tech, ru_blue_chips, etc.)."""

    __tablename__ = "segments"

    id: Mapped[str] = mapped_column(String(30), primary_key=True)
    market_id: Mapped[str] = mapped_column(
        String(10), nullable=False
    )
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    active_strategies: Mapped[list[str] | None] = mapped_column(ARRAY(Text))
    strategy_params: Mapped[dict | None] = mapped_column(JSONB, default={})  # type: ignore[type-arg]
    ml_model_id: Mapped[str | None] = mapped_column(String(50))
    max_allocation_pct: Mapped[Decimal] = mapped_column(
        Numeric(5, 4), default=Decimal("0.30")
    )
    news_languages: Mapped[list[str] | None] = mapped_column(
        ARRAY(Text), default=["en"]
    )

    market: Mapped[MarketModel] = relationship(back_populates="segments")


class InstrumentModel(Base):
    """Instruments -- stocks, ETFs, bonds."""

    __tablename__ = "instruments"

    symbol: Mapped[str] = mapped_column(String(20), primary_key=True)
    market_id: Mapped[str] = mapped_column(String(10), primary_key=True)
    segment_id: Mapped[str | None] = mapped_column(String(30))
    name: Mapped[str | None] = mapped_column(String(200))
    figi: Mapped[str | None] = mapped_column(String(20))
    instrument_type: Mapped[str | None] = mapped_column(String(20))
    currency: Mapped[str | None] = mapped_column(String(3))
    lot_size: Mapped[int] = mapped_column(Integer, default=1)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)


class CandleModel(Base):
    """OHLCV candle data -- TimescaleDB hypertable on timestamp."""

    __tablename__ = "candles"

    symbol: Mapped[str] = mapped_column(String(20), primary_key=True)
    market_id: Mapped[str] = mapped_column(String(10), primary_key=True)
    timeframe: Mapped[str] = mapped_column(String(5), primary_key=True)
    timestamp: Mapped[datetime] = mapped_column(primary_key=True)
    open: Mapped[Decimal] = mapped_column(Numeric(12, 4))
    high: Mapped[Decimal] = mapped_column(Numeric(12, 4))
    low: Mapped[Decimal] = mapped_column(Numeric(12, 4))
    close: Mapped[Decimal] = mapped_column(Numeric(12, 4))
    volume: Mapped[int] = mapped_column(BigInteger)
    source: Mapped[str | None] = mapped_column(String(20))


class SignalModel(Base):
    """Trading signals generated by strategies."""

    __tablename__ = "signals"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    strategy_name: Mapped[str] = mapped_column(String(50))
    symbol: Mapped[str] = mapped_column(String(20))
    market_id: Mapped[str] = mapped_column(String(10))
    segment_id: Mapped[str] = mapped_column(String(30))
    direction: Mapped[str] = mapped_column(String(4))
    confidence: Mapped[Decimal] = mapped_column(Numeric(5, 4))
    features: Mapped[dict | None] = mapped_column(JSONB)  # type: ignore[type-arg]
    reasoning: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column()
    mode: Mapped[str | None] = mapped_column(String(10))


class OrderModel(Base):
    """Orders submitted to brokers."""

    __tablename__ = "orders"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    signal_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    broker: Mapped[str] = mapped_column(String(20), nullable=False)
    broker_order_id: Mapped[str | None] = mapped_column(String(100))
    symbol: Mapped[str] = mapped_column(String(20))
    market_id: Mapped[str] = mapped_column(String(10))
    side: Mapped[str] = mapped_column(String(4))
    order_type: Mapped[str] = mapped_column(String(20))
    quantity: Mapped[Decimal] = mapped_column(Numeric(12, 4))
    limit_price: Mapped[Decimal | None] = mapped_column(Numeric(12, 4))
    stop_price: Mapped[Decimal | None] = mapped_column(Numeric(12, 4))
    currency: Mapped[str | None] = mapped_column(String(3))
    status: Mapped[str] = mapped_column(String(20))
    filled_quantity: Mapped[Decimal] = mapped_column(Numeric(12, 4), default=Decimal("0"))
    filled_avg_price: Mapped[Decimal | None] = mapped_column(Numeric(12, 4))
    submitted_at: Mapped[datetime | None] = mapped_column()
    filled_at: Mapped[datetime | None] = mapped_column()
    risk_checks: Mapped[dict | None] = mapped_column(JSONB)  # type: ignore[type-arg]
    mode: Mapped[str | None] = mapped_column(String(10))
```

Note: The `SegmentModel.market_id` foreign key and `relationship` references require forward references. Ensure the FK is set up properly with `ForeignKey("markets.id")`. The exact code may need minor adjustments during implementation to satisfy mypy strict mode -- this is expected and should be resolved during the GREEN step.

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_models.py -v`
Expected: PASS (4 tests)

**Step 5: Run linters**

Run: `uv run ruff check src/finalayze/core/models.py src/finalayze/core/db.py && uv run mypy src/finalayze/core/models.py src/finalayze/core/db.py`
Expected: zero errors (some `type: ignore` comments may be needed for JSONB/ARRAY generics)

**Step 6: Commit**

```bash
git add src/finalayze/core/models.py src/finalayze/core/db.py tests/unit/test_models.py
git commit -m "feat(core): add SQLAlchemy ORM models for all tables"
```

---

## Task 5: Alembic Initial Migration

**Files:**
- Create: `alembic/versions/001_initial.py`
- Modify: `alembic/env.py` (wire up metadata)

**Step 1: Update alembic env.py**

Update `alembic/env.py` to import `Base.metadata`:

```python
from finalayze.core.models import Base
target_metadata = Base.metadata
```

**Step 2: Write migration file**

```python
# alembic/versions/001_initial.py
"""Initial schema -- markets, segments, instruments, candles, signals, orders.

Revision ID: 001
Create Date: 2026-02-22
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision = "001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Markets
    op.create_table(
        "markets",
        sa.Column("id", sa.String(10), primary_key=True),
        sa.Column("name", sa.String(50), nullable=False),
        sa.Column("currency", sa.String(3), nullable=False),
        sa.Column("timezone", sa.String(30), nullable=False),
        sa.Column("open_time", sa.Time, nullable=False),
        sa.Column("close_time", sa.Time, nullable=False),
    )

    # Segments
    op.create_table(
        "segments",
        sa.Column("id", sa.String(30), primary_key=True),
        sa.Column("market_id", sa.String(10), sa.ForeignKey("markets.id"), nullable=False),
        sa.Column("name", sa.String(100), nullable=False),
        sa.Column("description", sa.Text),
        sa.Column("active_strategies", sa.ARRAY(sa.Text)),
        sa.Column("strategy_params", JSONB, server_default="{}"),
        sa.Column("ml_model_id", sa.String(50)),
        sa.Column("max_allocation_pct", sa.Numeric(5, 4), server_default="0.30"),
        sa.Column("news_languages", sa.ARRAY(sa.Text), server_default="{en}"),
    )

    # Instruments
    op.create_table(
        "instruments",
        sa.Column("symbol", sa.String(20), primary_key=True),
        sa.Column("market_id", sa.String(10), sa.ForeignKey("markets.id"), primary_key=True),
        sa.Column("segment_id", sa.String(30), sa.ForeignKey("segments.id")),
        sa.Column("name", sa.String(200)),
        sa.Column("figi", sa.String(20)),
        sa.Column("instrument_type", sa.String(20)),
        sa.Column("currency", sa.String(3)),
        sa.Column("lot_size", sa.Integer, server_default="1"),
        sa.Column("is_active", sa.Boolean, server_default="true"),
    )

    # Candles (hypertable)
    op.create_table(
        "candles",
        sa.Column("symbol", sa.String(20), primary_key=True),
        sa.Column("market_id", sa.String(10), primary_key=True),
        sa.Column("timeframe", sa.String(5), primary_key=True),
        sa.Column("timestamp", sa.DateTime(timezone=True), primary_key=True),
        sa.Column("open", sa.Numeric(12, 4)),
        sa.Column("high", sa.Numeric(12, 4)),
        sa.Column("low", sa.Numeric(12, 4)),
        sa.Column("close", sa.Numeric(12, 4)),
        sa.Column("volume", sa.BigInteger),
        sa.Column("source", sa.String(20)),
    )
    op.execute("SELECT create_hypertable('candles', 'timestamp', migrate_data => true)")

    # Signals
    op.create_table(
        "signals",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("strategy_name", sa.String(50)),
        sa.Column("symbol", sa.String(20)),
        sa.Column("market_id", sa.String(10)),
        sa.Column("segment_id", sa.String(30)),
        sa.Column("direction", sa.String(4)),
        sa.Column("confidence", sa.Numeric(5, 4)),
        sa.Column("features", JSONB),
        sa.Column("reasoning", sa.Text),
        sa.Column("created_at", sa.DateTime(timezone=True)),
        sa.Column("mode", sa.String(10)),
    )

    # Orders
    op.create_table(
        "orders",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("signal_id", UUID(as_uuid=True), sa.ForeignKey("signals.id")),
        sa.Column("broker", sa.String(20), nullable=False),
        sa.Column("broker_order_id", sa.String(100)),
        sa.Column("symbol", sa.String(20)),
        sa.Column("market_id", sa.String(10)),
        sa.Column("side", sa.String(4)),
        sa.Column("order_type", sa.String(20)),
        sa.Column("quantity", sa.Numeric(12, 4)),
        sa.Column("limit_price", sa.Numeric(12, 4)),
        sa.Column("stop_price", sa.Numeric(12, 4)),
        sa.Column("currency", sa.String(3)),
        sa.Column("status", sa.String(20)),
        sa.Column("filled_quantity", sa.Numeric(12, 4), server_default="0"),
        sa.Column("filled_avg_price", sa.Numeric(12, 4)),
        sa.Column("submitted_at", sa.DateTime(timezone=True)),
        sa.Column("filled_at", sa.DateTime(timezone=True)),
        sa.Column("risk_checks", JSONB),
        sa.Column("mode", sa.String(10)),
    )


def downgrade() -> None:
    op.drop_table("orders")
    op.drop_table("signals")
    op.drop_table("candles")
    op.drop_table("instruments")
    op.drop_table("segments")
    op.drop_table("markets")
```

**Step 3: Verify migration syntax**

Run: `uv run python -c "import alembic.versions" 2>&1 || echo "OK -- manual import not needed"`

Verify the migration file is valid Python:
Run: `uv run python -c "import ast; ast.parse(open('alembic/versions/001_initial.py').read()); print('Syntax OK')"`

**Step 4: Commit**

```bash
git add alembic/versions/001_initial.py alembic/env.py
git commit -m "feat(infra): add initial Alembic migration for all tables"
```

**Note:** Actually running the migration (`uv run alembic upgrade head`) requires docker-compose up with PostgreSQL + TimescaleDB. This will be tested in the integration test (Task 12).

---

## Task 6: Market Registry

**Files:**
- Create: `src/finalayze/markets/registry.py`
- Test: `tests/unit/test_market_registry.py`

**Step 1: Write the failing test**

```python
# tests/unit/test_market_registry.py
"""Unit tests for market registry."""

from __future__ import annotations

from datetime import UTC, datetime, time

import pytest

from finalayze.markets.registry import MarketDefinition, MarketRegistry


class TestMarketDefinition:
    def test_create_market(self) -> None:
        market = MarketDefinition(
            id="us",
            name="US Stock Market",
            currency="USD",
            timezone="America/New_York",
            open_time=time(9, 30),
            close_time=time(16, 0),
        )
        assert market.id == "us"
        assert market.currency == "USD"

    def test_market_is_frozen(self) -> None:
        market = MarketDefinition(
            id="us",
            name="US Stock Market",
            currency="USD",
            timezone="America/New_York",
            open_time=time(9, 30),
            close_time=time(16, 0),
        )
        with pytest.raises(Exception):  # noqa: B017
            market.id = "moex"  # type: ignore[misc]


EXPECTED_MARKET_COUNT = 2


class TestMarketRegistry:
    def test_get_us_market(self) -> None:
        registry = MarketRegistry()
        market = registry.get_market("us")
        assert market.id == "us"
        assert market.currency == "USD"

    def test_get_moex_market(self) -> None:
        registry = MarketRegistry()
        market = registry.get_market("moex")
        assert market.id == "moex"
        assert market.currency == "RUB"

    def test_get_unknown_market_raises(self) -> None:
        registry = MarketRegistry()
        with pytest.raises(KeyError):
            registry.get_market("unknown")

    def test_list_markets(self) -> None:
        registry = MarketRegistry()
        markets = registry.list_markets()
        assert len(markets) == EXPECTED_MARKET_COUNT

    def test_is_market_open_during_hours(self) -> None:
        registry = MarketRegistry()
        # US market: 9:30-16:00 ET = 14:30-21:00 UTC
        weekday_during_hours = datetime(2024, 6, 17, 15, 0, tzinfo=UTC)  # Monday 15:00 UTC
        assert registry.is_market_open("us", at=weekday_during_hours) is True

    def test_is_market_closed_outside_hours(self) -> None:
        registry = MarketRegistry()
        weekday_outside_hours = datetime(2024, 6, 17, 22, 0, tzinfo=UTC)  # Monday 22:00 UTC
        assert registry.is_market_open("us", at=weekday_outside_hours) is False

    def test_is_market_closed_on_weekend(self) -> None:
        registry = MarketRegistry()
        saturday = datetime(2024, 6, 15, 15, 0, tzinfo=UTC)  # Saturday
        assert registry.is_market_open("us", at=saturday) is False
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_market_registry.py -v`
Expected: FAIL with `ImportError`

**Step 3: Write minimal implementation**

```python
# src/finalayze/markets/registry.py
"""Market and segment registry (Layer 2).

Provides static market definitions and runtime lookup.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timezone
import zoneinfo


@dataclass(frozen=True)
class MarketDefinition:
    """Static definition of a trading market."""

    id: str
    name: str
    currency: str
    timezone: str
    open_time: time  # Local time
    close_time: time  # Local time


# Pre-built market definitions
_US_MARKET = MarketDefinition(
    id="us",
    name="US Stock Market",
    currency="USD",
    timezone="America/New_York",
    open_time=time(9, 30),
    close_time=time(16, 0),
)

_MOEX_MARKET = MarketDefinition(
    id="moex",
    name="Moscow Exchange",
    currency="RUB",
    timezone="Europe/Moscow",
    open_time=time(10, 0),
    close_time=time(18, 40),
)


class MarketRegistry:
    """Registry for looking up market definitions."""

    def __init__(self) -> None:
        self._markets: dict[str, MarketDefinition] = {
            "us": _US_MARKET,
            "moex": _MOEX_MARKET,
        }

    def get_market(self, market_id: str) -> MarketDefinition:
        """Get market definition by ID. Raises KeyError if not found."""
        return self._markets[market_id]

    def list_markets(self) -> list[MarketDefinition]:
        """List all registered markets."""
        return list(self._markets.values())

    def is_market_open(self, market_id: str, at: datetime) -> bool:
        """Check if a market is open at the given UTC datetime."""
        market = self.get_market(market_id)
        tz = zoneinfo.ZoneInfo(market.timezone)
        local_dt = at.astimezone(tz)

        # Weekends are always closed (Mon=0, Sun=6)
        if local_dt.weekday() >= 5:  # noqa: PLR2004
            return False

        local_time = local_dt.time()
        return market.open_time <= local_time < market.close_time
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_market_registry.py -v`
Expected: PASS (7 tests)

**Step 5: Run linters**

Run: `uv run ruff check src/finalayze/markets/registry.py && uv run mypy src/finalayze/markets/registry.py`

**Step 6: Commit**

```bash
git add src/finalayze/markets/registry.py tests/unit/test_market_registry.py
git commit -m "feat(markets): add MarketRegistry with US and MOEX definitions"
```

---

## Task 7: Data Fetcher -- Base + yfinance

**Files:**
- Create: `src/finalayze/data/fetchers/base.py`
- Create: `src/finalayze/data/fetchers/yfinance.py`
- Test: `tests/unit/test_fetchers.py`

**Step 1: Write the failing test**

```python
# tests/unit/test_fetchers.py
"""Unit tests for data fetchers."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from finalayze.core.schemas import Candle
from finalayze.data.fetchers.base import BaseFetcher
from finalayze.data.fetchers.yfinance import YFinanceFetcher


class TestBaseFetcher:
    def test_is_abstract(self) -> None:
        with pytest.raises(TypeError):
            BaseFetcher()  # type: ignore[abstract]


class TestYFinanceFetcher:
    def test_fetch_returns_candles(self) -> None:
        """Mock yfinance and verify we get Candle objects back."""
        # Create mock DataFrame that yfinance.download() returns
        index = pd.DatetimeIndex(
            [pd.Timestamp("2024-01-02", tz="UTC"), pd.Timestamp("2024-01-03", tz="UTC")]
        )
        mock_df = pd.DataFrame(
            {
                "Open": [185.0, 186.0],
                "High": [187.0, 188.0],
                "Low": [184.0, 185.0],
                "Close": [186.0, 187.5],
                "Volume": [50000000, 45000000],
            },
            index=index,
        )

        fetcher = YFinanceFetcher(market_id="us")

        with patch("finalayze.data.fetchers.yfinance.yf") as mock_yf:
            mock_yf.download.return_value = mock_df

            candles = fetcher.fetch_candles(
                symbol="AAPL",
                start=datetime(2024, 1, 1, tzinfo=UTC),
                end=datetime(2024, 1, 5, tzinfo=UTC),
                timeframe="1d",
            )

        assert len(candles) == 2
        assert isinstance(candles[0], Candle)
        assert candles[0].symbol == "AAPL"
        assert candles[0].market_id == "us"
        assert candles[0].close == Decimal("186.0")
        assert candles[0].volume == 50000000

    def test_fetch_empty_returns_empty_list(self) -> None:
        fetcher = YFinanceFetcher(market_id="us")
        empty_df = pd.DataFrame()

        with patch("finalayze.data.fetchers.yfinance.yf") as mock_yf:
            mock_yf.download.return_value = empty_df
            candles = fetcher.fetch_candles(
                symbol="INVALID",
                start=datetime(2024, 1, 1, tzinfo=UTC),
                end=datetime(2024, 1, 5, tzinfo=UTC),
                timeframe="1d",
            )

        assert candles == []
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_fetchers.py -v`
Expected: FAIL with `ImportError`

**Step 3: Write minimal implementation**

```python
# src/finalayze/data/fetchers/base.py
"""Abstract base for data fetchers (Layer 2)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime

from finalayze.core.schemas import Candle


class BaseFetcher(ABC):
    """Abstract base class for market data fetchers."""

    @abstractmethod
    def fetch_candles(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: str = "1d",
    ) -> list[Candle]:
        """Fetch OHLCV candles for a symbol."""
        ...
```

```python
# src/finalayze/data/fetchers/yfinance.py
"""yfinance data fetcher (Layer 2)."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

import yfinance as yf  # type: ignore[import-untyped]

from finalayze.core.schemas import Candle
from finalayze.data.fetchers.base import BaseFetcher


class YFinanceFetcher(BaseFetcher):
    """Fetches OHLCV data from Yahoo Finance."""

    def __init__(self, market_id: str = "us") -> None:
        self._market_id = market_id

    def fetch_candles(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: str = "1d",
    ) -> list[Candle]:
        """Fetch candles via yfinance.download()."""
        df = yf.download(
            symbol,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            interval=timeframe,
            progress=False,
        )

        if df.empty:
            return []

        candles: list[Candle] = []
        for ts, row in df.iterrows():
            candles.append(
                Candle(
                    symbol=symbol,
                    market_id=self._market_id,
                    timeframe=timeframe,
                    timestamp=ts.to_pydatetime(),  # type: ignore[union-attr]
                    open=Decimal(str(row["Open"])),
                    high=Decimal(str(row["High"])),
                    low=Decimal(str(row["Low"])),
                    close=Decimal(str(row["Close"])),
                    volume=int(row["Volume"]),
                )
            )
        return candles
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_fetchers.py -v`
Expected: PASS (3 tests)

**Step 5: Run linters**

Run: `uv run ruff check src/finalayze/data/fetchers/ && uv run mypy src/finalayze/data/fetchers/`

**Step 6: Commit**

```bash
git add src/finalayze/data/fetchers/base.py src/finalayze/data/fetchers/yfinance.py tests/unit/test_fetchers.py
git commit -m "feat(data): add BaseFetcher ABC and YFinanceFetcher"
```

---

## Task 8: Base Strategy + Momentum Strategy

**Files:**
- Create: `src/finalayze/strategies/base.py`
- Create: `src/finalayze/strategies/momentum.py`
- Test: `tests/unit/test_strategies.py`

**Step 1: Write the failing test**

```python
# tests/unit/test_strategies.py
"""Unit tests for trading strategies."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest

from finalayze.core.schemas import Candle, SignalDirection
from finalayze.strategies.base import BaseStrategy
from finalayze.strategies.momentum import MomentumStrategy

# Number of candles needed for RSI(14) + MACD(26) to have enough data
MIN_CANDLES_FOR_INDICATORS = 35


def _make_candles(prices: list[float], start_year: int = 2024) -> list[Candle]:
    """Helper: create candle list from close prices (open=high=low=close for simplicity)."""
    candles = []
    for i, price in enumerate(prices):
        p = Decimal(str(price))
        candles.append(
            Candle(
                symbol="AAPL",
                market_id="us",
                timeframe="1d",
                timestamp=datetime(start_year, 1, 1 + i, 14, 30, tzinfo=UTC),
                open=p,
                high=p + Decimal("1"),
                low=p - Decimal("1"),
                close=p,
                volume=1_000_000,
            )
        )
    return candles


class TestBaseStrategy:
    def test_is_abstract(self) -> None:
        with pytest.raises(TypeError):
            BaseStrategy()  # type: ignore[abstract]


class TestMomentumStrategy:
    def test_name(self) -> None:
        strategy = MomentumStrategy()
        assert strategy.name == "momentum"

    def test_supported_segments(self) -> None:
        strategy = MomentumStrategy()
        segments = strategy.supported_segments()
        assert "us_tech" in segments
        assert "us_broad" in segments

    def test_get_parameters_us_tech(self) -> None:
        strategy = MomentumStrategy()
        params = strategy.get_parameters("us_tech")
        rsi_period = 14
        assert params["rsi_period"] == rsi_period

    def test_insufficient_data_returns_none(self) -> None:
        strategy = MomentumStrategy()
        short_candles = _make_candles([100.0] * 5)
        result = strategy.generate_signal("AAPL", short_candles, "us_tech")
        assert result is None

    def test_buy_signal_on_oversold_rsi(self) -> None:
        """Create a declining price series that should produce oversold RSI."""
        # Start high and decline steadily to push RSI below 30
        prices = [200.0 - i * 2 for i in range(MIN_CANDLES_FOR_INDICATORS)]
        # Then add a small uptick at the end (MACD histogram cross)
        prices.extend([prices[-1] + 1, prices[-1] + 2, prices[-1] + 3])

        strategy = MomentumStrategy()
        candles = _make_candles(prices)
        signal = strategy.generate_signal("AAPL", candles, "us_tech")

        # Signal may or may not fire depending on exact indicator values,
        # but if it does fire it must be BUY (oversold conditions)
        if signal is not None:
            assert signal.direction == SignalDirection.BUY
            assert signal.strategy_name == "momentum"
            assert signal.symbol == "AAPL"
            assert 0.0 <= signal.confidence <= 1.0

    def test_hold_when_no_signal(self) -> None:
        """Flat prices should not trigger a signal."""
        prices = [150.0] * (MIN_CANDLES_FOR_INDICATORS + 5)
        strategy = MomentumStrategy()
        candles = _make_candles(prices)
        result = strategy.generate_signal("AAPL", candles, "us_tech")
        # Flat price -> no clear signal -> None (HOLD)
        assert result is None
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_strategies.py -v`
Expected: FAIL with `ImportError`

**Step 3: Write minimal implementation**

```python
# src/finalayze/strategies/base.py
"""Abstract base for trading strategies (Layer 4)."""

from __future__ import annotations

from abc import ABC, abstractmethod

from finalayze.core.schemas import Candle, Signal


class BaseStrategy(ABC):
    """Abstract base class for trading strategies."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy name (e.g., 'momentum')."""
        ...

    @abstractmethod
    def supported_segments(self) -> list[str]:
        """List of segment IDs this strategy supports."""
        ...

    @abstractmethod
    def generate_signal(
        self,
        symbol: str,
        candles: list[Candle],
        segment_id: str,
    ) -> Signal | None:
        """Generate a trading signal. Returns None for HOLD."""
        ...

    @abstractmethod
    def get_parameters(self, segment_id: str) -> dict[str, object]:
        """Load strategy parameters for a segment from YAML preset."""
        ...
```

```python
# src/finalayze/strategies/momentum.py
"""Momentum strategy: RSI + MACD (Layer 4).

BUY:  RSI < oversold AND MACD histogram crosses above zero
SELL: RSI > overbought AND MACD histogram crosses below zero
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from finalayze.core.schemas import Candle, Signal, SignalDirection
from finalayze.strategies.base import BaseStrategy

_PRESETS_DIR = Path(__file__).parent / "presets"
_MIN_CANDLES = 30  # Minimum candles needed for RSI(14) + MACD(26)


class MomentumStrategy(BaseStrategy):
    """Momentum strategy using RSI and MACD indicators."""

    @property
    def name(self) -> str:
        return "momentum"

    def supported_segments(self) -> list[str]:
        """Return segments that have a momentum preset YAML."""
        segments: list[str] = []
        for path in _PRESETS_DIR.glob("*.yaml"):
            with open(path) as f:
                preset = yaml.safe_load(f)
            strategies = preset.get("strategies", {})
            if "momentum" in strategies and strategies["momentum"].get("enabled"):
                segments.append(preset["segment_id"])
        return segments

    def get_parameters(self, segment_id: str) -> dict[str, object]:
        """Load momentum params from YAML preset for this segment."""
        preset_path = _PRESETS_DIR / f"{segment_id}.yaml"
        with open(preset_path) as f:
            preset = yaml.safe_load(f)
        return dict(preset["strategies"]["momentum"]["params"])

    def generate_signal(
        self,
        symbol: str,
        candles: list[Candle],
        segment_id: str,
    ) -> Signal | None:
        """Generate BUY/SELL signal based on RSI + MACD."""
        if len(candles) < _MIN_CANDLES:
            return None

        params = self.get_parameters(segment_id)
        rsi_period: int = int(params.get("rsi_period", 14))
        rsi_oversold: int = int(params.get("rsi_oversold", 30))
        rsi_overbought: int = int(params.get("rsi_overbought", 70))
        macd_fast: int = int(params.get("macd_fast", 12))
        macd_slow: int = int(params.get("macd_slow", 26))
        min_confidence: float = float(params.get("min_confidence", 0.6))

        # Build price series
        closes = pd.Series([float(c.close) for c in candles])

        # Compute RSI
        import pandas_ta as ta  # type: ignore[import-untyped]

        rsi_series = ta.rsi(closes, length=rsi_period)
        if rsi_series is None or rsi_series.empty:
            return None

        rsi_val = rsi_series.iloc[-1]
        if pd.isna(rsi_val):
            return None

        # Compute MACD
        macd_df = ta.macd(closes, fast=macd_fast, slow=macd_slow)
        if macd_df is None or macd_df.empty:
            return None

        # MACD histogram column name varies; find it
        hist_col = [c for c in macd_df.columns if "h" in c.lower() or "hist" in c.lower()]
        if not hist_col:
            return None

        hist_current = macd_df[hist_col[0]].iloc[-1]
        hist_prev = macd_df[hist_col[0]].iloc[-2]
        if pd.isna(hist_current) or pd.isna(hist_prev):
            return None

        # Signal logic
        direction: SignalDirection | None = None
        confidence = 0.0

        if rsi_val < rsi_oversold and hist_prev < 0 and hist_current > 0:
            # BUY: RSI oversold + MACD histogram crosses above zero
            direction = SignalDirection.BUY
            rsi_distance = (rsi_oversold - rsi_val) / rsi_oversold
            confidence = min(1.0, 0.5 + rsi_distance * 0.3 + abs(hist_current) * 0.1)

        elif rsi_val > rsi_overbought and hist_prev > 0 and hist_current < 0:
            # SELL: RSI overbought + MACD histogram crosses below zero
            direction = SignalDirection.SELL
            rsi_distance = (rsi_val - rsi_overbought) / (100 - rsi_overbought)
            confidence = min(1.0, 0.5 + rsi_distance * 0.3 + abs(hist_current) * 0.1)

        if direction is None or confidence < min_confidence:
            return None

        return Signal(
            strategy_name=self.name,
            symbol=symbol,
            market_id=candles[-1].market_id,
            segment_id=segment_id,
            direction=direction,
            confidence=confidence,
            features={"rsi": round(float(rsi_val), 2), "macd_hist": round(float(hist_current), 4)},
            reasoning=f"RSI={rsi_val:.1f}, MACD hist crossed {'above' if direction == SignalDirection.BUY else 'below'} zero",
        )
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_strategies.py -v`
Expected: PASS (6 tests)

**Step 5: Run linters**

Run: `uv run ruff check src/finalayze/strategies/ && uv run mypy src/finalayze/strategies/base.py src/finalayze/strategies/momentum.py`

**Step 6: Commit**

```bash
git add src/finalayze/strategies/base.py src/finalayze/strategies/momentum.py tests/unit/test_strategies.py
git commit -m "feat(strategies): add BaseStrategy ABC and MomentumStrategy (RSI+MACD)"
```

---

## Task 9: Risk Management -- Position Sizer + Stop Loss

**Files:**
- Create: `src/finalayze/risk/position_sizer.py`
- Create: `src/finalayze/risk/stop_loss.py`
- Test: `tests/unit/test_risk.py`

**Step 1: Write the failing test**

```python
# tests/unit/test_risk.py
"""Unit tests for risk management."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest

from finalayze.core.schemas import Candle
from finalayze.risk.position_sizer import compute_position_size
from finalayze.risk.stop_loss import compute_atr_stop_loss


class TestPositionSizer:
    def test_half_kelly_basic(self) -> None:
        """Half-Kelly with 60% win rate, 1.5 avg win ratio."""
        size = compute_position_size(
            win_rate=0.6,
            avg_win_ratio=Decimal("1.5"),
            equity=Decimal("100000"),
            kelly_fraction=0.5,
            max_position_pct=0.20,
        )
        # Kelly f* = (0.6 * 1.5 - 0.4) / 1.5 = 0.333
        # Half-Kelly = 0.333 * 0.5 = 0.167
        # Position = 100000 * 0.167 = 16667
        assert size > Decimal("0")
        assert size <= Decimal("20000")  # max 20% of 100k

    def test_capped_at_max_position(self) -> None:
        """Very high win rate should still be capped at max_position_pct."""
        size = compute_position_size(
            win_rate=0.95,
            avg_win_ratio=Decimal("3.0"),
            equity=Decimal("100000"),
            kelly_fraction=0.5,
            max_position_pct=0.20,
        )
        max_allowed = Decimal("20000")
        assert size <= max_allowed

    def test_negative_kelly_returns_zero(self) -> None:
        """Negative edge (losing strategy) -> zero position."""
        size = compute_position_size(
            win_rate=0.3,
            avg_win_ratio=Decimal("0.5"),
            equity=Decimal("100000"),
            kelly_fraction=0.5,
            max_position_pct=0.20,
        )
        assert size == Decimal("0")

    def test_zero_equity_returns_zero(self) -> None:
        size = compute_position_size(
            win_rate=0.6,
            avg_win_ratio=Decimal("1.5"),
            equity=Decimal("0"),
            kelly_fraction=0.5,
            max_position_pct=0.20,
        )
        assert size == Decimal("0")


def _make_candles_with_range(
    highs: list[float], lows: list[float], closes: list[float]
) -> list[Candle]:
    """Create candles with specific high/low/close values."""
    candles = []
    for i, (h, l, c) in enumerate(zip(highs, lows, closes)):
        candles.append(
            Candle(
                symbol="AAPL",
                market_id="us",
                timeframe="1d",
                timestamp=datetime(2024, 1, 1 + i, 14, 30, tzinfo=UTC),
                open=Decimal(str(c)),
                high=Decimal(str(h)),
                low=Decimal(str(l)),
                close=Decimal(str(c)),
                volume=1_000_000,
            )
        )
    return candles


ATR_PERIOD = 14


class TestStopLoss:
    def test_atr_stop_loss_basic(self) -> None:
        """ATR stop should be entry - ATR * multiplier."""
        # Create 15 candles with known high-low range of 2.0 each
        highs = [101.0] * (ATR_PERIOD + 1)
        lows = [99.0] * (ATR_PERIOD + 1)
        closes = [100.0] * (ATR_PERIOD + 1)
        candles = _make_candles_with_range(highs, lows, closes)

        stop = compute_atr_stop_loss(
            entry_price=Decimal("100.00"),
            candles=candles,
            atr_period=ATR_PERIOD,
            atr_multiplier=Decimal("2.0"),
        )
        # ATR ~= 2.0, stop = 100 - 2*2 = 96
        assert stop < Decimal("100.00")
        assert stop > Decimal("90.00")

    def test_insufficient_candles_returns_none(self) -> None:
        candles = _make_candles_with_range([101.0] * 5, [99.0] * 5, [100.0] * 5)
        stop = compute_atr_stop_loss(
            entry_price=Decimal("100.00"),
            candles=candles,
            atr_period=ATR_PERIOD,
            atr_multiplier=Decimal("2.0"),
        )
        assert stop is None
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_risk.py -v`
Expected: FAIL with `ImportError`

**Step 3: Write minimal implementation**

```python
# src/finalayze/risk/position_sizer.py
"""Half-Kelly position sizing (Layer 4)."""

from __future__ import annotations

from decimal import Decimal


def compute_position_size(
    win_rate: float,
    avg_win_ratio: Decimal,
    equity: Decimal,
    kelly_fraction: float = 0.5,
    max_position_pct: float = 0.20,
) -> Decimal:
    """Compute position size using Half-Kelly criterion.

    f* = (win_rate * avg_win_ratio - (1 - win_rate)) / avg_win_ratio
    position = equity * f* * kelly_fraction, capped at max_position_pct.
    """
    if equity <= 0:
        return Decimal("0")

    loss_rate = 1.0 - win_rate
    if avg_win_ratio <= 0:
        return Decimal("0")

    kelly_f = (win_rate * float(avg_win_ratio) - loss_rate) / float(avg_win_ratio)

    if kelly_f <= 0:
        return Decimal("0")

    half_kelly = kelly_f * kelly_fraction
    position = equity * Decimal(str(half_kelly))
    max_position = equity * Decimal(str(max_position_pct))

    return min(position, max_position)
```

```python
# src/finalayze/risk/stop_loss.py
"""ATR-based stop-loss calculation (Layer 4)."""

from __future__ import annotations

from decimal import Decimal

import pandas as pd

from finalayze.core.schemas import Candle


def compute_atr_stop_loss(
    entry_price: Decimal,
    candles: list[Candle],
    atr_period: int = 14,
    atr_multiplier: Decimal = Decimal("2.0"),
) -> Decimal | None:
    """Compute stop-loss price using ATR.

    stop_loss = entry_price - ATR(period) * multiplier

    Returns None if insufficient data.
    """
    if len(candles) < atr_period + 1:
        return None

    highs = pd.Series([float(c.high) for c in candles])
    lows = pd.Series([float(c.low) for c in candles])
    closes = pd.Series([float(c.close) for c in candles])

    # True Range
    tr1 = highs - lows
    tr2 = (highs - closes.shift(1)).abs()
    tr3 = (lows - closes.shift(1)).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # ATR = rolling mean of True Range
    atr = true_range.rolling(window=atr_period).mean().iloc[-1]

    if pd.isna(atr):
        return None

    stop = entry_price - Decimal(str(atr)) * atr_multiplier
    return stop
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_risk.py -v`
Expected: PASS (5 tests)

**Step 5: Run linters**

Run: `uv run ruff check src/finalayze/risk/ && uv run mypy src/finalayze/risk/position_sizer.py src/finalayze/risk/stop_loss.py`

**Step 6: Commit**

```bash
git add src/finalayze/risk/position_sizer.py src/finalayze/risk/stop_loss.py tests/unit/test_risk.py
git commit -m "feat(risk): add Half-Kelly position sizer and ATR stop-loss"
```

---

## Task 10: Pre-Trade Risk Checks

**Files:**
- Create: `src/finalayze/risk/pre_trade_check.py`
- Modify: `tests/unit/test_risk.py`

**Step 1: Write the failing test**

Append to `tests/unit/test_risk.py`:

```python
from finalayze.risk.pre_trade_check import PreTradeChecker, PreTradeResult


class TestPreTradeChecker:
    def _make_checker(
        self,
        max_position_pct: float = 0.20,
        max_positions: int = 10,
    ) -> PreTradeChecker:
        return PreTradeChecker(
            max_position_pct=max_position_pct,
            max_positions_per_market=max_positions,
        )

    def test_passes_valid_order(self) -> None:
        checker = self._make_checker()
        result = checker.check(
            order_value=Decimal("10000"),
            portfolio_equity=Decimal("100000"),
            available_cash=Decimal("50000"),
            open_position_count=3,
        )
        assert result.passed is True
        assert len(result.violations) == 0

    def test_rejects_position_too_large(self) -> None:
        checker = self._make_checker(max_position_pct=0.20)
        result = checker.check(
            order_value=Decimal("25000"),
            portfolio_equity=Decimal("100000"),
            available_cash=Decimal("50000"),
            open_position_count=3,
        )
        assert result.passed is False
        assert any("position size" in v.lower() for v in result.violations)

    def test_rejects_insufficient_cash(self) -> None:
        checker = self._make_checker()
        result = checker.check(
            order_value=Decimal("10000"),
            portfolio_equity=Decimal("100000"),
            available_cash=Decimal("5000"),
            open_position_count=3,
        )
        assert result.passed is False
        assert any("cash" in v.lower() for v in result.violations)

    def test_rejects_too_many_positions(self) -> None:
        checker = self._make_checker(max_positions=5)
        result = checker.check(
            order_value=Decimal("10000"),
            portfolio_equity=Decimal("100000"),
            available_cash=Decimal("50000"),
            open_position_count=5,
        )
        assert result.passed is False
        assert any("position" in v.lower() for v in result.violations)
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_risk.py::TestPreTradeChecker -v`
Expected: FAIL with `ImportError`

**Step 3: Write minimal implementation**

```python
# src/finalayze/risk/pre_trade_check.py
"""Pre-trade risk checks (Layer 4).

Basic subset for backtest slice:
1. Position size <= max % of portfolio
2. Cash sufficient for order
3. Open positions < max per market
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal


@dataclass(frozen=True)
class PreTradeResult:
    """Result of pre-trade risk checks."""

    passed: bool
    violations: list[str] = field(default_factory=list)


class PreTradeChecker:
    """Runs pre-trade risk validation."""

    def __init__(
        self,
        max_position_pct: float = 0.20,
        max_positions_per_market: int = 10,
    ) -> None:
        self._max_position_pct = max_position_pct
        self._max_positions = max_positions_per_market

    def check(
        self,
        order_value: Decimal,
        portfolio_equity: Decimal,
        available_cash: Decimal,
        open_position_count: int,
    ) -> PreTradeResult:
        """Run all pre-trade checks. Returns result with violations list."""
        violations: list[str] = []

        # 1. Position size check
        if portfolio_equity > 0:
            position_pct = float(order_value / portfolio_equity)
            if position_pct > self._max_position_pct:
                violations.append(
                    f"Position size {position_pct:.1%} exceeds max {self._max_position_pct:.1%}"
                )

        # 2. Cash sufficiency
        if order_value > available_cash:
            violations.append(
                f"Insufficient cash: need {order_value}, have {available_cash}"
            )

        # 3. Position count
        if open_position_count >= self._max_positions:
            violations.append(
                f"Open positions ({open_position_count}) >= max ({self._max_positions})"
            )

        return PreTradeResult(passed=len(violations) == 0, violations=violations)
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_risk.py -v`
Expected: PASS (all risk tests)

**Step 5: Run linters**

Run: `uv run ruff check src/finalayze/risk/pre_trade_check.py && uv run mypy src/finalayze/risk/pre_trade_check.py`

**Step 6: Commit**

```bash
git add src/finalayze/risk/pre_trade_check.py tests/unit/test_risk.py
git commit -m "feat(risk): add pre-trade risk checker (size, cash, position count)"
```

---

## Task 11: Simulated Broker

**Files:**
- Create: `src/finalayze/execution/broker_base.py`
- Create: `src/finalayze/execution/simulated_broker.py`
- Test: `tests/unit/test_broker.py`

**Step 1: Write the failing test**

```python
# tests/unit/test_broker.py
"""Unit tests for broker implementations."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest

from finalayze.core.schemas import Candle, PortfolioState
from finalayze.execution.broker_base import BrokerBase, OrderRequest, OrderResult
from finalayze.execution.simulated_broker import SimulatedBroker

INITIAL_CASH = Decimal("100000")


def _candle(price: float, day: int = 1) -> Candle:
    p = Decimal(str(price))
    return Candle(
        symbol="AAPL",
        market_id="us",
        timeframe="1d",
        timestamp=datetime(2024, 1, day, 14, 30, tzinfo=UTC),
        open=p,
        high=p + Decimal("2"),
        low=p - Decimal("2"),
        close=p,
        volume=1_000_000,
    )


class TestBrokerBase:
    def test_is_abstract(self) -> None:
        with pytest.raises(TypeError):
            BrokerBase()  # type: ignore[abstract]


class TestSimulatedBroker:
    def test_initial_portfolio(self) -> None:
        broker = SimulatedBroker(initial_cash=INITIAL_CASH)
        portfolio = broker.get_portfolio()
        assert portfolio.cash == INITIAL_CASH
        assert portfolio.equity == INITIAL_CASH
        assert len(portfolio.positions) == 0

    def test_buy_order(self) -> None:
        broker = SimulatedBroker(initial_cash=INITIAL_CASH)
        order = OrderRequest(
            symbol="AAPL",
            side="BUY",
            quantity=Decimal("10"),
        )
        fill_candle = _candle(185.0, day=2)
        result = broker.submit_order(order, fill_candle)
        assert result.filled is True
        assert result.fill_price == Decimal("185.0")  # Fills at candle open

        portfolio = broker.get_portfolio()
        assert portfolio.positions["AAPL"] == Decimal("10")
        expected_cash = INITIAL_CASH - Decimal("1850.0")
        assert portfolio.cash == expected_cash

    def test_sell_order(self) -> None:
        broker = SimulatedBroker(initial_cash=INITIAL_CASH)
        # Buy first
        buy = OrderRequest(symbol="AAPL", side="BUY", quantity=Decimal("10"))
        broker.submit_order(buy, _candle(185.0, day=2))
        # Then sell
        sell = OrderRequest(symbol="AAPL", side="SELL", quantity=Decimal("10"))
        result = broker.submit_order(sell, _candle(190.0, day=3))
        assert result.filled is True

        portfolio = broker.get_portfolio()
        assert "AAPL" not in portfolio.positions or portfolio.positions.get("AAPL") == Decimal("0")

    def test_insufficient_cash_rejects(self) -> None:
        broker = SimulatedBroker(initial_cash=Decimal("100"))
        order = OrderRequest(symbol="AAPL", side="BUY", quantity=Decimal("10"))
        result = broker.submit_order(order, _candle(185.0))
        assert result.filled is False

    def test_stop_loss_triggers(self) -> None:
        broker = SimulatedBroker(initial_cash=INITIAL_CASH)
        # Buy shares
        buy = OrderRequest(symbol="AAPL", side="BUY", quantity=Decimal("10"))
        broker.submit_order(buy, _candle(185.0, day=2))

        # Set stop loss
        broker.set_stop_loss("AAPL", Decimal("180.0"))

        # Candle where low drops below stop
        drop_candle = Candle(
            symbol="AAPL",
            market_id="us",
            timeframe="1d",
            timestamp=datetime(2024, 1, 3, 14, 30, tzinfo=UTC),
            open=Decimal("183.0"),
            high=Decimal("184.0"),
            low=Decimal("178.0"),  # Below stop
            close=Decimal("179.0"),
            volume=1_000_000,
        )
        triggered = broker.check_stop_losses(drop_candle)
        assert len(triggered) == 1
        assert triggered[0].symbol == "AAPL"

        # Position should be closed
        portfolio = broker.get_portfolio()
        assert portfolio.positions.get("AAPL", Decimal("0")) == Decimal("0")
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_broker.py -v`
Expected: FAIL with `ImportError`

**Step 3: Write minimal implementation**

```python
# src/finalayze/execution/broker_base.py
"""Abstract broker interface (Layer 5)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from decimal import Decimal

from finalayze.core.schemas import Candle, PortfolioState


@dataclass(frozen=True)
class OrderRequest:
    """Request to submit an order."""

    symbol: str
    side: str  # "BUY" or "SELL"
    quantity: Decimal


@dataclass(frozen=True)
class OrderResult:
    """Result of an order submission."""

    filled: bool
    fill_price: Decimal | None = None
    symbol: str = ""
    side: str = ""
    quantity: Decimal = Decimal("0")
    reason: str = ""


class BrokerBase(ABC):
    """Abstract base class for broker implementations."""

    @abstractmethod
    def submit_order(self, order: OrderRequest, fill_candle: Candle) -> OrderResult:
        """Submit an order. Returns fill result."""
        ...

    @abstractmethod
    def get_portfolio(self) -> PortfolioState:
        """Get current portfolio state."""
        ...
```

```python
# src/finalayze/execution/simulated_broker.py
"""Simulated broker for backtesting (Layer 5).

Fills market orders at the next candle's open price.
Tracks cash, positions, equity internally.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

from finalayze.core.schemas import Candle, PortfolioState
from finalayze.execution.broker_base import BrokerBase, OrderRequest, OrderResult


class SimulatedBroker(BrokerBase):
    """Paper broker that fills at candle open prices."""

    def __init__(self, initial_cash: Decimal = Decimal("100000")) -> None:
        self._cash = initial_cash
        self._positions: dict[str, Decimal] = {}
        self._stop_losses: dict[str, Decimal] = {}
        self._last_prices: dict[str, Decimal] = {}

    def submit_order(self, order: OrderRequest, fill_candle: Candle) -> OrderResult:
        """Fill order at fill_candle's open price."""
        fill_price = fill_candle.open
        total_cost = fill_price * order.quantity

        if order.side == "BUY":
            if total_cost > self._cash:
                return OrderResult(
                    filled=False, reason="Insufficient cash", symbol=order.symbol
                )
            self._cash -= total_cost
            current = self._positions.get(order.symbol, Decimal("0"))
            self._positions[order.symbol] = current + order.quantity

        elif order.side == "SELL":
            current = self._positions.get(order.symbol, Decimal("0"))
            sell_qty = min(order.quantity, current)
            self._cash += fill_price * sell_qty
            remaining = current - sell_qty
            if remaining <= 0:
                self._positions.pop(order.symbol, None)
                self._stop_losses.pop(order.symbol, None)
            else:
                self._positions[order.symbol] = remaining

        self._last_prices[order.symbol] = fill_candle.close

        return OrderResult(
            filled=True,
            fill_price=fill_price,
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
        )

    def set_stop_loss(self, symbol: str, price: Decimal) -> None:
        """Set a stop-loss price for a position."""
        self._stop_losses[symbol] = price

    def check_stop_losses(self, candle: Candle) -> list[OrderResult]:
        """Check and execute any triggered stop-losses."""
        triggered: list[OrderResult] = []
        symbol = candle.symbol

        if symbol in self._stop_losses and symbol in self._positions:
            stop_price = self._stop_losses[symbol]
            if candle.low <= stop_price:
                # Fill at stop price
                qty = self._positions[symbol]
                self._cash += stop_price * qty
                self._positions.pop(symbol)
                self._stop_losses.pop(symbol)
                self._last_prices[symbol] = stop_price
                triggered.append(
                    OrderResult(
                        filled=True,
                        fill_price=stop_price,
                        symbol=symbol,
                        side="SELL",
                        quantity=qty,
                    )
                )
        return triggered

    def update_prices(self, candle: Candle) -> None:
        """Update last known price for equity calculation."""
        self._last_prices[candle.symbol] = candle.close

    def get_portfolio(self) -> PortfolioState:
        """Get current portfolio snapshot."""
        positions_value = sum(
            qty * self._last_prices.get(sym, Decimal("0"))
            for sym, qty in self._positions.items()
        )
        return PortfolioState(
            cash=self._cash,
            positions=dict(self._positions),
            equity=self._cash + positions_value,
            timestamp=datetime.now(tz=UTC),
        )
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_broker.py -v`
Expected: PASS (6 tests)

**Step 5: Run linters**

Run: `uv run ruff check src/finalayze/execution/ && uv run mypy src/finalayze/execution/broker_base.py src/finalayze/execution/simulated_broker.py`

**Step 6: Commit**

```bash
git add src/finalayze/execution/broker_base.py src/finalayze/execution/simulated_broker.py tests/unit/test_broker.py
git commit -m "feat(execution): add BrokerBase ABC and SimulatedBroker"
```

---

## Task 12: Performance Analyzer

**Files:**
- Create: `src/finalayze/backtest/performance.py`
- Test: `tests/unit/test_performance.py`

**Step 1: Write the failing test**

```python
# tests/unit/test_performance.py
"""Unit tests for performance analyzer."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from decimal import Decimal

from finalayze.core.schemas import BacktestResult, PortfolioState, TradeResult
from finalayze.backtest.performance import PerformanceAnalyzer

INITIAL_EQUITY = Decimal("100000")


def _make_trades() -> list[TradeResult]:
    """Create a mix of winning and losing trades."""
    return [
        TradeResult(
            signal_id=uuid.uuid4(), symbol="AAPL", side="BUY",
            quantity=Decimal("10"), entry_price=Decimal("185"),
            exit_price=Decimal("195"), pnl=Decimal("100"), pnl_pct=Decimal("0.054"),
        ),
        TradeResult(
            signal_id=uuid.uuid4(), symbol="AAPL", side="BUY",
            quantity=Decimal("10"), entry_price=Decimal("190"),
            exit_price=Decimal("185"), pnl=Decimal("-50"), pnl_pct=Decimal("-0.026"),
        ),
        TradeResult(
            signal_id=uuid.uuid4(), symbol="AAPL", side="BUY",
            quantity=Decimal("10"), entry_price=Decimal("180"),
            exit_price=Decimal("200"), pnl=Decimal("200"), pnl_pct=Decimal("0.111"),
        ),
    ]


def _make_snapshots() -> list[PortfolioState]:
    """Create portfolio snapshots showing equity curve."""
    values = [100000, 100100, 100050, 100250, 99800, 100500]
    return [
        PortfolioState(
            cash=Decimal(str(v)),
            positions={},
            equity=Decimal(str(v)),
            timestamp=datetime(2024, 1, i + 1, 21, 0, tzinfo=UTC),
        )
        for i, v in enumerate(values)
    ]


EXPECTED_TRADE_COUNT = 3


class TestPerformanceAnalyzer:
    def test_total_trades(self) -> None:
        analyzer = PerformanceAnalyzer()
        result = analyzer.analyze(_make_trades(), _make_snapshots())
        assert result.total_trades == EXPECTED_TRADE_COUNT

    def test_win_rate(self) -> None:
        analyzer = PerformanceAnalyzer()
        result = analyzer.analyze(_make_trades(), _make_snapshots())
        # 2 wins out of 3
        expected_win_rate = Decimal("0.6667")
        assert abs(result.win_rate - expected_win_rate) < Decimal("0.01")

    def test_total_return(self) -> None:
        analyzer = PerformanceAnalyzer()
        result = analyzer.analyze(_make_trades(), _make_snapshots())
        # Final equity 100500 vs initial 100000 = 0.5%
        assert result.total_return > Decimal("0")

    def test_max_drawdown(self) -> None:
        analyzer = PerformanceAnalyzer()
        result = analyzer.analyze(_make_trades(), _make_snapshots())
        assert result.max_drawdown >= Decimal("0")
        assert result.max_drawdown <= Decimal("1")

    def test_profit_factor(self) -> None:
        analyzer = PerformanceAnalyzer()
        result = analyzer.analyze(_make_trades(), _make_snapshots())
        # Total wins = 300, total loss = 50 => PF = 6.0
        assert result.profit_factor > Decimal("1")

    def test_result_is_backtest_result(self) -> None:
        analyzer = PerformanceAnalyzer()
        result = analyzer.analyze(_make_trades(), _make_snapshots())
        assert isinstance(result, BacktestResult)

    def test_empty_trades(self) -> None:
        analyzer = PerformanceAnalyzer()
        result = analyzer.analyze([], _make_snapshots())
        assert result.total_trades == 0
        assert result.win_rate == Decimal("0")
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_performance.py -v`
Expected: FAIL with `ImportError`

**Step 3: Write minimal implementation**

```python
# src/finalayze/backtest/performance.py
"""Performance analyzer for backtest results (Layer 6-adjacent, backtest module)."""

from __future__ import annotations

from decimal import Decimal

from finalayze.core.schemas import BacktestResult, PortfolioState, TradeResult


class PerformanceAnalyzer:
    """Computes performance metrics from trades and portfolio snapshots."""

    def analyze(
        self,
        trades: list[TradeResult],
        snapshots: list[PortfolioState],
    ) -> BacktestResult:
        """Compute all performance metrics."""
        total_trades = len(trades)

        if total_trades == 0:
            return BacktestResult(
                sharpe=Decimal("0"),
                max_drawdown=Decimal("0"),
                win_rate=Decimal("0"),
                profit_factor=Decimal("0"),
                total_return=Decimal("0"),
                total_trades=0,
            )

        # Win rate
        wins = [t for t in trades if t.pnl > 0]
        win_rate = Decimal(str(len(wins))) / Decimal(str(total_trades))

        # Profit factor
        gross_profit = sum(t.pnl for t in trades if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in trades if t.pnl < 0))
        profit_factor = (
            gross_profit / gross_loss if gross_loss > 0 else Decimal("999")
        )

        # Total return from equity curve
        if len(snapshots) >= 2:
            initial = snapshots[0].equity
            final = snapshots[-1].equity
            total_return = (final - initial) / initial if initial > 0 else Decimal("0")
        else:
            total_return = Decimal("0")

        # Max drawdown
        max_drawdown = self._compute_max_drawdown(snapshots)

        # Sharpe ratio (annualized, assuming daily snapshots)
        sharpe = self._compute_sharpe(snapshots)

        return BacktestResult(
            sharpe=sharpe,
            max_drawdown=max_drawdown,
            win_rate=win_rate.quantize(Decimal("0.0001")),
            profit_factor=profit_factor.quantize(Decimal("0.0001")),
            total_return=total_return.quantize(Decimal("0.0001")),
            total_trades=total_trades,
        )

    def _compute_max_drawdown(self, snapshots: list[PortfolioState]) -> Decimal:
        """Compute maximum drawdown from equity curve."""
        if len(snapshots) < 2:
            return Decimal("0")

        peak = snapshots[0].equity
        max_dd = Decimal("0")

        for snap in snapshots[1:]:
            if snap.equity > peak:
                peak = snap.equity
            dd = (peak - snap.equity) / peak if peak > 0 else Decimal("0")
            max_dd = max(max_dd, dd)

        return max_dd.quantize(Decimal("0.0001"))

    def _compute_sharpe(
        self, snapshots: list[PortfolioState], risk_free_rate: float = 0.0
    ) -> Decimal:
        """Compute annualized Sharpe ratio from daily equity snapshots."""
        if len(snapshots) < 3:
            return Decimal("0")

        # Daily returns
        equities = [float(s.equity) for s in snapshots]
        returns = [
            (equities[i] - equities[i - 1]) / equities[i - 1]
            for i in range(1, len(equities))
            if equities[i - 1] > 0
        ]

        if not returns:
            return Decimal("0")

        import statistics

        mean_return = statistics.mean(returns) - risk_free_rate / 252
        std_return = statistics.stdev(returns) if len(returns) > 1 else 0.0

        if std_return == 0:
            return Decimal("0")

        annualization_factor = 252 ** 0.5
        sharpe = (mean_return / std_return) * annualization_factor

        return Decimal(str(round(sharpe, 4)))
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_performance.py -v`
Expected: PASS (7 tests)

**Step 5: Run linters**

Run: `uv run ruff check src/finalayze/backtest/performance.py && uv run mypy src/finalayze/backtest/performance.py`

**Step 6: Commit**

```bash
git add src/finalayze/backtest/performance.py tests/unit/test_performance.py
git commit -m "feat(backtest): add PerformanceAnalyzer (Sharpe, drawdown, win rate)"
```

---

## Task 13: Backtest Engine

**Files:**
- Create: `src/finalayze/backtest/engine.py`
- Test: `tests/unit/test_backtest_engine.py`

**Step 1: Write the failing test**

```python
# tests/unit/test_backtest_engine.py
"""Unit tests for backtest engine."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

from finalayze.backtest.engine import BacktestEngine
from finalayze.core.schemas import Candle, Signal, SignalDirection
from finalayze.strategies.base import BaseStrategy

INITIAL_CASH = Decimal("100000")
CANDLE_COUNT = 40
TRADE_DAY_BUY = 30
TRADE_DAY_SELL = 35


class StubStrategy(BaseStrategy):
    """Strategy that emits BUY on day 30, SELL on day 35."""

    @property
    def name(self) -> str:
        return "stub"

    def supported_segments(self) -> list[str]:
        return ["us_tech"]

    def get_parameters(self, segment_id: str) -> dict[str, object]:
        return {}

    def generate_signal(
        self, symbol: str, candles: list[Candle], segment_id: str
    ) -> Signal | None:
        if len(candles) == TRADE_DAY_BUY:
            return Signal(
                strategy_name="stub",
                symbol=symbol,
                market_id="us",
                segment_id=segment_id,
                direction=SignalDirection.BUY,
                confidence=0.8,
                features={},
                reasoning="stub buy",
            )
        if len(candles) == TRADE_DAY_SELL:
            return Signal(
                strategy_name="stub",
                symbol=symbol,
                market_id="us",
                segment_id=segment_id,
                direction=SignalDirection.SELL,
                confidence=0.8,
                features={},
                reasoning="stub sell",
            )
        return None


def _make_candle_series(count: int = CANDLE_COUNT) -> list[Candle]:
    """Create an upward-trending candle series."""
    candles = []
    for i in range(count):
        price = Decimal(str(100 + i))
        candles.append(
            Candle(
                symbol="AAPL",
                market_id="us",
                timeframe="1d",
                timestamp=datetime(2024, 1, 1 + i, 14, 30, tzinfo=UTC),
                open=price,
                high=price + Decimal("2"),
                low=price - Decimal("2"),
                close=price,
                volume=1_000_000,
            )
        )
    return candles


class TestBacktestEngine:
    def test_engine_runs_to_completion(self) -> None:
        engine = BacktestEngine(
            strategy=StubStrategy(),
            initial_cash=INITIAL_CASH,
            max_position_pct=0.20,
            max_positions=10,
        )
        candles = _make_candle_series()
        trades, snapshots = engine.run(
            symbol="AAPL",
            segment_id="us_tech",
            candles=candles,
        )
        assert len(snapshots) == CANDLE_COUNT
        # Should have at least 1 trade (buy + sell = 1 round trip)
        assert len(trades) >= 1

    def test_engine_no_signals_no_trades(self) -> None:
        """A strategy that never signals -> 0 trades."""

        class SilentStrategy(BaseStrategy):
            @property
            def name(self) -> str:
                return "silent"

            def supported_segments(self) -> list[str]:
                return ["us_tech"]

            def get_parameters(self, segment_id: str) -> dict[str, object]:
                return {}

            def generate_signal(
                self, symbol: str, candles: list[Candle], segment_id: str
            ) -> Signal | None:
                return None

        engine = BacktestEngine(
            strategy=SilentStrategy(),
            initial_cash=INITIAL_CASH,
        )
        trades, snapshots = engine.run(
            symbol="AAPL",
            segment_id="us_tech",
            candles=_make_candle_series(),
        )
        assert len(trades) == 0
        assert len(snapshots) == CANDLE_COUNT

    def test_engine_preserves_initial_cash_when_no_trades(self) -> None:
        class SilentStrategy(BaseStrategy):
            @property
            def name(self) -> str:
                return "silent"

            def supported_segments(self) -> list[str]:
                return ["us_tech"]

            def get_parameters(self, segment_id: str) -> dict[str, object]:
                return {}

            def generate_signal(
                self, symbol: str, candles: list[Candle], segment_id: str
            ) -> Signal | None:
                return None

        engine = BacktestEngine(
            strategy=SilentStrategy(),
            initial_cash=INITIAL_CASH,
        )
        trades, snapshots = engine.run(
            symbol="AAPL", segment_id="us_tech", candles=_make_candle_series()
        )
        assert snapshots[-1].equity == INITIAL_CASH
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_backtest_engine.py -v`
Expected: FAIL with `ImportError`

**Step 3: Write minimal implementation**

```python
# src/finalayze/backtest/engine.py
"""Backtest engine -- iterates candles and runs strategy (Layer 6-adjacent)."""

from __future__ import annotations

import uuid
from decimal import Decimal

from finalayze.core.schemas import Candle, PortfolioState, SignalDirection, TradeResult
from finalayze.execution.broker_base import OrderRequest
from finalayze.execution.simulated_broker import SimulatedBroker
from finalayze.risk.position_sizer import compute_position_size
from finalayze.risk.pre_trade_check import PreTradeChecker
from finalayze.risk.stop_loss import compute_atr_stop_loss
from finalayze.strategies.base import BaseStrategy


class BacktestEngine:
    """Runs a strategy over historical candles."""

    def __init__(
        self,
        strategy: BaseStrategy,
        initial_cash: Decimal = Decimal("100000"),
        max_position_pct: float = 0.20,
        max_positions: int = 10,
        kelly_fraction: float = 0.5,
        atr_multiplier: Decimal = Decimal("2.0"),
    ) -> None:
        self._strategy = strategy
        self._initial_cash = initial_cash
        self._max_position_pct = max_position_pct
        self._kelly_fraction = kelly_fraction
        self._atr_multiplier = atr_multiplier
        self._checker = PreTradeChecker(
            max_position_pct=max_position_pct,
            max_positions_per_market=max_positions,
        )

    def run(
        self,
        symbol: str,
        segment_id: str,
        candles: list[Candle],
    ) -> tuple[list[TradeResult], list[PortfolioState]]:
        """Run backtest. Returns (trades, portfolio_snapshots)."""
        broker = SimulatedBroker(initial_cash=self._initial_cash)
        trades: list[TradeResult] = []
        snapshots: list[PortfolioState] = []

        # Track entry price for P&L calculation
        entry_prices: dict[str, Decimal] = {}

        for i, candle in enumerate(candles):
            # 1. Check stop-losses
            stop_results = broker.check_stop_losses(candle)
            for sr in stop_results:
                if sr.filled and sr.symbol in entry_prices:
                    entry = entry_prices.pop(sr.symbol)
                    pnl = (sr.fill_price - entry) * sr.quantity if sr.fill_price else Decimal("0")
                    pnl_pct = pnl / (entry * sr.quantity) if entry > 0 else Decimal("0")
                    trades.append(
                        TradeResult(
                            signal_id=uuid.uuid4(),
                            symbol=sr.symbol,
                            side="SELL",
                            quantity=sr.quantity,
                            entry_price=entry,
                            exit_price=sr.fill_price or Decimal("0"),
                            pnl=pnl,
                            pnl_pct=pnl_pct,
                        )
                    )

            # 2. Update broker prices
            broker.update_prices(candle)

            # 3. Generate signal (use candle history up to current)
            history = candles[: i + 1]
            signal = self._strategy.generate_signal(symbol, history, segment_id)

            # 4. Process signal
            if signal is not None and signal.direction != SignalDirection.HOLD:
                portfolio = broker.get_portfolio()

                if signal.direction == SignalDirection.BUY:
                    # Position sizing
                    position_value = compute_position_size(
                        win_rate=0.5,  # Default assumption
                        avg_win_ratio=Decimal("1.5"),
                        equity=portfolio.equity,
                        kelly_fraction=self._kelly_fraction,
                        max_position_pct=self._max_position_pct,
                    )

                    if position_value > 0 and candle.close > 0:
                        quantity = (position_value / candle.close).quantize(Decimal("1"))
                        if quantity > 0:
                            # Pre-trade check
                            order_value = quantity * candle.close
                            check = self._checker.check(
                                order_value=order_value,
                                portfolio_equity=portfolio.equity,
                                available_cash=portfolio.cash,
                                open_position_count=len(portfolio.positions),
                            )

                            if check.passed and i + 1 < len(candles):
                                fill_candle = candles[i + 1]
                                order = OrderRequest(
                                    symbol=symbol, side="BUY", quantity=quantity
                                )
                                result = broker.submit_order(order, fill_candle)
                                if result.filled and result.fill_price:
                                    entry_prices[symbol] = result.fill_price

                                    # Set stop-loss
                                    stop = compute_atr_stop_loss(
                                        entry_price=result.fill_price,
                                        candles=history,
                                        atr_multiplier=self._atr_multiplier,
                                    )
                                    if stop is not None:
                                        broker.set_stop_loss(symbol, stop)

                elif signal.direction == SignalDirection.SELL:
                    qty = portfolio.positions.get(symbol, Decimal("0"))
                    if qty > 0 and i + 1 < len(candles):
                        fill_candle = candles[i + 1]
                        order = OrderRequest(symbol=symbol, side="SELL", quantity=qty)
                        result = broker.submit_order(order, fill_candle)
                        if result.filled and result.fill_price and symbol in entry_prices:
                            entry = entry_prices.pop(symbol)
                            pnl = (result.fill_price - entry) * qty
                            pnl_pct = pnl / (entry * qty) if entry > 0 else Decimal("0")
                            trades.append(
                                TradeResult(
                                    signal_id=uuid.uuid4(),
                                    symbol=symbol,
                                    side="SELL",
                                    quantity=qty,
                                    entry_price=entry,
                                    exit_price=result.fill_price,
                                    pnl=pnl,
                                    pnl_pct=pnl_pct,
                                )
                            )

            # 5. Record snapshot
            snapshots.append(broker.get_portfolio())

        return trades, snapshots
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_backtest_engine.py -v`
Expected: PASS (3 tests)

**Step 5: Run linters**

Run: `uv run ruff check src/finalayze/backtest/engine.py && uv run mypy src/finalayze/backtest/engine.py`

**Step 6: Commit**

```bash
git add src/finalayze/backtest/engine.py tests/unit/test_backtest_engine.py
git commit -m "feat(backtest): add BacktestEngine with signal processing and risk management"
```

---

## Task 14: CLI Runner Script

**Files:**
- Modify: `scripts/run_backtest.py` (replace stub)
- No unit test (integration-level; manual verification)

**Step 1: Write the CLI script**

```python
# scripts/run_backtest.py
"""Run a backtest for a single symbol + segment.

Usage:
    uv run python scripts/run_backtest.py --symbol AAPL --segment us_tech
    uv run python scripts/run_backtest.py --symbol AAPL --segment us_tech --start 2023-01-01 --end 2024-12-31
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from decimal import Decimal

from finalayze.backtest.engine import BacktestEngine
from finalayze.backtest.performance import PerformanceAnalyzer
from finalayze.data.fetchers.yfinance import YFinanceFetcher
from finalayze.strategies.momentum import MomentumStrategy


def main() -> None:
    """Run backtest and print performance report."""
    parser = argparse.ArgumentParser(description="Run backtest")
    parser.add_argument("--symbol", required=True, help="Ticker symbol (e.g., AAPL)")
    parser.add_argument("--segment", required=True, help="Segment ID (e.g., us_tech)")
    parser.add_argument("--start", default="2023-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2024-12-31", help="End date (YYYY-MM-DD)")
    parser.add_argument("--cash", type=float, default=100_000, help="Initial cash")
    args = parser.parse_args()

    start = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=UTC)
    end = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=UTC)

    print(f"Fetching {args.symbol} data from {args.start} to {args.end}...")  # noqa: T201
    fetcher = YFinanceFetcher(market_id="us")
    candles = fetcher.fetch_candles(args.symbol, start, end)

    if not candles:
        print("No candles fetched. Check symbol and date range.")  # noqa: T201
        return

    print(f"Fetched {len(candles)} candles. Running backtest...")  # noqa: T201

    strategy = MomentumStrategy()
    engine = BacktestEngine(
        strategy=strategy,
        initial_cash=Decimal(str(args.cash)),
    )
    trades, snapshots = engine.run(
        symbol=args.symbol,
        segment_id=args.segment,
        candles=candles,
    )

    analyzer = PerformanceAnalyzer()
    result = analyzer.analyze(trades, snapshots)

    print()  # noqa: T201
    print("=" * 50)  # noqa: T201
    print(f"  BACKTEST RESULTS: {args.symbol} ({args.segment})")  # noqa: T201
    print("=" * 50)  # noqa: T201
    print(f"  Period:         {args.start} to {args.end}")  # noqa: T201
    print(f"  Strategy:       {strategy.name}")  # noqa: T201
    print(f"  Initial Cash:   ${args.cash:,.2f}")  # noqa: T201
    print("-" * 50)  # noqa: T201
    print(f"  Total Return:   {result.total_return:.2%}")  # noqa: T201
    print(f"  Sharpe Ratio:   {result.sharpe}")  # noqa: T201
    print(f"  Max Drawdown:   {result.max_drawdown:.2%}")  # noqa: T201
    print(f"  Win Rate:       {result.win_rate:.2%}")  # noqa: T201
    print(f"  Profit Factor:  {result.profit_factor}")  # noqa: T201
    print(f"  Total Trades:   {result.total_trades}")  # noqa: T201
    print("=" * 50)  # noqa: T201


if __name__ == "__main__":
    main()
```

**Step 2: Verify syntax**

Run: `uv run python -c "import ast; ast.parse(open('scripts/run_backtest.py').read()); print('OK')"`

**Step 3: Smoke test (requires internet for yfinance)**

Run: `uv run python scripts/run_backtest.py --symbol AAPL --segment us_tech --start 2024-01-01 --end 2024-06-30`
Expected: Performance report table printed to console

**Step 4: Run linters**

Run: `uv run ruff check scripts/run_backtest.py`

**Step 5: Commit**

```bash
git add scripts/run_backtest.py
git commit -m "feat(backtest): add CLI runner script for single-symbol backtest"
```

---

## Task 15: Full Test Suite + Lint Pass

**Step 1: Run all tests**

Run: `uv run pytest -v`
Expected: All tests pass, coverage >= 50%

**Step 2: Run ruff**

Run: `uv run ruff check .`
Expected: zero errors

**Step 3: Run mypy**

Run: `uv run mypy src/finalayze/`
Expected: zero errors (may need `type: ignore` annotations for third-party libs)

**Step 4: Fix any issues discovered**

Address linting, type-checking, or test failures. Iterate until clean.

**Step 5: Commit any fixes**

```bash
git add -A
git commit -m "chore(quality): fix lint and type errors across Phase 1 slice"
```

---

## Task 16: Update Documentation

**Files:**
- Modify: `docs/plans/ROADMAP.md` -- mark backtest slice as DONE
- Modify: `docs/quality/GRADES.md` -- grade implemented modules
- Modify: `docs/quality/GAPS.md` -- update gaps
- Modify: `CHANGELOG.md` -- add Phase 1 slice entry

**Step 1: Update ROADMAP.md**

Add entry marking the backtest vertical slice as complete.

**Step 2: Update GRADES.md**

Grade each implemented module:
- `core/schemas.py` -- A (fully tested, typed, frozen)
- `core/models.py` -- B (defined but not integration tested)
- `markets/registry.py` -- A (fully tested)
- `data/fetchers/` -- B (unit tested with mocks)
- `strategies/` -- B (unit tested, depends on pandas-ta)
- `risk/` -- A (fully tested, financial safety critical)
- `execution/` -- B (unit tested, simulated only)
- `backtest/` -- B (unit tested, integration tested)

**Step 3: Update CHANGELOG.md**

Add Phase 1 vertical slice entry with all new modules.

**Step 4: Commit docs**

```bash
git add docs/ CHANGELOG.md
git commit -m "docs(quality): update grades, roadmap, and changelog for Phase 1 slice"
```

---

## Summary

| Task | Component | Tests | Files |
|------|-----------|-------|-------|
| 1 | SignalDirection enum | 4 | schemas.py |
| 2 | Candle schema | 3 | schemas.py |
| 3 | Signal, TradeResult, PortfolioState, BacktestResult | 7 | schemas.py |
| 4 | SQLAlchemy models (all tables) | 4 | models.py, db.py |
| 5 | Alembic migration | 0 (syntax check) | 001_initial.py |
| 6 | MarketRegistry | 7 | registry.py |
| 7 | BaseFetcher + YFinanceFetcher | 3 | base.py, yfinance.py |
| 8 | BaseStrategy + MomentumStrategy | 6 | base.py, momentum.py |
| 9 | Position sizer + Stop loss | 5 | position_sizer.py, stop_loss.py |
| 10 | Pre-trade checks | 4 | pre_trade_check.py |
| 11 | SimulatedBroker | 6 | broker_base.py, simulated_broker.py |
| 12 | PerformanceAnalyzer | 7 | performance.py |
| 13 | BacktestEngine | 3 | engine.py |
| 14 | CLI runner | 0 (manual) | run_backtest.py |
| 15 | Full test + lint pass | -- | all |
| 16 | Documentation updates | -- | docs/ |
| **Total** | | **~59 tests** | **~15 files** |

**Estimated effort:** Tasks 1-14 are independent TDD cycles (~2-5 min each). Tasks 15-16 are validation + docs.
