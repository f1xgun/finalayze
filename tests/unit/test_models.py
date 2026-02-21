"""Unit tests for SQLAlchemy ORM models."""

from __future__ import annotations

from finalayze.core.models import (
    Base,
    CandleModel,
    InstrumentModel,
    MarketModel,
    OrderModel,
    SegmentModel,
    SignalModel,
)


class TestBase:
    """Tests for the SQLAlchemy declarative base."""

    def test_base_has_metadata(self) -> None:
        assert Base.metadata is not None

    def test_base_registry_exists(self) -> None:
        assert Base.registry is not None


class TestMarketModel:
    """Tests for MarketModel ORM mapping."""

    TABLE_NAME = "markets"

    def test_table_name(self) -> None:
        assert MarketModel.__tablename__ == self.TABLE_NAME

    def test_primary_key_columns(self) -> None:
        pk_cols = [c.name for c in MarketModel.__table__.primary_key.columns]
        assert pk_cols == ["id"]

    def test_required_columns_exist(self) -> None:
        col_names = {c.name for c in MarketModel.__table__.columns}
        expected = {"id", "name", "currency", "timezone", "open_time", "close_time"}
        assert expected.issubset(col_names)


class TestSegmentModel:
    """Tests for SegmentModel ORM mapping."""

    TABLE_NAME = "segments"

    def test_table_name(self) -> None:
        assert SegmentModel.__tablename__ == self.TABLE_NAME

    def test_primary_key_columns(self) -> None:
        pk_cols = [c.name for c in SegmentModel.__table__.primary_key.columns]
        assert pk_cols == ["id"]

    def test_foreign_key_to_markets(self) -> None:
        market_id_col = SegmentModel.__table__.c.market_id
        fk_targets = {fk.target_fullname for fk in market_id_col.foreign_keys}
        assert "markets.id" in fk_targets

    def test_required_columns_exist(self) -> None:
        col_names = {c.name for c in SegmentModel.__table__.columns}
        expected = {
            "id", "market_id", "name", "description",
            "active_strategies", "strategy_params", "ml_model_id",
            "max_allocation_pct", "news_languages",
        }
        assert expected.issubset(col_names)


class TestInstrumentModel:
    """Tests for InstrumentModel ORM mapping."""

    TABLE_NAME = "instruments"

    def test_table_name(self) -> None:
        assert InstrumentModel.__tablename__ == self.TABLE_NAME

    def test_composite_primary_key(self) -> None:
        pk_cols = sorted(c.name for c in InstrumentModel.__table__.primary_key.columns)
        assert pk_cols == ["market_id", "symbol"]

    def test_foreign_key_to_segments(self) -> None:
        segment_id_col = InstrumentModel.__table__.c.segment_id
        fk_targets = {fk.target_fullname for fk in segment_id_col.foreign_keys}
        assert "segments.id" in fk_targets

    def test_required_columns_exist(self) -> None:
        col_names = {c.name for c in InstrumentModel.__table__.columns}
        expected = {
            "symbol", "market_id", "segment_id", "name", "figi",
            "instrument_type", "currency", "lot_size", "is_active",
        }
        assert expected.issubset(col_names)


class TestCandleModel:
    """Tests for CandleModel ORM mapping."""

    TABLE_NAME = "candles"
    COMPOSITE_PK_COUNT = 4

    def test_table_name(self) -> None:
        assert CandleModel.__tablename__ == self.TABLE_NAME

    def test_composite_primary_key_has_four_columns(self) -> None:
        pk_cols = list(CandleModel.__table__.primary_key.columns)
        assert len(pk_cols) == self.COMPOSITE_PK_COUNT

    def test_composite_primary_key_column_names(self) -> None:
        pk_cols = sorted(c.name for c in CandleModel.__table__.primary_key.columns)
        assert pk_cols == ["market_id", "symbol", "timeframe", "timestamp"]

    def test_required_columns_exist(self) -> None:
        col_names = {c.name for c in CandleModel.__table__.columns}
        expected = {
            "symbol", "market_id", "timeframe", "timestamp",
            "open", "high", "low", "close", "volume", "source",
        }
        assert expected.issubset(col_names)


class TestSignalModel:
    """Tests for SignalModel ORM mapping."""

    TABLE_NAME = "signals"

    def test_table_name(self) -> None:
        assert SignalModel.__tablename__ == self.TABLE_NAME

    def test_primary_key_is_uuid(self) -> None:
        pk_cols = [c.name for c in SignalModel.__table__.primary_key.columns]
        assert pk_cols == ["id"]

    def test_required_columns_exist(self) -> None:
        col_names = {c.name for c in SignalModel.__table__.columns}
        expected = {
            "id", "strategy_name", "symbol", "market_id", "segment_id",
            "direction", "confidence", "features", "reasoning",
            "created_at", "mode",
        }
        assert expected.issubset(col_names)


class TestOrderModel:
    """Tests for OrderModel ORM mapping."""

    TABLE_NAME = "orders"

    def test_table_name(self) -> None:
        assert OrderModel.__tablename__ == self.TABLE_NAME

    def test_primary_key_is_uuid(self) -> None:
        pk_cols = [c.name for c in OrderModel.__table__.primary_key.columns]
        assert pk_cols == ["id"]

    def test_foreign_key_to_signals(self) -> None:
        signal_id_col = OrderModel.__table__.c.signal_id
        fk_targets = {fk.target_fullname for fk in signal_id_col.foreign_keys}
        assert "signals.id" in fk_targets

    def test_required_columns_exist(self) -> None:
        col_names = {c.name for c in OrderModel.__table__.columns}
        expected = {
            "id", "signal_id", "broker", "broker_order_id",
            "symbol", "market_id", "side", "order_type",
            "quantity", "limit_price", "stop_price",
            "currency", "status",
            "filled_quantity", "filled_avg_price",
            "submitted_at", "filled_at",
            "risk_checks", "mode",
        }
        assert expected.issubset(col_names)


class TestAllTablesRegistered:
    """Verify all models are registered in Base metadata."""

    EXPECTED_TABLES = {"markets", "segments", "instruments", "candles", "signals", "orders"}

    def test_all_tables_in_metadata(self) -> None:
        registered = set(Base.metadata.tables.keys())
        assert self.EXPECTED_TABLES.issubset(registered)
