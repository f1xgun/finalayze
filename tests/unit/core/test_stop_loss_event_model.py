from __future__ import annotations

from decimal import Decimal
from datetime import UTC, datetime


def test_stop_loss_event_model_tablename() -> None:
    from finalayze.core.models import StopLossEventModel
    assert StopLossEventModel.__tablename__ == "stop_loss_events"


def test_stop_loss_event_model_composite_primary_key() -> None:
    from finalayze.core.models import StopLossEventModel
    pk_columns = {c.name for c in StopLossEventModel.__table__.primary_key.columns}
    assert pk_columns == {"timestamp", "symbol", "market_id"}


def test_stop_loss_event_model_columns_present() -> None:
    from finalayze.core.models import StopLossEventModel
    cols = {c.name for c in StopLossEventModel.__table__.columns}
    assert cols == {
        "timestamp", "symbol", "market_id", "event_type",
        "entry_price", "current_stop", "highest_price", "atr_value",
        "activation_atr", "trail_atr", "trail_activated", "current_price",
    }


def test_stop_loss_event_model_event_type_not_null() -> None:
    from finalayze.core.models import StopLossEventModel
    col = StopLossEventModel.__table__.columns["event_type"]
    assert col.nullable is False


def test_stop_loss_event_model_instantiation() -> None:
    from finalayze.core.models import StopLossEventModel
    row = StopLossEventModel(
        timestamp=datetime.now(UTC),
        symbol="SBER",
        market_id="moex",
        event_type="snapshot",
        entry_price=Decimal("280.50"),
        current_stop=Decimal("275.00"),
        highest_price=Decimal("285.00"),
        atr_value=Decimal("5.50"),
        activation_atr=Decimal("1.0"),
        trail_atr=Decimal("1.5"),
        trail_activated=True,
        current_price=Decimal("283.00"),
    )
    assert row.symbol == "SBER"
    assert row.event_type == "snapshot"
