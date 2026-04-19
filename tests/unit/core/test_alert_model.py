"""Tests for AlertModel ORM class (Phase 57-01, ALRT-03).

Validates:
  - All 10 columns are present
  - Composite primary key (timestamp, id)
  - parent_id has FK to alerts.id with ON DELETE SET NULL
  - alert_metadata Python attribute maps to "metadata" DB column
    (SQLAlchemy reserves `metadata` on DeclarativeBase)
  - delivery_status defaults to "queued"
"""

from __future__ import annotations


def test_alert_model_fields() -> None:
    """AlertModel must define all 10 columns from the schema."""
    from finalayze.core.models import AlertModel

    cols = {c.name for c in AlertModel.__table__.columns}
    assert cols == {
        "timestamp",
        "id",
        "alert_type",
        "priority",
        "symbol",
        "market_id",
        "message",
        "parent_id",
        "delivery_status",
        "metadata",
    }


def test_alert_model_composite_pk() -> None:
    """Primary key must be composite over (timestamp, id) for hypertable."""
    from finalayze.core.models import AlertModel

    pk_cols = list(AlertModel.__table__.primary_key.columns)
    pk_names = {c.name for c in pk_cols}
    expected_pk_size = 2
    assert len(pk_cols) == expected_pk_size
    assert pk_names == {"timestamp", "id"}


def test_alert_model_parent_id_fk() -> None:
    """parent_id must FK to alerts.id with ON DELETE SET NULL."""
    from finalayze.core.models import AlertModel

    fks = list(AlertModel.__table__.foreign_keys)
    expected_fk_count = 1
    assert len(fks) == expected_fk_count, f"expected exactly 1 FK, found {len(fks)}"
    fk = fks[0]
    assert fk.column.table.name == "alerts"
    assert fk.column.name == "id"
    assert fk.ondelete == "SET NULL"


def test_alert_metadata_column_name() -> None:
    """alert_metadata Python attr must map to "metadata" DB column.

    SQLAlchemy's DeclarativeBase reserves `metadata`; the Python attribute
    is renamed to `alert_metadata` while the column name stays "metadata".
    """
    from finalayze.core.models import AlertModel

    col = AlertModel.alert_metadata.property.columns[0]
    assert col.name == "metadata"


def test_alert_model_delivery_status_default() -> None:
    """delivery_status defaults to "queued" on construction."""
    from finalayze.core.models import AlertModel

    row = AlertModel()
    assert row.delivery_status == "queued"
