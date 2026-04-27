"""Tests for MetaAgentDecisionModel ORM (Phase 58-01, META-03).

Mirrors tests/unit/core/test_alert_model.py:
  - column set
  - composite primary key (timestamp, id)
  - parent_decision_id has NO database FK
  - decision_metadata Python attr maps to "metadata" DB column
  - __init__ defaults: status='queued', dry_run=True, actions=[]
"""

from __future__ import annotations

_PK_SIZE = 2
_EXPECTED_COLUMNS = {
    "timestamp",
    "id",
    "severity",
    "summary",
    "rationale",
    "actions",
    "outcome",
    "dry_run",
    "metadata",  # DB column name; Python attr is decision_metadata
    "parent_decision_id",
    "status",
    "created_at",
}


def test_meta_agent_decision_model_columns() -> None:
    """All 12 columns from SPEC §Requirement 3 are declared."""
    from finalayze.core.models import MetaAgentDecisionModel

    cols = {c.name for c in MetaAgentDecisionModel.__table__.columns}
    assert cols == _EXPECTED_COLUMNS


def test_composite_pk() -> None:
    """Primary key is composite (timestamp, id) per hypertable convention."""
    from finalayze.core.models import MetaAgentDecisionModel

    pk_cols = list(MetaAgentDecisionModel.__table__.primary_key.columns)
    pk_names = {c.name for c in pk_cols}
    assert len(pk_cols) == _PK_SIZE
    assert pk_names == {"timestamp", "id"}


def test_metadata_column_name_is_metadata_python_attr_is_decision_metadata() -> None:
    """`decision_metadata` Python attr → `metadata` DB column (AP-3)."""
    from finalayze.core.models import MetaAgentDecisionModel

    db_col = MetaAgentDecisionModel.__table__.c["metadata"]
    assert db_col.name == "metadata"
    py_attr = MetaAgentDecisionModel.decision_metadata
    assert py_attr.key == "decision_metadata"
    # Confirm the Python attr is bound to the DB column.
    assert py_attr.property.columns[0].name == "metadata"


def test_init_defaults_status_dry_run_actions() -> None:
    """SQLAlchemy 2.0 default= only fires at flush; __init__ override applies
    Python-side defaults so callers can construct without explicit values."""
    from finalayze.core.models import MetaAgentDecisionModel

    row = MetaAgentDecisionModel(severity="HEALTHY", summary="s", rationale="r")
    assert row.status == "queued"
    assert row.dry_run is True
    assert row.actions == []


def test_parent_decision_id_no_fk() -> None:
    """parent_decision_id is a plain nullable UUID — TimescaleDB hypertables
    forbid the UNIQUE (id) constraint that a self-FK would require (AP-2)."""
    from finalayze.core.models import MetaAgentDecisionModel

    fks = list(MetaAgentDecisionModel.__table__.foreign_keys)
    assert fks == [], f"MetaAgentDecisionModel must have NO FKs; found {fks!r}"
    parent_col = MetaAgentDecisionModel.__table__.c["parent_decision_id"]
    assert parent_col.nullable is True
    assert str(parent_col.type) == "UUID", (
        f"parent_decision_id must be UUID; got {parent_col.type!r}"
    )
