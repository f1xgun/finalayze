"""Phase 82: DB-free ORM column-type parity guard for the rebalance audit tables (AH-02).

Pins the ORM model's column types + FK ondelete via ``__table__`` introspection (no DB). Together
with the static-AST migration test, a type/length drift on EITHER the ORM or the migration fails a
test -- enforcing the L-04 byte-for-byte parity the phase exists to guarantee.
"""

from __future__ import annotations

from sqlalchemy import Date, DateTime, Numeric, String, Text

from finalayze.core.models import SaaRebalanceOrderModel, SaaRebalanceRunModel


def _cols(model: type) -> dict:
    return {c.name: c for c in model.__table__.columns}  # type: ignore[attr-defined]


def test_run_model_column_types() -> None:
    cols = _cols(SaaRebalanceRunModel)
    budget = cols["budget_rub"].type
    assert isinstance(budget, Numeric)
    assert budget.precision == 20
    assert budget.scale == 2
    fill_rate = cols["fill_rate"].type
    assert isinstance(fill_rate, Numeric)
    assert fill_rate.precision == 8
    assert fill_rate.scale == 4
    assert isinstance(cols["as_of"].type, Date)
    assert not isinstance(cols["as_of"].type, DateTime)  # Date, not DateTime
    created = cols["created_at"].type
    assert isinstance(created, DateTime)
    assert created.timezone is True
    assert isinstance(cols["plan_id"].type, String)
    assert cols["plan_id"].type.length == 120
    assert cols["mode"].type.length == 12


def test_order_model_column_types() -> None:
    cols = _cols(SaaRebalanceOrderModel)
    for qty_col in ("requested_qty", "filled_qty"):
        col_type = cols[qty_col].type
        assert isinstance(col_type, Numeric)
        assert col_type.precision == 28
        assert col_type.scale == 8
    assert isinstance(cols["symbol"].type, String)
    assert cols["symbol"].type.length == 40
    assert cols["client_order_id"].type.length == 64
    assert cols["status"].type.length == 20
    assert isinstance(cols["reason"].type, Text)  # unbounded (CR-CORR-01)


def test_fk_ondelete_semantics() -> None:
    run_fk = next(iter(SaaRebalanceRunModel.__table__.columns["portfolio_id"].foreign_keys))
    assert run_fk.ondelete == "RESTRICT"  # no silent loss of a portfolio with run history
    order_fk = next(iter(SaaRebalanceOrderModel.__table__.columns["run_id"].foreign_keys))
    assert order_fk.ondelete == "CASCADE"  # orders belong to their run
