"""Phase 83 P83-01: pure rebalance-reader mapping + limit clamp (no DB)."""

from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal
from uuid import uuid4

from finalayze.core.models import SaaRebalanceOrderModel, SaaRebalanceRunModel
from finalayze.execution.rebalance_reader import _clamp_limit, _to_record


def test_to_record_maps_run_and_orders() -> None:
    run = SaaRebalanceRunModel(
        id=uuid4(),
        portfolio_id=uuid4(),
        plan_id="pid:2026-06-23",
        as_of=date(2026, 6, 23),
        mode="SANDBOX",
        budget_rub=Decimal(1_000_000),
        status="COMPLETE",
        fill_rate=Decimal("1.0000"),
        created_at=datetime(2026, 6, 23, 12, 0, tzinfo=UTC),
    )
    run.orders = [
        SaaRebalanceOrderModel(
            asset_class="ofz_pk",
            symbol="SU29024RMFS5",
            side="BUY",
            requested_qty=Decimal(50),
            filled_qty=Decimal(50),
            status="FILLED",
            client_order_id="fnz-ofz",
            reason=None,
        ),
        SaaRebalanceOrderModel(
            asset_class="equity",
            symbol="EQMX",
            side="BUY",
            requested_qty=Decimal(100),
            filled_qty=Decimal(100),
            status="FILLED",
            client_order_id="fnz-eq",
            reason=None,
        ),
    ]
    rec = _to_record(run)
    assert rec.run_id == run.id
    assert rec.plan_id == "pid:2026-06-23"
    assert rec.mode == "SANDBOX"
    assert rec.status == "COMPLETE"
    assert rec.fill_rate == Decimal("1.0000")
    assert len(rec.orders) == 2
    # orders sorted by asset_class -> equity before ofz_pk
    assert [o.asset_class for o in rec.orders] == ["equity", "ofz_pk"]
    assert rec.orders[0].symbol == "EQMX"
    assert rec.orders[0].filled_qty == Decimal(100)


def test_clamp_limit() -> None:
    assert _clamp_limit(0) == 1
    assert _clamp_limit(20) == 20
    assert _clamp_limit(5000) == 100
