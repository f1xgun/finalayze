"""Phase 81 P81-01/02: SAA target-allocation endpoint -- auth, 404, 200, structure, token-free.

The endpoint is token-free: these tests patch ONLY the two DB reads (get_active_portfolio,
load_deposit_broker_from_db); the regime weights + leg instruments resolve for real. No
TinkoffBroker is ever constructed -- proving the target-allocation view needs no Tinkoff token.
"""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from fastapi.testclient import TestClient

from finalayze.main import create_app

_PATH = "/api/v1/saa/target-allocation"
_HTTP_OK = 200
_HTTP_UNAUTH = 401
_HTTP_NOT_FOUND = 404


def _client() -> TestClient:
    return TestClient(create_app())


def _auth() -> dict[str, str]:
    from config.settings import Settings  # noqa: PLC0415

    return {"X-API-Key": Settings().api_key}


def test_requires_auth() -> None:
    """No X-API-Key -> 401."""
    assert _client().get(_PATH).status_code == _HTTP_UNAUTH


def test_no_active_portfolio_returns_404() -> None:
    """No active SAA portfolio -> 404 (clear, not a silent empty)."""
    with (
        patch("finalayze.core.db.get_async_session_factory", return_value=object()),
        patch("finalayze.api.v1.saa.get_active_portfolio", new=AsyncMock(return_value=None)),
    ):
        resp = _client().get(_PATH, headers=_auth())
    assert resp.status_code == _HTTP_NOT_FOUND


def test_returns_target_allocation() -> None:
    """200 with budget, regime-tilted weights, per-leg targets (budget*weight), deposit mark."""
    pid = uuid4()
    with (
        patch("finalayze.core.db.get_async_session_factory", return_value=object()),
        patch(
            "finalayze.api.v1.saa.get_active_portfolio",
            new=AsyncMock(return_value=(pid, "balanced", Decimal(1_000_000))),
        ),
        patch(
            "finalayze.api.v1.saa.load_deposit_broker_from_db",
            new=AsyncMock(return_value=None),
        ),
    ):
        resp = _client().get(_PATH, headers=_auth())

    assert resp.status_code == _HTTP_OK
    body = resp.json()
    assert body["portfolio_id"] == str(pid)
    assert body["risk_profile"] == "balanced"
    assert body["budget_rub"] == "1000000"
    assert body["deposit_current_notional_rub"] == "0"  # no persisted deposit
    assert body["as_of"]  # serialized ISO as-of date (AH-03)
    assert set(body["legs"]) == {"deposit", "ofz_pk", "equity"}

    budget = Decimal(body["budget_rub"])
    total_weight = Decimal(0)
    for asset_class, leg in body["legs"].items():
        weight = Decimal(leg["weight"])
        total_weight += weight
        # P81-R4: per-leg target == budget * weight, exactly.
        assert Decimal(leg["target_notional_rub"]) == budget * weight
        if asset_class == "deposit":
            assert leg["symbol"] is None  # deposit is not a tradeable instrument
        else:
            assert leg["symbol"]  # equity + OFZ-PK carry a tradeable ticker
    assert total_weight == Decimal(1)  # a valid allocation


# --- Phase 83: rebalance-runs history endpoint --------------------------------------------------

_RUNS_PATH = "/api/v1/saa/rebalance-runs"


def _sample_record() -> object:
    from datetime import UTC, date, datetime  # noqa: PLC0415

    from finalayze.execution.rebalance_reader import (  # noqa: PLC0415
        OrderRecord,
        RebalanceRunRecord,
    )

    return RebalanceRunRecord(
        run_id=uuid4(),
        plan_id="pid:2026-06-23",
        as_of=date(2026, 6, 23),
        mode="SANDBOX",
        status="COMPLETE",
        fill_rate=Decimal("1.0000"),
        created_at=datetime(2026, 6, 23, 12, 0, tzinfo=UTC),
        orders=(
            OrderRecord(
                asset_class="equity",
                symbol="EQMX",
                side="BUY",
                requested_qty=Decimal(100),
                filled_qty=Decimal(100),
                status="FILLED",
                reason=None,
            ),
        ),
    )


def test_rebalance_runs_requires_auth() -> None:
    assert _client().get(_RUNS_PATH).status_code == _HTTP_UNAUTH


def test_rebalance_runs_no_active_portfolio_404() -> None:
    with (
        patch("finalayze.core.db.get_async_session_factory", return_value=object()),
        patch("finalayze.api.v1.saa.get_active_portfolio", new=AsyncMock(return_value=None)),
    ):
        resp = _client().get(_RUNS_PATH, headers=_auth())
    assert resp.status_code == _HTTP_NOT_FOUND


def test_rebalance_runs_returns_history() -> None:
    pid = uuid4()
    with (
        patch("finalayze.core.db.get_async_session_factory", return_value=object()),
        patch(
            "finalayze.api.v1.saa.get_active_portfolio",
            new=AsyncMock(return_value=(pid, "balanced", Decimal(1_000_000))),
        ),
        patch(
            "finalayze.api.v1.saa.list_rebalance_runs",
            new=AsyncMock(return_value=[_sample_record()]),
        ),
    ):
        resp = _client().get(_RUNS_PATH, headers=_auth())
    assert resp.status_code == _HTTP_OK
    body = resp.json()
    assert body["portfolio_id"] == str(pid)
    assert len(body["runs"]) == 1
    run = body["runs"][0]
    assert run["plan_id"] == "pid:2026-06-23"
    assert run["status"] == "COMPLETE"
    assert run["fill_rate"] == "1.0000"
    assert run["mode"] == "SANDBOX"
    assert len(run["orders"]) == 1
    assert run["orders"][0]["symbol"] == "EQMX"
    assert run["orders"][0]["filled_qty"] == "100"


def test_rebalance_runs_empty_when_no_runs() -> None:
    pid = uuid4()
    with (
        patch("finalayze.core.db.get_async_session_factory", return_value=object()),
        patch(
            "finalayze.api.v1.saa.get_active_portfolio",
            new=AsyncMock(return_value=(pid, "balanced", Decimal(1_000_000))),
        ),
        patch("finalayze.api.v1.saa.list_rebalance_runs", new=AsyncMock(return_value=[])),
    ):
        resp = _client().get(_RUNS_PATH, headers=_auth())
    assert resp.status_code == _HTTP_OK
    assert resp.json()["runs"] == []


def test_rebalance_runs_forwards_limit_to_reader() -> None:
    """?limit=N is honored: the value reaches list_rebalance_runs (P83-R6, AH-01)."""
    pid = uuid4()
    reader = AsyncMock(return_value=[])
    with (
        patch("finalayze.core.db.get_async_session_factory", return_value=object()),
        patch(
            "finalayze.api.v1.saa.get_active_portfolio",
            new=AsyncMock(return_value=(pid, "balanced", Decimal(1_000_000))),
        ),
        patch("finalayze.api.v1.saa.list_rebalance_runs", new=reader),
    ):
        resp = _client().get(_RUNS_PATH, headers=_auth(), params={"limit": 3})
    assert resp.status_code == _HTTP_OK
    assert reader.await_args.kwargs["limit"] == 3  # the ?limit=3 reached the reader
