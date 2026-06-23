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
