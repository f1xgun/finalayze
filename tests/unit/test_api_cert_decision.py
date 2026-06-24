"""Phase 87: GET /api/v1/saa/cert-decision -- auth, 200 (matches committed cert), 503 fail-closed.

The endpoint is read-only + token-free (a filesystem read of the committed cert; no Tinkoff token,
no DB). The 200 body must BYTE-match the committed cert (anti-hollow); a missing cert is a clean
503, never a 200-with-zeros.
"""

from __future__ import annotations

import json
from unittest.mock import patch

from fastapi.testclient import TestClient

from finalayze.backtest.cert_reader import select_latest_cert_dir
from finalayze.core.exceptions import CertNotFoundError
from finalayze.main import create_app

_PATH = "/api/v1/saa/cert-decision"
_HTTP_OK = 200
_HTTP_UNAUTH = 401
_HTTP_NO_CERT = 503


def _client() -> TestClient:
    return TestClient(create_app())


def _auth() -> dict[str, str]:
    from config.settings import Settings  # noqa: PLC0415

    return {"X-API-Key": Settings().api_key}


def _committed_cert() -> dict:
    return json.loads((select_latest_cert_dir() / "summary.json").read_text(encoding="utf-8"))


def test_requires_auth() -> None:
    """No X-API-Key -> 401 (behind the gateway like every /api/v1 route)."""
    assert _client().get(_PATH).status_code == _HTTP_UNAUTH


def test_cert_decision_200_matches_committed_cert() -> None:
    """200 with verdict/headline/stories DERIVED from -- and byte-matching -- the committed cert."""
    resp = _client().get(_PATH, headers=_auth())
    assert resp.status_code == _HTTP_OK
    body = resp.json()
    cert = _committed_cert()

    # anti-hollow: surfaced verdict + escalation + caveat are the committed cert's, not literals.
    assert body["phase_verdict"] == cert["phase_verdict"]
    assert body["escalation"] == cert["escalation"]
    assert body["n1_caveat"] == cert["n1_caveat"]
    assert body["high_rate_caveat"] == cert["high_rate_caveat"]
    # the full-window best-naive bar is equity_100, NOT the deposit (the honesty trap).
    assert body["best_naive_sharpe_full"] == cert["naive"]["equity_100_sharpe"]
    assert body["best_naive_sharpe_full"] != cert["naive"]["deposit_100_sharpe"]
    # HARD_FAIL not softened.
    if cert["phase_verdict"] == "HARD_FAIL":
        assert "does not beat" in body["headline"]
    # per-regime stories present (high_rate first); deposit wins ONLY in the high_rate row.
    high = next(s for s in body["regime_stories"] if s["unit_key"] == "high_rate")
    assert high["best_naive_sharpe"] > 0
    assert high["allocation_sharpe"] < 0
    # no fabricated rate threshold in the framing.
    assert "no rate threshold" in body["when_framing"].lower()


def test_cert_decision_503_when_no_cert() -> None:
    """When no committed cert exists, the endpoint fails CLOSED with 503 (not 200-with-zeros)."""
    with patch(
        "finalayze.backtest.cert_reader.load_latest_cert",
        side_effect=CertNotFoundError("no committed cert"),
    ):
        resp = _client().get(_PATH, headers=_auth())
    assert resp.status_code == _HTTP_NO_CERT
    assert "cert" in resp.json()["detail"].lower()
