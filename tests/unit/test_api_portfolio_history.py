"""Unit tests for GET /api/v1/portfolio/history (EQTY-02 D-05..D-08).

The history endpoint must:
* Read primarily from `daily_equity_snapshots` (D-05 primary source).
* Fall back to `sandbox_metrics` when `daily_equity_snapshots` row count
  inside the requested window is < 5 (D-05 hybrid).
* Honor `?days=N` (default 30) and `?market_id=X` query params (D-06).
* Compute `drawdown_pct` server-side, per `market_id`, via running peak (D-07).
* Emit a structlog `portfolio_history_served` line containing
  `history_source` and `row_count` keys (D-08).

Tests mock `finalayze.api.v1.portfolio.get_async_session_factory` so the
handler executes against in-memory ORM rows (no DB required).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
import structlog
from fastapi.testclient import TestClient

from finalayze.main import create_app

# ---------- Fixtures and helpers -------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_structlog() -> None:
    """Reset structlog so capture_logs() works after setup_logging() ran.

    The API tests typically call setup_logging() once; structlog caches the
    module-level _log with the JSONRenderer config and bypasses capture_logs.
    Mirrors the pattern from tests/unit/test_portfolio_review_integration.py.
    """
    structlog.reset_defaults()
    import finalayze.api.v1.portfolio as portfolio_mod  # noqa: PLC0415

    portfolio_mod._log = structlog.get_logger()


def _auth() -> dict[str, str]:
    from config.settings import Settings

    return {"X-API-Key": Settings().api_key}


def _make_daily_row(
    timestamp: datetime,
    market_id: str,
    equity: float,
) -> SimpleNamespace:
    """Build a duck-typed stand-in for a DailyEquitySnapshot ORM row.

    The handler accesses only `.timestamp`, `.market_id`, `.equity`. Using
    SimpleNamespace avoids the SQLAlchemy mapper machinery — the rows never
    enter a session, so attribute setters via `__set__` don't apply.
    """
    return SimpleNamespace(
        timestamp=timestamp,
        market_id=market_id,
        equity=Decimal(str(equity)),
        currency="RUB" if market_id.startswith(("moex", "ru_")) else "USD",
    )


def _make_sandbox_row(
    timestamp: datetime,
    market_id: str,
    equity_rub: float,
    drawdown_pct: float | None = None,
) -> SimpleNamespace:
    """Build a duck-typed stand-in for a SandboxMetricRow ORM row."""
    return SimpleNamespace(
        timestamp=timestamp,
        market_id=market_id,
        equity_rub=Decimal(str(equity_rub)),
        drawdown_pct=Decimal(str(drawdown_pct)) if drawdown_pct is not None else None,
    )


def _patch_session_factory(
    daily_rows: list[SimpleNamespace],
    sandbox_rows: list[SimpleNamespace],
) -> Any:
    """Patch `get_async_session_factory` so the handler reads our seeded rows.

    The handler issues two queries (primary daily_equity_snapshots, optional
    fallback sandbox_metrics). We dispatch on the SQL text to decide which
    list to return, then apply the cutoff and market_id filters that the
    real query would have applied.
    """

    class _Result:
        def __init__(self, rows: list[Any]) -> None:
            self._rows = rows

        def scalars(self) -> _Result:
            return self

        def all(self) -> list[Any]:
            return list(self._rows)

    class _Session:
        async def __aenter__(self) -> _Session:
            return self

        async def __aexit__(self, *args: object) -> None:  # noqa: D401
            return None

        async def execute(self, stmt: Any) -> _Result:
            # Render the compiled SQL (with bound markers) for table detection
            # and market_id filter detection. SQLAlchemy clauses raise
            # TypeError on truthiness checks, so we MUST avoid bool(stmt).
            sql = str(stmt)
            if "daily_equity_snapshots" in sql:
                rows: list[Any] = list(daily_rows)
            else:
                rows = list(sandbox_rows)

            # market_id filter: each test sets _Session._market_filter to the
            # value it passed via ?market_id=...; if None, no filter applied.
            expected = _Session._market_filter
            if expected is not None:
                rows = [r for r in rows if r.market_id == expected]
            # Sort by timestamp ascending to mirror ORDER BY timestamp asc.
            rows.sort(key=lambda r: r.timestamp)
            return _Result(rows)

    _Session._market_filter = None  # type: ignore[attr-defined]

    def _factory_callable() -> _Session:
        return _Session()

    def _get_factory() -> Any:
        return _factory_callable

    # Patch on the source module: the handler does a function-local import
    # `from finalayze.core.db import get_async_session_factory`, so the
    # symbol resolves through the source module each invocation.
    patcher = patch(
        "finalayze.core.db.get_async_session_factory",
        side_effect=_get_factory,
    )
    return patcher, _Session


# ---------- Tests ----------------------------------------------------------------


def test_history_uses_daily_equity_snapshots_when_sufficient() -> None:
    """When daily_equity_snapshots has >=5 rows, handler reads from primary source (D-05)."""
    now = datetime.now(UTC)
    daily = [_make_daily_row(now - timedelta(days=i), "moex", 1000.0 + i) for i in range(6)]

    patcher, _ = _patch_session_factory(daily, [])
    with patcher, structlog.testing.capture_logs() as captured:
        resp = TestClient(create_app()).get("/api/v1/portfolio/history?days=30", headers=_auth())

    assert resp.status_code == 200
    snapshots = resp.json()["snapshots"]
    assert len(snapshots) == 6, f"Expected 6 daily snapshots, got {len(snapshots)}"
    served = [e for e in captured if e.get("event") == "portfolio_history_served"]
    assert served, "Expected a portfolio_history_served log line"
    assert served[-1]["history_source"] == "daily_equity_snapshots", (
        f"Expected history_source=daily_equity_snapshots, got {served[-1].get('history_source')!r}"
    )


def test_history_falls_back_to_sandbox_metrics_when_sparse() -> None:
    """When daily_equity_snapshots has <5 rows, handler falls back to sandbox_metrics (D-05)."""
    now = datetime.now(UTC)
    # 2 daily rows (under the threshold of 5)
    daily = [_make_daily_row(now - timedelta(days=i), "moex", 1000.0) for i in range(2)]
    # Sandbox source returns the actual data
    sandbox = [
        _make_sandbox_row(now - timedelta(days=i), "moex", 5000.0 + i, drawdown_pct=0.01)
        for i in range(3)
    ]

    patcher, _ = _patch_session_factory(daily, sandbox)
    with patcher, structlog.testing.capture_logs() as captured:
        resp = TestClient(create_app()).get("/api/v1/portfolio/history?days=30", headers=_auth())

    assert resp.status_code == 200
    snapshots = resp.json()["snapshots"]
    # Fallback rows reflect sandbox seed (3 rows with equity 5000+i), NOT daily seed
    assert len(snapshots) == 3, (
        f"Expected 3 sandbox_metrics rows in fallback path, got {len(snapshots)}"
    )
    equities = sorted(s["equity"] for s in snapshots)
    assert equities == sorted([5000.0, 5001.0, 5002.0]), (
        f"Expected sandbox equities {[5000.0, 5001.0, 5002.0]}, got {equities}"
    )
    served = [e for e in captured if e.get("event") == "portfolio_history_served"]
    assert served, "Expected a portfolio_history_served log line"
    actual_source = served[-1].get("history_source")
    assert served[-1]["history_source"] == "sandbox_metrics", (
        f"Expected fallback history_source=sandbox_metrics, got {actual_source!r}"
    )


def test_history_query_params() -> None:
    """?days=N narrows the window; ?market_id=X filters to one market (D-06)."""
    now = datetime.now(UTC)
    # 5 us + 5 moex rows, all inside a 30-day window
    daily: list[SimpleNamespace] = []
    for i in range(5):
        daily.append(_make_daily_row(now - timedelta(days=i), "us", 100.0 + i))
        daily.append(_make_daily_row(now - timedelta(days=i), "moex", 1000.0 + i))

    # --- Sub-case A: market_id=moex returns ONLY moex rows ---
    patcher, sess = _patch_session_factory(daily, [])
    sess._market_filter = "moex"  # type: ignore[attr-defined]
    with patcher:
        resp = TestClient(create_app()).get(
            "/api/v1/portfolio/history?days=30&market_id=moex",
            headers=_auth(),
        )
    assert resp.status_code == 200
    snaps = resp.json()["snapshots"]
    assert snaps, "Expected non-empty snapshots when filtering by market_id=moex"
    assert all(s["market_id"] == "moex" for s in snaps), (
        f"Expected only moex rows, got markets {sorted({s['market_id'] for s in snaps})}"
    )

    # --- Sub-case B: no params -> defaults (days=30, all markets) ---
    patcher2, sess2 = _patch_session_factory(daily, [])
    sess2._market_filter = None  # type: ignore[attr-defined]
    with patcher2:
        resp = TestClient(create_app()).get("/api/v1/portfolio/history", headers=_auth())
    assert resp.status_code == 200
    snaps_all = resp.json()["snapshots"]
    assert len(snaps_all) == 10, (
        f"Expected 10 rows (5 us + 5 moex) when no market_id, got {len(snaps_all)}"
    )
    markets = {s["market_id"] for s in snaps_all}
    assert markets == {"us", "moex"}, f"Expected both us and moex, got {markets}"


def test_drawdown_pct_per_market_running_peak() -> None:
    """drawdown_pct is computed per market, independently, via running peak (D-07)."""
    now = datetime.now(UTC)
    # Series ordered chronologically: equity = 100, 110, 105, 120, 90, 95
    # Expected per-row drawdown_pct (running peak per market):
    #   100 -> peak=100 dd=0
    #   110 -> peak=110 dd=0
    #   105 -> peak=110 dd=(110-105)/110 ≈ 0.04545
    #   120 -> peak=120 dd=0
    #    90 -> peak=120 dd=(120-90)/120 = 0.25
    #    95 -> peak=120 dd=(120-95)/120 ≈ 0.20833
    equities = [100.0, 110.0, 105.0, 120.0, 90.0, 95.0]
    expected_dd = [
        0.0,
        0.0,
        (110.0 - 105.0) / 110.0,
        0.0,
        (120.0 - 90.0) / 120.0,
        (120.0 - 95.0) / 120.0,
    ]
    # Stagger timestamps in strictly ascending order (5 minutes apart) so
    # the handler's ORDER BY timestamp asc returns the same sequence.
    daily_one = [
        _make_daily_row(now - timedelta(hours=10) + timedelta(minutes=5 * i), "moex", eq)
        for i, eq in enumerate(equities)
    ]

    patcher, sess = _patch_session_factory(daily_one, [])
    sess._market_filter = "moex"  # type: ignore[attr-defined]
    with patcher:
        resp = TestClient(create_app()).get(
            "/api/v1/portfolio/history?days=30&market_id=moex",
            headers=_auth(),
        )
    assert resp.status_code == 200
    snaps = resp.json()["snapshots"]
    actual_dd = [s["drawdown_pct"] for s in snaps]
    assert len(actual_dd) == len(expected_dd), (
        f"Expected {len(expected_dd)} snapshots, got {len(actual_dd)}"
    )
    for idx, (a, e) in enumerate(zip(actual_dd, expected_dd, strict=True)):
        assert abs(a - e) < 1e-6, f"row {idx}: expected drawdown {e:.6f}, got {a:.6f}"

    # --- Sub-case: per-market peaks track INDEPENDENTLY ---
    # market A: rises 100 -> 200 (no drawdown)
    # market B: peaks at 50 then falls to 25 (50% drawdown)
    # The drawdown for market B's row must be 0.5, NOT influenced by market A's higher equity.
    daily_two: list[SimpleNamespace] = []
    base_a = now - timedelta(days=2)
    base_b = now - timedelta(days=2)
    for i, (eq_a, eq_b) in enumerate([(100.0, 50.0), (150.0, 50.0), (200.0, 25.0)]):
        daily_two.append(_make_daily_row(base_a + timedelta(hours=i), "us", eq_a))
        daily_two.append(_make_daily_row(base_b + timedelta(hours=i, minutes=30), "moex", eq_b))

    # Pad to >=5 rows so primary path is taken
    daily_two.append(_make_daily_row(now - timedelta(minutes=5), "us", 200.0))
    daily_two.append(_make_daily_row(now - timedelta(minutes=4), "moex", 25.0))

    patcher2, sess2 = _patch_session_factory(daily_two, [])
    sess2._market_filter = None  # type: ignore[attr-defined]
    with patcher2:
        resp2 = TestClient(create_app()).get("/api/v1/portfolio/history?days=30", headers=_auth())
    snaps2 = resp2.json()["snapshots"]
    moex_snaps = [s for s in snaps2 if s["market_id"] == "moex"]
    us_snaps = [s for s in snaps2 if s["market_id"] == "us"]
    # us is monotonically rising → all drawdowns should be 0
    for s in us_snaps:
        assert abs(s["drawdown_pct"]) < 1e-6, (
            f"US drawdown should be 0 (monotonic rise), got {s['drawdown_pct']}"
        )
    # moex final row equity=25 vs peak=50 → drawdown=0.5
    assert moex_snaps, "Expected moex rows in mixed-market response"
    final_moex = moex_snaps[-1]
    final_dd = final_moex["drawdown_pct"]
    assert abs(final_dd - 0.5) < 1e-6, (
        f"Expected moex final drawdown 0.5 (independent of us peak), got {final_dd}"
    )


def test_logs_source_and_row_count() -> None:
    """Both primary and fallback paths emit structlog with history_source + row_count (D-08)."""
    now = datetime.now(UTC)

    # --- Primary path: ≥5 daily rows ---
    daily = [_make_daily_row(now - timedelta(days=i), "moex", 1000.0 + i) for i in range(6)]
    patcher, _ = _patch_session_factory(daily, [])
    with patcher, structlog.testing.capture_logs() as captured1:
        TestClient(create_app()).get("/api/v1/portfolio/history?days=30", headers=_auth())
    served1 = [e for e in captured1 if e.get("event") == "portfolio_history_served"]
    assert served1, "Primary path must emit portfolio_history_served"
    assert "history_source" in served1[-1], "Primary log missing history_source"
    assert "row_count" in served1[-1], "Primary log missing row_count"
    assert served1[-1]["history_source"] == "daily_equity_snapshots"
    assert served1[-1]["row_count"] == 6

    # --- Fallback path: <5 daily rows ---
    daily_sparse = [_make_daily_row(now - timedelta(days=i), "moex", 1000.0) for i in range(2)]
    sandbox = [_make_sandbox_row(now - timedelta(days=i), "moex", 7000.0) for i in range(3)]
    patcher2, _ = _patch_session_factory(daily_sparse, sandbox)
    with patcher2, structlog.testing.capture_logs() as captured2:
        TestClient(create_app()).get("/api/v1/portfolio/history?days=30", headers=_auth())
    served2 = [e for e in captured2 if e.get("event") == "portfolio_history_served"]
    assert served2, "Fallback path must emit portfolio_history_served"
    assert "history_source" in served2[-1], "Fallback log missing history_source"
    assert "row_count" in served2[-1], "Fallback log missing row_count"
    assert served2[-1]["history_source"] == "sandbox_metrics"
    assert served2[-1]["row_count"] == 3
