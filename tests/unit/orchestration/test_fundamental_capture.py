"""RED-phase failing tests for the Phase-63 fundamental-snapshot capture writer.

Wave 0 (TDD mandatory — CLAUDE.md #2): these tests define the observable contract
for the daily MOEX fundamental-capture cycle, its idempotent (as_of, symbol) upsert,
and the freshness/coverage alert BEFORE any production code exists. Every test here
MUST fail now — the methods they target are implemented in plans 63-02 / 63-03:

  - TradingLoop._fundamental_capture_cycle              (plan 02)
  - TradingPersistence.persist_fundamental_snapshot     (plan 02, sync fire-and-forget guard)
  - TradingPersistence.persist_fundamental_snapshot_async (plan 02, ON CONFLICT upsert)
  - TradingLoop._fundamental_freshness_cycle            (plan 03)

D-03 refinement (CONTEXT/RESEARCH A5): freshness is measured as *job-run liveness* —
the age of the last successful capture RUN — NOT the age of the newest snapshot
(fundamentals legitimately stay constant for weeks). The contract the implementing
plans MUST honor is documented in MARKER ATTRIBUTE CONTRACT below.

MARKER ATTRIBUTE CONTRACT (plans 02 sets, plan 03 reads):
  loop._last_fundamental_capture_at: datetime | None  — wall-clock of last successful run
  loop._last_fundamental_coverage_ratio: float | None — captured / len(universe) of last run
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from finalayze.api.alerts import AlertPriority
from finalayze.core.exceptions import DataFetchError
from finalayze.core.schemas import FundamentalSnapshot
from finalayze.orchestration.db_persistence import TradingPersistence
from finalayze.orchestration.trading_loop import TradingLoop, TradingLoopDeps

# ── Named constants (no magic numbers — ruff PLR2004) ────────────────────────
_UNIVERSE_SYMBOLS = ("SBER", "GAZP", "LKOH")
_EXPECTED_PERSIST_CALLS = 2  # 3 symbols, one returns None → 2 persisted
_AS_OF = datetime(2026, 3, 31, tzinfo=UTC)
_STALENESS_THRESHOLD_HOURS = 36  # job-run liveness window (D-03)
_STALE_RUN_AGE_HOURS = 72  # last successful run older than threshold → alert
_FRESH_RUN_AGE_HOURS = 1  # recent successful run → no alert
_COVERAGE_FLOOR = 0.5  # captured/universe ratio floor (Q2 resolution)
_LOW_COVERAGE_RATIO = 0.1  # 1/10 of universe → alert
_HEALTHY_COVERAGE_RATIO = 0.95  # above floor → no alert
_TEST_DB_URL = "postgresql+asyncpg://stub/stub"  # never connected (factory mocked)


def _make_loop() -> TradingLoop:
    """Create a minimal TradingLoop with mocked dependencies (mirrors
    tests/unit/core/test_db_persistence.py:_make_loop)."""
    settings = MagicMock()
    settings.mode = MagicMock()
    settings.mode.value = "sandbox"
    settings.effective_risk_limits.return_value = MagicMock(
        max_position_pct=Decimal("0.1"),
        max_positions_per_market=10,
        max_sector_concentration_pct=Decimal("0.3"),
        min_cash_reserve_pct=Decimal("0.1"),
        daily_loss_limit_pct=0.02,
    )
    settings.kelly_fraction = 0.5
    # Numeric scheduler knobs: APScheduler interval/cron triggers reject MagicMock
    # numerics, so _setup_scheduler() needs real ints to register every job and
    # reach the fundamental_capture add_job. Enable-flags are False so the optional
    # (meta-agent / ML / bond) branches stay off in the unit harness.
    settings.news_cycle_minutes = 2
    settings.news_poll_interval_minutes = 5
    settings.strategy_cycle_minutes = 60
    settings.daily_reset_hour_utc = 0
    settings.weekly_digest_hour_utc = 16
    settings.fundamental_capture_hour_utc = 7
    settings.meta_agent_enabled = False
    settings.ml_enabled = False
    settings.bond_cycle_enabled = False

    return TradingLoop(
        TradingLoopDeps(
            settings=settings,
            fetchers={},
            news_fetcher=MagicMock(),
            news_analyzer=MagicMock(),
            event_classifier=MagicMock(),
            impact_estimator=MagicMock(),
            strategy=MagicMock(),
            broker_router=MagicMock(),
            circuit_breakers={},
            cross_market_breaker=MagicMock(),
            alerter=MagicMock(),
            instrument_registry=MagicMock(),
        )
    )


def _make_loop_with_moex_universe() -> TradingLoop:
    """Loop whose registry/fetcher/persistence/alerter are independently patchable."""
    loop = _make_loop()
    loop._registry = MagicMock()
    loop._registry.list_by_market.return_value = [
        SimpleNamespace(symbol=s) for s in _UNIVERSE_SYMBOLS
    ]
    loop._fetchers = {"moex": MagicMock()}
    loop._persistence = MagicMock()
    loop._alerter = MagicMock()
    return loop


def _snap(symbol: str) -> FundamentalSnapshot:
    return FundamentalSnapshot(
        symbol=symbol,
        as_of=_AS_OF,
        pe_ratio=5.1,
        eps_ttm=80.0,
        currency="RUB",
    )


# ── CAPTURE-01: cycle iterates the universe and persists each non-None snapshot ──
def test_capture_cycle_persists_each_non_none_snapshot() -> None:  # -k iterates
    """registry returns 3 symbols; fetcher returns a snapshot for two and None for
    one → persist called exactly twice (the None is skipped)."""
    loop = _make_loop_with_moex_universe()
    loop._fetchers["moex"].fetch_fundamentals.side_effect = [
        _snap("SBER"),
        None,
        _snap("LKOH"),
    ]

    loop._fundamental_capture_cycle()

    loop._registry.list_by_market.assert_called_once_with("moex")
    assert loop._persistence.persist_fundamental_snapshot.call_count == _EXPECTED_PERSIST_CALLS


# ── CAPTURE-01: cron job id registered (Phase-57 Pitfall-6 guard) ────────────
def test_capture_cron_registered() -> None:  # -k registered
    """After _setup_scheduler(), 'fundamental_capture' must be a registered job id."""
    loop = _make_loop()
    loop._setup_scheduler()
    assert loop._scheduler is not None
    job_ids = {j.id for j in loop._scheduler.get_jobs()}
    assert "fundamental_capture" in job_ids


# ── CAPTURE-01 / D-02: one failing symbol does NOT abort the run ─────────────
def test_capture_degrades_per_symbol() -> None:  # -k degrade
    """A DataFetchError on the FIRST symbol must not abort the run; the remaining
    good symbols are still persisted (graceful per-symbol degrade, D-02)."""
    loop = _make_loop_with_moex_universe()
    loop._fetchers["moex"].fetch_fundamentals.side_effect = [
        DataFetchError("gRPC error for SBER"),
        _snap("GAZP"),
        _snap("LKOH"),
    ]

    loop._fundamental_capture_cycle()

    # First symbol raised; the other two still persisted (run not aborted).
    assert loop._persistence.persist_fundamental_snapshot.call_count == _EXPECTED_PERSIST_CALLS


# ── CAPTURE-02: upsert compiles to ON CONFLICT (as_of, symbol) DO UPDATE ──────
def test_upsert_stmt_is_on_conflict() -> None:  # -k upsert
    """persist_fundamental_snapshot_async must build a pg upsert: capture the
    executed statement, compile against the postgresql dialect, and assert the
    ON CONFLICT shape plus non-key columns in the SET clause."""
    import asyncio

    from sqlalchemy.dialects import postgresql

    persistence = TradingPersistence(db_url=_TEST_DB_URL, async_loop=None)

    captured: dict[str, object] = {}

    class _StubSession:
        async def __aenter__(self) -> _StubSession:
            return self

        async def __aexit__(self, *exc: object) -> None:
            return None

        async def execute(self, stmt: object) -> None:
            captured["stmt"] = stmt

        async def commit(self) -> None:
            return None

    factory = MagicMock(return_value=_StubSession())
    with patch.object(persistence, "_get_bg_session_factory", return_value=factory):
        asyncio.run(persistence.persist_fundamental_snapshot_async(_snap("SBER")))

    stmt = captured["stmt"]
    sql = str(stmt.compile(dialect=postgresql.dialect()))  # type: ignore[attr-defined]
    assert "ON CONFLICT (as_of, symbol) DO UPDATE" in sql
    # Non-key columns must appear in the SET clause; key columns must not be SET.
    set_clause = sql.split("DO UPDATE")[1]
    assert "pe_ratio" in set_clause
    assert "eps_ttm" in set_clause
    assert "currency" in set_clause


# ── CAPTURE-02 / D-04 / Pitfall-4: table absent → fail safe (no crash) ────────
def test_table_absent_fails_safe() -> None:  # -k table_absent
    """If the session.execute raises (UndefinedTable — migration not yet applied),
    the sync-guarded persist entry point must NOT raise; the loop continues."""
    persistence = TradingPersistence(db_url=_TEST_DB_URL, async_loop=None)

    class _RaisingSession:
        async def __aenter__(self) -> _RaisingSession:
            return self

        async def __aexit__(self, *exc: object) -> None:
            return None

        async def execute(self, stmt: object) -> None:
            raise RuntimeError('relation "fundamental_snapshots" does not exist')

        async def commit(self) -> None:
            return None

    factory = MagicMock(return_value=_RaisingSession())
    with patch.object(persistence, "_get_bg_session_factory", return_value=factory):
        # Must swallow via the _persist_to_db fire-and-forget guard — never raises.
        persistence.persist_fundamental_snapshot(_snap("SBER"))


# ── CAPTURE-03 / D-03: stale capture-RUN liveness → alert ─────────────────────
def test_freshness_alerts_when_run_stale() -> None:  # -k freshness
    """When the last successful capture RUN is older than the staleness threshold,
    _fundamental_freshness_cycle alerts via send_alert(IMPORTANT). Asserts on
    job-run liveness (D-03), NOT newest-snapshot as_of-age."""
    loop = _make_loop_with_moex_universe()
    loop._last_fundamental_capture_at = datetime.now(UTC) - timedelta(hours=_STALE_RUN_AGE_HOURS)
    loop._last_fundamental_coverage_ratio = _HEALTHY_COVERAGE_RATIO

    loop._fundamental_freshness_cycle()

    loop._alerter.send_alert.assert_called_once()
    _, kwargs = loop._alerter.send_alert.call_args
    assert kwargs.get("priority") == AlertPriority.IMPORTANT
    # Security (T-63-01): alert message carries only counts/ages — never a secret.
    message = loop._alerter.send_alert.call_args[0][0]
    assert "token" not in message.lower()
    assert "postgresql" not in message.lower()
    assert "://" not in message


# ── CAPTURE-03 / Q2: low coverage ratio → alert (catches gRPC-wide outage) ────
def test_freshness_alerts_when_coverage_low() -> None:  # -k freshness
    """A recent run that captured a ratio below the coverage floor must still
    alert — catches a gRPC-wide outage the per-symbol degrade swallows."""
    loop = _make_loop_with_moex_universe()
    loop._last_fundamental_capture_at = datetime.now(UTC) - timedelta(hours=_FRESH_RUN_AGE_HOURS)
    loop._last_fundamental_coverage_ratio = _LOW_COVERAGE_RATIO

    loop._fundamental_freshness_cycle()

    loop._alerter.send_alert.assert_called_once()
    _, kwargs = loop._alerter.send_alert.call_args
    assert kwargs.get("priority") == AlertPriority.IMPORTANT


# ── CAPTURE-03: healthy run → NO alert ────────────────────────────────────────
def test_freshness_healthy_no_alert() -> None:  # -k healthy
    """A recent successful run with coverage above the floor must NOT alert."""
    loop = _make_loop_with_moex_universe()
    loop._last_fundamental_capture_at = datetime.now(UTC) - timedelta(hours=_FRESH_RUN_AGE_HOURS)
    loop._last_fundamental_coverage_ratio = _HEALTHY_COVERAGE_RATIO

    loop._fundamental_freshness_cycle()

    assert loop._alerter.send_alert.call_count == 0


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-q"])
