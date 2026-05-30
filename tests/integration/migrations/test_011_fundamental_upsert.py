"""Integration-gated idempotency proof for migration 011 + the (as_of, symbol)
upsert (CAPTURE-02).

This is the 9th DB-integration test. It needs a real Postgres/TimescaleDB and is
operator/CI-gated: it SKIPS cleanly when ``FINALAYZE_DATABASE_URL`` is unset (no
dev Postgres in the worktree — D-04). The skip-gating mirrors
``tests/integration/test_alembic_upgrade.py``.

The "no duplicate row" guarantee can only be proven against real Postgres
(``ON CONFLICT`` is a server-side primitive), so this single behavioral assertion
lives here while the statement *shape* is unit-tested in
``tests/unit/orchestration/test_fundamental_capture.py``.

RED note: ``persist_fundamental_snapshot_async`` does not exist yet (plan 02).
Because this test SKIPS without a DB, it neither passes nor fails in the worktree —
it is the operator-deferred behavioral proof.
"""

from __future__ import annotations

import asyncio
import os
from datetime import UTC, datetime

import pytest

pytestmark = pytest.mark.integration

# ── Named constants (no magic numbers — ruff PLR2004) ────────────────────────
_SYMBOL = "SBER"
_AS_OF = datetime(2026, 3, 31, tzinfo=UTC)
_PE_FIRST = 5.1  # initial upsert value
_PE_SECOND = 6.2  # second upsert overwrites pe_ratio
_EXPECTED_ROW_COUNT = 1  # second upsert of same (as_of, symbol) → still one row


def _db_url() -> str:
    url = os.environ.get("FINALAYZE_DATABASE_URL") or os.environ.get("DATABASE_URL")
    if not url:
        pytest.skip("FINALAYZE_DATABASE_URL not set; integration DB unavailable")
    return url


def test_upsert_no_duplicate_rows() -> None:
    """Two upserts of the same (as_of, symbol) yield exactly one row, and the
    second upsert's value overwrites the first (idempotent — CAPTURE-02)."""
    import sqlalchemy as sa
    from alembic import command
    from alembic.config import Config

    from finalayze.core.schemas import FundamentalSnapshot
    from finalayze.orchestration.db_persistence import TradingPersistence

    url = _db_url()

    # Apply migration 011 (idempotent via if_not_exists guards).
    cfg = Config("alembic/alembic.ini")
    cfg.set_main_option("sqlalchemy.url", url)
    command.upgrade(cfg, "head")

    async def _run() -> None:
        loop = asyncio.get_running_loop()
        persistence = TradingPersistence(db_url=url, async_loop=loop)
        first = FundamentalSnapshot(
            symbol=_SYMBOL, as_of=_AS_OF, pe_ratio=_PE_FIRST, currency="RUB"
        )
        second = FundamentalSnapshot(
            symbol=_SYMBOL, as_of=_AS_OF, pe_ratio=_PE_SECOND, currency="RUB"
        )
        await persistence.persist_fundamental_snapshot_async(first)
        await persistence.persist_fundamental_snapshot_async(second)

    asyncio.run(_run())

    # Verify exactly one row and that pe_ratio reflects the second upsert.
    sync_url = url.replace("+asyncpg", "").replace("postgresql+asyncpg", "postgresql")
    engine = sa.create_engine(sync_url)
    try:
        with engine.connect() as conn:
            count = conn.execute(
                sa.text(
                    "SELECT count(*) FROM fundamental_snapshots "
                    "WHERE symbol = :sym AND as_of = :asof"
                ),
                {"sym": _SYMBOL, "asof": _AS_OF},
            ).scalar_one()
            assert count == _EXPECTED_ROW_COUNT, "second upsert must not create a duplicate row"

            pe = conn.execute(
                sa.text(
                    "SELECT pe_ratio FROM fundamental_snapshots "
                    "WHERE symbol = :sym AND as_of = :asof"
                ),
                {"sym": _SYMBOL, "asof": _AS_OF},
            ).scalar_one()
            assert float(pe) == _PE_SECOND, "upsert must overwrite pe_ratio with the latest value"
    finally:
        # Clean up the test row so the gated DB stays reusable.
        with engine.begin() as conn:
            conn.execute(
                sa.text("DELETE FROM fundamental_snapshots WHERE symbol = :sym AND as_of = :asof"),
                {"sym": _SYMBOL, "asof": _AS_OF},
            )
        engine.dispose()
