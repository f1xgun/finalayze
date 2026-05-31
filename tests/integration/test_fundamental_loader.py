"""DB-backed integration test for the FUNDML-01 fundamental loader path.

Inserts a backfilled-style ``fundamental_snapshots`` row for a ru_blue_chips
peer and asserts:

  1. ``fetch_fundamental_snapshots([...peers...], end_dt, settings)`` returns it.
  2. ``MarketDataLoader(...).load(SimpleNamespace(market="moex", symbols=...))``
     populates ``moex_data.fundamentals`` non-empty with the inserted snapshot.
  3. A future-dated snapshot (``as_of > end_dt``) is NOT returned by the reader.

Operator/CI-gated: the test SKIPS cleanly when no DB is reachable (mirrors the
candle reader's swallow-to-empty resilience). It never errors. The reader and
``_load_moex`` themselves degrade to ``()/None`` on DB failure, so we probe
connectivity up front and skip rather than assert against an empty result.
"""

from __future__ import annotations

import asyncio
import os
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.integration

# ── Named constants (no magic numbers — ruff PLR2004) ────────────────────────
_PEERS = ["SBER", "LKOH", "GMKN", "ROSN"]
_SYMBOL = "SBER"
_AS_OF = datetime(2023, 6, 1, tzinfo=UTC)
_FUTURE_AS_OF = datetime(2099, 1, 1, tzinfo=UTC)
_END_DT = datetime(2023, 12, 31, tzinfo=UTC)
_START = datetime(2023, 1, 1, tzinfo=UTC).date()
_END = _END_DT.date()
_PE_RATIO = 8.0
_EPS_TTM = 50.0
_EXPECTED_MIN_ROWS = 1


def _db_url() -> str:
    url = os.environ.get("FINALAYZE_DATABASE_URL") or os.environ.get("DATABASE_URL")
    if not url:
        pytest.skip("FINALAYZE_DATABASE_URL not set; integration DB unavailable")
    return url


def _settings_for(url: str) -> object:
    # The reader only accesses ``settings.database_url``; a lightweight duck-typed
    # object exposing exactly that is sufficient and keeps the test self-contained.
    return SimpleNamespace(database_url=url)


async def _insert_and_cleanup(url: str) -> None:
    """Insert the test rows (idempotent upsert) and ensure the table exists."""
    from sqlalchemy.dialects.postgresql import insert as pg_insert  # noqa: PLC0415
    from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine  # noqa: PLC0415

    from finalayze.core.models import FundamentalSnapshotModel  # noqa: PLC0415

    engine = create_async_engine(url, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(FundamentalSnapshotModel.__table__.create, checkfirst=True)
    async with AsyncSession(engine) as session:
        for as_of, pe in ((_AS_OF, _PE_RATIO), (_FUTURE_AS_OF, _PE_RATIO)):
            stmt = pg_insert(FundamentalSnapshotModel).values(
                as_of=as_of,
                symbol=_SYMBOL,
                pe_ratio=pe,
                eps_ttm=_EPS_TTM,
                currency="RUB",
            )
            stmt = stmt.on_conflict_do_update(
                index_elements=["as_of", "symbol"],
                set_={"pe_ratio": pe, "eps_ttm": _EPS_TTM},
            )
            await session.execute(stmt)
        await session.commit()
    await engine.dispose()


async def _delete_rows(url: str) -> None:
    from sqlalchemy import delete  # noqa: PLC0415
    from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine  # noqa: PLC0415

    from finalayze.core.models import FundamentalSnapshotModel  # noqa: PLC0415

    engine = create_async_engine(url, echo=False)
    async with AsyncSession(engine) as session:
        await session.execute(
            delete(FundamentalSnapshotModel).where(
                FundamentalSnapshotModel.symbol == _SYMBOL,
                FundamentalSnapshotModel.as_of.in_([_AS_OF, _FUTURE_AS_OF]),
            )
        )
        await session.commit()
    await engine.dispose()


def _probe_db(url: str) -> None:
    """Skip the test if the DB is unreachable (never error)."""
    try:
        asyncio.run(_insert_and_cleanup(url))
    except Exception as exc:  # connection refused, auth, etc.
        pytest.skip(f"integration DB unreachable: {exc}")


def test_fundamental_loader_db_path() -> None:
    """fetch_fundamental_snapshots + _load_moex deliver fundamental_snapshots
    from the DB, look-ahead-safe (future as_of excluded)."""
    from scripts.training.data_loader import fetch_fundamental_snapshots  # noqa: PLC0415

    from finalayze.data.loader import MarketDataLoader  # noqa: PLC0415

    url = _db_url()
    _probe_db(url)
    settings = _settings_for(url)

    try:
        # 1) The reader returns the inserted snapshot and excludes the future one.
        snaps = asyncio.run(fetch_fundamental_snapshots(_PEERS, _END_DT, settings))  # type: ignore[arg-type]
        assert len(snaps) >= _EXPECTED_MIN_ROWS
        as_ofs = {s.as_of for s in snaps}
        assert _AS_OF in as_ofs
        assert _FUTURE_AS_OF not in as_ofs, "future-dated snapshot must be excluded"
        sber = next(s for s in snaps if s.symbol == _SYMBOL and s.as_of == _AS_OF)
        assert sber.pe_ratio == pytest.approx(_PE_RATIO)

        # 2) The loader populates moex_data.fundamentals non-empty.
        loader = MarketDataLoader(settings=settings)  # type: ignore[arg-type]
        seg_cfg = SimpleNamespace(market="moex", symbols=_PEERS)
        ctx = loader.load(seg_cfg, _START, _END)
        assert ctx.moex_data is not None
        assert ctx.moex_data.fundamentals is not None
        loaded_as_ofs = {s.as_of for s in ctx.moex_data.fundamentals}
        assert _AS_OF in loaded_as_ofs
        assert _FUTURE_AS_OF not in loaded_as_ofs
    finally:
        asyncio.run(_delete_rows(url))
