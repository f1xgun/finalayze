"""Live-DB integration tests for SAA persistence migration 012 (Phase 77 P2-04).

Skipped locally (no FINALAYZE_DATABASE_URL); runs in CI. Mirrors the proven
``test_alembic_migrations.py`` pattern: actually RUN ``alembic upgrade head`` (so
migration 012 executes against a real Postgres) and introspect ``information_schema``
for byte-for-byte shape parity — NOT ``Base.metadata.create_all`` (which would test the
ORM, not the migration, and let a migration-only bug ship past the static AST test).
"""

from __future__ import annotations

import os
import uuid
from datetime import UTC, date, datetime
from decimal import Decimal

import pytest

pytestmark = pytest.mark.integration


def _db_url() -> str:
    url = os.environ.get("FINALAYZE_DATABASE_URL") or os.environ.get("DATABASE_URL")
    if not url:
        pytest.skip("FINALAYZE_DATABASE_URL not set; integration DB unavailable")
    return url


def _sync_url(url: str) -> str:
    return url.replace("+asyncpg", "").replace("postgresql+asyncpg", "postgresql")


def _upgrade_head(url: str) -> None:
    from alembic import command  # noqa: PLC0415
    from alembic.config import Config  # noqa: PLC0415

    cfg = Config("alembic/alembic.ini")
    cfg.set_main_option("sqlalchemy.url", url)
    command.upgrade(cfg, "head")


def test_012_creates_saa_tables_with_correct_shape() -> None:
    """Migration 012 creates saa_portfolios + deposit_tranches with byte-exact column shape.

    Runs the REAL migration (alembic upgrade head) and introspects information_schema:
    money columns Numeric(20,2), annual_rate Numeric(8,4), timestamps are tz-aware, the
    deposit dates are plain DATE, and the FK to saa_portfolios.id exists.
    """
    import sqlalchemy as sa  # noqa: PLC0415

    url = _db_url()
    _upgrade_head(url)
    engine = sa.create_engine(_sync_url(url))
    try:
        insp = sa.inspect(engine)
        assert insp.has_table("saa_portfolios")
        assert insp.has_table("deposit_tranches")

        with engine.connect() as conn:
            shape = {
                (r[0], r[1]): (r[2], r[3], r[4])
                for r in conn.execute(
                    sa.text(
                        "SELECT table_name, column_name, data_type, "
                        "numeric_precision, numeric_scale "
                        "FROM information_schema.columns "
                        "WHERE table_name IN ('saa_portfolios', 'deposit_tranches')"
                    )
                ).fetchall()
            }
            # Money: Numeric(20, 2).
            for col in ("principal", "accrued_net", "accrued_gross"):
                assert shape[("deposit_tranches", col)][1:] == (20, 2), col
            assert shape[("saa_portfolios", "budget_rub")][1:] == (20, 2)
            # Rate: Numeric(8, 4).
            assert shape[("deposit_tranches", "annual_rate")][1:] == (8, 4)
            # Dates are plain DATE; timestamps are tz-aware.
            assert shape[("deposit_tranches", "open_date")][0] == "date"
            assert shape[("deposit_tranches", "maturity_date")][0] == "date"
            assert shape[("saa_portfolios", "created_at")][0] == "timestamp with time zone"
            assert shape[("deposit_tranches", "updated_at")][0] == "timestamp with time zone"

            # The FK deposit_tranches.portfolio_id -> saa_portfolios exists.
            fk_count = conn.execute(
                sa.text(
                    "SELECT COUNT(*) FROM pg_constraint "
                    "WHERE conrelid = 'deposit_tranches'::regclass AND contype = 'f'"
                )
            ).scalar()
            assert fk_count == 1, f"deposit_tranches must have one FK; got {fk_count}"

        # Both secondary indexes were created (design-doc success criterion).
        idx_names = {ix["name"] for ix in insp.get_indexes("saa_portfolios")} | {
            ix["name"] for ix in insp.get_indexes("deposit_tranches")
        }
        assert "ix_saa_portfolios_is_active" in idx_names
        assert "ix_deposit_tranches_portfolio_id" in idx_names
    finally:
        engine.dispose()


def test_012_on_delete_restrict_blocks_parent_delete() -> None:
    """ON DELETE RESTRICT: deleting a portfolio that still has tranches raises (D-3 safety)."""
    import asyncio  # noqa: PLC0415

    import sqlalchemy as sa  # noqa: PLC0415
    from sqlalchemy.exc import IntegrityError  # noqa: PLC0415
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine  # noqa: PLC0415

    from finalayze.core.models import DepositTrancheModel, SaaPortfolioModel  # noqa: PLC0415

    url = _db_url()
    _upgrade_head(url)
    pid = uuid.uuid4()
    tid = uuid.uuid4()

    async def _run() -> None:
        engine = create_async_engine(url)
        try:
            session_factory = async_sessionmaker(engine, expire_on_commit=False)
            async with session_factory() as session:
                session.add(
                    SaaPortfolioModel(
                        id=pid,
                        risk_profile="balanced",
                        budget_rub=Decimal(100000),
                        is_active=True,
                        created_at=datetime.now(UTC),
                        updated_at=datetime.now(UTC),
                    )
                )
                session.add(
                    DepositTrancheModel(
                        id=tid,
                        portfolio_id=pid,
                        principal=Decimal(50000),
                        term_months=3,
                        annual_rate=Decimal("0.1800"),
                        open_date=date(2026, 1, 15),
                        maturity_date=date(2026, 4, 15),
                        updated_at=datetime.now(UTC),
                    )
                )
                await session.commit()
            async with session_factory() as session:
                await session.execute(
                    sa.delete(SaaPortfolioModel).where(SaaPortfolioModel.id == pid)
                )
                with pytest.raises(IntegrityError):
                    await session.commit()
        finally:
            await engine.dispose()

    asyncio.run(_run())


def test_012_fk_enforced_rejects_bogus_portfolio_id() -> None:
    """The portfolio_id FK is ENFORCED: a tranche pointing at a nonexistent portfolio fails."""
    import asyncio  # noqa: PLC0415

    import sqlalchemy as sa  # noqa: PLC0415
    from sqlalchemy.exc import IntegrityError  # noqa: PLC0415
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine  # noqa: PLC0415

    from finalayze.core.models import DepositTrancheModel  # noqa: PLC0415

    url = _db_url()
    _upgrade_head(url)

    async def _run() -> None:
        engine = create_async_engine(url)
        try:
            session_factory = async_sessionmaker(engine, expire_on_commit=False)
            async with session_factory() as session:
                session.add(
                    DepositTrancheModel(
                        id=uuid.uuid4(),
                        portfolio_id=uuid.uuid4(),  # bogus — no such portfolio
                        principal=Decimal(50000),
                        term_months=3,
                        annual_rate=Decimal("0.1800"),
                        open_date=date(2026, 1, 15),
                        maturity_date=date(2026, 4, 15),
                        updated_at=datetime.now(UTC),
                    )
                )
                with pytest.raises(IntegrityError):
                    await session.commit()
        finally:
            await engine.dispose()

    asyncio.run(_run())


def test_012_orm_round_trip_preserves_decimal_precision() -> None:
    """A portfolio + tranche round-trip through the migrated tables with exact Decimal precision."""
    import asyncio  # noqa: PLC0415

    import sqlalchemy as sa  # noqa: PLC0415
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine  # noqa: PLC0415

    from finalayze.core.models import DepositTrancheModel, SaaPortfolioModel  # noqa: PLC0415

    url = _db_url()
    _upgrade_head(url)
    portfolio_id = uuid.uuid4()
    tranche_id = uuid.uuid4()

    async def _run() -> None:
        engine = create_async_engine(url)
        try:
            session_factory = async_sessionmaker(engine, expire_on_commit=False)
            async with session_factory() as session:
                session.add(
                    SaaPortfolioModel(
                        id=portfolio_id,
                        risk_profile="balanced",
                        budget_rub=Decimal("1234567.89"),
                        is_active=True,
                        created_at=datetime.now(UTC),
                        updated_at=datetime.now(UTC),
                    )
                )
                session.add(
                    DepositTrancheModel(
                        id=tranche_id,
                        portfolio_id=portfolio_id,
                        principal=Decimal("50000.00"),
                        term_months=3,
                        annual_rate=Decimal("0.1800"),
                        open_date=date(2026, 1, 15),
                        maturity_date=date(2026, 4, 15),
                        accrued_net=Decimal("123.45"),
                        accrued_gross=Decimal("141.89"),
                        broken=False,
                        bank_id=None,
                        updated_at=datetime.now(UTC),
                    )
                )
                await session.commit()

            async with session_factory() as session:
                p = (
                    await session.execute(
                        sa.select(SaaPortfolioModel).where(SaaPortfolioModel.id == portfolio_id)
                    )
                ).scalar_one()
                t = (
                    await session.execute(
                        sa.select(DepositTrancheModel).where(DepositTrancheModel.id == tranche_id)
                    )
                ).scalar_one()
                assert p.budget_rub == Decimal("1234567.89")
                assert p.risk_profile == "balanced"
                assert t.annual_rate == Decimal("0.1800")
                assert t.accrued_net == Decimal("123.45")
                assert t.accrued_gross == Decimal("141.89")
                assert t.broken is False
        finally:
            await engine.dispose()

    asyncio.run(_run())


def test_012_deposit_accumulators_persist_and_reload() -> None:
    """deposit_accumulators JSONB persists + the loader restores broker state (P2-06)."""
    import asyncio  # noqa: PLC0415

    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine  # noqa: PLC0415

    from finalayze.core.models import DepositTrancheModel, SaaPortfolioModel  # noqa: PLC0415
    from finalayze.execution.deposit_loader import load_deposit_broker_from_db  # noqa: PLC0415

    url = _db_url()
    _upgrade_head(url)
    pid = uuid.uuid4()
    accumulators = {
        "ytd_deposit_gross": "1234.56",
        "running_max_key_rate": "0.1800",
        "current_year": 2026,
        "total_interest_gross": "1234.56",
        "total_interest_net": "1100.00",
        "total_tax_paid": "134.56",
        "last_accrual_date": "2026-03-15",
    }

    async def _run() -> None:
        engine = create_async_engine(url)
        try:
            session_factory = async_sessionmaker(engine, expire_on_commit=False)
            async with session_factory() as session:
                session.add(
                    SaaPortfolioModel(
                        id=pid,
                        risk_profile="balanced",
                        budget_rub=Decimal(100000),
                        is_active=True,
                        created_at=datetime.now(UTC),
                        updated_at=datetime.now(UTC),
                        deposit_accumulators=accumulators,
                    )
                )
                session.add(
                    DepositTrancheModel(
                        id=uuid.uuid4(),
                        portfolio_id=pid,
                        principal=Decimal(50000),
                        term_months=3,
                        annual_rate=Decimal("0.1800"),
                        open_date=date(2026, 1, 15),
                        maturity_date=date(2026, 4, 15),
                        accrued_net=Decimal("12.34"),
                        accrued_gross=Decimal("14.18"),
                        updated_at=datetime.now(UTC),
                    )
                )
                await session.commit()

            broker = await load_deposit_broker_from_db(pid, date(2026, 2, 1), session_factory)
            assert broker is not None
            assert broker._ytd_deposit_gross == Decimal("1234.56")  # noqa: SLF001
            assert broker._running_max_key_rate == Decimal("0.1800")  # noqa: SLF001
            assert broker._current_year == 2026  # noqa: SLF001
            assert len(broker._tranches) == 1  # noqa: SLF001
            assert broker._tranches[0].accrued_net == Decimal("12.34")  # noqa: SLF001
        finally:
            await engine.dispose()

    asyncio.run(_run())
