"""Budget-driver ANTI-HOLLOW gate (Phase 78 P3-05).

The binding test: rescaling every leg to ``budget * weight`` makes ``total_return_pct``
SCALE-INVARIANT while ``rebalance_cost`` and ``realized_ndfl`` scale EXACTLY linearly with
the budget -- proving the budget genuinely drives the economics through the REAL
``AllocationOrchestrator.run()`` path (no DB, no test hook). Because the rescale is purely
multiplicative, a 10x budget scales the whole book by exactly 10x: the rebalance UNIT scales
are budget-invariant, but the per-leg RUB prices are 10x, so the traded notional (and thus
cost + realized NDFL) is exactly 10x while the return ratio is unchanged. A test that compared
the driver to itself, or only asserted a number is non-zero, would be HOLLOW; this asserts the
exact 10x relationship through the real run().
"""

from __future__ import annotations

import os
from datetime import date, timedelta
from decimal import Decimal

import pytest

from finalayze.config.allocation_profiles import load_allocation_profiles
from finalayze.core.schemas import AssetClass, RiskProfile
from finalayze.orchestration.allocation import AllocationOrchestrator
from finalayze.orchestration.budget_driver import _rescale_curve

_START = date(2025, 1, 1)
_N_BARS = 365  # ~1y daily; spans the 2025-06-06 regime boundary so the tilt + rebalances fire
_SMALL = Decimal(100_000)
_LARGE = Decimal(1_000_000)  # 10x
_TEN = Decimal(10)


def _curve(base: int, daily: str) -> list[tuple[date, Decimal]]:
    """A deterministic geometric daily TR curve (drifting so quarterly rebalances trade)."""
    factor = Decimal(daily)
    return [(_START + timedelta(days=i), Decimal(base) * factor**i) for i in range(_N_BARS)]


def _run_at(budget: Decimal) -> object:
    """Rescale each leg to budget*weight (the driver's core) and run the FROZEN orchestrator."""
    weights = load_allocation_profiles()[RiskProfile.BALANCED].weights
    # MCFTR equity is a real index level (~3000), rescaled MULTIPLICATIVELY -- not a base swap.
    deposit = _curve(100, "1.00055")
    ofz = _curve(110, "1.0004")
    equity = _curve(3000, "1.0008")
    dep = _rescale_curve(deposit, budget * weights[AssetClass.DEPOSIT])
    o = _rescale_curve(ofz, budget * weights[AssetClass.OFZ_PK])
    eq = _rescale_curve(equity, budget * weights[AssetClass.EQUITY])
    return AllocationOrchestrator(risk_profile=RiskProfile.BALANCED).run(dep, o, eq)


def test_budget_driver_scale_invariant_and_cost_linear() -> None:
    """total_return_pct scale-invariant; cost + realized_ndfl EXACTLY 10x at a 10x budget."""
    small = _run_at(_SMALL)
    large = _run_at(_LARGE)

    # The opening notional equals the budget (sum of leg targets = budget * sum(weights) = budget).
    assert abs(small.merged_equity_curve[0] - _SMALL) < Decimal(1)
    assert abs(large.merged_equity_curve[0] - _LARGE) < Decimal(1)

    # total_return_pct is a RATIO -> scale-invariant across budgets.
    assert abs(small.total_return_pct - large.total_return_pct) < 1e-9

    # rebalance_cost arises from the REAL per-leg rescale delta and scales EXACTLY 10x.
    assert small.rebalance_cost > Decimal(0), "the drifting book must actually trade (non-hollow)"
    assert large.rebalance_cost == small.rebalance_cost * _TEN

    # realized NDFL likewise scales exactly 10x (FIFO gain on the real, 10x-priced delta).
    if small.realized_ndfl > Decimal(0):
        assert large.realized_ndfl == small.realized_ndfl * _TEN


@pytest.mark.integration
@pytest.mark.skipif(
    not (os.environ.get("FINALAYZE_DATABASE_URL") or os.environ.get("DATABASE_URL")),
    reason="FINALAYZE_DATABASE_URL not set; integration DB unavailable",
)
def test_run_with_active_budget_drives_opening_notional_from_db() -> None:
    """End-to-end: run_with_active_budget reads the persisted budget and opens the book at it."""
    import asyncio  # noqa: PLC0415

    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine  # noqa: PLC0415

    from finalayze.execution.saa_portfolio_writer import create_active_portfolio  # noqa: PLC0415
    from finalayze.orchestration.budget_driver import run_with_active_budget  # noqa: PLC0415

    url = os.environ.get("FINALAYZE_DATABASE_URL") or os.environ["DATABASE_URL"]
    from alembic import command  # noqa: PLC0415
    from alembic.config import Config  # noqa: PLC0415

    cfg = Config("alembic/alembic.ini")
    cfg.set_main_option("sqlalchemy.url", url)
    command.upgrade(cfg, "head")

    async def _run() -> None:
        engine = create_async_engine(url)
        try:
            sf = async_sessionmaker(engine, expire_on_commit=False)
            await create_active_portfolio(sf, budget_rub=_LARGE, risk_profile=RiskProfile.BALANCED)
            result = await run_with_active_budget(
                AllocationOrchestrator(risk_profile=RiskProfile.BALANCED),
                _curve(100, "1.00055"),
                _curve(110, "1.0004"),
                _curve(3000, "1.0008"),
                sf,
            )
            assert abs(result.merged_equity_curve[0] - _LARGE) < Decimal(1)
        finally:
            await engine.dispose()

    asyncio.run(_run())
