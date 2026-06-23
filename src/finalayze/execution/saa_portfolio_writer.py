"""Create and read SAA portfolios from the database (Phase 78).

This module provides the missing CREATE and READ paths for SAA portfolios
(Phase 77 shipped only the read-only deposit_loader). The single-operator
MVP persists a budget_rub + risk_profile as the "active" portfolio, with
exactly one active at a time via deactivate-then-insert in a transaction.

Validation is fail-closed: unknown profiles or invalid budgets raise
ConfigurationError before any database write.
"""

from __future__ import annotations

from decimal import ROUND_HALF_EVEN, Decimal
from typing import TYPE_CHECKING

import structlog

from finalayze.config.allocation_profiles import load_allocation_profiles
from finalayze.core.exceptions import ConfigurationError
from finalayze.core.schemas import RiskProfile

if TYPE_CHECKING:
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

_log = structlog.get_logger()

# Validation constants
_ZERO = Decimal(0)
_BUDGET_PRECISION = Decimal("0.01")


def resolve_risk_profile(name: str | None) -> RiskProfile:
    """Resolve a risk profile name to RiskProfile enum, fail-closed.

    Args:
        name: The profile name string to resolve (e.g., 'conservative').

    Returns:
        The resolved RiskProfile enum value.

    Raises:
        ConfigurationError: If name is None, empty, or unknown, listing valid choices.
    """
    if name is None or not name:
        valid = [p.value for p in RiskProfile]
        msg = f"risk profile name required; valid choices: {valid}"
        raise ConfigurationError(msg)

    try:
        return RiskProfile(name)
    except ValueError as exc:
        valid = [p.value for p in RiskProfile]
        msg = f"unknown risk profile '{name}'; valid choices: {valid}"
        raise ConfigurationError(msg) from exc


def coerce_budget(value: int | str | float | Decimal) -> Decimal:
    """Coerce budget to Decimal(0.01) precision, reject non-positive.

    Converts the input to Decimal via str() (avoiding float precision loss),
    quantizes to 0.01 using ROUND_HALF_EVEN, and rejects zero or negative.

    Args:
        value: Budget value (int, str, float, or Decimal).

    Returns:
        Decimal budget exact to 0.01 (two decimal places).

    Raises:
        ConfigurationError: If value <= 0 or cannot be converted to Decimal.
    """
    try:
        decimal_value = Decimal(str(value))
    except (ValueError, TypeError) as exc:
        msg = f"budget must be convertible to Decimal; got {value!r}: {exc}"
        raise ConfigurationError(msg) from exc

    if decimal_value <= _ZERO:
        msg = f"budget must be positive; got {decimal_value}"
        raise ConfigurationError(msg)

    return decimal_value.quantize(_BUDGET_PRECISION, rounding=ROUND_HALF_EVEN)


async def get_active_portfolio(
    session_factory: async_sessionmaker[AsyncSession],
) -> tuple[UUID, str, Decimal] | None:
    """Fetch the single active portfolio (budget_rub, risk_profile, id).

    Returns:
        Tuple of (portfolio_id, risk_profile, budget_rub) if found,
        None if no active portfolio exists.
    """
    from sqlalchemy import select  # noqa: PLC0415

    from finalayze.core.models import SaaPortfolioModel  # noqa: PLC0415

    async with session_factory() as session:
        portfolio = (
            await session.execute(
                select(SaaPortfolioModel)
                .where(SaaPortfolioModel.is_active.is_(True))
                .order_by(SaaPortfolioModel.created_at.desc())
                .limit(1)
            )
        ).scalar_one_or_none()

    if portfolio is None:
        return None

    return (portfolio.id, portfolio.risk_profile, portfolio.budget_rub)


async def create_active_portfolio(
    session_factory: async_sessionmaker[AsyncSession],
    *,
    budget_rub: Decimal,
    risk_profile: RiskProfile,
) -> UUID:
    """Create the active portfolio, deactivating any existing active in one transaction.

    This is the CREATE path for Phase 78: exactly one active portfolio at a time.
    The transaction ensures atomicity:
      1. Deactivate all rows with is_active=True.
      2. Insert the new active portfolio row.

    All validation (budget, profile, profile availability in config) must occur
    BEFORE this call. Validation failures in this function are precondition
    violations (should never happen in production code).

    Args:
        session_factory: AsyncSessionMaker for database access.
        budget_rub: Validated Decimal budget (≥ 0.01, quantized to 0.01).
        risk_profile: Validated RiskProfile enum.

    Returns:
        UUID of the newly created portfolio.

    Raises:
        ConfigurationError: If the risk profile is not in the loaded allocation profiles.
    """
    from sqlalchemy import update  # noqa: PLC0415

    from finalayze.core.models import SaaPortfolioModel  # noqa: PLC0415

    # Pre-flight: ensure the profile exists in the config (fail-closed).
    profiles = load_allocation_profiles()
    if risk_profile not in profiles:
        msg = f"risk profile {risk_profile.value!r} not in allocation_profiles.yaml"
        raise ConfigurationError(msg)

    new_portfolio_id: UUID | None = None
    async with session_factory() as session, session.begin():
        # Deactivate all existing active portfolios.
        await session.execute(
            update(SaaPortfolioModel)
            .where(SaaPortfolioModel.is_active.is_(True))
            .values(is_active=False)
        )

        # Insert the new active portfolio.
        new_portfolio = SaaPortfolioModel(
            risk_profile=risk_profile.value,
            budget_rub=budget_rub,
            is_active=True,
            deposit_accumulators=None,
        )
        session.add(new_portfolio)
        await session.flush()
        new_portfolio_id = new_portfolio.id

    if new_portfolio_id is None:
        msg = "portfolio creation failed: no id assigned"
        raise RuntimeError(msg)

    _log.info(
        "create_active_portfolio_success",
        portfolio_id=str(new_portfolio_id),
        budget_rub=str(budget_rub),
        risk_profile=risk_profile.value,
    )
    return new_portfolio_id
