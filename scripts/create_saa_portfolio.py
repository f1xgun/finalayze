"""Create a new SAA portfolio with budget + risk profile (Phase 78 P3-04).

A thin CLI harness that accepts command-line budget and risk-profile arguments,
validates them fail-closed, and persists the active portfolio to the database.

Usage:
    uv run python scripts/create_saa_portfolio.py --budget-rub 100000 --risk-profile balanced

On success: prints the new portfolio UUID to stdout and exits 0.
On error: logs the failure and exits 1.

The script requires FINALAYZE_DATABASE_URL to be set (non-zero exit if unset).
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import structlog
from dotenv import load_dotenv

from finalayze.core.db import get_async_session_factory
from finalayze.core.schemas import RiskProfile
from finalayze.execution.saa_portfolio_writer import (
    coerce_budget,
    create_active_portfolio,
    resolve_risk_profile,
)

_log = structlog.get_logger(__name__)


def main() -> int:
    """Entry point: parse args, validate, create portfolio, print UUID or fail."""
    parser = argparse.ArgumentParser(description="Create a new active SAA portfolio.")
    parser.add_argument(
        "--budget-rub",
        type=int,
        required=True,
        help="Budget in RUB (integer, required).",
    )
    parser.add_argument(
        "--risk-profile",
        required=True,
        choices=[p.value for p in RiskProfile],
        help="Risk profile (conservative, balanced, or growth).",
    )

    args = parser.parse_args()

    # Load .env
    load_dotenv()

    # Fail loud on a money path: get_async_session_factory() silently defaults to a localhost
    # DB in DEBUG mode, which would write the portfolio to the WRONG database. Require an explicit
    # DB URL env var so a missing config is a clear error, not a silent mis-target (IN-01).
    if not (os.environ.get("FINALAYZE_DATABASE_URL") or os.environ.get("DATABASE_URL")):
        _log.error(
            "database_url_not_set",
            error="set FINALAYZE_DATABASE_URL (or DATABASE_URL) before creating a portfolio",
        )
        return 1

    # Get session factory (will raise if DB env unset)
    try:
        session_factory = get_async_session_factory()
    except ValueError as exc:
        _log.error("database_url_not_set", error=str(exc))
        return 1

    # Validate inputs (fail-closed)
    try:
        budget = coerce_budget(args.budget_rub)
        risk_profile = resolve_risk_profile(args.risk_profile)
    except Exception as exc:
        _log.error("validation_failed", error=str(exc))
        return 1

    # Create portfolio
    try:
        portfolio_id = asyncio.run(
            create_active_portfolio(
                session_factory,
                budget_rub=budget,
                risk_profile=risk_profile,
            )
        )
    except Exception as exc:
        _log.error("create_failed", error=str(exc))
        return 1

    # Success: print UUID
    print(str(portfolio_id))
    _log.info(
        "portfolio_created_success",
        portfolio_id=str(portfolio_id),
        budget_rub=str(budget),
        risk_profile=risk_profile.value,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
