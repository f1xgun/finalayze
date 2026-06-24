#!/usr/bin/env python3
"""Run a one-shot SAA rebalance: build the plan from live data and (sandbox) submit it.

This is the operator-checkpoint entry for the Phase 80 execution wiring. It reads the active SAA
portfolio, fetches current positions + last prices from the Tinkoff SANDBOX, builds the rebalance
plan (Phase 79 ``plan_rebalance``) and -- unless previewing -- submits it.

Usage:
    # safe PREVIEW (real positions/prices, NO orders placed) -- the default:
    FINALAYZE_TINKOFF_TOKEN=... FINALAYZE_DATABASE_URL=... uv run python scripts/run_rebalance.py
    # place SANDBOX orders:
    ... uv run python scripts/run_rebalance.py --mode sandbox
    # LIVE (real money) is a HARD STOP -- requires --confirm AND WorkMode.REAL
    # (FINALAYZE_REAL_CONFIRMED=true):
    ... uv run python scripts/run_rebalance.py --mode live --confirm

Required env: FINALAYZE_TINKOFF_TOKEN (T-Bank sandbox token) + a DB URL with an active SAA
portfolio (FINALAYZE_DATABASE_URL or DATABASE_URL).
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

# gRPC env vars MUST be set before importing grpc (via t_tech.invest), mirroring run_sandbox.py.
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
os.environ.setdefault("GRPC_DNS_RESOLVER", "native")
_GRPC_ROOTS = Path(_PROJECT_ROOT) / "certs" / "grpc_roots.pem"
if _GRPC_ROOTS.exists():
    os.environ.setdefault("GRPC_DEFAULT_SSL_ROOTS_FILE_PATH", str(_GRPC_ROOTS))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import structlog

_log = structlog.get_logger(__name__)

# CLI --mode -> (plan mode stamped on the plan / used by the live gate, submit?)
_MODE_MAP: dict[str, tuple[str, bool]] = {
    "preview": ("DRY_RUN", False),  # real data, NO orders
    "sandbox": ("SANDBOX", True),  # place sandbox orders
    "live": ("LIVE", True),  # real money -- triple-gated
}


def resolve_run_mode(cli_mode: str) -> tuple[str, bool]:
    """Map the CLI ``--mode`` to ``(plan_mode, submit)``."""
    return _MODE_MAP[cli_mode]


def missing_env_error() -> str | None:
    """Return an error message if a required env var is unset, else ``None`` (fail-loud)."""
    if not (os.environ.get("FINALAYZE_DATABASE_URL") or os.environ.get("DATABASE_URL")):
        return "set FINALAYZE_DATABASE_URL (or DATABASE_URL) to a DB with an active SAA portfolio"
    if not os.environ.get("FINALAYZE_TINKOFF_TOKEN"):
        return "set FINALAYZE_TINKOFF_TOKEN (T-Bank sandbox API token)"
    return None


def fetch_nkd_by_symbol(fetcher: object, ofz_symbol: str, as_of: object) -> dict[str, object]:
    """Best-effort: today's accrued coupon (NKD, RUB/bond) for the OFZ leg (Phase 82 P82-R8).

    Returns ``{ofz_symbol: nkd_value}`` so run_rebalance sizes the bond off the DIRTY price; an
    empty dict on a fetch failure or no data -> sizing falls back to the clean price (no exception).
    """
    from datetime import UTC, datetime, time  # noqa: PLC0415

    try:
        records = fetcher.fetch_accrued_interest(  # type: ignore[attr-defined]
            ofz_symbol,
            datetime.combine(as_of, time.min, tzinfo=UTC),  # type: ignore[arg-type]
            datetime.combine(as_of, time.max, tzinfo=UTC),  # type: ignore[arg-type]
        )
    except Exception as exc:  # NKD is an enhancement -- fall back to clean-only sizing
        _log.warning("nkd_fetch_failed", symbol=ofz_symbol, error=str(exc))
        return {}
    return {ofz_symbol: records[-1].value} if records else {}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a one-shot SAA rebalance.")
    parser.add_argument(
        "--mode",
        choices=["preview", "sandbox", "live"],
        default="preview",
        help=(
            "preview = real data, NO orders (default); sandbox = place sandbox orders; "
            "live = real money (requires --confirm + WorkMode.REAL)."
        ),
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Required for --mode live (real-money hard stop).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse args, validate env (fail-loud), then wire + run the rebalance."""
    from dotenv import load_dotenv  # noqa: PLC0415

    args = _build_parser().parse_args(argv)
    load_dotenv(Path(_PROJECT_ROOT) / ".env")

    err = missing_env_error()
    if err:
        _log.error("run_rebalance_env_missing", error=err)
        return 1
    if args.mode == "live" and not args.confirm:
        _log.error(
            "run_rebalance_live_requires_confirm",
            error="--mode live requires --confirm (real-money hard stop)",
        )
        return 1

    plan_mode, submit = resolve_run_mode(args.mode)
    return _run(plan_mode=plan_mode, submit=submit, confirm=args.confirm)


def _run(*, plan_mode: str, submit: bool, confirm: bool) -> int:
    """Wire the sandbox broker + session factory and run (operator checkpoint; needs a token)."""
    from config.settings import Settings  # noqa: PLC0415

    from finalayze.config.rebalance_config import (  # noqa: PLC0415
        get_equity_point_value,
        get_equity_symbol,
        get_ofz_pk_symbol,
    )
    from finalayze.core.clock import RealClock  # noqa: PLC0415
    from finalayze.core.db import get_async_session_factory  # noqa: PLC0415
    from finalayze.core.modes import ModeManager, WorkMode  # noqa: PLC0415
    from finalayze.data.fetchers.tinkoff_data import TinkoffFetcher  # noqa: PLC0415
    from finalayze.execution.broker_router import BrokerRouter  # noqa: PLC0415
    from finalayze.execution.retry import RetryPolicy  # noqa: PLC0415
    from finalayze.execution.tinkoff_broker import TinkoffBroker  # noqa: PLC0415
    from finalayze.markets.instruments import build_default_registry  # noqa: PLC0415
    from finalayze.orchestration.rebalance_execution import (  # noqa: PLC0415
        format_rebalance_plan,
        run_rebalance,
    )

    settings = Settings()
    registry = build_default_registry()
    broker = TinkoffBroker(
        token=settings.tinkoff_token,
        registry=registry,
        sandbox=True,  # never the production endpoint from this CLI
        retry_policy=RetryPolicy(max_retries=3, base_delay=1.0),
    )
    broker_router = BrokerRouter({"moex": broker})
    session_factory = get_async_session_factory()
    # LIVE requires WorkMode.REAL (which itself requires FINALAYZE_REAL_CONFIRMED=true); the
    # ModeManager construction is an extra hard-stop layer for a LIVE run.
    mode_manager = ModeManager(WorkMode.REAL if plan_mode == "LIVE" else WorkMode.SANDBOX)
    clock = RealClock()
    # Best-effort NKD (accrued coupon) so the OFZ leg sizes off the dirty price; {} -> clean-only.
    fetcher = TinkoffFetcher(token=settings.tinkoff_token, registry=registry, sandbox=True)
    nkd_by_symbol = fetch_nkd_by_symbol(fetcher, get_ofz_pk_symbol(), clock.now().date())
    # The equity FUTURE (IMOEXF) is sized by exposure: contract notional = points * point_value.
    point_value_by_symbol = {get_equity_symbol(): get_equity_point_value()}

    async def _go() -> int:
        plan, outcomes = await run_rebalance(
            broker_router=broker_router,
            mode_manager=mode_manager,
            registry=registry,
            session_factory=session_factory,
            clock=clock,
            fetch_last_prices=broker.get_last_prices,
            nkd_by_symbol=nkd_by_symbol,  # type: ignore[arg-type]
            point_value_by_symbol=point_value_by_symbol,
            mode=plan_mode,  # type: ignore[arg-type]
            confirm=confirm,
            submit=submit,
        )
        print(format_rebalance_plan(plan, outcomes))  # CLI stdout output
        return 0

    try:
        return asyncio.run(_go())
    except Exception as exc:  # operator CLI: surface any wiring/gRPC failure as a clean exit
        _log.error("run_rebalance_failed", error=str(exc))
        return 1


if __name__ == "__main__":
    sys.exit(main())
