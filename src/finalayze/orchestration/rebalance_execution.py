"""Execution wiring -- assemble the real inputs for the weights->orders engine (Phase 80).

Connects the active SAA portfolio to an end-to-end (sandbox) rebalance run: read the active
portfolio, resolve the leg instruments, fetch current positions + last prices, load the deposit
mark, compute regime-tilted weights, build the plan (Phase 79 ``plan_rebalance``) and (dry-run)
submit it (Phase 79 ``submit_rebalance_plan``).

The orchestration takes its token-dependent collaborators as INJECTED arguments (an already-wired
``BrokerRouter``, a ``fetch_last_prices`` callable, a ``session_factory``, a ``Clock``), so the
whole flow is unit-testable with a ``SimulatedBroker`` + a fake price source -- no Tinkoff token.
The CLI (``scripts/run_rebalance.py``) injects the real sandbox broker; that real run is the
operator checkpoint. DRY_RUN is the default; real-money LIVE stays a hard stop (Phase 79
``_enforce_live_gate``).
"""

from __future__ import annotations

import asyncio
from decimal import Decimal
from typing import TYPE_CHECKING

import structlog

from finalayze.config.rebalance_config import get_equity_symbol, get_ofz_pk_symbol
from finalayze.core.exceptions import InstrumentNotFoundError
from finalayze.core.schemas import AssetClass, RiskProfile
from finalayze.execution.deposit_loader import load_deposit_broker_from_db
from finalayze.execution.saa_portfolio_writer import get_active_portfolio
from finalayze.orchestration.allocation import AllocationOrchestrator
from finalayze.orchestration.rebalance_executor import submit_rebalance_plan
from finalayze.orchestration.rebalance_planner import plan_rebalance

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

    from finalayze.core.clock import Clock
    from finalayze.core.modes import ModeManager
    from finalayze.execution.broker_router import BrokerRouter
    from finalayze.markets.instruments import Instrument, InstrumentRegistry
    from finalayze.orchestration.rebalance_planner import LegOutcome, Mode, RebalancePlan

_log = structlog.get_logger()

_MOEX = "moex"
_HUNDRED = Decimal(100)
_ZERO = Decimal(0)


def normalize_positions_to_symbols(
    positions: Mapping[str, Decimal], registry: InstrumentRegistry
) -> dict[str, Decimal]:
    """Convert broker positions to symbol-keyed quantities (P80-R1).

    TinkoffBroker ``get_positions()`` is FIGI-keyed; SimulatedBroker is symbol-keyed. For each key:
    resolve a FIGI to its instrument symbol; else accept an already-known MOEX symbol as-is; else
    skip an unrecognized key with a debug log (a holding outside the SAA leg universe is harmless --
    ``plan_rebalance`` only reads the leg symbols).
    """
    out: dict[str, Decimal] = {}
    for key, qty in positions.items():
        try:
            out[registry.get_by_figi(key).symbol] = qty
            continue
        except InstrumentNotFoundError:
            pass
        try:
            registry.get(key, _MOEX)  # validate it is a known MOEX symbol
        except InstrumentNotFoundError:
            _log.debug("normalize_positions_skip_unknown_key", key=key)
            continue
        out[key] = qty
    return out


def resolve_leg_instruments(registry: InstrumentRegistry) -> dict[AssetClass, Instrument]:
    """Resolve the configured equity + OFZ-PK tickers to ``Instrument``s, fail-loud (P80-R2).

    Symbols come from the fail-closed config (``get_equity_symbol``/``get_ofz_pk_symbol``); a symbol
    that does not resolve in the registry raises ``InstrumentNotFoundError``.
    """
    return {
        AssetClass.EQUITY: registry.get(get_equity_symbol(), _MOEX),
        AssetClass.OFZ_PK: registry.get(get_ofz_pk_symbol(), _MOEX),
    }


def to_rub_price(instrument: Instrument, raw_price: Decimal) -> Decimal:
    """Convert a broker quote to a RUB-per-unit price (P80-R3).

    Bonds are quoted as a PERCENT of face value (TinkoffBroker.get_last_prices), so the RUB price
    per bond is ``raw/100 * face_value`` -- skipping this would mis-size the OFZ-PK leg by ~100x.
    Shares/ETFs are already RUB-per-unit and pass through unchanged. A bond with no ``face_value``
    cannot be priced and fails loud.
    """
    if instrument.instrument_type == "bond":
        if instrument.face_value is None:
            msg = f"bond {instrument.symbol} has no face_value; cannot convert a % quote to RUB"
            raise ValueError(msg)
        return raw_price / _HUNDRED * instrument.face_value
    return raw_price


async def run_rebalance(
    *,
    broker_router: BrokerRouter,
    mode_manager: ModeManager,
    registry: InstrumentRegistry,
    session_factory: async_sessionmaker[AsyncSession],
    clock: Clock,
    fetch_last_prices: Callable[[list[str]], Mapping[str, Decimal]],
    mode: Mode = "DRY_RUN",
    confirm: bool = False,
) -> tuple[RebalancePlan, list[LegOutcome]]:
    """Assemble the real inputs and run an end-to-end (sandbox) SAA rebalance (P80-R4..R7).

    Reads the active portfolio, computes the regime-tilted target weights for today, resolves the
    leg instruments, normalizes current positions to symbols, converts last-price quotes to RUB
    (bond %-of-face -> RUB), loads the deposit mark, then builds (``plan_rebalance``) and submits
    (``submit_rebalance_plan``) the plan. DRY_RUN by default; LIVE stays triple-gated.

    The token-dependent collaborators are injected (``broker_router``, ``fetch_last_prices``,
    ``session_factory``, ``clock``), so the whole flow is unit-testable with a fake broker + price
    source and a stubbed portfolio/deposit -- no Tinkoff token.

    Raises:
        ValueError: If no active portfolio exists, or a leg has no last price.
        InstrumentNotFoundError: If a configured leg symbol does not resolve (via resolve helpers).
    """
    now_dt = clock.now()
    as_of = now_dt.date()

    active = await get_active_portfolio(session_factory)
    if active is None:
        msg = "no active SAA portfolio found; create one (scripts/create_saa_portfolio.py) first"
        raise ValueError(msg)
    portfolio_id, risk_profile_str, _budget_rub = active

    weights = AllocationOrchestrator(
        risk_profile=RiskProfile(risk_profile_str)
    ).get_rebalance_weights(as_of)

    leg_instruments = resolve_leg_instruments(registry)

    current_positions = normalize_positions_to_symbols(
        broker_router.route(_MOEX).get_positions(), registry
    )

    raw_prices = fetch_last_prices([inst.symbol for inst in leg_instruments.values()])
    last_prices: dict[str, Decimal] = {}
    for asset_class, instrument in leg_instruments.items():
        if instrument.symbol not in raw_prices:
            msg = f"no last price for the {asset_class.value} leg {instrument.symbol!r}"
            raise ValueError(msg)
        last_prices[instrument.symbol] = to_rub_price(instrument, raw_prices[instrument.symbol])

    deposit_broker = await load_deposit_broker_from_db(portfolio_id, as_of, session_factory)
    deposit_current = deposit_broker.deposit_value() if deposit_broker is not None else _ZERO

    plan = plan_rebalance(
        active_portfolio=active,
        target_weights=weights,
        current_positions=current_positions,
        last_prices=last_prices,
        leg_instruments=leg_instruments,
        deposit_current_notional=deposit_current,
        plan_id=f"{portfolio_id}:{as_of.isoformat()}",
        created_at=now_dt,
        mode=mode,
        deposit_broker=deposit_broker,
        as_of=as_of,
    )

    # submit_rebalance_plan is synchronous (blocking gRPC bridge); offload off the event loop.
    outcomes = await asyncio.to_thread(
        submit_rebalance_plan, plan, broker_router, mode_manager, confirm=confirm
    )
    return plan, outcomes
