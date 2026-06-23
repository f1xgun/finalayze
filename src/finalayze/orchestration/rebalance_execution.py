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

from decimal import Decimal
from typing import TYPE_CHECKING

import structlog

from finalayze.config.rebalance_config import get_equity_symbol, get_ofz_pk_symbol
from finalayze.core.exceptions import InstrumentNotFoundError
from finalayze.core.schemas import AssetClass

if TYPE_CHECKING:
    from collections.abc import Mapping

    from finalayze.markets.instruments import Instrument, InstrumentRegistry

_log = structlog.get_logger()

_MOEX = "moex"
_HUNDRED = Decimal(100)


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
