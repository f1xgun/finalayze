"""Instrument registry -- symbol lookup and metadata (Layer 2).

Maps (symbol, market_id) pairs to instrument metadata.
For MOEX, instruments also carry a FIGI identifier used by Tinkoff Invest API.

See docs/architecture/DEPENDENCY_LAYERS.md for layering rules.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any, get_args

import structlog

from finalayze.core.exceptions import ConfigurationError, InstrumentNotFoundError
from finalayze.core.schemas import InstrumentType

_log = structlog.get_logger()

# Valid instrument types, derived from the InstrumentType Literal alias (single
# source of truth -- never hardcode a duplicate list that can drift). Used by the
# fail-closed snapshot loader to reject an unknown type from the committed file
# (IN-05: the loader is the trust boundary for the attacker-influenceable file).
_VALID_INSTRUMENT_TYPES: frozenset[str] = frozenset(get_args(InstrumentType.__value__))


@dataclass(frozen=True)
class Instrument:
    """Metadata for a tradeable instrument."""

    symbol: str
    market_id: str  # "us" | "moex"
    name: str
    instrument_type: InstrumentType = "stock"
    figi: str | None = None  # Tinkoff FIGI identifier (MOEX only)
    lot_size: int = 1  # MOEX instruments often have lot sizes > 1
    currency: str = "USD"
    is_active: bool = True
    segment_id: str = ""  # optional segment the instrument belongs to
    # Bond-specific fields (None for stocks/ETFs)
    face_value: Decimal | None = None
    coupon_rate: Decimal | None = None  # annual % (e.g. 7.10)
    coupon_frequency: int | None = None  # payments per year
    maturity_date: date | None = None
    floating_coupon: bool = False
    # NEW (additive, all default) -- per-class metadata for futures/currency (D-06)
    isin: str | None = None
    class_code: str | None = None  # board code: TQBR / TQTF / SPBFUT / ...
    expiration_date: date | None = None  # futures
    basic_asset: str | None = None  # futures underlying ticker
    asset_uid: str | None = None  # used by the fundamentals path
    short_history: bool = False  # D-05 flag; set False for now (sub-area 3 fills)


class InstrumentRegistry:
    """Registry mapping (symbol, market_id) to Instrument metadata."""

    def __init__(self) -> None:
        self._instruments: dict[tuple[str, str], Instrument] = {}

    def register(self, instrument: Instrument) -> None:
        """Register an instrument. Overwrites if already exists.

        Logs a warning (WR-01) when the overwrite drops a DISTINCT instrument
        (different FIGI) under the same (symbol, market_id) key -- last-write-wins
        silently loses the prior row otherwise.
        """
        key = (instrument.symbol, instrument.market_id)
        prior = self._instruments.get(key)
        if prior is not None and prior.figi != instrument.figi:
            _log.warning(
                "instrument_overwrite_distinct_figi",
                symbol=instrument.symbol,
                market_id=instrument.market_id,
                figi_a=prior.figi,
                figi_b=instrument.figi,
            )
        self._instruments[key] = instrument

    def get(self, symbol: str, market_id: str) -> Instrument:
        """Return instrument by symbol+market. Raises InstrumentNotFoundError if missing."""
        key = (symbol, market_id)
        if key not in self._instruments:
            msg = f"Instrument '{symbol}' not found in market '{market_id}'"
            raise InstrumentNotFoundError(msg)
        return self._instruments[key]

    def get_by_figi(self, figi: str) -> Instrument:
        """Return instrument by FIGI. Raises InstrumentNotFoundError if not found."""
        for instrument in self._instruments.values():
            if instrument.figi == figi:
                return instrument
        msg = f"Instrument with FIGI '{figi}' not found"
        raise InstrumentNotFoundError(msg)

    def list_by_market(self, market_id: str) -> list[Instrument]:
        """Return all active instruments for a given market, sorted by symbol."""
        return sorted(
            [i for i in self._instruments.values() if i.market_id == market_id and i.is_active],
            key=lambda i: i.symbol,
        )

    def list_by_type(self, market_id: str, instrument_type: str) -> list[Instrument]:
        """Return active instruments of a given type in a market, sorted by symbol."""
        return sorted(
            [
                i
                for i in self._instruments.values()
                if i.market_id == market_id and i.instrument_type == instrument_type and i.is_active
            ],
            key=lambda i: i.symbol,
        )

    def __len__(self) -> int:
        return len(self._instruments)


# Default US instruments for Phase 1
DEFAULT_US_INSTRUMENTS: list[Instrument] = [
    Instrument(
        symbol="AAPL",
        market_id="us",
        name="Apple Inc.",
        instrument_type="stock",
        currency="USD",
    ),
    Instrument(
        symbol="MSFT",
        market_id="us",
        name="Microsoft Corporation",
        instrument_type="stock",
        currency="USD",
    ),
    Instrument(
        symbol="GOOGL",
        market_id="us",
        name="Alphabet Inc.",
        instrument_type="stock",
        currency="USD",
    ),
    Instrument(
        symbol="AMZN",
        market_id="us",
        name="Amazon.com Inc.",
        instrument_type="stock",
        currency="USD",
    ),
    Instrument(
        symbol="NVDA",
        market_id="us",
        name="NVIDIA Corporation",
        instrument_type="stock",
        currency="USD",
    ),
    Instrument(
        symbol="SPY",
        market_id="us",
        name="SPDR S&P 500 ETF Trust",
        instrument_type="etf",
        currency="USD",
    ),
    Instrument(
        symbol="QQQ",
        market_id="us",
        name="Invesco QQQ Trust",
        instrument_type="etf",
        currency="USD",
    ),
]


# ---------------------------------------------------------------------------
# MOEX universe -- loaded fail-closed from the committed snapshot (Plan 65-03)
# ---------------------------------------------------------------------------
# The runtime loader reads a committed JSON snapshot ONLY -- no network, no DB,
# no `scripts.*` import (Layer-2 rule, CLAUDE.md invariant #1). A missing or
# corrupt snapshot raises ConfigurationError (fail-closed, D-04) -- there is no
# silent fallback to a stale hand-maintained list.
_SNAPSHOT = Path(__file__).parent / "data" / "moex_universe.json"


def _row_to_instrument(row: dict[str, Any]) -> Instrument:
    """Re-hydrate one JSON snapshot row into an Instrument.

    JSON serializes Decimal -> str and date -> ISO string, so we coerce them
    back; currency is upper-cased (Pitfall 3); absent keys stay None.
    """

    def _dec(key: str) -> Decimal | None:
        val = row.get(key)
        return None if val is None else Decimal(str(val))

    def _date(key: str) -> date | None:
        val = row.get(key)
        return None if val is None else date.fromisoformat(str(val))

    # IN-05: validate instrument_type against the InstrumentType Literal (fail-closed).
    # The frozen dataclass does no runtime validation, so a corrupt/attacker-influenced
    # snapshot row with an unknown type would otherwise be accepted and silently dropped
    # from every list_by_type query. The committed file is the trust boundary -- raise.
    itype = row.get("instrument_type", "stock")
    if itype not in _VALID_INSTRUMENT_TYPES:
        msg = (
            f"unknown instrument_type {itype!r} for {row.get('symbol')!r} "
            f"(valid: {sorted(_VALID_INSTRUMENT_TYPES)})"
        )
        raise ConfigurationError(msg)

    currency = row.get("currency") or "RUB"
    return Instrument(
        symbol=row["symbol"],
        market_id=row.get("market_id", "moex"),
        name=row.get("name", ""),
        instrument_type=itype,
        figi=row.get("figi") or None,
        lot_size=row.get("lot_size", 1),
        currency=str(currency).upper(),
        is_active=row.get("is_active", True),
        segment_id=row.get("segment_id", ""),
        face_value=_dec("face_value"),
        coupon_rate=_dec("coupon_rate"),
        coupon_frequency=row.get("coupon_frequency"),
        maturity_date=_date("maturity_date"),
        floating_coupon=row.get("floating_coupon", False),
        isin=row.get("isin") or None,
        class_code=row.get("class_code") or None,
        expiration_date=_date("expiration_date"),
        basic_asset=row.get("basic_asset") or None,
        asset_uid=row.get("asset_uid") or None,
        short_history=row.get("short_history", False),
    )


def _load_moex_snapshot() -> list[Instrument]:
    """Read the committed MOEX universe snapshot, fail-closed.

    Raises ConfigurationError on a missing or corrupt snapshot -- never falls
    back to a stale hand-list (D-04 / T-65-08).
    """
    try:
        raw = json.loads(_SNAPSHOT.read_text(encoding="utf-8"))
        rows = raw["instruments"]
    except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError) as exc:
        msg = f"MOEX universe snapshot missing/corrupt at {_SNAPSHOT}: {exc}"
        raise ConfigurationError(msg) from exc  # NO fallback to a stale hand-list (D-04)
    return [_row_to_instrument(r) for r in rows]


# Compat shims (UNIV-07): the ~10 direct importers iterate these module-level
# lists. They are computed ONCE at import from the real committed snapshot so
# those importers need zero edits. build_default_registry does NOT iterate
# these -- it reads the snapshot lazily so a monkeypatched _SNAPSHOT (UNIV-04
# fail-closed test) is honoured.
_ALL_MOEX_SNAPSHOT: list[Instrument] = _load_moex_snapshot()
DEFAULT_MOEX_INSTRUMENTS: list[Instrument] = [
    i for i in _ALL_MOEX_SNAPSHOT if i.instrument_type != "bond"
]
DEFAULT_MOEX_OFZ_INSTRUMENTS: list[Instrument] = [
    i for i in _ALL_MOEX_SNAPSHOT if i.instrument_type == "bond"
]


def build_default_registry() -> InstrumentRegistry:
    """Build and return a registry pre-populated with default instruments."""
    registry = InstrumentRegistry()
    for instrument in DEFAULT_US_INSTRUMENTS:  # KEEP unchanged (D-04, US out of scope)
        registry.register(instrument)
    for inst in _load_moex_snapshot():  # LAZY read -- honours a patched _SNAPSHOT;
        registry.register(inst)  # fail-closed ConfigurationError reachable (UNIV-04)
    return registry
