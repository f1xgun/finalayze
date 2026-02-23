"""Instrument registry -- symbol lookup and metadata (Layer 2).

Maps (symbol, market_id) pairs to instrument metadata.
For MOEX, instruments also carry a FIGI identifier used by Tinkoff Invest API.

See docs/architecture/DEPENDENCY_LAYERS.md for layering rules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from finalayze.core.exceptions import InstrumentNotFoundError

type InstrumentType = Literal["stock", "etf", "bond"]


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


class InstrumentRegistry:
    """Registry mapping (symbol, market_id) to Instrument metadata."""

    def __init__(self) -> None:
        self._instruments: dict[tuple[str, str], Instrument] = {}

    def register(self, instrument: Instrument) -> None:
        """Register an instrument. Overwrites if already exists."""
        key = (instrument.symbol, instrument.market_id)
        self._instruments[key] = instrument

    def get(self, symbol: str, market_id: str) -> Instrument:
        """Return instrument by symbol+market. Raises InstrumentNotFoundError if missing."""
        key = (symbol, market_id)
        if key not in self._instruments:
            msg = f"Instrument '{symbol}' not found in market '{market_id}'"
            raise InstrumentNotFoundError(msg)
        return self._instruments[key]

    def list_by_market(self, market_id: str) -> list[Instrument]:
        """Return all active instruments for a given market, sorted by symbol."""
        return sorted(
            [i for i in self._instruments.values() if i.market_id == market_id and i.is_active],
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


# Default MOEX instruments for Phase 2
# FIGI identifiers from Tinkoff Invest API instrument catalogue.
DEFAULT_MOEX_INSTRUMENTS: list[Instrument] = [
    Instrument(
        symbol="SBER",
        market_id="moex",
        name="Sberbank",
        instrument_type="stock",
        figi="BBG004730N88",
        lot_size=10,
        currency="RUB",
    ),
    Instrument(
        symbol="GAZP",
        market_id="moex",
        name="Gazprom",
        instrument_type="stock",
        figi="BBG004730RP0",
        lot_size=10,
        currency="RUB",
    ),
    Instrument(
        symbol="LKOH",
        market_id="moex",
        name="Lukoil",
        instrument_type="stock",
        figi="BBG004731032",
        lot_size=1,
        currency="RUB",
    ),
    Instrument(
        symbol="GMKN",
        market_id="moex",
        name="Norilsk Nickel",
        instrument_type="stock",
        figi="BBG004731489",
        lot_size=1,
        currency="RUB",
    ),
    Instrument(
        symbol="YNDX",
        market_id="moex",
        name="Yandex",
        instrument_type="stock",
        figi="BBG006L8G4H1",
        lot_size=1,
        currency="RUB",
    ),
    Instrument(
        symbol="NVTK",
        market_id="moex",
        name="Novatek",
        instrument_type="stock",
        figi="BBG00475KKY8",
        lot_size=1,
        currency="RUB",
    ),
    Instrument(
        symbol="ROSN",
        market_id="moex",
        name="Rosneft",
        instrument_type="stock",
        figi="BBG004731354",
        lot_size=1,
        currency="RUB",
    ),
    Instrument(
        symbol="VTBR",
        market_id="moex",
        name="VTB Bank",
        instrument_type="stock",
        figi="BBG004730ZJ9",
        lot_size=10,
        currency="RUB",
    ),
]


def build_default_registry() -> InstrumentRegistry:
    """Build and return a registry pre-populated with default instruments."""
    registry = InstrumentRegistry()
    for instrument in DEFAULT_US_INSTRUMENTS:
        registry.register(instrument)
    for instrument in DEFAULT_MOEX_INSTRUMENTS:
        registry.register(instrument)
    return registry
