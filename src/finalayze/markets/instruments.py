"""Instrument registry -- symbol lookup and metadata (Layer 2).

Maps (symbol, market_id) pairs to instrument metadata.
For MOEX, instruments also carry a FIGI identifier used by Tinkoff Invest API.

See docs/architecture/DEPENDENCY_LAYERS.md for layering rules.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from typing import TYPE_CHECKING

from finalayze.core.exceptions import InstrumentNotFoundError

if TYPE_CHECKING:
    from finalayze.core.schemas import InstrumentType


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
    Instrument(
        symbol="TATN",
        market_id="moex",
        name="Tatneft",
        instrument_type="stock",
        figi="BBG004RVFFC0",
        lot_size=1,
        currency="RUB",
    ),
    Instrument(
        symbol="SBERP",
        market_id="moex",
        name="Sberbank Preferred",
        instrument_type="stock",
        figi="BBG0047315Y7",
        lot_size=10,
        currency="RUB",
    ),
    Instrument(
        symbol="MGNT",
        market_id="moex",
        name="Magnit",
        instrument_type="stock",
        figi="BBG004RVFCY3",
        lot_size=1,
        currency="RUB",
    ),
    Instrument(
        symbol="POLY",
        market_id="moex",
        name="Polymetal International",
        instrument_type="stock",
        figi="BBG004PYF2N3",
        lot_size=1,
        currency="RUB",
    ),
    Instrument(
        symbol="ALRS",
        market_id="moex",
        name="Alrosa",
        instrument_type="stock",
        figi="BBG004S68B31",
        lot_size=10,
        currency="RUB",
    ),
    Instrument(
        symbol="SNGS",
        market_id="moex",
        name="Surgutneftegas",
        instrument_type="stock",
        figi="BBG004S681W1",
        lot_size=10,
        currency="RUB",
    ),
    Instrument(
        symbol="TRNFP",
        market_id="moex",
        name="Transneft Preferred",
        instrument_type="stock",
        figi="BBG00475K6C3",
        lot_size=1,
        currency="RUB",
    ),
    Instrument(
        symbol="IRAO",
        market_id="moex",
        name="Inter RAO",
        instrument_type="stock",
        figi="BBG004S68473",
        lot_size=100,
        currency="RUB",
    ),
    Instrument(
        symbol="OZON",
        market_id="moex",
        name="Ozon Holdings",
        instrument_type="stock",
        figi="BBG00Y91R9T3",
        lot_size=1,
        currency="RUB",
    ),
    Instrument(
        symbol="MOEX",
        market_id="moex",
        name="Moscow Exchange",
        instrument_type="stock",
        figi="BBG004730JJ5",
        lot_size=10,
        currency="RUB",
    ),
]


# Default OFZ bond instruments (Phase 0 validated via T-Bank API 2026-03-11)
# FIGIs confirmed via services.instruments.bond_by()
DEFAULT_MOEX_OFZ_INSTRUMENTS: list[Instrument] = [
    # OFZ-PD (Fixed Coupon) — Strategic/Tactical layers
    Instrument(
        symbol="SU26238RMFS4",
        market_id="moex",
        name="ОФЗ 26238",
        instrument_type="bond",
        figi="BBG011FJ4HS6",
        lot_size=1,
        currency="RUB",
        face_value=Decimal(1000),
        coupon_rate=Decimal("7.10"),
        coupon_frequency=2,
        maturity_date=date(2041, 5, 15),
    ),
    Instrument(
        symbol="SU26239RMFS2",
        market_id="moex",
        name="ОФЗ 26239",
        instrument_type="bond",
        figi="BBG011FHF1F7",
        lot_size=1,
        currency="RUB",
        face_value=Decimal(1000),
        coupon_rate=Decimal("6.90"),
        coupon_frequency=2,
        maturity_date=date(2031, 7, 23),
    ),
    Instrument(
        symbol="SU26241RMFS8",
        market_id="moex",
        name="ОФЗ 26241",
        instrument_type="bond",
        figi="BBG01BJBR2W0",
        lot_size=1,
        currency="RUB",
        face_value=Decimal(1000),
        coupon_rate=Decimal("9.50"),
        coupon_frequency=2,
        maturity_date=date(2032, 11, 17),
    ),
    Instrument(
        symbol="SU26243RMFS4",
        market_id="moex",
        name="ОФЗ 26243",
        instrument_type="bond",
        figi="TCS00A106E90",
        lot_size=1,
        currency="RUB",
        face_value=Decimal(1000),
        coupon_rate=Decimal("9.80"),
        coupon_frequency=2,
        maturity_date=date(2038, 5, 19),
    ),
    Instrument(
        symbol="SU26244RMFS2",
        market_id="moex",
        name="ОФЗ 26244",
        instrument_type="bond",
        figi="TCS00A1074G2",
        lot_size=1,
        currency="RUB",
        face_value=Decimal(1000),
        coupon_rate=Decimal("11.25"),
        coupon_frequency=2,
        maturity_date=date(2034, 3, 15),
    ),
    Instrument(
        symbol="SU26246RMFS7",
        market_id="moex",
        name="ОФЗ 26246",
        instrument_type="bond",
        figi="BBG01N0CVG83",
        lot_size=1,
        currency="RUB",
        face_value=Decimal(1000),
        coupon_rate=Decimal("12.00"),
        coupon_frequency=2,
        maturity_date=date(2036, 3, 12),
    ),
    Instrument(
        symbol="SU26252RMFS5",
        market_id="moex",
        name="ОФЗ 26252",
        instrument_type="bond",
        figi="TCS00A10D4Y2",
        lot_size=1,
        currency="RUB",
        face_value=Decimal(1000),
        coupon_rate=Decimal("12.50"),
        coupon_frequency=2,
        maturity_date=date(2033, 10, 12),
    ),
    Instrument(
        symbol="SU26253RMFS3",
        market_id="moex",
        name="ОФЗ 26253",
        instrument_type="bond",
        figi="TCS00A10D517",
        lot_size=1,
        currency="RUB",
        face_value=Decimal(1000),
        coupon_rate=Decimal("13.00"),
        coupon_frequency=2,
        maturity_date=date(2038, 10, 6),
    ),
    # OFZ-PK (Floating Coupon) — Core layer
    Instrument(
        symbol="SU29007RMFS0",
        market_id="moex",
        name="ОФЗ 29007",
        instrument_type="bond",
        figi="BBG007Z5DF79",
        lot_size=1,
        currency="RUB",
        face_value=Decimal(1000),
        coupon_rate=Decimal("1.30"),  # spread over RUONIA
        coupon_frequency=2,
        maturity_date=date(2027, 3, 3),
        floating_coupon=True,
    ),
    Instrument(
        symbol="SU29008RMFS8",
        market_id="moex",
        name="ОФЗ 29008",
        instrument_type="bond",
        figi="BBG007Z5DZS2",
        lot_size=1,
        currency="RUB",
        face_value=Decimal(1000),
        coupon_rate=Decimal("1.40"),
        coupon_frequency=2,
        maturity_date=date(2029, 10, 3),
        floating_coupon=True,
    ),
    Instrument(
        symbol="SU29009RMFS6",
        market_id="moex",
        name="ОФЗ 29009",
        instrument_type="bond",
        figi="BBG007Z5F748",
        lot_size=1,
        currency="RUB",
        face_value=Decimal(1000),
        coupon_rate=Decimal("1.50"),
        coupon_frequency=2,
        maturity_date=date(2032, 5, 5),
        floating_coupon=True,
    ),
    Instrument(
        symbol="SU29010RMFS4",
        market_id="moex",
        name="ОФЗ 29010",
        instrument_type="bond",
        figi="BBG007Z5FFL1",
        lot_size=1,
        currency="RUB",
        face_value=Decimal(1000),
        coupon_rate=Decimal("1.60"),
        coupon_frequency=2,
        maturity_date=date(2034, 12, 6),
        floating_coupon=True,
    ),
]


def build_default_registry() -> InstrumentRegistry:
    """Build and return a registry pre-populated with default instruments."""
    registry = InstrumentRegistry()
    for instrument in DEFAULT_US_INSTRUMENTS:
        registry.register(instrument)
    for instrument in DEFAULT_MOEX_INSTRUMENTS:
        registry.register(instrument)
    for instrument in DEFAULT_MOEX_OFZ_INSTRUMENTS:
        registry.register(instrument)
    return registry
