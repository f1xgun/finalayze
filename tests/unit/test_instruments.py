"""Unit tests for the instrument registry (Layer 2)."""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from finalayze.core.exceptions import InstrumentNotFoundError
from finalayze.markets.instruments import (
    DEFAULT_MOEX_OFZ_INSTRUMENTS,
    Instrument,
    InstrumentRegistry,
    build_default_registry,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
AAPL_SYMBOL = "AAPL"
MSFT_SYMBOL = "MSFT"
US_MARKET = "us"
UNKNOWN_SYMBOL = "UNKN"
UNKNOWN_MARKET = "unknown"

EXPECTED_DEFAULT_SYMBOLS = {"AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "SPY", "QQQ"}
EXPECTED_DEFAULT_US_COUNT = 7
EXPECTED_DEFAULT_COUNT = 68  # 7 US + 49 MOEX stocks + 12 OFZ bonds
EXPECTED_COUNT_AFTER_TWO = 2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_instrument(
    symbol: str = AAPL_SYMBOL,
    market_id: str = US_MARKET,
    name: str = "Apple Inc.",
) -> Instrument:
    return Instrument(symbol=symbol, market_id=market_id, name=name)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_register_and_get() -> None:
    registry = InstrumentRegistry()
    instrument = make_instrument()
    registry.register(instrument)
    result = registry.get(AAPL_SYMBOL, US_MARKET)
    assert result is instrument


def test_get_raises_for_unknown() -> None:
    registry = InstrumentRegistry()
    with pytest.raises(InstrumentNotFoundError):
        registry.get(UNKNOWN_SYMBOL, UNKNOWN_MARKET)


def test_list_by_market_returns_sorted() -> None:
    registry = InstrumentRegistry()
    # Register in reverse order
    registry.register(make_instrument(symbol=MSFT_SYMBOL, name="Microsoft Corporation"))
    registry.register(make_instrument(symbol=AAPL_SYMBOL, name="Apple Inc."))
    results = registry.list_by_market(US_MARKET)
    assert [i.symbol for i in results] == [AAPL_SYMBOL, MSFT_SYMBOL]


def test_list_by_market_excludes_inactive() -> None:
    registry = InstrumentRegistry()
    active = Instrument(symbol=AAPL_SYMBOL, market_id=US_MARKET, name="Apple Inc.", is_active=True)
    inactive = Instrument(
        symbol=MSFT_SYMBOL, market_id=US_MARKET, name="Microsoft", is_active=False
    )
    registry.register(active)
    registry.register(inactive)
    results = registry.list_by_market(US_MARKET)
    symbols = [i.symbol for i in results]
    assert AAPL_SYMBOL in symbols
    assert MSFT_SYMBOL not in symbols


def test_build_default_registry_has_us_instruments() -> None:
    registry = build_default_registry()
    for symbol in EXPECTED_DEFAULT_SYMBOLS:
        instrument = registry.get(symbol, US_MARKET)
        assert instrument.market_id == US_MARKET


def test_build_default_registry_has_expected_count() -> None:
    registry = build_default_registry()
    assert len(registry) == EXPECTED_DEFAULT_COUNT
    assert len(registry.list_by_market(US_MARKET)) == EXPECTED_DEFAULT_US_COUNT


def test_register_overwrites() -> None:
    registry = InstrumentRegistry()
    original = Instrument(symbol=AAPL_SYMBOL, market_id=US_MARKET, name="Apple Original")
    updated = Instrument(symbol=AAPL_SYMBOL, market_id=US_MARKET, name="Apple Updated")
    registry.register(original)
    registry.register(updated)
    result = registry.get(AAPL_SYMBOL, US_MARKET)
    assert result.name == "Apple Updated"


def test_len() -> None:
    registry = InstrumentRegistry()
    assert len(registry) == 0
    registry.register(make_instrument(symbol=AAPL_SYMBOL))
    assert len(registry) == 1
    registry.register(make_instrument(symbol=MSFT_SYMBOL))
    assert len(registry) == EXPECTED_COUNT_AFTER_TWO


EXPECTED_MOEX_STOCK_COUNT = 49
EXPECTED_MOEX_OFZ_COUNT = 12
EXPECTED_MOEX_INSTRUMENT_COUNT = 61  # 49 stocks + 12 OFZ bonds
# Number of statically-defined instruments with hardcoded FIGIs
EXPECTED_MOEX_WITH_FIGI = 31  # 19 stocks with FIGI + 12 OFZ bonds
EXPECTED_MOEX_SYMBOLS = {
    # Original blue chips
    "SBER", "SBERP", "GAZP", "LKOH", "GMKN", "YNDX", "NVTK", "ROSN",
    "VTBR", "TATN", "SNGS", "ALRS", "MGNT", "POLY", "IRAO", "TRNFP",
    "OZON", "MOEX",
    # Finance / Tech / Energy expansions
    "TCSG", "VKCO", "CBOM", "BSPB", "HHRU", "POSI",
    "SNGSP", "SIBN", "TATNP",
    # Metals & Mining
    "CHMF", "NLMK", "MAGN", "PLZL", "RUAL", "MTLR",
    # Consumer / Telecom / Utilities
    "FIVE", "FIXP", "LENT", "MTSS", "RTKM",
    "HYDR", "FEES", "MSNG", "UPRO",
    # Construction / Chemicals / Transport
    "PIKK", "SMLT", "PHOR", "AKRN", "AFLT", "FLOT", "NMTP",
    # OFZ bonds
    "SU26238RMFS4", "SU26239RMFS2", "SU26241RMFS8", "SU26243RMFS4",
    "SU26244RMFS2", "SU26246RMFS7", "SU26252RMFS5", "SU26253RMFS3",
    "SU29007RMFS0", "SU29008RMFS8", "SU29009RMFS6", "SU29010RMFS4",
}


def test_default_registry_includes_moex_instruments() -> None:
    """Default registry must include all 16 MOEX instruments."""
    registry = build_default_registry()
    moex_instruments = registry.list_by_market("moex")
    assert len(moex_instruments) == EXPECTED_MOEX_INSTRUMENT_COUNT


def test_moex_instruments_with_static_figi() -> None:
    """MOEX instruments with hardcoded FIGIs must have non-empty values.

    New instruments added without FIGIs get them resolved at runtime via T-Bank API.
    """
    registry = build_default_registry()
    with_figi = [inst for inst in registry.list_by_market("moex") if inst.figi is not None]
    assert len(with_figi) == EXPECTED_MOEX_WITH_FIGI
    for inst in with_figi:
        assert inst.figi != "", f"{inst.symbol} has empty FIGI"


def test_moex_instruments_symbols() -> None:
    """Default registry must contain exactly the expected MOEX symbols."""
    registry = build_default_registry()
    symbols = {i.symbol for i in registry.list_by_market("moex")}
    assert symbols == EXPECTED_MOEX_SYMBOLS


def test_moex_instruments_currency_is_rub() -> None:
    """All MOEX instruments must be denominated in RUB."""
    registry = build_default_registry()
    for inst in registry.list_by_market("moex"):
        assert inst.currency == "RUB", (
            f"{inst.symbol} currency is {inst.currency!r}, expected 'RUB'"
        )


# ---------------------------------------------------------------------------
# Bond instrument tests
# ---------------------------------------------------------------------------
BOND_FACE_VALUE = Decimal(1000)
BOND_COUPON_RATE = Decimal("7.10")
BOND_COUPON_FREQUENCY = 2
BOND_MATURITY = date(2041, 5, 15)


def test_bond_instrument_creation_with_all_fields() -> None:
    """Bond instrument should carry all bond-specific fields."""
    bond = Instrument(
        symbol="SU26238RMFS4",
        market_id="moex",
        name="OFZ 26238",
        instrument_type="bond",
        figi="BBG011FJ4HS6",
        lot_size=1,
        currency="RUB",
        face_value=BOND_FACE_VALUE,
        coupon_rate=BOND_COUPON_RATE,
        coupon_frequency=BOND_COUPON_FREQUENCY,
        maturity_date=BOND_MATURITY,
        floating_coupon=False,
    )
    assert bond.face_value == BOND_FACE_VALUE
    assert bond.coupon_rate == BOND_COUPON_RATE
    assert bond.coupon_frequency == BOND_COUPON_FREQUENCY
    assert bond.maturity_date == BOND_MATURITY
    assert bond.floating_coupon is False
    assert bond.instrument_type == "bond"


def test_bond_floating_coupon_default_is_false() -> None:
    """floating_coupon should default to False when not specified."""
    bond = Instrument(
        symbol="TEST_BOND",
        market_id="moex",
        name="Test Bond",
        instrument_type="bond",
    )
    assert bond.floating_coupon is False


def test_stock_has_none_bond_fields() -> None:
    """Stock instruments should have None for all bond-specific fields."""
    stock = Instrument(
        symbol=AAPL_SYMBOL,
        market_id=US_MARKET,
        name="Apple Inc.",
        instrument_type="stock",
    )
    assert stock.face_value is None
    assert stock.coupon_rate is None
    assert stock.coupon_frequency is None
    assert stock.maturity_date is None
    assert stock.floating_coupon is False


def test_list_by_type_returns_only_bonds() -> None:
    """list_by_type('moex', 'bond') should return only bond instruments."""
    registry = build_default_registry()
    bonds = registry.list_by_type("moex", "bond")
    assert len(bonds) == EXPECTED_MOEX_OFZ_COUNT
    for bond in bonds:
        assert bond.instrument_type == "bond"
        assert bond.market_id == "moex"


def test_list_by_type_returns_only_stocks() -> None:
    """list_by_type('moex', 'stock') should return only stock instruments."""
    registry = build_default_registry()
    stocks = registry.list_by_type("moex", "stock")
    assert len(stocks) == EXPECTED_MOEX_STOCK_COUNT
    for stock in stocks:
        assert stock.instrument_type == "stock"
        assert stock.market_id == "moex"


def test_list_by_type_excludes_inactive() -> None:
    """list_by_type should exclude inactive instruments."""
    registry = InstrumentRegistry()
    active_bond = Instrument(
        symbol="BOND1",
        market_id="moex",
        name="Active Bond",
        instrument_type="bond",
        is_active=True,
    )
    inactive_bond = Instrument(
        symbol="BOND2",
        market_id="moex",
        name="Inactive Bond",
        instrument_type="bond",
        is_active=False,
    )
    registry.register(active_bond)
    registry.register(inactive_bond)
    bonds = registry.list_by_type("moex", "bond")
    assert len(bonds) == 1
    assert bonds[0].symbol == "BOND1"


def test_list_by_type_sorted_by_symbol() -> None:
    """list_by_type should return instruments sorted by symbol."""
    registry = build_default_registry()
    bonds = registry.list_by_type("moex", "bond")
    symbols = [b.symbol for b in bonds]
    assert symbols == sorted(symbols)


def test_default_registry_includes_ofz_instruments() -> None:
    """Default registry must include all 12 OFZ bond instruments."""
    registry = build_default_registry()
    bonds = registry.list_by_type("moex", "bond")
    assert len(bonds) == EXPECTED_MOEX_OFZ_COUNT


def test_ofz_instruments_have_correct_figis() -> None:
    """OFZ instruments in DEFAULT_MOEX_OFZ_INSTRUMENTS must have non-empty FIGIs."""
    expected_figis = {
        "SU26238RMFS4": "BBG011FJ4HS6",
        "SU26239RMFS2": "BBG011FHF1F7",
        "SU26241RMFS8": "BBG01BJBR2W0",
        "SU26243RMFS4": "TCS00A106E90",
        "SU26244RMFS2": "TCS00A1074G2",
        "SU26246RMFS7": "BBG01N0CVG83",
        "SU26252RMFS5": "TCS00A10D4Y2",
        "SU26253RMFS3": "TCS00A10D517",
        "SU29007RMFS0": "BBG007Z5DF79",
        "SU29008RMFS8": "BBG007Z5DZS2",
        "SU29009RMFS6": "BBG007Z5F748",
        "SU29010RMFS4": "BBG007Z5FFL1",
    }
    for inst in DEFAULT_MOEX_OFZ_INSTRUMENTS:
        assert inst.figi == expected_figis[inst.symbol], (
            f"{inst.symbol}: expected FIGI {expected_figis[inst.symbol]!r}, got {inst.figi!r}"
        )


def test_ofz_fixed_coupon_bonds_not_floating() -> None:
    """OFZ-PD (fixed coupon) bonds should have floating_coupon=False."""
    fixed_symbols = {
        "SU26238RMFS4",
        "SU26239RMFS2",
        "SU26241RMFS8",
        "SU26243RMFS4",
        "SU26244RMFS2",
        "SU26246RMFS7",
        "SU26252RMFS5",
        "SU26253RMFS3",
    }
    for inst in DEFAULT_MOEX_OFZ_INSTRUMENTS:
        if inst.symbol in fixed_symbols:
            assert inst.floating_coupon is False, (
                f"{inst.symbol} should be fixed coupon (floating_coupon=False)"
            )


def test_ofz_floating_coupon_bonds() -> None:
    """OFZ-PK (floating coupon) bonds should have floating_coupon=True."""
    floating_symbols = {"SU29007RMFS0", "SU29008RMFS8", "SU29009RMFS6", "SU29010RMFS4"}
    for inst in DEFAULT_MOEX_OFZ_INSTRUMENTS:
        if inst.symbol in floating_symbols:
            assert inst.floating_coupon is True, (
                f"{inst.symbol} should be floating coupon (floating_coupon=True)"
            )


def test_ofz_instruments_all_have_face_value() -> None:
    """All OFZ instruments must have face_value set."""
    for inst in DEFAULT_MOEX_OFZ_INSTRUMENTS:
        assert inst.face_value is not None, f"{inst.symbol} missing face_value"
        assert inst.face_value == BOND_FACE_VALUE, (
            f"{inst.symbol} face_value={inst.face_value}, expected {BOND_FACE_VALUE}"
        )
