"""Unit tests for the instrument registry (Layer 2)."""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from finalayze.core.exceptions import ConfigurationError, InstrumentNotFoundError
from finalayze.markets.instruments import (
    DEFAULT_MOEX_INSTRUMENTS,
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
EXPECTED_DEFAULT_US_COUNT = 7  # exact (UNIV-05)
EXPECTED_COUNT_AFTER_TWO = 2

# Per-class FLOORS for the snapshot-derived MOEX universe (UNIV-03/05). The full
# universe drifts as MOEX lists/delists, so we assert floors, not exact counts
# (Pitfall 4). Floors sit conservatively below the committed snapshot's counts.
MIN_MOEX_SHARES = 250  # floors (Pitfall 4 -- universe drifts)
MIN_MOEX_ETFS = 40
MIN_MOEX_BONDS = 1400
MIN_MOEX_FUTURES = 350
MIN_MOEX_CURRENCIES = 10
SBER_FIGI = "BBG004730N88"  # UNIV-09
SBER_SYMBOL = "SBER"
MOEX_MARKET = "moex"


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
    """US count is exact (UNIV-05); MOEX is a floor (UNIV-03/05, universe drifts)."""
    registry = build_default_registry()
    assert len(registry.list_by_market(US_MARKET)) == EXPECTED_DEFAULT_US_COUNT
    assert len(registry.list_by_market(MOEX_MARKET)) >= MIN_MOEX_SHARES + MIN_MOEX_BONDS


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


# Snapshot-universe core blue chips that MUST always be present (subset check,
# not an exact universe -- the full snapshot has 2300+ instruments, UNIV-01/03).
EXPECTED_MOEX_CORE_SYMBOLS = {
    "SBER",
    "GAZP",
    "LKOH",
    "GMKN",
    "ROSN",
    "MOEX",
    "T",
}
# Traded OFZ derived from config segments (ru_ofz_pd + ru_ofz_pk), TCSG->T n/a.
# These are the YTM-able bonds the snapshot must carry (UNIV-06 / UNIV-10).
TRADED_OFZ_SYMBOLS = {
    "SU26239RMFS2",
    "SU26241RMFS8",
    "SU26243RMFS4",
    "SU26244RMFS2",
    "SU26246RMFS7",
    "SU26252RMFS5",
    "SU26253RMFS3",
    "SU29007RMFS0",
    "SU29008RMFS8",
    "SU29009RMFS6",
    "SU29010RMFS4",
}
OFZ_FLOATING_SYMBOLS = {"SU29007RMFS0", "SU29008RMFS8", "SU29009RMFS6", "SU29010RMFS4"}
OFZ_FIXED_SYMBOLS = TRADED_OFZ_SYMBOLS - OFZ_FLOATING_SYMBOLS
OFZ_FACE_VALUE = Decimal(1000)
TCSG_ALIAS = {"TCSG": "T"}  # historical rebrand reconciliation (Pitfall 2)


def test_default_registry_includes_moex_instruments() -> None:
    """Default registry must include the full snapshot MOEX universe (floor, UNIV-03)."""
    registry = build_default_registry()
    moex_instruments = registry.list_by_market(MOEX_MARKET)
    assert len(moex_instruments) >= MIN_MOEX_SHARES + MIN_MOEX_BONDS


def test_moex_core_symbols_present(monkeypatch: pytest.MonkeyPatch) -> None:
    """Core MOEX blue chips must always resolve in the snapshot universe (UNIV-01)."""
    registry = build_default_registry()
    for sym in EXPECTED_MOEX_CORE_SYMBOLS:
        inst = registry.get(sym, MOEX_MARKET)
        assert inst.market_id == MOEX_MARKET


def test_moex_instruments_with_figi_nonempty() -> None:
    """Every snapshot MOEX instrument carrying a FIGI must have a non-empty value."""
    registry = build_default_registry()
    with_figi = [inst for inst in registry.list_by_market(MOEX_MARKET) if inst.figi is not None]
    assert len(with_figi) > 0
    for inst in with_figi:
        assert inst.figi != "", f"{inst.symbol} has empty FIGI"


def test_universe_counts() -> None:
    """Per-class floors hold for the snapshot universe (UNIV-03, named floors PLR2004)."""
    registry = build_default_registry()
    assert len(registry.list_by_type(MOEX_MARKET, "stock")) >= MIN_MOEX_SHARES
    assert len(registry.list_by_type(MOEX_MARKET, "etf")) >= MIN_MOEX_ETFS
    assert len(registry.list_by_type(MOEX_MARKET, "bond")) >= MIN_MOEX_BONDS
    assert len(registry.list_by_type(MOEX_MARKET, "future")) >= MIN_MOEX_FUTURES
    assert len(registry.list_by_type(MOEX_MARKET, "currency")) >= MIN_MOEX_CURRENCIES


def _required_moex_symbols() -> set[str]:
    """Derive the enabled-MOEX required-symbol set from config segments (UNIV-02)."""
    from config.segments import DEFAULT_SEGMENTS

    req: set[str] = set()
    for seg in DEFAULT_SEGMENTS:
        if seg.market != "moex" or not seg.enabled:
            continue
        req |= {TCSG_ALIAS.get(s, s) for s in seg.symbols}
    return req


def test_required_symbols_resolve() -> None:
    """Every enabled-MOEX required symbol resolves via the registry (UNIV-02, TCSG->T)."""
    registry = build_default_registry()
    required = _required_moex_symbols()
    assert required  # sanity: derivation is non-empty
    for sym in required:
        # raises InstrumentNotFoundError if absent -- the assertion
        inst = registry.get(sym, MOEX_MARKET)
        assert inst.symbol == sym


def test_get_by_figi_resolves_sber() -> None:
    """FIGI lookup resolves to SBER from the snapshot (UNIV-09)."""
    registry = build_default_registry()
    assert registry.get_by_figi(SBER_FIGI).symbol == SBER_SYMBOL


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
    """list_by_type('moex', 'bond') should return only bond instruments (floor)."""
    registry = build_default_registry()
    bonds = registry.list_by_type("moex", "bond")
    assert len(bonds) >= MIN_MOEX_BONDS
    for bond in bonds:
        assert bond.instrument_type == "bond"
        assert bond.market_id == "moex"


def test_list_by_type_returns_only_stocks() -> None:
    """list_by_type('moex', 'stock') should return only stock instruments (floor)."""
    registry = build_default_registry()
    stocks = registry.list_by_type("moex", "stock")
    assert len(stocks) >= MIN_MOEX_SHARES
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
    """Default registry must include the traded OFZ as resolvable bonds (UNIV-10)."""
    registry = build_default_registry()
    for sym in TRADED_OFZ_SYMBOLS:
        inst = registry.get(sym, MOEX_MARKET)
        assert inst.instrument_type == "bond"


def _ofz_by_symbol() -> dict[str, Instrument]:
    """Index the snapshot OFZ shim by symbol for the traded-OFZ assertions."""
    return {i.symbol: i for i in DEFAULT_MOEX_OFZ_INSTRUMENTS}


def test_ofz_instruments_have_correct_figis() -> None:
    """Traded OFZ in the snapshot shim must carry their known non-empty FIGIs."""
    expected_figis = {
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
    ofz = _ofz_by_symbol()
    for sym, figi in expected_figis.items():
        assert ofz[sym].figi == figi, f"{sym}: expected FIGI {figi!r}, got {ofz[sym].figi!r}"


def test_ofz_fixed_coupon_bonds_not_floating() -> None:
    """Traded OFZ-PD (fixed coupon) bonds should have floating_coupon=False."""
    ofz = _ofz_by_symbol()
    for sym in OFZ_FIXED_SYMBOLS:
        assert ofz[sym].floating_coupon is False, (
            f"{sym} should be fixed coupon (floating_coupon=False)"
        )


def test_ofz_floating_coupon_bonds() -> None:
    """Traded OFZ-PK (floating coupon) bonds should have floating_coupon=True."""
    ofz = _ofz_by_symbol()
    for sym in OFZ_FLOATING_SYMBOLS:
        assert ofz[sym].floating_coupon is True, (
            f"{sym} should be floating coupon (floating_coupon=True)"
        )


def test_ofz_instruments_all_have_face_value() -> None:
    """All traded OFZ must have the par face_value set (UNIV-06)."""
    ofz = _ofz_by_symbol()
    for sym in TRADED_OFZ_SYMBOLS:
        inst = ofz[sym]
        assert inst.face_value is not None, f"{sym} missing face_value"
        assert inst.face_value == OFZ_FACE_VALUE, (
            f"{sym} face_value={inst.face_value}, expected {OFZ_FACE_VALUE}"
        )


def test_ofz_yieldable() -> None:
    """Every traded OFZ has the four YTM inputs non-None (UNIV-06 / UNIV-10, D-01)."""
    registry = build_default_registry()
    for sym in TRADED_OFZ_SYMBOLS:
        inst = registry.get(sym, MOEX_MARKET)
        assert inst.coupon_rate is not None, f"{sym} missing coupon_rate"
        assert inst.coupon_frequency is not None, f"{sym} missing coupon_frequency"
        assert inst.face_value is not None, f"{sym} missing face_value"
        assert inst.maturity_date is not None, f"{sym} missing maturity_date"


def test_all_moex_equity_segment_symbols_have_figi() -> None:
    """Sprint 8 / audit #16: every MOEX equity segment symbol must resolve to a
    registered instrument carrying a Tinkoff FIGI.

    Without a FIGI the TinkoffFetcher raises InstrumentNotFoundError and the
    symbol cannot be backtested or traded — so a sector preset on such a segment
    is dead weight. This guard ties the segment universe (config) to the
    instrument registry (markets) so a newly declared symbol can't slip through
    without a FIGI.
    """
    from config.segments import DEFAULT_SEGMENTS

    registry = build_default_registry()
    missing: list[str] = []
    for seg in DEFAULT_SEGMENTS:
        if seg.market != "moex" or "bond_carry" in seg.active_strategies:
            continue
        for sym in seg.symbols:
            try:
                inst = registry.get(sym, "moex")
            except InstrumentNotFoundError:
                missing.append(f"{seg.segment_id}:{sym} (unregistered)")
                continue
            if not inst.figi:
                missing.append(f"{seg.segment_id}:{sym} (no FIGI)")
    assert not missing, f"MOEX equity symbols without a FIGI: {missing}"


# ---------------------------------------------------------------------------
# Snapshot-loader fail-closed contract (Plan 65-03)
# ---------------------------------------------------------------------------
CORRUPT_SNAPSHOT_BODY = "{ this is not valid json ::: ]"


def test_build_default_registry_fail_closed_on_missing_snapshot(
    monkeypatch: pytest.MonkeyPatch, tmp_path: object
) -> None:
    """A missing snapshot file makes build_default_registry raise ConfigurationError
    (no silent fallback to a stale hand-list -- UNIV-04 / D-04)."""
    from pathlib import Path

    import finalayze.markets.instruments as instruments_mod

    missing = Path(str(tmp_path)) / "does_not_exist" / "moex_universe.json"
    monkeypatch.setattr(instruments_mod, "_SNAPSHOT", missing)
    with pytest.raises(ConfigurationError):
        build_default_registry()


def test_build_default_registry_fail_closed_on_corrupt_snapshot(
    monkeypatch: pytest.MonkeyPatch, tmp_path: object
) -> None:
    """A corrupt snapshot file makes build_default_registry raise ConfigurationError
    (no silent fallback -- UNIV-04 / D-04)."""
    from pathlib import Path

    import finalayze.markets.instruments as instruments_mod

    corrupt = Path(str(tmp_path)) / "moex_universe.json"
    corrupt.write_text(CORRUPT_SNAPSHOT_BODY, encoding="utf-8")
    monkeypatch.setattr(instruments_mod, "_SNAPSHOT", corrupt)
    with pytest.raises(ConfigurationError):
        build_default_registry()


def test_build_default_registry_has_us_and_moex(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default registry resolves both the US set (AAPL) and the MOEX snapshot (SBER) -- UNIV-01."""
    registry = build_default_registry()
    assert registry.get(AAPL_SYMBOL, US_MARKET).market_id == US_MARKET
    assert registry.get(SBER_SYMBOL, MOEX_MARKET).market_id == MOEX_MARKET


def test_default_moex_compat_shims_are_nonempty_instrument_lists() -> None:
    """DEFAULT_MOEX_INSTRUMENTS / _OFZ_ survive as snapshot-derived module-level lists
    so the ~10 direct importers need zero edits -- UNIV-07 compat shim."""
    assert len(DEFAULT_MOEX_INSTRUMENTS) > 0
    assert len(DEFAULT_MOEX_OFZ_INSTRUMENTS) > 0
    assert all(isinstance(i, Instrument) for i in DEFAULT_MOEX_INSTRUMENTS)
    assert all(isinstance(i, Instrument) for i in DEFAULT_MOEX_OFZ_INSTRUMENTS)
    # the direct-importer contract: iterate-and-register works
    registry = InstrumentRegistry()
    for inst in DEFAULT_MOEX_INSTRUMENTS:
        registry.register(inst)
    assert len(registry) == len(DEFAULT_MOEX_INSTRUMENTS)
