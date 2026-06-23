"""Phase 80 P80-01/02/03: pure execution-wiring helpers.

normalize_positions_to_symbols (FIGI/symbol -> symbol), resolve_leg_instruments (config ->
Instrument, fail-loud), to_rub_price (bond %-of-face -> RUB; ETF passthrough).
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from finalayze.core.exceptions import InstrumentNotFoundError
from finalayze.core.schemas import AssetClass
from finalayze.markets.instruments import Instrument, build_default_registry
from finalayze.orchestration.rebalance_execution import (
    normalize_positions_to_symbols,
    resolve_leg_instruments,
    to_rub_price,
)

_EQUITY_SYMBOL = "EQMX"
_OFZ_SYMBOL = "SU29024RMFS5"


def _registry() -> object:
    return build_default_registry()


class TestNormalizePositions:
    def test_figi_key_maps_to_symbol(self) -> None:
        """A FIGI-keyed position (TinkoffBroker) maps to its instrument symbol."""
        registry = _registry()
        figi = registry.get(_EQUITY_SYMBOL, "moex").figi
        assert figi is not None
        out = normalize_positions_to_symbols({figi: Decimal(100)}, registry)
        assert out == {_EQUITY_SYMBOL: Decimal(100)}

    def test_symbol_key_passthrough(self) -> None:
        """A symbol-keyed position (SimulatedBroker) passes through unchanged."""
        registry = _registry()
        out = normalize_positions_to_symbols({_OFZ_SYMBOL: Decimal(50)}, registry)
        assert out == {_OFZ_SYMBOL: Decimal(50)}

    def test_unknown_key_skipped(self) -> None:
        """A key that is neither a known FIGI nor a known MOEX symbol is skipped, not an error."""
        registry = _registry()
        out = normalize_positions_to_symbols({"NOT_A_REAL_KEY_XYZ": Decimal(5)}, registry)
        assert out == {}

    def test_mixed_keys(self) -> None:
        """Mixed FIGI + symbol + junk normalizes the recognized ones only."""
        registry = _registry()
        figi = registry.get(_EQUITY_SYMBOL, "moex").figi
        out = normalize_positions_to_symbols(
            {figi: Decimal(10), _OFZ_SYMBOL: Decimal(20), "JUNK": Decimal(1)}, registry
        )
        assert out == {_EQUITY_SYMBOL: Decimal(10), _OFZ_SYMBOL: Decimal(20)}


class TestResolveLegInstruments:
    def test_resolves_equity_and_ofz(self) -> None:
        """The default config tickers resolve to the EQMX ETF + SU29024 bond instruments."""
        legs = resolve_leg_instruments(_registry())
        assert legs[AssetClass.EQUITY].symbol == _EQUITY_SYMBOL
        assert legs[AssetClass.OFZ_PK].symbol == _OFZ_SYMBOL
        assert legs[AssetClass.EQUITY].figi is not None
        assert legs[AssetClass.OFZ_PK].figi is not None

    def test_unresolvable_symbol_fails_loud(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A configured equity ticker absent from the registry raises InstrumentNotFoundError."""
        monkeypatch.setenv("FINALAYZE_SAA_EQUITY_SYMBOL", "NOPE_NOT_LISTED")
        with pytest.raises(InstrumentNotFoundError):
            resolve_leg_instruments(_registry())


class TestToRubPrice:
    def test_bond_percent_of_face_converts_to_rub(self) -> None:
        """A bond quote (% of face) converts to RUB: 95.5% of 1000 face = 955 RUB."""
        bond = Instrument(
            symbol="SU29024RMFS5",
            market_id="moex",
            name="OFZ 29024",
            instrument_type="bond",
            face_value=Decimal(1000),
            floating_coupon=True,
        )
        assert to_rub_price(bond, Decimal("95.5")) == Decimal("955.0")

    def test_etf_price_passes_through(self) -> None:
        """An ETF/share quote is already RUB-per-unit and passes through unchanged."""
        etf = Instrument(symbol="EQMX", market_id="moex", name="x", instrument_type="etf")
        assert to_rub_price(etf, Decimal("123.45")) == Decimal("123.45")

    def test_bond_without_face_value_fails_loud(self) -> None:
        """A bond lacking face_value cannot be priced and raises ValueError."""
        bond = Instrument(
            symbol="X", market_id="moex", name="x", instrument_type="bond", face_value=None
        )
        with pytest.raises(ValueError, match="face_value"):
            to_rub_price(bond, Decimal("95.5"))
