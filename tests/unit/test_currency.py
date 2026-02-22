"""Unit tests for the currency conversion stub (Layer 2)."""

from __future__ import annotations

from decimal import Decimal

import pytest

from finalayze.markets.currency import _FALLBACK_USDRUB, CurrencyConverter

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
USD = "USD"
RUB = "RUB"
UNSUPPORTED_CURRENCY = "JPY"
UNSUPPORTED_PAIR = f"{USD}{UNSUPPORTED_CURRENCY}"

AMOUNT_100_USD = Decimal(100)
AMOUNT_9000_RUB = Decimal(9000)
NEW_RATE = Decimal(95)
ZERO_RATE = Decimal(0)
NEGATIVE_RATE = Decimal(-1)
ONE = Decimal(1)

MATCH_RATE_POSITIVE = "Rate must be positive"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_convert_usd_to_usd_identity() -> None:
    converter = CurrencyConverter()
    result = converter.convert(AMOUNT_100_USD, USD, USD)
    assert result == AMOUNT_100_USD


def test_convert_usd_to_rub() -> None:
    converter = CurrencyConverter()
    result = converter.convert(AMOUNT_100_USD, USD, RUB)
    assert result == AMOUNT_100_USD * _FALLBACK_USDRUB


def test_convert_rub_to_usd() -> None:
    converter = CurrencyConverter()
    result = converter.convert(AMOUNT_9000_RUB, RUB, USD)
    expected = AMOUNT_9000_RUB * (ONE / _FALLBACK_USDRUB)
    assert result == expected


def test_unsupported_pair_raises() -> None:
    converter = CurrencyConverter()
    with pytest.raises(ValueError, match=UNSUPPORTED_PAIR):
        converter.convert(AMOUNT_100_USD, USD, UNSUPPORTED_CURRENCY)


def test_set_rate_updates_and_inverse() -> None:
    converter = CurrencyConverter()
    converter.set_rate("USDRUB", NEW_RATE)
    assert converter.convert(ONE, USD, RUB) == NEW_RATE
    expected_inverse = ONE / NEW_RATE
    assert converter.convert(ONE, RUB, USD) == expected_inverse


def test_set_rate_zero_raises() -> None:
    converter = CurrencyConverter()
    with pytest.raises(ValueError, match=MATCH_RATE_POSITIVE):
        converter.set_rate("USDRUB", ZERO_RATE)


def test_to_base_converts_to_base_currency() -> None:
    converter = CurrencyConverter(base_currency=USD)
    result = converter.to_base(AMOUNT_9000_RUB, RUB)
    expected = AMOUNT_9000_RUB * (ONE / _FALLBACK_USDRUB)
    assert result == expected
