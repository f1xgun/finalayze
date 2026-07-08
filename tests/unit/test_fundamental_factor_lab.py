"""Unit tests for the MOEX fundamental factor primitives.

Pins the anti-fabrication invariants: a missing input never becomes a 0.0 factor,
a degenerate cross-section raises rather than inventing a spread, and a name
without both a factor and a forward return is dropped, not imputed.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from finalayze.backtest.fundamental_factor_lab import (
    deposit_accrual,
    detect_splits,
    excess_over_deposit,
    forward_total_return,
    gross_profit_to_assets,
    long_short_spread,
    split_factor_at,
    tercile_labels,
)


def test_gp_to_assets_basic() -> None:
    # GMKN FY2021: revenue 1317, COGS 427.6, assets 1741 -> ~0.511
    got = gross_profit_to_assets(Decimal(1317), Decimal("427.6"), Decimal(1741))
    assert got is not None
    assert abs(got - Decimal("0.5109")) < Decimal("0.001")


def test_gp_to_assets_missing_returns_none() -> None:
    assert gross_profit_to_assets(None, Decimal(1), Decimal(1)) is None  # type: ignore[arg-type]
    assert gross_profit_to_assets(Decimal(1), None, Decimal(1)) is None  # type: ignore[arg-type]


def test_gp_to_assets_zero_assets_returns_none_not_zero() -> None:
    # A zero/negative denominator must NOT fabricate a 0.0 factor.
    assert gross_profit_to_assets(Decimal(100), Decimal(40), Decimal(0)) is None
    assert gross_profit_to_assets(Decimal(100), Decimal(40), Decimal(-5)) is None


def test_tercile_labels_top_and_bottom() -> None:
    vals = [("a", Decimal("0.1")), ("b", Decimal("0.5")), ("c", Decimal("0.9"))]
    labs = tercile_labels(vals)
    assert labs["c"] == "top"
    assert labs["a"] == "bottom"
    assert labs["b"] == "mid"


def test_tercile_labels_monotonic_six() -> None:
    vals = [(chr(97 + i), Decimal(i)) for i in range(6)]  # a..f, 0..5
    labs = tercile_labels(vals)
    assert labs["a"] == "bottom" and labs["b"] == "bottom"
    assert labs["f"] == "top" and labs["e"] == "top"
    assert labs["c"] == "mid" and labs["d"] == "mid"


def test_tercile_labels_raises_on_degenerate() -> None:
    with pytest.raises(ValueError, match=">=3 names"):
        tercile_labels([("a", Decimal(1)), ("b", Decimal(2))])


def test_forward_total_return_with_dividends() -> None:
    # entry 100, exit 110, div 5 -> 0.15
    got = forward_total_return(Decimal(100), Decimal(110), Decimal(5))
    assert got == Decimal("0.15")


def test_forward_total_return_price_only() -> None:
    assert forward_total_return(Decimal(100), Decimal(90)) == Decimal("-0.10")


def test_forward_total_return_bad_entry_none() -> None:
    assert forward_total_return(Decimal(0), Decimal(90)) is None
    assert forward_total_return(Decimal(-1), Decimal(90)) is None


def test_deposit_accrual_one_year() -> None:
    # 18% for 365 days -> 0.18
    assert abs(deposit_accrual(Decimal(18), 365) - Decimal("0.18")) < Decimal("1e-9")


def test_deposit_accrual_half_year() -> None:
    assert abs(deposit_accrual(Decimal(20), 182) - Decimal("0.09973")) < Decimal("0.001")


def test_deposit_accrual_negative_days_raises() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        deposit_accrual(Decimal(18), -1)


def test_long_short_spread_basic() -> None:
    labels = {"a": "bottom", "b": "mid", "c": "top"}
    fwd = {"a": Decimal("0.0"), "b": Decimal("0.1"), "c": Decimal("0.3")}
    # top mean 0.3 - bottom mean 0.0 = 0.3
    assert long_short_spread(labels, fwd) == Decimal("0.3")


def test_long_short_spread_drops_names_without_forward() -> None:
    # 'c' (top) has no forward return -> top leg empty -> None, never imputed.
    labels = {"a": "bottom", "c": "top"}
    fwd = {"a": Decimal("0.0")}
    assert long_short_spread(labels, fwd) is None


def test_long_short_spread_empty_leg_none() -> None:
    labels = {"a": "mid", "b": "mid"}
    fwd = {"a": Decimal("0.1"), "b": Decimal("0.2")}
    assert long_short_spread(labels, fwd) is None


def test_excess_over_deposit() -> None:
    assert excess_over_deposit(Decimal("0.25"), Decimal("0.18")) == Decimal("0.07")
    assert excess_over_deposit(Decimal("0.10"), Decimal("0.18")) == Decimal("-0.08")


def test_detect_splits_forward() -> None:
    # PLZL-like 1:10 forward split: price drops ~10x on the split day.
    prices = {
        "2025-03-25": 19000.0,
        "2025-03-26": 19011.0,
        "2025-03-27": 1867.0,
        "2025-03-28": 1900.0,
    }
    splits = detect_splits(prices)
    assert len(splits) == 1
    sdate, factor = splits[0]
    assert sdate == "2025-03-27"
    assert abs(factor - 1867.0 / 19011.0) < 1e-6  # <1 for a forward split


def test_detect_splits_reverse() -> None:
    # VTBR-like reverse split: price jumps way up on the split day.
    prices = {"2024-07-11": 0.021, "2024-07-12": 0.02, "2024-07-15": 92.95}
    splits = detect_splits(prices)
    assert len(splits) == 1
    assert splits[0][0] == "2024-07-15"
    assert splits[0][1] > 3  # >1 for a reverse split


def test_detect_splits_ignores_real_moves() -> None:
    # A brutal -40% crash day is NOT a split (ratio 1.67 < 3).
    prices = {"2022-02-24": 100.0, "2022-02-25": 60.0, "2022-02-28": 55.0}
    assert detect_splits(prices) == []


def test_split_factor_at_applies_only_before() -> None:
    splits = [("2025-03-27", 0.0982)]
    # a value dated BEFORE the split is scaled down; on/after the split is unchanged.
    assert abs(split_factor_at(splits, "2024-01-01") - 0.0982) < 1e-9
    assert split_factor_at(splits, "2025-03-27") == 1.0
    assert split_factor_at(splits, "2025-06-01") == 1.0


def test_split_factor_at_compounds_multiple() -> None:
    splits = [("2024-04-08", 0.5), ("2025-03-27", 0.1)]
    # dated before both -> both apply (0.5 * 0.1); between -> only the later (0.1); after -> 1.0
    assert abs(split_factor_at(splits, "2023-01-01") - 0.05) < 1e-9
    assert abs(split_factor_at(splits, "2024-06-01") - 0.1) < 1e-9
    assert split_factor_at(splits, "2025-12-01") == 1.0
