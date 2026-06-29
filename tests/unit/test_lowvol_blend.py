"""Tests for step-3 index cap-weight + low-vol-blend policies.

These run the low-vol tilt against a REAL IMOEX cap-weight baseline (from the
committed index-weight snapshot), not the ADV proxy. The data-correctness control
is that lambda=0 reduces to the cap-weight baseline byte-for-byte.
"""

from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal

from finalayze.backtest.equity_tilt_lab import (
    PricePoint,
    make_index_cap_weight_policy,
    make_low_vol_blend_policy,
)

_TOL = Decimal("0.0000001")


def _series(price_fn, n: int = 60, vol_shares: Decimal = Decimal(1000)) -> list[PricePoint]:
    start = date(2024, 1, 1)
    return [(start + timedelta(days=i), price_fn(i), vol_shares) for i in range(n)]


def _panel() -> dict[str, list[PricePoint]]:
    # A: smooth drift (low vol). B: choppy (high vol). Both full history.
    return {
        "A": _series(lambda i: Decimal(100) + Decimal(i) / Decimal(10)),
        "B": _series(lambda i: Decimal(110) if i % 2 == 0 else Decimal(90)),
    }


def test_index_cap_weight_as_of_and_renormalizes() -> None:
    weights = {
        date(2024, 1, 1): {"A": Decimal("0.7"), "B": Decimal("0.3")},
        date(2024, 3, 1): {"A": Decimal("0.4"), "B": Decimal("0.6")},
    }
    pol = make_index_cap_weight_policy(weights)
    # as-of 2024-02-01 -> uses the 2024-01-01 snapshot
    w = pol(date(2024, 2, 15), _panel())
    assert w["A"] == Decimal("0.7")
    assert w["B"] == Decimal("0.3")
    # as-of after 2024-03-01 -> uses the newer snapshot
    w2 = pol(date(2024, 3, 15), _panel())
    assert w2["A"] == Decimal("0.4")
    assert abs(sum(w2.values()) - Decimal(1)) < _TOL


def test_index_cap_weight_restricts_to_available_and_renormalizes() -> None:
    weights = {date(2024, 1, 1): {"A": Decimal("0.6"), "B": Decimal("0.3"), "GONE": Decimal("0.1")}}
    pol = make_index_cap_weight_policy(weights)
    # GONE has no candles -> dropped, A/B renormalized to sum 1
    w = pol(date(2024, 2, 1), _panel())
    assert set(w) == {"A", "B"}
    assert abs(sum(w.values()) - Decimal(1)) < _TOL
    assert abs(w["A"] - (Decimal("0.6") / Decimal("0.9"))) < _TOL


def test_low_vol_blend_lambda_zero_equals_cap() -> None:
    weights = {date(2024, 1, 1): {"A": Decimal("0.5"), "B": Decimal("0.5")}}
    cap = make_index_cap_weight_policy(weights)
    blend0 = make_low_vol_blend_policy(weights, lam=Decimal(0))
    asof = date(2024, 2, 29)
    assert blend0(asof, _panel()) == cap(asof, _panel())  # data-correctness control


def test_low_vol_blend_shifts_weight_to_the_calm_name() -> None:
    weights = {date(2024, 1, 1): {"A": Decimal("0.5"), "B": Decimal("0.5")}}
    cap = make_index_cap_weight_policy(weights)
    blend = make_low_vol_blend_policy(weights, lam=Decimal("0.25"))
    asof = date(2024, 2, 29)
    c = cap(asof, _panel())
    b = blend(asof, _panel())
    assert b["A"] > c["A"]  # low-vol name overweighted vs cap
    assert b["B"] < c["B"]
    assert abs(sum(b.values()) - Decimal(1)) < _TOL
