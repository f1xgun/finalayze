"""Unit tests for the PEAD sleeve lab (pure NAV primitives)."""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from finalayze.backtest.pead_lab import (
    blend_pead_nav,
    daily_factors,
    net_window_factor,
    realpath_window,
    spread_factor,
)

_ZERO = Decimal(0)
_ONE = Decimal(1)
_D0 = date(2025, 1, 1)
_D1 = date(2025, 1, 2)
_D2 = date(2025, 1, 3)


def test_net_window_factor_taxes_only_gains() -> None:
    # +10% gross, no cost, 13% NDFL -> 1 + 0.10*0.87 = 1.087.
    assert net_window_factor(Decimal(100), Decimal(110), _ZERO, Decimal("0.13")) == Decimal("1.087")
    # a loss is not taxed.
    assert net_window_factor(Decimal(100), Decimal(90), _ZERO, Decimal("0.13")) == Decimal("0.90")


def test_net_window_factor_charges_both_legs() -> None:
    # flat price, 1% per side -> (0.99)^2 < 1 (a round trip of a non-mover loses).
    f = net_window_factor(Decimal(100), Decimal(100), Decimal("0.01"), _ZERO)
    assert f == (Decimal(1) - Decimal("0.01")) * (Decimal(1) - Decimal("0.01"))
    assert f < _ONE


def test_net_window_factor_nonpositive_entry_is_identity() -> None:
    assert net_window_factor(_ZERO, Decimal(50), Decimal("0.01"), Decimal("0.13")) == _ONE


def test_spread_factor_product_recovers_total() -> None:
    total = Decimal("1.20")
    n = 4
    per = spread_factor(total, n)
    assert float(per**n) == pytest.approx(float(total), rel=1e-9)
    assert spread_factor(total, 0) == _ONE
    assert spread_factor(_ZERO, 4) == _ONE


def test_blend_pead_nav_empty_reproduces_deposit() -> None:
    axis = [_D0, _D1, _D2]
    dep = {_D1: Decimal("1.001"), _D2: Decimal("1.002")}
    nav = blend_pead_nav(axis, dep, {})
    assert nav[0] == (_D0, _ONE)
    assert nav[1][1] == Decimal("1.001")
    assert nav[2][1] == Decimal("1.001") * Decimal("1.002")


def test_blend_pead_nav_uses_window_on_active_bar() -> None:
    axis = [_D0, _D1, _D2]
    dep = {_D1: Decimal("1.001"), _D2: Decimal("1.001")}
    # a single window active on D2 with a per-bar factor of 1.05 overrides the deposit.
    nav = blend_pead_nav(axis, dep, {_D2: [Decimal("1.05")]})
    assert nav[1][1] == Decimal("1.001")  # idle -> deposit
    assert nav[2][1] == Decimal("1.001") * Decimal("1.05")  # active -> window


def test_blend_pead_nav_equal_weights_concurrent_windows() -> None:
    axis = [_D0, _D1]
    nav = blend_pead_nav(axis, {}, {_D1: [Decimal("1.10"), Decimal("0.90")]})
    assert nav[1][1] == _ONE  # (1.10 + 0.90) / 2 = 1.00


def test_realpath_window_preserves_daily_moves_and_nets_to_target() -> None:
    active = [_D1, _D2]
    name_daily = {_D1: Decimal("1.05"), _D2: Decimal("0.98")}  # real up then down
    target = Decimal("1.10")  # the net_window_factor terminal
    out = realpath_window(active, name_daily, target)
    # interior bar keeps its REAL move; only the last bar is rescaled.
    assert out[_D1] == Decimal("1.05")
    # window product equals the target (to Decimal rounding).
    prod = out[_D1] * out[_D2]
    assert abs(prod - target) < Decimal("1e-20")
    # the last bar carries the adjustment (not the flat spread).
    assert out[_D2] != Decimal("0.98")


def test_realpath_window_empty_is_empty() -> None:
    assert realpath_window([], {_D1: Decimal("1.01")}, Decimal("1.05")) == {}


def test_daily_factors_from_levels() -> None:
    curve = [(_D0, Decimal(100)), (_D1, Decimal(101)), (_D2, Decimal(101))]
    f = daily_factors(curve)
    assert f[_D1] == Decimal("1.01")
    assert f[_D2] == _ONE
    assert _D0 not in f  # first bar has no prior


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
