"""Phase 79 P79-02: fail-closed SAA rebalance config (band + leg symbols).

The equity/OFZ-PK tradeable tickers are operator-overridable config (env), default to the
verified snapshot instruments (EQMX / SU29024RMFS5), and FAIL CLOSED on an empty value -- a
blank ticker on a money path must raise, never silently resolve to nothing (P79-R4/R14).
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from finalayze.config.rebalance_config import (
    SAA_REBALANCE_BAND_PCT,
    get_equity_drawdown_survival_pct,
    get_equity_im_hike_mult,
    get_equity_margin_rate,
    get_equity_point_value,
    get_equity_symbol,
    get_ofz_pk_symbol,
)
from finalayze.core.exceptions import ConfigurationError

_DRAWDOWN_ENV = "FINALAYZE_SAA_EQUITY_DRAWDOWN_SURVIVAL_PCT"
_IM_HIKE_ENV = "FINALAYZE_SAA_EQUITY_IM_HIKE_MULT"
_MARGIN_RATE_ENV = "FINALAYZE_SAA_EQUITY_MARGIN_RATE"


def test_rebalance_band_is_two_percent() -> None:
    """The no-churn band is an exact Decimal 2% (no binary-float drift)."""
    assert Decimal("0.02") == SAA_REBALANCE_BAND_PCT


def test_default_leg_symbols_are_verified_snapshot_instruments() -> None:
    """Defaults are API-tradeable: IMOEXF future (equity) / SU29024RMFS5 (OFZ-PK)."""
    assert get_equity_symbol() == "IMOEXF"
    assert get_ofz_pk_symbol() == "SU29024RMFS5"


def test_env_override_equity_symbol(monkeypatch: pytest.MonkeyPatch) -> None:
    """The operator overrides the equity ticker via env (e.g. SBMX) with no code change."""
    monkeypatch.setenv("FINALAYZE_SAA_EQUITY_SYMBOL", "SBMX")
    assert get_equity_symbol() == "SBMX"


def test_env_override_ofz_pk_symbol(monkeypatch: pytest.MonkeyPatch) -> None:
    """The operator overrides the OFZ-PK ticker via env to any SU29* issue."""
    monkeypatch.setenv("FINALAYZE_SAA_OFZ_PK_SYMBOL", "SU29025RMFS2")
    assert get_ofz_pk_symbol() == "SU29025RMFS2"


def test_empty_equity_symbol_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty equity ticker raises ConfigurationError, never resolves to ''."""
    monkeypatch.setenv("FINALAYZE_SAA_EQUITY_SYMBOL", "")
    with pytest.raises(ConfigurationError):
        get_equity_symbol()


def test_whitespace_ofz_pk_symbol_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    """A whitespace-only OFZ-PK ticker fails closed (stripped to empty)."""
    monkeypatch.setenv("FINALAYZE_SAA_OFZ_PK_SYMBOL", "   ")
    with pytest.raises(ConfigurationError):
        get_ofz_pk_symbol()


def test_default_equity_point_value() -> None:
    """IMOEXF point value defaults to 10 RUB/point."""
    assert get_equity_point_value() == Decimal(10)


def test_env_override_equity_point_value(monkeypatch: pytest.MonkeyPatch) -> None:
    """The point value is operator-overridable via env."""
    monkeypatch.setenv("FINALAYZE_SAA_EQUITY_POINT_VALUE", "5")
    assert get_equity_point_value() == Decimal(5)


def test_invalid_equity_point_value_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-numeric point value raises ConfigurationError."""
    monkeypatch.setenv("FINALAYZE_SAA_EQUITY_POINT_VALUE", "abc")
    with pytest.raises(ConfigurationError):
        get_equity_point_value()


def test_nonpositive_equity_point_value_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-positive point value raises ConfigurationError."""
    monkeypatch.setenv("FINALAYZE_SAA_EQUITY_POINT_VALUE", "-1")
    with pytest.raises(ConfigurationError):
        get_equity_point_value()


def test_infinity_equity_point_value_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    """An Infinity point value is rejected (WR-01) -- Decimal('inf') > 0 would slip past."""
    monkeypatch.setenv("FINALAYZE_SAA_EQUITY_POINT_VALUE", "inf")
    with pytest.raises(ConfigurationError):
        get_equity_point_value()


# ── Phase 86: funded-equity reserve parameters (drawdown survival, IM-hike, static margin) ──


def test_default_drawdown_survival_pct() -> None:
    """The drawdown-survival fraction defaults to 0.45 (operator's '-45%' intent)."""
    assert get_equity_drawdown_survival_pct() == Decimal("0.45")


def test_env_override_drawdown_survival_pct(monkeypatch: pytest.MonkeyPatch) -> None:
    """The drawdown-survival fraction is operator-overridable via env."""
    monkeypatch.setenv(_DRAWDOWN_ENV, "0.50")
    assert get_equity_drawdown_survival_pct() == Decimal("0.50")


@pytest.mark.parametrize("bad", ["abc", "0", "-0.1", "1.5", "inf", "nan"])
def test_drawdown_survival_pct_fails_closed(monkeypatch: pytest.MonkeyPatch, bad: str) -> None:
    """Non-numeric / non-finite / out-of-(0,1] drawdown fractions fail closed."""
    monkeypatch.setenv(_DRAWDOWN_ENV, bad)
    with pytest.raises(ConfigurationError):
        get_equity_drawdown_survival_pct()


def test_default_im_hike_mult() -> None:
    """The IM-hike multiplier defaults to 2.5 (the Feb-2022 overnight ratio)."""
    assert get_equity_im_hike_mult() == Decimal("2.5")


def test_env_override_im_hike_mult(monkeypatch: pytest.MonkeyPatch) -> None:
    """The IM-hike multiplier is operator-overridable via env."""
    monkeypatch.setenv(_IM_HIKE_ENV, "3")
    assert get_equity_im_hike_mult() == Decimal(3)


@pytest.mark.parametrize("bad", ["abc", "0.9", "0", "-1", "inf", "nan"])
def test_im_hike_mult_fails_closed(monkeypatch: pytest.MonkeyPatch, bad: str) -> None:
    """A non-numeric / non-finite / <1 IM-hike multiplier fails closed."""
    monkeypatch.setenv(_IM_HIKE_ENV, bad)
    with pytest.raises(ConfigurationError):
        get_equity_im_hike_mult()


def test_im_hike_mult_one_is_allowed(monkeypatch: pytest.MonkeyPatch) -> None:
    """A multiplier of exactly 1.0 (no IM-hike headroom) is allowed."""
    monkeypatch.setenv(_IM_HIKE_ENV, "1")
    assert get_equity_im_hike_mult() == Decimal(1)


def test_equity_margin_rate_none_when_unset() -> None:
    """The static margin rate is None unless the operator explicitly sets it (no auto-fallback)."""
    assert get_equity_margin_rate() is None


def test_equity_margin_rate_when_set(monkeypatch: pytest.MonkeyPatch) -> None:
    """An explicit static margin rate is parsed for offline planning."""
    monkeypatch.setenv(_MARGIN_RATE_ENV, "0.10")
    assert get_equity_margin_rate() == Decimal("0.10")


@pytest.mark.parametrize("bad", ["abc", "0", "-0.1", "1.5", "inf", "nan"])
def test_equity_margin_rate_fails_closed(monkeypatch: pytest.MonkeyPatch, bad: str) -> None:
    """A set-but-malformed static margin rate fails closed (never a silent unsafe guess)."""
    monkeypatch.setenv(_MARGIN_RATE_ENV, bad)
    with pytest.raises(ConfigurationError):
        get_equity_margin_rate()
