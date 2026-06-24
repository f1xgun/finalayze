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
    get_equity_point_value,
    get_equity_symbol,
    get_ofz_pk_symbol,
)
from finalayze.core.exceptions import ConfigurationError


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
