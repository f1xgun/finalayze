"""SAA rebalance config -- no-churn band + operator-overridable leg tickers (Phase 79 P79-02).

The weights-to-orders engine needs concrete TRADEABLE MOEX tickers for the equity and OFZ-PK
legs. These are operator-overridable via env (no code change) and default to the verified
committed-snapshot instruments:

- equity   -> ``IMOEXF`` (perpetual cash-settled MOEX-index FUTURE, figi FUTIMOEXF000) -- the
  API-tradeable broad-index instrument. MOEX index ETFs (EQMX/SBMX) are forbidden for API
  trading (sandbox cert, Phase 84), so the equity sleeve is the index future (Phase 85). It is
  leveraged: sized to match the target index EXPOSURE (contract notional = points * point_value).
- ofz_pk   -> ``SU29024RMFS5`` (ОФЗ 29024, mat 2035) -- a liquid federal floating-coupon issue
  from the 21-strong ``SU29*`` universe.

Resolution is FAIL-CLOSED (D-04 / saa_portfolio_writer pattern): a blank/whitespace ticker
raises ``ConfigurationError`` rather than silently resolving to an empty symbol on a money path.
"""

from __future__ import annotations

import os
from decimal import Decimal, InvalidOperation

from finalayze.core.exceptions import ConfigurationError

# No-churn / dust band: a leg whose |delta| is under this fraction of the budget does not trade.
SAA_REBALANCE_BAND_PCT = Decimal("0.02")

# Verified-snapshot defaults (overridable via env). Equity = the IMOEXF index future (ETFs are
# API-forbidden); OFZ-PK = a federal floating-coupon bond.
_DEFAULT_EQUITY_SYMBOL = "IMOEXF"
_DEFAULT_OFZ_PK_SYMBOL = "SU29024RMFS5"
# RUB value of one index point for the equity FUTURE (IMOEXF: min_price_increment_amount /
# min_price_increment = 5 / 0.5 = 10). Used to size the future by exposure (contract notional =
# points * point_value). Override if the equity future is changed.
_DEFAULT_EQUITY_POINT_VALUE = "10"

# Phase 86 "fully-funded synthetic equity": the equity FUTURE is funded fully -- only its margin is
# charged, plus a CASH reserve sized to survive a deep IMOEX drawdown EVEN if MOEX hikes the initial
# margin (IM) mid-crash, and the freed cash is swept into the deposit anchor (deposit-as-plug):
#   reserve = exposure * drawdown_survival_pct + margin * (im_hike_mult - 1)
# so the survivable drawdown stays >= drawdown_survival_pct even after the IM is hiked im_hike_mult
# times. Defaults: 0.45 (operator's "-45%" intent, brackets the 2022 -33% cluster) and 2.5 (the
# observed Feb-2022 overnight IM ratio). Both env-overridable + fail-closed.
SAA_EQUITY_DRAWDOWN_SURVIVAL_PCT_DEFAULT = Decimal("0.45")
SAA_EQUITY_IM_HIKE_MULT_DEFAULT = Decimal("2.5")

_EQUITY_ENV = "FINALAYZE_SAA_EQUITY_SYMBOL"
_EQUITY_POINT_VALUE_ENV = "FINALAYZE_SAA_EQUITY_POINT_VALUE"
_EQUITY_DRAWDOWN_SURVIVAL_PCT_ENV = "FINALAYZE_SAA_EQUITY_DRAWDOWN_SURVIVAL_PCT"
_EQUITY_IM_HIKE_MULT_ENV = "FINALAYZE_SAA_EQUITY_IM_HIKE_MULT"
_EQUITY_MARGIN_RATE_ENV = "FINALAYZE_SAA_EQUITY_MARGIN_RATE"
_OFZ_PK_ENV = "FINALAYZE_SAA_OFZ_PK_SYMBOL"


def _resolve(env_var: str, default: str, leg: str) -> str:
    """Return the configured ticker for *leg*, fail-closed on an empty value.

    Reads ``env_var`` (falling back to *default*), strips it, and raises
    ConfigurationError if the result is empty -- a blank ticker must never reach
    instrument resolution on a money path.
    """
    raw = os.environ.get(env_var, default)
    symbol = raw.strip() if raw else ""
    if not symbol:
        msg = f"SAA {leg} symbol is unset/empty; set {env_var} to a tradeable MOEX ticker"
        raise ConfigurationError(msg)
    return symbol


def get_equity_symbol() -> str:
    """Return the configured equity (IMOEXF index future) ticker, fail-closed."""
    return _resolve(_EQUITY_ENV, _DEFAULT_EQUITY_SYMBOL, "equity")


def get_equity_point_value() -> Decimal:
    """Return the RUB value of one index point for the equity FUTURE leg, fail-closed (Phase 85).

    Used to size the future by exposure (contract notional = quoted points * this value). Defaults
    to IMOEXF's 10 RUB/point; raises ConfigurationError on a non-numeric or non-positive override.
    """
    raw = os.environ.get(_EQUITY_POINT_VALUE_ENV, _DEFAULT_EQUITY_POINT_VALUE)
    try:
        point_value = Decimal(str(raw).strip())
    except InvalidOperation as exc:
        msg = f"{_EQUITY_POINT_VALUE_ENV} must be a positive number; got {raw!r}"
        raise ConfigurationError(msg) from exc
    # Reject NaN/Infinity too: Decimal('inf') > 0 is True and would slip past a bare positivity
    # check, then blow up far away as 0*inf -> InvalidOperation in sizing (WR-01).
    if not point_value.is_finite() or point_value <= 0:
        msg = f"{_EQUITY_POINT_VALUE_ENV} must be a positive finite number; got {point_value}"
        raise ConfigurationError(msg)
    return point_value


def get_ofz_pk_symbol() -> str:
    """Return the configured OFZ-PK (federal floating-coupon bond) ticker, fail-closed."""
    return _resolve(_OFZ_PK_ENV, _DEFAULT_OFZ_PK_SYMBOL, "ofz_pk")


def _resolve_decimal(env_var: str, default: Decimal | None) -> Decimal | None:
    """Parse an env var to a finite Decimal (or *default* when unset), fail-closed on garbage.

    Returns ``default`` (possibly ``None``) when the var is unset/blank; otherwise the parsed
    finite Decimal. A non-numeric or non-finite value raises ConfigurationError -- a malformed
    risk parameter must never silently fall back to a (possibly unsafe) guess on a money path.
    """
    raw = os.environ.get(env_var)
    if raw is None or not raw.strip():
        return default
    try:
        value = Decimal(raw.strip())
    except InvalidOperation as exc:
        msg = f"{env_var} must be a finite number; got {raw!r}"
        raise ConfigurationError(msg) from exc
    if not value.is_finite():
        msg = f"{env_var} must be a finite number; got {value}"
        raise ConfigurationError(msg)
    return value


def get_equity_drawdown_survival_pct() -> Decimal:
    """Return the equity-future drawdown-survival fraction, fail-closed (Phase 86).

    The CASH reserve held against the equity FUTURE is sized so the position survives an index
    drawdown of at least this fraction (default 0.45) even after an initial-margin hike. Must be a
    finite fraction in ``(0, 1]``; anything else raises ConfigurationError.
    """
    value = _resolve_decimal(
        _EQUITY_DRAWDOWN_SURVIVAL_PCT_ENV, SAA_EQUITY_DRAWDOWN_SURVIVAL_PCT_DEFAULT
    )
    assert value is not None  # default is non-None
    if not (Decimal(0) < value <= Decimal(1)):
        msg = f"{_EQUITY_DRAWDOWN_SURVIVAL_PCT_ENV} must be a fraction in (0, 1]; got {value}"
        raise ConfigurationError(msg)
    return value


def get_equity_im_hike_mult() -> Decimal:
    """Return the initial-margin (IM) hike multiplier the reserve must withstand, fail-closed.

    MOEX can raise the IM mid-crash (~2.5x overnight in Feb-2022); the reserve carries headroom so
    the position is not force-liquidated when the IM is hiked by up to this multiple. Must be a
    finite value ``>= 1`` (1.0 = no hike headroom); anything else raises ConfigurationError.
    """
    value = _resolve_decimal(_EQUITY_IM_HIKE_MULT_ENV, SAA_EQUITY_IM_HIKE_MULT_DEFAULT)
    assert value is not None  # default is non-None
    if value < Decimal(1):
        msg = f"{_EQUITY_IM_HIKE_MULT_ENV} must be >= 1 (1.0 = no IM-hike headroom); got {value}"
        raise ConfigurationError(msg)
    return value


def get_equity_margin_rate() -> Decimal | None:
    """Return the OPTIONAL static equity-future margin rate, or ``None`` when unset (Phase 86).

    This is a fraction of the contract notional used ONLY for offline/weekend planning when a live
    broker margin fetch is not possible. It is NEVER an automatic fallback on a live fetch failure
    (a too-low guess would silently under-reserve). Returns ``None`` when the operator has not
    explicitly set it; when set, must be a finite fraction in ``(0, 1]``, else ConfigurationError.
    """
    value = _resolve_decimal(_EQUITY_MARGIN_RATE_ENV, None)
    if value is None:
        return None
    if not (Decimal(0) < value <= Decimal(1)):
        msg = f"{_EQUITY_MARGIN_RATE_ENV} must be a fraction in (0, 1]; got {value}"
        raise ConfigurationError(msg)
    return value
