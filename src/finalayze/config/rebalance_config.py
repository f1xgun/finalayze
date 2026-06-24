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

_EQUITY_ENV = "FINALAYZE_SAA_EQUITY_SYMBOL"
_EQUITY_POINT_VALUE_ENV = "FINALAYZE_SAA_EQUITY_POINT_VALUE"
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
