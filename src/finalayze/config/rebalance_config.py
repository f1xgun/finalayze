"""SAA rebalance config -- no-churn band + operator-overridable leg tickers (Phase 79 P79-02).

The weights-to-orders engine needs concrete TRADEABLE MOEX tickers for the equity and OFZ-PK
legs. These are operator-overridable via env (no code change) and default to the verified
committed-snapshot instruments:

- equity   -> ``EQMX`` (VIM MOEX-Index ETF, figi TCS00A101EJ5) -- the tradeable MOEX-index
  ETF that is the real-instrument proxy for the MCFTR analytics leg. ``SBMX`` is a fallback.
- ofz_pk   -> ``SU29024RMFS5`` (ОФЗ 29024, mat 2035) -- a liquid federal floating-coupon issue
  from the 21-strong ``SU29*`` universe.

Resolution is FAIL-CLOSED (D-04 / saa_portfolio_writer pattern): a blank/whitespace ticker
raises ``ConfigurationError`` rather than silently resolving to an empty symbol on a money path.
"""

from __future__ import annotations

import os
from decimal import Decimal

from finalayze.core.exceptions import ConfigurationError

# No-churn / dust band: a leg whose |delta| is under this fraction of the budget does not trade.
SAA_REBALANCE_BAND_PCT = Decimal("0.02")

# Verified-snapshot defaults (overridable via env).
_DEFAULT_EQUITY_SYMBOL = "EQMX"
_DEFAULT_OFZ_PK_SYMBOL = "SU29024RMFS5"

_EQUITY_ENV = "FINALAYZE_SAA_EQUITY_SYMBOL"
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
    """Return the configured equity (MCFTR-tracking ETF) ticker, fail-closed."""
    return _resolve(_EQUITY_ENV, _DEFAULT_EQUITY_SYMBOL, "equity")


def get_ofz_pk_symbol() -> str:
    """Return the configured OFZ-PK (federal floating-coupon bond) ticker, fail-closed."""
    return _resolve(_OFZ_PK_ENV, _DEFAULT_OFZ_PK_SYMBOL, "ofz_pk")
