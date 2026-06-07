"""RED scaffold: SAA-04 / R-1 passive MCFTR equity sleeve, OFFLINE (Phase 72 Wave-0).

Pins the L2 MCFTR total-return-index loader contract before it exists:
- ``load_mcftr_series`` returns an ORDERED ``(date, Decimal close)`` series of
  total-return index levels, loaded from the offline fixture (no live ISS call
  in unit tests -- R-1 verified the MCFTR secid is reachable via the same index
  endpoint as IMOEX, but tests must stay offline);
- the MCFTR sleeve must NOT run ``process_dividends`` -- the index already
  reinvests dividends gross, so accruing them again would double-count (R-1 / A1).

The offline fixture uses the R-1 VERIFIED probe rows (2022-01-03 CLOSE 7374.93,
2025-12-30 CLOSE 7321.15) monkeypatched into the fetcher, so no network is hit.

RED now: the Plan-04 MCFTR loader ``finalayze.data.loader.load_mcftr_series``
does not exist yet. (Loader path pinned here = ``finalayze.data.loader.load_mcftr_series``;
Plan 04 wires this exact path or a thin wrapper at it.)
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

from finalayze.data.loader import load_mcftr_series

# -- Constants (named -- no magic numbers, ruff PLR2004) ----------------------

# R-1 verified offline probe rows (TRADEDATE, CLOSE) for MCFTR.
_FIRST_DATE = date(2022, 1, 3)
_FIRST_CLOSE = Decimal("7374.93")
_LAST_DATE = date(2025, 12, 30)
_LAST_CLOSE = Decimal("7321.15")

_MCFTR_FIXTURE: tuple[tuple[date, Decimal], ...] = (
    (_FIRST_DATE, _FIRST_CLOSE),
    (_LAST_DATE, _LAST_CLOSE),
)

_MCFTR_SECID = "MCFTR"


def test_mcftr_series_loads_offline(monkeypatch) -> None:  # noqa: ANN001
    """The MCFTR loader returns an ordered Decimal series from the offline fixture (R-1)."""
    # Stub the underlying fetch so no live ISS endpoint is hit.
    monkeypatch.setattr(
        "finalayze.data.loader._fetch_mcftr_rows",
        lambda *_args, **_kwargs: list(_MCFTR_FIXTURE),
        raising=False,
    )
    series = load_mcftr_series(_MCFTR_SECID)
    assert series[0][0] == _FIRST_DATE
    # Total-return index levels are Decimal-typed (no float).
    for _when, close in series:
        assert isinstance(close, Decimal)


def test_no_dividend_double_accrual(monkeypatch) -> None:  # noqa: ANN001
    """The MCFTR sleeve never calls process_dividends -- the index reinvests gross (R-1).

    MCFTR already embeds dividend reinvestment, so accruing dividends again on the
    equity sleeve would double-count. A spy on ``process_dividends`` must record
    zero calls during MCFTR series construction.
    """
    calls: list[object] = []

    def _spy(*args: object, **kwargs: object) -> None:
        calls.append((args, kwargs))

    monkeypatch.setattr(
        "finalayze.data.loader._fetch_mcftr_rows",
        lambda *_args, **_kwargs: list(_MCFTR_FIXTURE),
        raising=False,
    )
    monkeypatch.setattr(
        "finalayze.data.loader.process_dividends",
        _spy,
        raising=False,
    )
    load_mcftr_series(_MCFTR_SECID)
    assert calls == []  # the MCFTR leg reinvests dividends gross -- no re-accrual
