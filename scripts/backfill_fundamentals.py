"""One-shot historical fundamental backfill driver (Layer 6, BACKFILL-H-01..04).

Operator-run script that composes the SmartLab per-quarter parser (Plan 63.1-02,
``SmartlabFundamentalsFetcher``) and the extended MOEX ISS fetcher (Plan 63.1-03,
``MoexISSFetcher``) into look-ahead-safe :class:`FundamentalSnapshot` rows and
writes each through the EXISTING Phase-63 idempotent ``(as_of, symbol)`` upsert
(:meth:`TradingPersistence.persist_fundamental_snapshot`). Blue-chips first; the
growth-tech short-history names are captured-what-exists and FLAGGED, never
aborting the run.

This is a ONE-SHOT script (D-04): it has NO scheduler/cron registration. The
LIVE run against the real site + a live TimescaleDB is an OPERATOR manual step
(network + robots gate + DB up). The unit tests mock both fetchers and the
persistence layer, so no live network/DB is touched under test.

────────────────────────────────────────────────────────────────────────────
Task-0 research confirms (resolved BEFORE the merge logic below):

UNIT SCALE (Open Q1 / Assumption A1) — RESOLVED:
    Phase-63 live capture (``tinkoff_data._map_fundamentals``) stores
    ``revenue_ttm`` and ``market_cap`` as the **raw** scalar values returned by
    ``get_asset_fundamentals`` (``opt(stat.revenue_ttm)`` /
    ``opt(stat.market_capitalization)``) — i.e. raw RUB, with NO additional
    scaling. The SmartLab parser (Plan 02, ``SmartlabFundamentalsFetcher._billions``)
    ALREADY multiplies its "mlrd rub" cells by ``1e9`` to land in raw RUB at parse
    time. Therefore the backfill driver must NOT scale a second time:
    ``_REVENUE_MARKETCAP_SCALE = 1`` (apply it as the identity so a future unit
    divergence has a single, documented knob). If a downstream consumer is ever
    found to expect billions, flip this constant to ``Decimal("1e-9")`` and update
    this note — DO NOT silently mismatch the live-capture unit.

HISTORY DEPTH (Open Q2) — RESOLVED (honest note):
    SmartLab's default quarterly view ``/q/{T}/f/q/MSFO/`` exposes only the most
    recent window (probes showed ~5 recent quarters); it does NOT accept a
    page/offset parameter that reliably deepens the quarterly history. Multi-year
    depth therefore requires the ANNUAL view ``/q/{T}/f/y/MSFO/`` as a supplement
    (operator-run, network-bound). This offline/quarterly driver captures only the
    realistically-reachable recent quarters per blue chip — it does NOT claim deep
    multi-year history. Deep history is an operator follow-up via the annual page.

DIVIDEND YIELD (Open Q3 / Assumption A4) — RESOLVED:
    SmartLab's ``div_yield`` is 0.0%/absent on some recent quarters (stale "not yet
    declared"). The driver PREFERS an ISS-derived yield (Σ dividends in the trailing
    12 months ÷ a contemporaneous close price) over a stale SmartLab 0.0%/None. The
    SmartLab value is kept only when the ISS-derived value is unavailable.
────────────────────────────────────────────────────────────────────────────

Usage (operator, with network + DB):
    uv run python scripts/backfill_fundamentals.py
    uv run python scripts/backfill_fundamentals.py --symbols SBER LKOH --dry-run
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

# Ensure project root is importable (config/ lives at the repo root — MEMORY convention).
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import structlog

from finalayze.core.exceptions import DataFetchError
from finalayze.core.schemas import FundamentalSnapshot

if TYPE_CHECKING:
    from datetime import date

    from finalayze.orchestration.db_persistence import TradingPersistence

_log = structlog.get_logger()

# ── Universe (D-03) ──────────────────────────────────────────────────────────

# Liquid blue chips with the deepest history — backfilled FIRST.
BLUE_CHIPS: tuple[str, ...] = (
    "SBER",
    "LKOH",
    "GMKN",
    "ROSN",
    "NVTK",
    "MGNT",
    "TATN",
    "SNGS",
    "SIBN",
    "VTBR",
    "MOEX",
)

# Growth-tech names with sparse/short fundamental history (D-03): captured
# whatever-exists and FLAGGED, never blocking the run.
SHORT_HISTORY_SYMBOLS: frozenset[str] = frozenset({"OZON", "VKCO", "CIAN", "YDEX"})

# ── Unit scale (Task-0, Open Q1 / A1) ────────────────────────────────────────
# SmartLab's parser already scales "mlrd rub" -> raw RUB (x1e9). Phase-63 live
# capture stores raw RUB unscaled. So the driver applies the IDENTITY here — a
# single documented knob to flip if a downstream unit divergence is ever found.
_REVENUE_MARKETCAP_SCALE: Decimal = Decimal(1)

# Trailing window for the ISS-derived dividend yield (Σ 12-month dividends ÷ price).
_TTM_DAYS = 365
# Half-window around as_of used to locate a contemporaneous close price for the yield.
_PRICE_LOOKBACK_DAYS = 30

# A SmartLab dividend_yield of exactly 0.0 (or None) is treated as stale/absent
# and superseded by the ISS-derived yield when available (A4).
_STALE_YIELD = 0.0


class _SmartlabLike(Protocol):
    """The subset of ``SmartlabFundamentalsFetcher`` the driver composes."""

    def assert_robots_allowed(self, path: str) -> None: ...

    def fetch_html(self, symbol: str, statement: str = ...) -> str: ...

    def parse_html(self, content: str, symbol: str) -> list[FundamentalSnapshot]: ...


class _IssLike(Protocol):
    """The subset of ``MoexISSFetcher`` the driver composes."""

    def fetch_dividends(self, secid: str) -> list[tuple[date, Decimal, str]]: ...

    def fetch_issuesize(self, secid: str) -> int | None: ...

    def fetch_close_history(
        self, secid: str, start: datetime, end: datetime
    ) -> list[tuple[date, Decimal]]: ...

    def reconstruct_market_cap(self, close: Decimal, issuesize: int | None) -> Decimal | None: ...


# ── Snapshot composition ─────────────────────────────────────────────────────


def _scaled(value: float | None) -> float | None:
    """Apply ``_REVENUE_MARKETCAP_SCALE`` to a revenue/market_cap value (identity)."""
    if value is None:
        return None
    return float(Decimal(str(value)) * _REVENUE_MARKETCAP_SCALE)


def _ttm_dividend_sum(dividends: list[tuple[date, Decimal, str]], as_of: date) -> Decimal | None:
    """Σ dividends with a record date in the trailing 12 months ending at *as_of*.

    Returns ``None`` when no dividend falls in the window (so the SmartLab value
    is preserved rather than fabricating a zero).
    """
    window_start = as_of - timedelta(days=_TTM_DAYS)
    total = Decimal(0)
    found = False
    for record_date, value, _currency in dividends:
        if window_start < record_date <= as_of:
            total += value
            found = True
    return total if found else None


def _price_near(closes: list[tuple[date, Decimal]], as_of: date) -> Decimal | None:
    """Pick the close whose trade date is nearest to (and not after) *as_of*.

    Falls back to the chronologically nearest available close when none precede
    *as_of*. Returns ``None`` for an empty history.
    """
    if not closes:
        return None
    on_or_before = [(d, c) for d, c in closes if d <= as_of]
    if on_or_before:
        return max(on_or_before, key=lambda dc: dc[0])[1]
    return min(closes, key=lambda dc: abs((dc[0] - as_of).days))[1]


def _iss_derived_yield(secid: str, as_of: datetime, iss: _IssLike) -> float | None:
    """Σ trailing-12m ISS dividends ÷ a contemporaneous close (A4 / Open Q3).

    Best-effort: any missing piece (no dividends in window, no price) yields
    ``None`` so the SmartLab value is preserved rather than overwritten.
    """
    as_of_date = as_of.date()
    div_sum = _ttm_dividend_sum(iss.fetch_dividends(secid), as_of_date)
    if div_sum is None or div_sum <= 0:
        return None
    start = as_of - timedelta(days=_PRICE_LOOKBACK_DAYS)
    end = as_of + timedelta(days=1)
    price = _price_near(iss.fetch_close_history(secid, start, end), as_of_date)
    if price is None or price <= 0:
        return None
    return float(div_sum / price)


def _gap_filled_market_cap(
    secid: str, as_of: datetime, smartlab_value: float | None, iss: _IssLike
) -> float | None:
    """SmartLab per-quarter market_cap primary; ISS reconstruction as flagged gap-fill.

    The ISS reconstruction (CLOSE x current ISSUESIZE) is APPROXIMATE (Pitfall 5)
    and is used ONLY when SmartLab has no value AND an issuesize exists; a None
    issuesize (e.g. CIAN — Pitfall 6) leaves market_cap None (never fabricated).
    """
    if smartlab_value is not None:
        return _scaled(smartlab_value)

    issuesize = iss.fetch_issuesize(secid)
    if issuesize is None:
        return None
    start = as_of - timedelta(days=_PRICE_LOOKBACK_DAYS)
    end = as_of + timedelta(days=1)
    price = _price_near(iss.fetch_close_history(secid, start, end), as_of.date())
    if price is None:
        return None
    approx = iss.reconstruct_market_cap(price, issuesize)
    if approx is None:
        return None
    _log.info("backfill_market_cap_reconstructed", symbol=secid, as_of=as_of.isoformat())
    return float(approx * _REVENUE_MARKETCAP_SCALE)


def build_snapshots(
    symbol: str,
    smartlab_fetcher: _SmartlabLike,
    iss_fetcher: _IssLike,
    statement: str = "MSFO",
) -> list[FundamentalSnapshot]:
    """Compose SmartLab + ISS into look-ahead-safe snapshots for *symbol*.

    SmartLab provides the base per-quarter snapshots (as_of already the disclosure
    date / +75d lag — never the fiscal-quarter end). For each snapshot:
      * dividend_yield: prefer the ISS-derived trailing-12m yield over a stale
        SmartLab 0.0%/None (A4 / Open Q3);
      * market_cap: SmartLab per-quarter primary, ISS reconstruction as a flagged
        approximate gap-fill (Pitfall 5/6);
      * revenue_ttm / market_cap: ``_REVENUE_MARKETCAP_SCALE`` applied (identity —
        the SmartLab parser already scaled "mlrd rub" -> raw RUB).
    """
    content = smartlab_fetcher.fetch_html(symbol, statement)
    base = smartlab_fetcher.parse_html(content, symbol)

    rebuilt: list[FundamentalSnapshot] = []
    for snap in base:
        dividend_yield = snap.dividend_yield
        if dividend_yield is None or dividend_yield == _STALE_YIELD:
            derived = _iss_derived_yield(symbol, snap.as_of, iss_fetcher)
            if derived is not None:
                dividend_yield = derived

        market_cap = _gap_filled_market_cap(symbol, snap.as_of, snap.market_cap, iss_fetcher)

        rebuilt.append(
            snap.model_copy(
                update={
                    "revenue_ttm": _scaled(snap.revenue_ttm),
                    "market_cap": market_cap,
                    "dividend_yield": dividend_yield,
                }
            )
        )
    return rebuilt


# ── Driver ───────────────────────────────────────────────────────────────────


def run_backfill(
    persistence: TradingPersistence,
    smartlab_fetcher: _SmartlabLike,
    iss_fetcher: _IssLike,
    symbols: tuple[str, ...] = BLUE_CHIPS,
    statement: str = "MSFO",
    *,
    dry_run: bool = False,
) -> int:
    """Backfill *symbols* (blue-chips first) through the existing Phase-63 upsert.

    Calls ``assert_robots_allowed`` ONCE up front (H-04 hard gate). Each symbol is
    wrapped in try/except so one failure never aborts the run (T-63.1-12): a
    short-history symbol logs ``backfill_short_history_skip``; a blue chip logs
    ``backfill_symbol_failed``. Persistence uses the existing idempotent
    ``persist_fundamental_snapshot`` upsert — it is NOT rebuilt here.

    Returns the number of snapshots persisted (0 under ``dry_run``).
    """
    # H-04: single robots gate before any pull (T-63.1-10).
    smartlab_fetcher.assert_robots_allowed(f"/q/SBER/f/q/{statement}/")

    persisted = 0
    for symbol in symbols:
        try:
            snapshots = build_snapshots(symbol, smartlab_fetcher, iss_fetcher, statement)
        except DataFetchError as exc:
            if symbol in SHORT_HISTORY_SYMBOLS:
                _log.warning("backfill_short_history_skip", symbol=symbol, error=str(exc))
            else:
                _log.warning("backfill_symbol_failed", symbol=symbol, error=str(exc))
            continue

        if symbol in SHORT_HISTORY_SYMBOLS and not snapshots:
            _log.warning("backfill_short_history_skip", symbol=symbol, reason="no_snapshots")

        for snap in snapshots:
            if dry_run:
                _log.info("backfill_dry_run", symbol=snap.symbol, as_of=snap.as_of.isoformat())
                continue
            persistence.persist_fundamental_snapshot(snap)
            persisted += 1

        _log.info("backfill_symbol_done", symbol=symbol, snapshots=len(snapshots))

    _log.info("backfill_complete", symbols=len(symbols), persisted=persisted, dry_run=dry_run)
    return persisted


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="One-shot MOEX fundamental backfill (D-03/D-04).")
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=list(BLUE_CHIPS),
        help="Symbols to backfill (default: blue chips first).",
    )
    parser.add_argument("--statement", default="MSFO", help="Statement type (MSFO|RSBU).")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build snapshots and log them without persisting.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Operator entry point: wire real fetchers + ``TradingPersistence`` and run.

    Heavy/optional imports are deferred so the test suite (which imports
    ``build_snapshots``/``SHORT_HISTORY_SYMBOLS``) never pulls in the DB stack.
    """
    args = _parse_args(argv)

    from config.settings import get_settings  # noqa: PLC0415
    from dotenv import load_dotenv  # noqa: PLC0415

    from finalayze.data.fetchers.moex_iss import MoexISSFetcher  # noqa: PLC0415
    from finalayze.data.fetchers.smartlab_fundamentals import (  # noqa: PLC0415
        SmartlabFundamentalsFetcher,
    )
    from finalayze.orchestration.db_persistence import TradingPersistence  # noqa: PLC0415

    load_dotenv()
    settings = get_settings()
    db_url = getattr(settings, "database_url", None)

    persistence = TradingPersistence(db_url=db_url, async_loop=None, settings=settings)

    with SmartlabFundamentalsFetcher() as smartlab, MoexISSFetcher() as iss:
        persisted = run_backfill(
            persistence,
            smartlab,
            iss,
            symbols=tuple(args.symbols),
            statement=args.statement,
            dry_run=args.dry_run,
        )

    _log.info("backfill_main_done", persisted=persisted)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
