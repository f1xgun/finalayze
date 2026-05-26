"""S3.2 — Stock-side position/equity reconciliation with the broker.

The bond ledger already has ``reconcile_with_broker`` in
``core.layer_ledger``. Stocks had no equivalent: after a container restart,
a partial fill, a manual broker action, or any external SELL we silently
drift from reality. ``maybe_register_retroactive_stop`` covers HALF the
problem (broker has a position we don't know about → re-attach a stop) but
leaves the other half open (we *think* we hold a position the broker no
longer reports). This module closes that gap and adds a small equity-drift
helper for daily-report sanity checks.

The reconciliation is intentionally side-effect-light:
  * Returns a structured ``StockReconcileReport`` (verdict + alert strings).
  * Only mutates ``PositionTracker`` when ``apply=True`` — the caller
    (typically the daily reporting service) decides whether to act.
  * Equity drift uses a pure helper (``compute_mtm_equity``) so backtest
    and live share the same formula.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from finalayze.api.alerts import TelegramAlerter
    from finalayze.core.schemas import PortfolioState
    from finalayze.execution.broker_base import BrokerBase
    from finalayze.markets.instruments import InstrumentRegistry
    from finalayze.orchestration.position_manager import PositionTracker

_log = structlog.get_logger(__name__)
_ZERO = Decimal(0)

# Default tolerance for equity drift between broker-reported total and our
# locally-computed mark-to-market: 0.5 % covers Tinkoff's rounding,
# bond-coupon accrual, and FX-mark differences without false-positives.
_DEFAULT_EQUITY_TOL_PCT = Decimal("0.005")


@dataclass(frozen=True)
class StockReconcileReport:
    """Outcome of a single stock-position reconciliation pass.

    All four collections are symbol-keyed (broker positions are normalised
    via the registry before comparison). Empty collections everywhere means
    perfect alignment — typically the daily report's "OK" case.
    """

    market_id: str
    broker_only: dict[str, Decimal] = field(default_factory=dict)
    tracker_only: list[str] = field(default_factory=list)
    matched: list[str] = field(default_factory=list)
    alerts: list[str] = field(default_factory=list)

    @property
    def has_drift(self) -> bool:
        return bool(self.broker_only) or bool(self.tracker_only)


def compute_mtm_equity(
    cash: Decimal,
    positions: dict[str, Decimal],
    last_prices: dict[str, Decimal],
) -> Decimal:
    """Mark-to-market equity formula shared by backtest + live reconcile.

    Equity = cash + Σ qty * last_price. Positions whose symbol is missing
    from ``last_prices`` contribute zero (callers should log this — we
    don't silently treat unknown prices as material loss).
    """
    pos_value = sum(
        (qty * last_prices.get(sym, _ZERO) for sym, qty in positions.items()),
        start=_ZERO,
    )
    return cash + pos_value


def compare_equity(
    broker_equity: Decimal,
    mtm_equity: Decimal,
    *,
    tolerance_pct: Decimal = _DEFAULT_EQUITY_TOL_PCT,
) -> tuple[Decimal, Decimal, bool]:
    """Return (abs_gap, pct_gap, within_tolerance).

    ``mtm_equity`` is the equity computed via ``compute_mtm_equity`` from
    our locally-tracked cash + positions + last prices.  ``broker_equity``
    is what the live broker reports (e.g. Tinkoff ``total_amount_portfolio``).
    Tolerance is expressed as a fraction of broker_equity to be invariant
    to portfolio size.

    A negative gap (mtm < broker) is the common case — broker accrual /
    bond coupon / FX revaluation that we don't replicate locally. A
    positive gap (mtm > broker) is the suspicious case (we think we hold
    more than we do); audit when seen.
    """
    abs_gap = mtm_equity - broker_equity
    base = abs(broker_equity)
    pct_gap = (abs_gap / base) if base > _ZERO else _ZERO
    return abs_gap, pct_gap, abs(pct_gap) <= tolerance_pct


def _normalise_broker_positions(
    broker_positions: dict[str, Decimal],
    market_id: str,
    registry: InstrumentRegistry | None,
) -> dict[str, Decimal]:
    """Convert FIGI-keyed MOEX positions to symbol-keyed.

    Tinkoff returns positions keyed by FIGI; backtest + Alpaca by symbol.
    Reconcile compares symbol-keyed sets so we need a single normalisation
    point. Unknown FIGIs (instrument not in our registry) are dropped with
    a debug log — those are universe items we don't trade and shouldn't
    raise false alerts.
    """
    if market_id != "moex" or registry is None:
        return dict(broker_positions)

    symbol_keyed: dict[str, Decimal] = {}
    for figi, qty in broker_positions.items():
        if qty <= _ZERO:
            continue
        symbol = _figi_to_symbol(registry, figi)
        if symbol is None:
            _log.debug("reconcile_unknown_figi", figi=figi, qty=float(qty))
            continue
        symbol_keyed[symbol] = qty
    return symbol_keyed


def _figi_to_symbol(registry: InstrumentRegistry, figi: str) -> str | None:
    """Best-effort FIGI -> symbol lookup against the registry.

    Returns None when the FIGI is not in any segment we know about — that
    is the harmless "broker holds an instrument outside our trading
    universe" case (e.g. money-market funds, currencies).
    """
    lookup = getattr(registry, "get_by_figi", None)
    if callable(lookup):
        instrument = lookup(figi)
        return getattr(instrument, "symbol", None)
    return None


def reconcile_stocks(
    broker: BrokerBase,
    tracker: PositionTracker,
    *,
    market_id: str,
    registry: InstrumentRegistry | None = None,
    alerter: TelegramAlerter | None = None,
    apply: bool = False,
) -> StockReconcileReport:
    """Reconcile broker-reported stock positions against PositionTracker.

    Compares two views of "what stock positions are open right now":

    * ``broker.get_positions()`` — authoritative source (Tinkoff /
      Alpaca / sandbox / simulator).  FIGI-keyed for Tinkoff, symbol-keyed
      for the others; normalised to symbol-keyed via ``registry``.

    * ``tracker._entry_prices`` — every symbol the live process believes
      it opened. Drives stop-loss state, Kelly P&L, and PresetApplicator
      ownership; drift here means stops fire on phantom positions or
      orphaned broker holdings carry no stop at all.

    Three outcome buckets in the returned report:

    * ``broker_only`` — broker has a position we don't track. Caller
      should normally trigger ``maybe_register_retroactive_stop`` — this
      function does NOT call it (single-responsibility; the wiring lives
      in the daily-report path so we can also fetch candles there).

    * ``tracker_only`` — we think we hold X but broker doesn't.  Stale
      after manual SELL, external stop fill, or post-restart corruption.
      When ``apply=True``, calls ``tracker.register_exit(symbol)`` to
      clear stop + entry tracking (no Kelly update — we don't know the
      exit price).

    * ``matched`` — symbol present in both. Reported for visibility, no
      action required. We cannot compare *quantity* because the tracker
      doesn't store it (qty is the broker's domain).

    The bond ``reconcile_with_broker`` trusts the broker for quantity
    mismatches; we apply the same principle here — the broker is the
    authoritative source of truth.
    """
    alerts: list[str] = []
    raw_positions = broker.get_positions()
    broker_positions = _normalise_broker_positions(raw_positions, market_id, registry)
    tracked = set(tracker._entry_prices.keys())

    broker_only: dict[str, Decimal] = {}
    matched: list[str] = []

    for symbol, qty in broker_positions.items():
        if qty <= _ZERO:
            continue
        if symbol in tracked:
            matched.append(symbol)
        else:
            broker_only[symbol] = qty
            alerts.append(
                f"Reconciliation [{market_id}]: broker holds {symbol} qty={qty} "
                f"but tracker has no entry. Wire a retroactive stop."
            )

    tracker_only: list[str] = []
    for symbol in tracked:
        if symbol not in broker_positions:
            tracker_only.append(symbol)
            alerts.append(
                f"Reconciliation [{market_id}]: tracker carries {symbol} but "
                f"broker reports no position. Clearing local state."
            )
            if apply:
                # register_exit clears stop_states + entry_strategy, but leaves
                # the _entry_prices entry behind (normally cleared by the
                # subsequent _update_kelly call on a real SELL fill). The
                # reconcile path has no fill price, so we drop the cached
                # entry price explicitly — otherwise the next BUY of the same
                # symbol would compute Kelly P&L against a phantom entry.
                tracker.register_exit(symbol)
                tracker._entry_prices.pop(symbol, None)

    if alerter is not None:
        for msg in alerts:
            alerter.on_error("equity_reconcile", msg)

    if alerts:
        _log.warning(
            "stock_reconcile_drift",
            market_id=market_id,
            broker_only=list(broker_only.keys()),
            tracker_only=tracker_only,
        )

    return StockReconcileReport(
        market_id=market_id,
        broker_only=broker_only,
        tracker_only=tracker_only,
        matched=matched,
        alerts=alerts,
    )


def reconcile_equity_drift(
    portfolio: PortfolioState,
    last_prices: dict[str, Decimal],
    *,
    tolerance_pct: Decimal = _DEFAULT_EQUITY_TOL_PCT,
    alerter: TelegramAlerter | None = None,
    market_id: str = "",
) -> tuple[Decimal, bool]:
    """Compare broker-reported equity against locally-computed MTM.

    Returns ``(pct_gap, within_tolerance)``.  Useful as a daily-report
    sanity check: if the gap exceeds tolerance, either the broker is
    accruing something we're not tracking (bond coupons, FX rev) or our
    last_prices cache is stale. Fires a Telegram alert when out of
    tolerance.
    """
    mtm = compute_mtm_equity(portfolio.cash, portfolio.positions, last_prices)
    abs_gap, pct_gap, within = compare_equity(portfolio.equity, mtm, tolerance_pct=tolerance_pct)
    if not within:
        msg = (
            f"Equity drift [{market_id or 'unknown'}]: broker={portfolio.equity} "
            f"mtm={mtm} gap={abs_gap} ({pct_gap * 100:.2f} %). "
            f"Investigate coupon accrual / stale prices / FX."
        )
        _log.warning(
            "equity_drift",
            market_id=market_id,
            broker_equity=float(portfolio.equity),
            mtm_equity=float(mtm),
            abs_gap=float(abs_gap),
            pct_gap=float(pct_gap),
        )
        if alerter is not None:
            alerter.on_error("equity_drift", msg)
    return pct_gap, within
