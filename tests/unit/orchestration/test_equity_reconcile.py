"""S3.2 — Stock-side reconcile + equity-drift checks.

Contract:
  STREC-01: broker_only — broker holds symbol that tracker doesn't → alert.
  STREC-02: tracker_only — tracker thinks holds symbol that broker doesn't
            → alert; with apply=True, register_exit fires.
  STREC-03: matched — perfect alignment yields empty alerts.
  STREC-04: MOEX FIGI-keyed broker positions normalise to symbol-keyed via
            registry; unknown FIGIs are dropped.
  STREC-05: compute_mtm_equity is pure: cash + Σ qty * last_price.
  STREC-06: equity drift over tolerance fires an alert + returns within=False.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import MagicMock

from finalayze.core.schemas import PortfolioState
from finalayze.execution.simulated_broker import StopLossState
from finalayze.orchestration.equity_reconcile import (
    StockReconcileReport,
    compare_equity,
    compute_mtm_equity,
    reconcile_equity_drift,
    reconcile_stocks,
)
from finalayze.orchestration.position_manager import PositionTracker

_QTY_SBER = Decimal(10)
_QTY_GAZP = Decimal(20)
_PRICE_SBER = Decimal(300)
_PRICE_GAZP = Decimal(150)
_CASH = Decimal(5000)
_TOL = Decimal("0.005")
_HIGH_DRIFT_PCT = Decimal("0.02")


def _make_tracker_with_entries(*symbols: str) -> PositionTracker:
    """Build a tracker that thinks it holds each symbol at price 100."""
    tracker = PositionTracker(
        kelly_sizer=MagicMock(),
        broker_router=MagicMock(),
        alerter=MagicMock(),
    )
    state = StopLossState(
        initial_stop=Decimal(95),
        current_stop=Decimal(95),
        highest_price=Decimal(100),
        trail_activated=False,
        activation_atr=Decimal("1.0"),
        trail_atr=Decimal("1.5"),
        entry_price=Decimal(100),
        atr_value=Decimal("2.5"),
    )
    for sym in symbols:
        tracker._entry_prices[sym] = Decimal(100)
        tracker._entry_strategy[sym] = "momentum"
        tracker._stop_states[sym] = state
    return tracker


def _make_broker(positions: dict[str, Decimal]) -> MagicMock:
    broker = MagicMock()
    broker.get_positions.return_value = positions
    return broker


# ─── STREC-01 ────────────────────────────────────────────────────────────────
def test_broker_only_position_flagged() -> None:
    tracker = _make_tracker_with_entries()  # tracker holds nothing
    broker = _make_broker({"SBER": _QTY_SBER})

    report = reconcile_stocks(broker, tracker, market_id="us")

    assert report.broker_only == {"SBER": _QTY_SBER}
    assert report.tracker_only == []
    assert any("SBER" in a and "tracker has no entry" in a for a in report.alerts)
    assert report.has_drift


# ─── STREC-02 ────────────────────────────────────────────────────────────────
def test_tracker_only_position_flagged_and_cleared_on_apply() -> None:
    tracker = _make_tracker_with_entries("SBER")
    broker = _make_broker({})  # broker reports empty

    report = reconcile_stocks(broker, tracker, market_id="us", apply=True)

    assert report.tracker_only == ["SBER"]
    assert "SBER" not in tracker._entry_prices  # cleared
    assert "SBER" not in tracker._stop_states  # cleared via register_exit


def test_tracker_only_does_not_mutate_without_apply() -> None:
    tracker = _make_tracker_with_entries("SBER")
    broker = _make_broker({})

    report = reconcile_stocks(broker, tracker, market_id="us")  # apply=False

    assert report.tracker_only == ["SBER"]
    assert "SBER" in tracker._entry_prices  # still there


# ─── STREC-03 ────────────────────────────────────────────────────────────────
def test_matched_positions_produce_no_alerts() -> None:
    tracker = _make_tracker_with_entries("SBER", "GAZP")
    broker = _make_broker({"SBER": _QTY_SBER, "GAZP": _QTY_GAZP})

    report = reconcile_stocks(broker, tracker, market_id="moex")

    assert report.alerts == []
    assert set(report.matched) == {"SBER", "GAZP"}
    assert not report.has_drift


# ─── STREC-04 ────────────────────────────────────────────────────────────────
def test_moex_figi_keyed_positions_normalised_via_registry() -> None:
    """MOEX positions arrive FIGI-keyed; reconcile converts to symbol-keyed."""
    tracker = _make_tracker_with_entries("SBER")
    broker = _make_broker({"BBG004730N88": _QTY_SBER})  # SBER's FIGI

    registry = MagicMock()
    sber_inst = MagicMock(symbol="SBER")
    registry.get_by_figi.side_effect = lambda figi: sber_inst if figi == "BBG004730N88" else None

    report = reconcile_stocks(broker, tracker, market_id="moex", registry=registry)

    assert report.matched == ["SBER"]
    assert report.broker_only == {}


def test_moex_unknown_figi_dropped_silently() -> None:
    """A FIGI that isn't in our registry shouldn't trigger a false alert."""
    tracker = _make_tracker_with_entries()
    broker = _make_broker({"BBG_UNKNOWN_FIGI": Decimal(5)})
    registry = MagicMock()
    registry.get_by_figi.return_value = None

    report = reconcile_stocks(broker, tracker, market_id="moex", registry=registry)

    assert report.broker_only == {}  # dropped, not flagged
    assert report.alerts == []


def test_zero_qty_broker_positions_ignored() -> None:
    """Positions with qty=0 (e.g. just closed) shouldn't appear as drift."""
    tracker = _make_tracker_with_entries()
    broker = _make_broker({"SBER": Decimal(0)})

    report = reconcile_stocks(broker, tracker, market_id="us")

    assert report.broker_only == {}
    assert report.matched == []


# ─── STREC-05 ────────────────────────────────────────────────────────────────
def test_compute_mtm_equity_cash_plus_positions() -> None:
    cash = _CASH
    positions = {"SBER": _QTY_SBER, "GAZP": _QTY_GAZP}
    last_prices = {"SBER": _PRICE_SBER, "GAZP": _PRICE_GAZP}

    mtm = compute_mtm_equity(cash, positions, last_prices)

    expected = _CASH + _QTY_SBER * _PRICE_SBER + _QTY_GAZP * _PRICE_GAZP
    assert mtm == expected


def test_compute_mtm_equity_missing_price_contributes_zero() -> None:
    """A position whose price isn't in the cache contributes zero (and should
    trigger an upstream stale-price warning — not this function's job)."""
    cash = _CASH
    positions = {"SBER": _QTY_SBER, "GAZP": _QTY_GAZP}
    last_prices = {"SBER": _PRICE_SBER}  # GAZP missing

    mtm = compute_mtm_equity(cash, positions, last_prices)

    assert mtm == _CASH + _QTY_SBER * _PRICE_SBER  # GAZP * 0


# ─── STREC-06 ────────────────────────────────────────────────────────────────
def test_equity_drift_within_tolerance_returns_true() -> None:
    portfolio = PortfolioState(
        cash=_CASH,
        positions={"SBER": _QTY_SBER},
        equity=_CASH + _QTY_SBER * _PRICE_SBER,  # exact match
        timestamp=datetime(2025, 1, 1, tzinfo=UTC),
    )
    last_prices = {"SBER": _PRICE_SBER}

    pct, within = reconcile_equity_drift(portfolio, last_prices)

    assert within
    assert abs(pct) < _TOL


def test_equity_drift_over_tolerance_fires_alert() -> None:
    # broker says equity = mtm + 2% — we should detect drift
    broker_equity = _CASH + _QTY_SBER * _PRICE_SBER
    mtm_diff = broker_equity * _HIGH_DRIFT_PCT  # 2% extra
    portfolio = PortfolioState(
        cash=_CASH,
        positions={"SBER": _QTY_SBER},
        equity=broker_equity + mtm_diff,
        timestamp=datetime(2025, 1, 1, tzinfo=UTC),
    )
    last_prices = {"SBER": _PRICE_SBER}
    alerter = MagicMock()

    _pct, within = reconcile_equity_drift(portfolio, last_prices, alerter=alerter, market_id="moex")

    assert not within
    alerter.on_error.assert_called_once()
    args = alerter.on_error.call_args
    assert args.args[0] == "equity_drift"


def test_compare_equity_pure_math() -> None:
    """compare_equity is a pure (no I/O) gap calculator."""
    broker = Decimal(10000)
    mtm = Decimal(9950)  # 0.5% gap
    abs_gap, pct_gap, within = compare_equity(broker, mtm, tolerance_pct=_TOL)
    assert abs_gap == Decimal(-50)
    assert abs(pct_gap) <= _TOL
    assert within

    mtm_drifted = Decimal(10200)  # 2 % gap
    _, _, within2 = compare_equity(broker, mtm_drifted, tolerance_pct=_TOL)
    assert not within2


# ─── Smoke: report dataclass shape ───────────────────────────────────────────
def test_report_dataclass_default_empty() -> None:
    """An empty report is the 'all clean' case."""
    rpt = StockReconcileReport(market_id="us")
    assert not rpt.has_drift
    assert rpt.broker_only == {}
    assert rpt.tracker_only == []
    assert rpt.alerts == []
