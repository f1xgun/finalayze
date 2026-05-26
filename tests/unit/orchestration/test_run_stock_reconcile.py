"""S4.1 — TradingLoop._run_stock_reconcile wires S3.2 into the daily hook.

Contract:
  RUNREC-01: Per-market loop iterates ``_circuit_breakers`` keys and calls
             both ``reconcile_stocks`` and ``reconcile_equity_drift``.
  RUNREC-02: A broker fetch failure for one market does not prevent
             reconcile from running on the other markets.
  RUNREC-03: A reconcile_stocks exception for one market doesn't suppress
             the equity-drift check on the same market.
  RUNREC-04: _daily_reset calls _run_stock_reconcile after updating
             baselines (so reconcile sees post-reset state).
"""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock, patch

from finalayze.orchestration.trading_loop import TradingLoop


def _make_loop_skeleton() -> TradingLoop:
    """Build a minimum TradingLoop instance for testing isolated methods.

    Uses __new__ to bypass the heavy __init__ wiring — we only set the
    attributes _run_stock_reconcile touches.
    """
    loop = TradingLoop.__new__(TradingLoop)
    loop._broker_router = MagicMock()
    loop._position_tracker = MagicMock()
    loop._registry = MagicMock()
    loop._alerter = MagicMock()
    loop._circuit_breakers = {"us": MagicMock(), "moex": MagicMock()}
    loop._signal_executor = MagicMock(_last_prices={"SBER": Decimal(300)})
    return loop


def _mock_portfolio(equity: Decimal = Decimal(100_000)) -> MagicMock:
    pf = MagicMock()
    pf.cash = Decimal(50_000)
    pf.positions = {}
    pf.equity = equity
    return pf


# ─── RUNREC-01 ───────────────────────────────────────────────────────────────
def test_run_stock_reconcile_iterates_all_markets() -> None:
    loop = _make_loop_skeleton()
    broker = MagicMock()
    broker.get_portfolio.return_value = _mock_portfolio()
    broker.get_positions.return_value = {}
    loop._broker_router.route.return_value = broker

    with (
        patch("finalayze.orchestration.equity_reconcile.reconcile_stocks") as mock_stocks,
        patch("finalayze.orchestration.equity_reconcile.reconcile_equity_drift") as mock_drift,
    ):
        loop._run_stock_reconcile()

    # Both markets visited once each
    market_args = [c.kwargs.get("market_id") for c in mock_stocks.call_args_list]
    assert sorted(market_args) == ["moex", "us"]
    assert mock_drift.call_count == 2


# ─── RUNREC-02 ───────────────────────────────────────────────────────────────
def test_broker_fetch_failure_doesnt_abort_other_markets() -> None:
    loop = _make_loop_skeleton()
    good_broker = MagicMock()
    good_broker.get_portfolio.return_value = _mock_portfolio()
    good_broker.get_positions.return_value = {}

    def _route(market_id: str) -> MagicMock:
        if market_id == "us":
            raise RuntimeError("Alpaca outage")
        return good_broker

    loop._broker_router.route.side_effect = _route

    with (
        patch("finalayze.orchestration.equity_reconcile.reconcile_stocks") as mock_stocks,
        patch("finalayze.orchestration.equity_reconcile.reconcile_equity_drift") as mock_drift,
    ):
        loop._run_stock_reconcile()

    # Only the working market (moex) is reconciled
    assert mock_stocks.call_count == 1
    assert mock_stocks.call_args.kwargs["market_id"] == "moex"
    assert mock_drift.call_count == 1


# ─── RUNREC-03 ───────────────────────────────────────────────────────────────
def test_reconcile_stocks_exception_does_not_skip_drift_check() -> None:
    loop = _make_loop_skeleton()
    loop._circuit_breakers = {"moex": MagicMock()}  # single market for simplicity
    broker = MagicMock()
    broker.get_portfolio.return_value = _mock_portfolio()
    broker.get_positions.return_value = {}
    loop._broker_router.route.return_value = broker

    with (
        patch(
            "finalayze.orchestration.equity_reconcile.reconcile_stocks",
            side_effect=ValueError("explode"),
        ),
        patch("finalayze.orchestration.equity_reconcile.reconcile_equity_drift") as mock_drift,
    ):
        loop._run_stock_reconcile()

    # Drift check still ran despite reconcile_stocks blowing up
    assert mock_drift.call_count == 1


# ─── RUNREC-04 ───────────────────────────────────────────────────────────────
def test_daily_reset_invokes_run_stock_reconcile() -> None:
    """_daily_reset must call _run_stock_reconcile after baseline update."""
    loop = TradingLoop.__new__(TradingLoop)
    loop._daily_reporter = MagicMock()
    loop._daily_reporter.daily_reset.return_value = {"us": Decimal(100), "moex": Decimal(200)}
    loop._metrics = MagicMock()
    loop._now = MagicMock(return_value=None)
    loop._baseline_equities = {"us": Decimal(99), "moex": Decimal(199)}

    with patch.object(loop, "_run_stock_reconcile") as mock_reconcile:
        # _daily_reset is unbound on a fresh __new__ — call via the class
        TradingLoop._daily_reset(loop)

    mock_reconcile.assert_called_once()
    # And the baselines must be updated BEFORE reconcile (so drift sees new state)
    assert loop._baseline_equities == {"us": Decimal(100), "moex": Decimal(200)}
