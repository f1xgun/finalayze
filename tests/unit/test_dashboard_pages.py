"""Smoke tests for dashboard page modules — verify importability and callable render()."""

from __future__ import annotations


def test_system_status_render_importable() -> None:
    from finalayze.dashboard.pages import system_status

    assert callable(system_status.render)


def test_portfolio_render_importable() -> None:
    from finalayze.dashboard.pages import portfolio

    assert callable(portfolio.render)


def test_trades_render_importable() -> None:
    from finalayze.dashboard.pages import trades

    assert callable(trades.render)


def test_signals_render_importable() -> None:
    from finalayze.dashboard.pages import signals

    assert callable(signals.render)


def test_risk_render_importable() -> None:
    from finalayze.dashboard.pages import risk

    assert callable(risk.render)


def test_sandbox_render_importable() -> None:
    from finalayze.dashboard.pages import sandbox

    assert callable(sandbox.render)


def test_experiments_list_render_importable() -> None:
    from finalayze.dashboard.pages import experiments_list

    assert callable(experiments_list.render)


def test_experiment_detail_importable() -> None:
    from finalayze.dashboard.pages import experiment_detail  # noqa: F401


def test_decision_history_importable() -> None:
    from finalayze.dashboard.pages import decision_history  # noqa: F401


def test_saa_allocation_render_importable() -> None:
    from finalayze.dashboard.pages import saa_allocation

    assert callable(saa_allocation.render)


def test_saa_allocation_build_leg_rows() -> None:
    from finalayze.dashboard.pages.saa_allocation import _build_leg_rows

    data = {
        "legs": {
            "deposit": {"symbol": None, "weight": "0.25", "target_notional_rub": "250000.00"},
            "equity": {"symbol": "EQMX", "weight": "0.35", "target_notional_rub": "350000.00"},
        }
    }
    rows = _build_leg_rows(data)
    by_class = {r["Asset class"]: r for r in rows}
    assert by_class["deposit"]["Symbol"] == "(deposit - manual)"
    assert by_class["equity"]["Symbol"] == "EQMX"
    assert by_class["equity"]["Target notional (RUB)"] == "350000.00"


def test_saa_allocation_build_leg_rows_empty() -> None:
    from finalayze.dashboard.pages.saa_allocation import _build_leg_rows

    assert _build_leg_rows({}) == []


def test_saa_allocation_module_runs_render_at_module_level(
    monkeypatch: object,
) -> None:
    """Executing the page module (as st.navigation does) invokes render() -- not a blank page.

    This drives the module-level page-execution path that a plain callable(render) smoke test
    misses: without the trailing `render(_api)` block, st.navigation renders a blank page.
    """
    import sys
    from pathlib import Path
    from unittest.mock import MagicMock

    from finalayze.dashboard.pages import saa_allocation

    st_mock = MagicMock()
    api_stub = MagicMock()
    api_stub.saa_target_allocation.return_value = {
        "legs": {"deposit": {"symbol": None, "weight": "0.25", "target_notional_rub": "250000"}},
        "risk_profile": "balanced",
        "budget_rub": "1000000",
        "deposit_current_notional_rub": "0",
        "as_of": "2026-06-23",
        "portfolio_id": "p",
    }
    st_mock.session_state = {"api": api_stub}
    st_mock.columns.return_value = (MagicMock(), MagicMock(), MagicMock())

    monkeypatch.setitem(sys.modules, "streamlit", st_mock)  # type: ignore[attr-defined]
    src_path = Path(saa_allocation.__file__)
    exec(compile(src_path.read_text(), str(src_path), "exec"), {"__name__": "saa_exec"})  # noqa: S102

    assert st_mock.title.called  # render ran via the module-level entry block
    api_stub.saa_target_allocation.assert_called_once()


def test_rebalance_history_render_importable() -> None:
    from finalayze.dashboard.pages import rebalance_history

    assert callable(rebalance_history.render)


def test_rebalance_history_build_run_rows() -> None:
    from finalayze.dashboard.pages.rebalance_history import _build_run_rows

    data = {
        "runs": [
            {
                "created_at": "2026-06-23T12:00:00+00:00",
                "as_of": "2026-06-23",
                "mode": "SANDBOX",
                "status": "COMPLETE",
                "fill_rate": "1.0000",
                "orders": [
                    {
                        "asset_class": "equity",
                        "symbol": "EQMX",
                        "side": "BUY",
                        "requested_qty": "100",
                        "filled_qty": "100",
                        "status": "FILLED",
                        "reason": None,
                    }
                ],
            }
        ]
    }
    rows = _build_run_rows(data)
    assert len(rows) == 1
    assert rows[0]["When"] == "2026-06-23 12:00:00"
    assert rows[0]["Status"] == "COMPLETE"
    assert rows[0]["Legs"] == 1


def test_rebalance_history_build_run_rows_empty() -> None:
    from finalayze.dashboard.pages.rebalance_history import _build_run_rows

    assert _build_run_rows({}) == []


def test_rebalance_history_module_runs_render_at_module_level(
    monkeypatch: object,
) -> None:
    """Executing the page module invokes render() (Phase 81 CR-01 -- not a blank page)."""
    import sys
    from pathlib import Path
    from unittest.mock import MagicMock

    from finalayze.dashboard.pages import rebalance_history

    st_mock = MagicMock()
    api_stub = MagicMock()
    api_stub.saa_rebalance_runs.return_value = {
        "portfolio_id": "p",
        "runs": [
            {
                "created_at": "2026-06-23T12:00:00+00:00",
                "as_of": "2026-06-23",
                "mode": "SANDBOX",
                "status": "COMPLETE",
                "fill_rate": "1.0000",
                "orders": [
                    {
                        "asset_class": "equity",
                        "symbol": "EQMX",
                        "side": "BUY",
                        "requested_qty": "100",
                        "filled_qty": "100",
                        "status": "FILLED",
                        "reason": None,
                    }
                ],
            }
        ],
    }
    st_mock.session_state = {"api": api_stub}

    monkeypatch.setitem(sys.modules, "streamlit", st_mock)  # type: ignore[attr-defined]
    src_path = Path(rebalance_history.__file__)
    exec(compile(src_path.read_text(), str(src_path), "exec"), {"__name__": "rh_exec"})  # noqa: S102

    assert st_mock.title.called
    api_stub.saa_rebalance_runs.assert_called_once()


# ── Phase 87: SAA cert-decision benchmark block (honest verdict surfaced on the page) ──


def test_saa_allocation_build_benchmark_rows() -> None:
    """The benchmark table shapes per-regime stories + a full-window row; Sharpe -> 4dp."""
    from finalayze.dashboard.pages.saa_allocation import _build_benchmark_rows

    cert = {
        "regime_stories": [
            {
                "unit_key": "high_rate",
                "unit_label": "high_rate",
                "window_start": "2024-01-02",
                "window_end": "2025-06-05",
                "allocation_sharpe": -0.783,
                "best_naive_sharpe": 0.8904,
                "unit_verdict": "HARD_FAIL",
            }
        ],
        "alloc_sharpe_full": -0.8589,
        "best_naive_sharpe_full": -0.6506,
        "full_verdict": "HARD_FAIL",
    }
    rows = _build_benchmark_rows(cert)
    assert rows[0]["Regime"] == "high_rate"
    assert rows[0]["Best-naive Sharpe"] == "0.8904"
    assert rows[-1]["Regime"] == "full window"
    assert rows[-1]["Best-naive Sharpe"] == "-0.6506"


def test_render_cert_decision_handles_empty_state() -> None:
    """render_cert_decision with no committed cert ({}) shows the info empty state, never raises."""
    from unittest.mock import MagicMock

    from finalayze.dashboard.pages import saa_allocation

    st_mock = MagicMock()
    api_stub = MagicMock()
    api_stub.saa_cert_decision.return_value = {}
    orig = saa_allocation.st
    saa_allocation.st = st_mock  # type: ignore[assignment]
    try:
        saa_allocation.render_cert_decision(api_stub)
    finally:
        saa_allocation.st = orig  # type: ignore[assignment]
    assert st_mock.info.called  # "No committed cert" empty state
    assert not st_mock.error.called  # not a crash / connection error


def test_render_cert_decision_hard_fail_uses_error_not_success() -> None:
    """A HARD_FAIL verdict renders RED (st.error), never softened to st.success (page honesty)."""
    from unittest.mock import MagicMock

    from finalayze.dashboard.pages import saa_allocation

    st_mock = MagicMock()
    api_stub = MagicMock()
    api_stub.saa_cert_decision.return_value = {
        "phase_verdict": "HARD_FAIL",
        "headline": "HOLD DEPOSIT-HEAVY: the allocator does not beat its best benchmark",
        "regime_stories": [],
        "alloc_sharpe_full": -0.86,
        "best_naive_sharpe_full": -0.65,
        "full_verdict": "HARD_FAIL",
        "when_framing": "qualitative; no rate threshold",
        "high_rate_caveat": "caveat",
        "escalation": "deposit_anchor_vs_redesign",
        "n1_caveat": True,
        "cert_timestamp": "2026-06-22T22:06:28+00:00",
        "git_sha": "44ef26ff",
        "staleness_days": 2,
    }
    orig = saa_allocation.st
    saa_allocation.st = st_mock  # type: ignore[assignment]
    try:
        saa_allocation.render_cert_decision(api_stub)
    finally:
        saa_allocation.st = orig  # type: ignore[assignment]
    assert st_mock.error.called  # HARD_FAIL is red...
    assert not st_mock.success.called  # ...never softened to green
