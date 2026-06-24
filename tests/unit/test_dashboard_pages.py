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
