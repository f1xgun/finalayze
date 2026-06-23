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
