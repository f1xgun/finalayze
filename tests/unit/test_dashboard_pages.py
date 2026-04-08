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


def test_experiment_detail_render_importable() -> None:
    from finalayze.dashboard.pages import experiment_detail

    assert callable(experiment_detail.render)


def test_decision_history_render_importable() -> None:
    from finalayze.dashboard.pages import decision_history

    assert callable(decision_history.render)
