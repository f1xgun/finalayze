from __future__ import annotations

from pathlib import Path


def test_portfolio_page_has_mini_badge_text() -> None:
    """D-11: portfolio summary row must mention 'Positions at risk'."""
    src = Path("src/finalayze/dashboard/pages/portfolio.py").read_text()
    assert "Positions at risk" in src


def test_portfolio_page_mini_badge_page_link() -> None:
    """D-11: mini-badge must link to the new /positions page."""
    src = Path("src/finalayze/dashboard/pages/portfolio.py").read_text()
    assert 'st.page_link("pages/positions.py"' in src


def test_count_at_risk_empty() -> None:
    from finalayze.dashboard.pages.portfolio import _count_at_risk

    assert _count_at_risk([]) == 0
    assert _count_at_risk(None) == 0


def test_count_at_risk_ignores_none_distance() -> None:
    from finalayze.dashboard.pages.portfolio import _count_at_risk

    positions = [
        {"symbol": "A", "distance_atr": None},
        {"symbol": "B", "distance_atr": 0.3},
        {"symbol": "C", "distance_atr": 2.0},
    ]
    # Only B is < 0.5
    assert _count_at_risk(positions) == 1


def test_count_at_risk_boundary() -> None:
    from finalayze.dashboard.pages.portfolio import _count_at_risk

    positions = [
        {"symbol": "A", "distance_atr": 0.5},  # exactly at threshold -> NOT red
        {"symbol": "B", "distance_atr": 0.49},  # red
        {"symbol": "C", "distance_atr": 0.1},  # red
    ]
    assert _count_at_risk(positions) == 2


def test_count_at_risk_rejects_bool_distance() -> None:
    """I-07: bool values must NOT be counted even though bool is a subclass of int.

    `isinstance(True, (int, float))` is True in Python, so without the
    explicit `not isinstance(da, bool)` guard, a malformed API payload
    with `"distance_atr": False` (=0 < 0.5) would incorrectly flag the
    position as at-risk.
    """
    from finalayze.dashboard.pages.portfolio import _count_at_risk

    positions = [
        {"symbol": "A", "distance_atr": False},  # must NOT count
        {"symbol": "B", "distance_atr": True},  # must NOT count
        {"symbol": "C", "distance_atr": 0.3},  # red
    ]
    # Only C is counted
    assert _count_at_risk(positions) == 1
