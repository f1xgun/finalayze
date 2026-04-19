"""Tests for the Trades page — period dropdown, extended analytics row, slippage rendering.

Warning #7 fix: `trades.py:render` calls `st.columns()` TWICE.
- First call: `col1, col2, col3 = st.columns(3)` for filter row.
- Second call: `mcols = st.columns(6)` for the TRAD-02 analytics metrics row.

Test fixtures MUST mock `streamlit.columns` with `side_effect=[3-list, 6-list]`,
NOT `return_value=` (a single return value would fail the 3-tuple unpack on the
first call, or hand 6 mocks where 3 are expected).
"""

from __future__ import annotations

import inspect
from unittest.mock import MagicMock

import pytest


def _columns_side_effect() -> list[list[MagicMock]]:
    """trades.py calls st.columns() twice: 3 for filters, 6 for metrics."""
    return [
        [MagicMock(), MagicMock(), MagicMock()],
        [MagicMock() for _ in range(6)],
    ]


def test_trades_page_importable() -> None:
    import finalayze.dashboard.pages.trades  # noqa: F401


def test_trades_page_has_render_function() -> None:
    from finalayze.dashboard.pages import trades as page

    assert callable(page.render)
    sig = inspect.signature(page.render)
    assert "api" in sig.parameters


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, "—"),
        (3.2, "3.20"),
        (0.0, "0.00"),
        (-1.5, "-1.50"),
        ("not-a-number", "—"),
    ],
)
def test_format_slippage(value: object, expected: str) -> None:
    """_format_slippage renders null and non-numeric as "—" (D-07 UI convention)."""
    from finalayze.dashboard.pages.trades import _format_slippage

    assert _format_slippage(value) == expected


def test_analytics_displays_win_rate_avg_win_avg_loss_pf(mocker: object) -> None:
    """55-05-05: metrics row labels include Win Rate / Avg Win / Avg Loss / Profit Factor."""
    trades_resp: dict[str, object] = {"trades": [], "total": 0}
    analytics_resp = {
        "period_days": 30,
        "total_trades": 50,
        "win_rate": 0.60,
        "avg_win": 150.5,
        "avg_loss": -80.0,
        "profit_factor": 1.9,
        "avg_slippage_bps": 3.5,
        "avg_fill_latency_ms": None,
        "rejection_rate_pct": None,
    }
    mocker.patch("streamlit.title")  # type: ignore[attr-defined]
    mocker.patch("streamlit.subheader")  # type: ignore[attr-defined]
    mocker.patch("streamlit.caption")  # type: ignore[attr-defined]
    mocker.patch("streamlit.selectbox", return_value="30d")  # type: ignore[attr-defined]
    mocker.patch("streamlit.text_input", return_value="")  # type: ignore[attr-defined]
    mocker.patch("streamlit.slider", return_value=100)  # type: ignore[attr-defined]
    # Warning #7 fix: st.columns() called twice -> side_effect with distinct lists.
    cols_sequence = _columns_side_effect()
    mocker.patch("streamlit.columns", side_effect=cols_sequence)  # type: ignore[attr-defined]
    mocker.patch("streamlit.info")  # type: ignore[attr-defined]
    mocker.patch("streamlit.dataframe")  # type: ignore[attr-defined]
    mocker.patch("streamlit.scatter_chart")  # type: ignore[attr-defined]

    # Stub the filter column selectbox too; first st.columns() returned 3 mocks.
    filter_cols = cols_sequence[0]
    filter_cols[0].selectbox = MagicMock(return_value="all")
    filter_cols[1].text_input = MagicMock(return_value="")
    filter_cols[2].slider = MagicMock(return_value=100)

    api = MagicMock()
    api.get.side_effect = lambda url, params=None: (
        MagicMock(json=lambda: analytics_resp)
        if "analytics" in url
        else MagicMock(json=lambda: trades_resp)
    )
    from finalayze.dashboard.pages.trades import render

    render(api)

    # Collect all metric-label kwargs from .metric() calls across the metrics-row column mocks.
    # The second st.columns() call returned cols_sequence[1] = the 6-list.
    metrics_cols = cols_sequence[1]
    metric_labels: list[str] = []
    for cm in metrics_cols:
        metric_labels.extend(call.args[0] if call.args else "" for call in cm.metric.call_args_list)
    joined = " ".join(metric_labels)
    assert "Win Rate" in joined
    assert "Avg Win" in joined
    assert "Avg Loss" in joined
    assert "Profit Factor" in joined


def test_period_dropdown_persists_selection(mocker: object) -> None:
    """55-05-06: period dropdown forwards ?period=N to the analytics endpoint."""
    mocker.patch("streamlit.title")  # type: ignore[attr-defined]
    mocker.patch("streamlit.subheader")  # type: ignore[attr-defined]
    mocker.patch("streamlit.caption")  # type: ignore[attr-defined]
    mocker.patch("streamlit.selectbox", return_value="7d")  # type: ignore[attr-defined]
    mocker.patch("streamlit.text_input", return_value="")  # type: ignore[attr-defined]
    mocker.patch("streamlit.slider", return_value=100)  # type: ignore[attr-defined]
    cols_sequence = _columns_side_effect()
    mocker.patch("streamlit.columns", side_effect=cols_sequence)  # type: ignore[attr-defined]
    mocker.patch("streamlit.info")  # type: ignore[attr-defined]
    mocker.patch("streamlit.dataframe")  # type: ignore[attr-defined]

    filter_cols = cols_sequence[0]
    filter_cols[0].selectbox = MagicMock(return_value="all")
    filter_cols[1].text_input = MagicMock(return_value="")
    filter_cols[2].slider = MagicMock(return_value=100)

    api = MagicMock()
    api.get.return_value = MagicMock(
        json=lambda: {
            "trades": [],
            "total": 0,
            "period_days": 7,
            "total_trades": 0,
            "win_rate": None,
            "avg_win": None,
            "avg_loss": None,
            "profit_factor": None,
            "avg_slippage_bps": None,
            "avg_fill_latency_ms": None,
            "rejection_rate_pct": None,
        }
    )
    from finalayze.dashboard.pages.trades import render

    render(api)

    analytics_calls = [c for c in api.get.call_args_list if "analytics" in c.args[0]]
    assert len(analytics_calls) == 1
    assert analytics_calls[0].kwargs.get("params", {}).get("period") == 7


def test_slippage_bps_renders_dash_for_null(mocker: object) -> None:
    """55-05-07: null slippage_bps renders as "—"; non-null renders as 2-decimal float (D-07)."""
    trades_resp = {
        "trades": [
            {
                "id": "a",
                "symbol": "SBER",
                "market_id": "moex",
                "side": "BUY",
                "quantity": 10.0,
                "fill_price": 280.0,
                "slippage_bps": None,
                "timestamp": "2026-04-17T10:00:00+00:00",
            },
            {
                "id": "b",
                "symbol": "GAZP",
                "market_id": "moex",
                "side": "BUY",
                "quantity": 5.0,
                "fill_price": 190.0,
                "slippage_bps": 3.2,
                "timestamp": "2026-04-17T11:00:00+00:00",
            },
        ],
        "total": 2,
    }
    mocker.patch("streamlit.title")  # type: ignore[attr-defined]
    mocker.patch("streamlit.subheader")  # type: ignore[attr-defined]
    mocker.patch("streamlit.caption")  # type: ignore[attr-defined]
    mocker.patch("streamlit.selectbox", return_value="30d")  # type: ignore[attr-defined]
    mocker.patch("streamlit.text_input", return_value="")  # type: ignore[attr-defined]
    mocker.patch("streamlit.slider", return_value=100)  # type: ignore[attr-defined]
    cols_sequence = _columns_side_effect()
    mocker.patch("streamlit.columns", side_effect=cols_sequence)  # type: ignore[attr-defined]
    mocker.patch("streamlit.info")  # type: ignore[attr-defined]
    df_call = mocker.patch("streamlit.dataframe")  # type: ignore[attr-defined]
    mocker.patch("streamlit.scatter_chart")  # type: ignore[attr-defined]

    filter_cols = cols_sequence[0]
    filter_cols[0].selectbox = MagicMock(return_value="all")
    filter_cols[1].text_input = MagicMock(return_value="")
    filter_cols[2].slider = MagicMock(return_value=100)

    api = MagicMock()
    api.get.side_effect = lambda url, params=None: (
        MagicMock(
            json=lambda: {
                "period_days": 30,
                "total_trades": 2,
                "win_rate": None,
                "avg_win": None,
                "avg_loss": None,
                "profit_factor": None,
                "avg_slippage_bps": 3.2,
                "avg_fill_latency_ms": None,
                "rejection_rate_pct": None,
            }
        )
        if "analytics" in url
        else MagicMock(json=lambda: trades_resp)
    )
    from finalayze.dashboard.pages.trades import render

    render(api)

    # First dataframe call receives the trades DF with slippage rendered.
    trades_df = df_call.call_args_list[0].args[0]
    slip_col = trades_df["slippage_bps"].astype(str).tolist()
    assert slip_col[0] == "—"
    assert slip_col[1] == "3.20"
