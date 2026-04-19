"""Tests for the Signals page — Plotly heatmap, metric toggle, period dropdown, N>=5 gate."""

from __future__ import annotations

import inspect
import math
from unittest.mock import MagicMock

import pytest


def test_signals_page_importable() -> None:
    import finalayze.dashboard.pages.signals  # noqa: F401


def test_signals_page_has_render_function() -> None:
    from finalayze.dashboard.pages import signals as page

    assert callable(page.render)
    sig = inspect.signature(page.render)
    assert "api" in sig.parameters


@pytest.mark.parametrize(
    ("win_rate", "trades_count", "expected"),
    [
        (None, 10, None),  # None stays None (no data)
        (0.56, 2, None),  # below gate -> None (gray)
        (0.56, 5, 0.56),  # at gate -> pass through
        (0.30, 10, 0.30),  # red bucket
        (0.50, 10, 0.50),  # yellow bucket
        (0.70, 10, 0.70),  # green bucket
    ],
)
def test_heatmap_cell_value_applies_sample_gate(
    win_rate: float | None, trades_count: int, expected: float | None
) -> None:
    """D-15 sample gate + D-11 buckets (covers 55-05-04)."""
    from finalayze.dashboard.pages.signals import _heatmap_cell_value

    assert _heatmap_cell_value(win_rate, None, trades_count, "win_rate") == expected


def _patch_streamlit_common(mocker: object, radio_value: str = "Win rate") -> None:
    mocker.patch("streamlit.title")  # type: ignore[attr-defined]
    mocker.patch("streamlit.subheader")  # type: ignore[attr-defined]
    mocker.patch("streamlit.caption")  # type: ignore[attr-defined]
    mocker.patch("streamlit.selectbox", return_value="30d")  # type: ignore[attr-defined]
    mocker.patch("streamlit.radio", return_value=radio_value)  # type: ignore[attr-defined]
    mocker.patch("streamlit.info")  # type: ignore[attr-defined]
    mocker.patch("streamlit.dataframe")  # type: ignore[attr-defined]


def test_heatmap_renders_with_win_rate_metric(mocker: object) -> None:
    """55-05-01: render() must call st.plotly_chart with a go.Heatmap figure."""
    import plotly.graph_objects as go

    strategies_resp = {
        "strategies": [
            {
                "strategy": "momentum",
                "market_id": "moex",
                "segment_id": "ru_blue_chips",
                "win_rate": 0.60,
                "profit_factor": 1.6,
                "trades_count": 10,
                "signal_count": 20,
                "last_signal_at": None,
            },
            {
                "strategy": "mean_reversion",
                "market_id": "moex",
                "segment_id": "ru_energy",
                "win_rate": 0.40,
                "profit_factor": 0.9,
                "trades_count": 12,
                "signal_count": 30,
                "last_signal_at": None,
            },
        ]
    }
    signals_resp: dict[str, list[object]] = {"signals": []}
    _patch_streamlit_common(mocker, radio_value="Win rate")
    plotly_chart = mocker.patch("streamlit.plotly_chart")  # type: ignore[attr-defined]

    api = MagicMock()
    api.get.side_effect = lambda url, params=None: (
        MagicMock(json=lambda: strategies_resp)
        if "strategies/performance" in url
        else MagicMock(json=lambda: signals_resp)
    )
    from finalayze.dashboard.pages.signals import render

    render(api)

    plotly_chart.assert_called_once()
    fig = plotly_chart.call_args.args[0]
    assert isinstance(fig.data[0], go.Heatmap)


def test_heatmap_toggle_switches_to_profit_factor(mocker: object) -> None:
    """55-05-02: metric toggle changes the heatmap z-values to profit_factor."""
    strategies_resp = {
        "strategies": [
            {
                "strategy": "m",
                "market_id": "moex",
                "segment_id": "ru_blue_chips",
                "win_rate": 0.60,
                "profit_factor": 2.0,
                "trades_count": 10,
                "signal_count": 5,
                "last_signal_at": None,
            }
        ]
    }
    _patch_streamlit_common(mocker, radio_value="Profit factor")
    plotly_chart = mocker.patch("streamlit.plotly_chart")  # type: ignore[attr-defined]

    api = MagicMock()
    api.get.side_effect = lambda url, params=None: (
        MagicMock(json=lambda: strategies_resp)
        if "strategies/performance" in url
        else MagicMock(json=lambda: {"signals": []})
    )
    from finalayze.dashboard.pages.signals import render

    render(api)
    fig = plotly_chart.call_args.args[0]
    hm = fig.data[0]
    # z matrix should contain 2.0 (the profit_factor), not 0.60 (the win_rate)
    zs = [
        v
        for row in hm.z
        for v in row
        if v is not None and not (isinstance(v, float) and math.isnan(v))
    ]
    assert 2.0 in zs


def test_heatmap_empty_cells_render_dash(mocker: object) -> None:
    """55-05-03: below-gate cells render with "—" text and hoverongaps=False."""
    strategies_resp = {
        "strategies": [
            {
                "strategy": "m",
                "market_id": "moex",
                "segment_id": "ru_blue_chips",
                "win_rate": 0.60,
                "profit_factor": 1.6,
                "trades_count": 2,
                "signal_count": 2,
                "last_signal_at": None,
            }
        ]
    }
    _patch_streamlit_common(mocker, radio_value="Win rate")
    plotly_chart = mocker.patch("streamlit.plotly_chart")  # type: ignore[attr-defined]

    api = MagicMock()
    api.get.side_effect = lambda url, params=None: (
        MagicMock(json=lambda: strategies_resp)
        if "strategies/performance" in url
        else MagicMock(json=lambda: {"signals": []})
    )
    from finalayze.dashboard.pages.signals import render

    render(api)
    fig = plotly_chart.call_args.args[0]
    hm = fig.data[0]
    assert hm.hoverongaps is False
    # text[0][0] is the single strategy x single segment cell
    assert hm.text[0][0] == "—"


def test_period_dropdown_passes_query_param(mocker: object) -> None:
    """D-14: period dropdown sends `period=N` as query param; "All" drops the param."""
    mocker.patch("streamlit.title")  # type: ignore[attr-defined]
    mocker.patch("streamlit.subheader")  # type: ignore[attr-defined]
    mocker.patch("streamlit.caption")  # type: ignore[attr-defined]
    mocker.patch("streamlit.selectbox", return_value="7d")  # type: ignore[attr-defined]
    mocker.patch("streamlit.radio", return_value="Win rate")  # type: ignore[attr-defined]
    mocker.patch("streamlit.info")  # type: ignore[attr-defined]
    mocker.patch("streamlit.dataframe")  # type: ignore[attr-defined]
    mocker.patch("streamlit.plotly_chart")  # type: ignore[attr-defined]

    api = MagicMock()
    api.get.return_value = MagicMock(json=lambda: {"strategies": [], "signals": []})
    from finalayze.dashboard.pages.signals import render

    render(api)

    # At least one api.get was called with params={"period": 7}
    calls = [c for c in api.get.call_args_list if "strategies/performance" in c.args[0]]
    assert len(calls) == 1
    params = calls[0].kwargs.get("params", {})
    assert params.get("period") == 7


def test_colorscale_module_constants_match_d11() -> None:
    """D-11 hardcoded thresholds: win_rate stops at 0.45/0.55; PF stops at 1.0/1.5 of zmax=3."""
    from finalayze.dashboard.pages.signals import (
        _PF_COLORSCALE,
        _PF_ZMAX,
        _WIN_RATE_COLORSCALE,
    )

    # Win-rate colorscale: stops at 0.45 and 0.55 (red->yellow and yellow->green transitions)
    stops = [stop for stop, _color in _WIN_RATE_COLORSCALE]
    assert 0.45 in stops
    assert 0.55 in stops
    # PF colorscale: stops at 1.0/3.0 and 1.5/3.0
    pf_stops = [stop for stop, _color in _PF_COLORSCALE]
    assert pytest.approx(1.0 / _PF_ZMAX, abs=1e-6) == pf_stops[1]
    assert pytest.approx(1.5 / _PF_ZMAX, abs=1e-6) == pf_stops[3]
