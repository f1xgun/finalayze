"""Signals page -- Strategy x Segment performance heatmap + recent signals table.

D-09: axes = Strategy x Segment.
D-10: metric toggle (win_rate / profit_factor).
D-11: discrete 3-band colorscale (green/yellow/red).
D-12: empty cells gray with "—".
D-14: period dropdown 7d / 30d / 90d / All.
D-15: sample-size gate N >= 5.
D-16: reuse /api/v1/strategies/performance endpoint.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from finalayze.dashboard.api_client import ApiClient

_MIN_TRADES = 5  # D-15 sample-size gate
_PERIOD_OPTIONS: dict[str, int | None] = {
    "7d": 7,
    "30d": 30,
    "90d": 90,
    "All": None,
}
_DEFAULT_PERIOD_LABEL = "30d"  # D-13
_METRIC_OPTIONS = ["Win rate", "Profit factor"]

# D-11 colorscales (exact stops -- research-verified, hardcoded per CONTEXT.md)
_WIN_RATE_COLORSCALE: list[list[Any]] = [
    [0.00, "#d73027"],
    [0.45, "#d73027"],
    [0.45, "#fee08b"],
    [0.55, "#fee08b"],
    [0.55, "#1a9850"],
    [1.00, "#1a9850"],
]
_PF_ZMIN = 0.0
_PF_ZMAX = 3.0
_PF_COLORSCALE: list[list[Any]] = [
    [0.0, "#d73027"],
    [1.0 / _PF_ZMAX, "#d73027"],
    [1.0 / _PF_ZMAX, "#fee08b"],
    [1.5 / _PF_ZMAX, "#fee08b"],
    [1.5 / _PF_ZMAX, "#1a9850"],
    [1.0, "#1a9850"],
]


def _heatmap_cell_value(
    win_rate: float | None,
    profit_factor: float | None,
    trades_count: int,
    metric: str,
) -> float | None:
    """Return the cell's numeric value or None for gray per D-12, D-15.

    Args:
        win_rate: fraction in [0, 1] or None.
        profit_factor: ratio in [0, inf) or None.
        trades_count: paired-trade count; below _MIN_TRADES triggers the D-15 gate.
        metric: "win_rate" or "profit_factor".
    """
    if trades_count < _MIN_TRADES:
        return None
    return win_rate if metric == "win_rate" else profit_factor


def _render_strategy_heatmap(rows: list[dict[str, Any]], metric: str) -> None:
    """Plotly go.Heatmap, Strategy x Segment axes, discrete 3-band colorscale."""
    if not rows:
        st.info("No strategy performance data yet.")
        return

    strategies = sorted({r["strategy"] for r in rows})
    segments = sorted({r["segment_id"] for r in rows})
    by_key = {(r["strategy"], r["segment_id"]): r for r in rows}

    z: list[list[float]] = []
    text: list[list[str]] = []
    for strat in strategies:
        z_row: list[float] = []
        t_row: list[str] = []
        for seg in segments:
            r = by_key.get((strat, seg))
            if r is None:
                # D-12: missing cell = gray "—"
                z_row.append(float("nan"))
                t_row.append("—")
                continue
            v = _heatmap_cell_value(
                r.get("win_rate"),
                r.get("profit_factor"),
                int(r.get("trades_count", 0)),
                metric,
            )
            if v is None:
                z_row.append(float("nan"))
                t_row.append("—")
            elif metric == "win_rate":
                z_row.append(float(v) * 100.0)  # domain 0-100 for the win_rate colorscale
                t_row.append(f"{float(v) * 100:.0f}%")
            else:
                z_row.append(min(float(v), _PF_ZMAX))  # cap display at zmax
                t_row.append(f"{float(v):.2f}")
        z.append(z_row)
        text.append(t_row)

    if metric == "win_rate":
        colorscale = _WIN_RATE_COLORSCALE
        zmin: float = 0.0
        zmax: float = 100.0
    else:
        colorscale = _PF_COLORSCALE
        zmin = _PF_ZMIN
        zmax = _PF_ZMAX

    fig = go.Figure(
        go.Heatmap(
            z=np.array(z),
            x=segments,
            y=strategies,
            colorscale=colorscale,
            zmin=zmin,
            zmax=zmax,
            text=text,
            texttemplate="%{text}",
            hoverongaps=False,
        )
    )
    fig.update_xaxes(side="top")
    fig.update_layout(height=max(300, 60 * len(strategies) + 80))
    st.plotly_chart(fig, use_container_width=True)


def render(api: ApiClient) -> None:
    """Render the Signals page (SIGP-01 table + SIGP-02 heatmap consumer)."""
    st.title("Signals")

    period_label = st.selectbox(
        "Period",
        list(_PERIOD_OPTIONS.keys()),
        index=list(_PERIOD_OPTIONS.keys()).index(_DEFAULT_PERIOD_LABEL),
    )
    period_days = _PERIOD_OPTIONS[period_label]
    metric_label = st.radio("Metric", _METRIC_OPTIONS, horizontal=True)
    metric = "win_rate" if metric_label == "Win rate" else "profit_factor"

    params: dict[str, int] = {}
    if period_days is not None:
        params["period"] = period_days

    try:
        strategies_resp = api.get(
            "/api/v1/strategies/performance",
            params=params,
        ).json()
        signals_resp = api.get("/api/v1/signals").json()
    except Exception:
        st.error("Cannot reach API server")
        return

    strategies = strategies_resp.get("strategies", [])
    signals = signals_resp.get("signals", [])

    st.subheader("Strategy Performance Heatmap")
    _render_strategy_heatmap(strategies, metric)

    st.subheader("Strategy Performance Table")
    if strategies:
        sdf = pd.DataFrame(strategies)
        # Apply D-15 gate visually in the table too: blank win_rate/PF when trades_count < 5
        if "trades_count" in sdf.columns:
            mask_gated = sdf["trades_count"] < _MIN_TRADES
            if "win_rate" in sdf.columns:
                sdf.loc[mask_gated, "win_rate"] = None
            if "profit_factor" in sdf.columns:
                sdf.loc[mask_gated, "profit_factor"] = None
        display_cols = [
            c
            for c in (
                "strategy",
                "market_id",
                "segment_id",
                "win_rate",
                "profit_factor",
                "trades_count",
                "signal_count",
                "last_signal_at",
            )
            if c in sdf.columns
        ]
        st.dataframe(sdf[display_cols], use_container_width=True)
    else:
        st.info("No strategy performance data yet.")

    st.subheader("Recent Signals")
    if signals:
        sig_df = pd.DataFrame(signals)
        cols = [
            c
            for c in (
                "symbol",
                "strategy",
                "market_id",
                "segment_id",
                "direction",
                "confidence",
                "created_at",
            )
            if c in sig_df.columns
        ]
        st.dataframe(sig_df[cols], use_container_width=True)
    else:
        st.info("No signals recorded yet.")


# Called by st.navigation/page.run(); app.py guarantees "api" is in session_state.
if (_api := st.session_state.get("api")) is not None:
    render(_api)
