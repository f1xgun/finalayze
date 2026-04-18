"""Positions page -- heatmap, table, and Plotly stop-loss history chart.

D-09 layout:
  - Top: risk heatmap grid (D-10 ATR-normalized)
  - Middle: positions table with stop columns (STOP-02)
  - Bottom: per-symbol detail with Plotly history chart (STOP-03, D-07)
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from finalayze.dashboard.api_client import ApiClient

# D-10 ATR-normalized heatmap thresholds (CONTEXT.md locked)
_ATR_GREEN_THRESHOLD = 1.5
_ATR_RED_THRESHOLD = 0.5

_HEATMAP_MAX_COLS = 6
_HISTORY_CHART_HEIGHT = 500
_HISTORY_DAYS_DEFAULT = 30


def _bucket_color(distance_atr: float | None) -> str:
    """Return 'green' | 'yellow' | 'red' | 'gray' per D-10.

    Thresholds (CONTEXT.md D-10 locked):
      - green : distance_atr > 1.5
      - yellow: 0.5 <= distance_atr <= 1.5
      - red   : distance_atr < 0.5
      - gray  : distance_atr is None (no active stop)
    """
    if distance_atr is None:
        return "gray"
    if distance_atr > _ATR_GREEN_THRESHOLD:
        return "green"
    if distance_atr >= _ATR_RED_THRESHOLD:
        return "yellow"
    return "red"


def _render_heatmap(positions: list[dict[str, Any]]) -> None:
    """STOP-04 heatmap grid. Each cell shows symbol + ATR distance label."""
    if not positions:
        st.info("No open positions.")
        return
    cols = st.columns(min(len(positions), _HEATMAP_MAX_COLS))
    for i, p in enumerate(positions):
        dist_atr = p.get("distance_atr")
        color = _bucket_color(dist_atr)
        label = "No stop" if dist_atr is None else f"{dist_atr:.2f} ATR"
        with cols[i % len(cols)]:
            st.markdown(
                f"<div style='background:{color};padding:12px;"
                f"border-radius:6px;text-align:center;color:black;'>"
                f"<b>{p.get('symbol', '-')}</b><br>{label}</div>",
                unsafe_allow_html=True,
            )


def _render_positions_table(positions: list[dict[str, Any]]) -> None:
    """STOP-02 positions table with stop columns next to price columns."""
    if not positions:
        return
    df = pd.DataFrame(positions)
    display_cols = [
        "symbol",
        "market_id",
        "quantity",
        "avg_price",
        "current_price",
        "stop_price",
        "distance_pct",
        "distance_atr",
        "trail_activated",
        "unrealized_pnl_pct",
    ]
    show_cols = [c for c in display_cols if c in df.columns]
    st.dataframe(df[show_cols], use_container_width=True)


def _render_history_chart(events: list[dict[str, Any]], symbol: str) -> None:
    """STOP-03 Plotly chart: price / trailing stop (step) / high-water / entry marker."""
    if not events:
        st.info("No stop-loss history yet for this position.")
        return
    df = pd.DataFrame(events)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    fig = go.Figure()
    if "current_price" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=df["current_price"],
                name="Price",
                mode="lines",
            )
        )
    if "current_stop" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=df["current_stop"],
                name="Trailing stop",
                mode="lines",
                line={"shape": "hv", "color": "red", "dash": "dash"},
            )
        )
    if "highest_price" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=df["highest_price"],
                name="High-water",
                mode="lines",
                line={"color": "green", "dash": "dot"},
            )
        )
    # Entry marker(s)
    if "event_type" in df.columns:
        entry_rows = df[df["event_type"] == "entry"]
        if not entry_rows.empty and "entry_price" in entry_rows.columns:
            fig.add_trace(
                go.Scatter(
                    x=entry_rows["timestamp"],
                    y=entry_rows["entry_price"],
                    name="Entry",
                    mode="markers",
                    marker={"size": 12, "symbol": "triangle-up", "color": "blue"},
                )
            )
        activation_rows = df[df["event_type"] == "activation"]
        for _, row in activation_rows.iterrows():
            fig.add_vline(
                x=row["timestamp"],
                line_dash="dot",
                line_color="orange",
                annotation_text="Trail activated",
            )

    fig.update_layout(title=f"{symbol} - stop-loss history", height=_HISTORY_CHART_HEIGHT)
    st.plotly_chart(fig, use_container_width=True)


def render(api: ApiClient) -> None:
    """Render the Positions page (STOP-02/03/04 consumer)."""
    st.title("Positions")
    try:
        positions_data = api.get("/api/v1/portfolio/positions").json()
    except Exception:
        st.error("Cannot reach API server")
        return
    positions = positions_data.get("positions", []) or []

    st.subheader("Risk heatmap (ATR-normalized distance to stop)")
    _render_heatmap(positions)

    st.subheader("Open positions")
    if positions:
        _render_positions_table(positions)
    else:
        st.info("No open positions.")

    st.subheader("Stop-loss history")
    symbols = [p.get("symbol", "") for p in positions if p.get("symbol")]
    if not symbols:
        return
    chosen = st.selectbox("Select position", symbols)
    if not chosen:
        return
    try:
        hist = api.get(
            f"/api/v1/portfolio/positions/{chosen}/stop-history",
            params={"days": _HISTORY_DAYS_DEFAULT},
        ).json()
    except Exception:
        st.error(f"Cannot fetch history for {chosen}")
        return
    _render_history_chart(hist.get("events", []), chosen)


# Called by st.navigation/page.run(); app.py guarantees "api" is in session_state.
if (_api := st.session_state.get("api")) is not None:
    render(_api)
