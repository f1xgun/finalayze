"""Sandbox monitoring page -- metrics table, equity curve, uptime, fill rate, slippage."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from finalayze.dashboard.api_client import ApiClient, get_sandbox_gonogo, get_sandbox_metrics

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_DEFAULT_DAYS = 7
_FILL_RATE_TARGET_PCT = 95.0
_SLIPPAGE_THRESHOLD_BPS = 50
_SLIPPAGE_HISTOGRAM_BINS = 30
_CACHE_TTL_SECONDS = 60
_EQUITY_ROW_HEIGHT = 0.7
_DD_ROW_HEIGHT = 0.3
_SUBPLOT_SPACING = 0.05


# ---------------------------------------------------------------------------
# Cached data fetchers
# ---------------------------------------------------------------------------


@st.cache_data(ttl=_CACHE_TTL_SECONDS)
def _fetch_metrics(
    base_url: str,
    api_key: str,
    days: int,
    market_id: str,
) -> list[dict[str, object]]:
    return get_sandbox_metrics(base_url, api_key, days=days, market_id=market_id)


@st.cache_data(ttl=_CACHE_TTL_SECONDS)
def _fetch_gonogo(base_url: str, api_key: str) -> dict[str, object]:
    return get_sandbox_gonogo(base_url, api_key)


# ---------------------------------------------------------------------------
# Page render
# ---------------------------------------------------------------------------


def render(api: ApiClient) -> None:
    """Render the Sandbox Monitoring page."""
    st.title("Sandbox Monitoring")

    # Date range selector
    col_start, col_end = st.columns(2)
    today = datetime.now(tz=UTC).date()
    with col_start:
        start_date = st.date_input("Start date", value=today - timedelta(days=_DEFAULT_DAYS))
    with col_end:
        end_date = st.date_input("End date", value=today)

    days = max((end_date - start_date).days, 1)  # type: ignore[operator]

    # Fetch data
    try:
        metrics = _fetch_metrics(api._base_url, api._headers["X-API-Key"], days, "moex")
        gonogo = _fetch_gonogo(api._base_url, api._headers["X-API-Key"])
    except Exception:
        st.error("Cannot reach API server")
        return

    # Go/no-go verdict badge
    verdict = gonogo.get("verdict", "UNKNOWN")
    if verdict == "PROCEED":
        st.success(f"Go/No-Go: **{verdict}**")
    elif verdict == "DEFER":
        st.warning(f"Go/No-Go: **{verdict}**")
    else:
        st.error(f"Go/No-Go: **{verdict}**")

    if not metrics:
        st.info("No sandbox metrics available. Start sandbox mode to collect data.")
        return

    df = pd.DataFrame(metrics)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Section 1: Metrics table
    st.subheader("Metrics Table")
    display_cols = [
        "timestamp",
        "trade_count",
        "pnl_rub",
        "equity_rub",
        "fill_rate",
        "max_slippage_bps",
    ]
    available_cols = [c for c in display_cols if c in df.columns]
    st.dataframe(df[available_cols], use_container_width=True)

    # Section 2: Equity curve with drawdown overlay
    st.subheader("Equity Curve & Drawdown")
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[_EQUITY_ROW_HEIGHT, _DD_ROW_HEIGHT],
        vertical_spacing=_SUBPLOT_SPACING,
    )
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df["equity_rub"],
            name="Equity (RUB)",
            mode="lines",
        ),
        row=1,
        col=1,
    )
    if "drawdown_pct" in df.columns:
        dd_pct = df["drawdown_pct"].apply(lambda v: float(v) * 100 if v is not None else 0.0)
        fig.add_trace(
            go.Bar(
                x=df["timestamp"],
                y=dd_pct,
                name="Drawdown %",
                marker_color="red",
            ),
            row=2,
            col=1,
        )
    fig.update_layout(height=500, margin={"t": 30, "b": 30})
    st.plotly_chart(fig, use_container_width=True)

    # Section 3: Uptime — total successful cycles over period
    st.subheader("Uptime")
    if "uptime_cycles" in df.columns:
        total_cycles = int(df["uptime_cycles"].sum())
        # Only count market hours (MOEX: ~8.7h/day, 5 days/week)
        market_hours = days * 8.7 * 5 / 7
        uptime_pct = min(total_cycles / market_hours * 100, 100.0) if market_hours > 0 else 0.0
        st.metric("Uptime", f"{uptime_pct:.1f}%", delta=f"{total_cycles} cycles")
    else:
        st.metric("Uptime", "N/A")

    # Section 4: Fill rate gauge
    st.subheader("Fill Rate")
    fill_values = df["fill_rate"].dropna() if "fill_rate" in df.columns else pd.Series(dtype=float)
    avg_fill = float(fill_values.mean()) * 100 if len(fill_values) > 0 else 0.0
    delta_fill = avg_fill - _FILL_RATE_TARGET_PCT
    delta_label = f"{delta_fill:+.1f}% vs {_FILL_RATE_TARGET_PCT}% target"
    st.metric("Fill Rate", f"{avg_fill:.1f}%", delta=delta_label)

    # Section 5: Slippage histogram with 50bps threshold
    st.subheader("Slippage Distribution")
    if "max_slippage_bps" in df.columns:
        slip_df = df[df["max_slippage_bps"].notna()].copy()
        if len(slip_df) > 0:
            hist_fig = px.histogram(
                slip_df,
                x="max_slippage_bps",
                nbins=_SLIPPAGE_HISTOGRAM_BINS,
                title="Slippage Distribution (bps)",
            )
            hist_fig.add_vline(
                x=_SLIPPAGE_THRESHOLD_BPS,
                line_dash="dash",
                line_color="red",
                annotation_text="50bps threshold",
            )
            st.plotly_chart(hist_fig, use_container_width=True)
        else:
            st.info("No slippage data available yet.")
    else:
        st.info("No slippage data available yet.")


# Streamlit multipage auto-discovery requires module-level execution
if not st.session_state.get("authenticated", False):
    st.warning("Please log in on the main page first.")
    st.stop()

render(st.session_state.get("api"))  # type: ignore[arg-type]
