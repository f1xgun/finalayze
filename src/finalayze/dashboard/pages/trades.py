"""Trades page — filterable trade log and slippage analytics."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from finalayze.dashboard.api_client import ApiClient


def render(api: ApiClient) -> None:
    """Render the Trades page."""
    st.title("Trades")

    # Filters
    col1, col2, col3 = st.columns(3)
    market_filter = col1.selectbox("Market", ["all", "us", "moex"])
    symbol_filter = col2.text_input("Symbol (optional)")
    limit = col3.slider("Limit", 10, 500, 100)

    params: dict[str, object] = {"limit": limit}
    if market_filter != "all":
        params["market"] = market_filter
    if symbol_filter:
        params["symbol"] = symbol_filter

    try:
        trades_resp = api.get("/api/v1/trades", params=params).json()
        analytics = api.get("/api/v1/trades/analytics").json()
    except Exception:
        st.error("Cannot reach API server")
        return

    trade_list = trades_resp.get("trades", [])
    total = trades_resp.get("total", 0)

    st.caption(f"Showing {len(trade_list)} of {total} trades")

    if trade_list:
        df = pd.DataFrame(trade_list)
        st.dataframe(df, use_container_width=True)

        # Slippage scatter chart
        if "slippage_bps" in df.columns and df["slippage_bps"].notna().any():
            st.subheader("Slippage by Time of Day")
            scatter_df = df.dropna(subset=["slippage_bps"]).copy()
            if "timestamp" in scatter_df.columns:
                scatter_df["timestamp"] = pd.to_datetime(scatter_df["timestamp"])
            st.scatter_chart(
                scatter_df,
                x="timestamp",
                y="slippage_bps",
                color="market_id" if "market_id" in scatter_df.columns else None,
            )
    else:
        st.info("No trades recorded yet.")

    # Analytics row
    st.subheader("Trade Analytics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Trades", analytics.get("total_trades", 0))
    avg_slip = analytics.get("avg_slippage_bps")
    col2.metric("Avg Slippage (bps)", f"{avg_slip:.1f}" if avg_slip is not None else "N/A")
    avg_lat = analytics.get("avg_fill_latency_ms")
    col3.metric("Avg Fill Latency (ms)", f"{avg_lat:.1f}" if avg_lat is not None else "N/A")
    rej_rate = analytics.get("rejection_rate_pct")
    col4.metric(
        "Rejection Rate",
        f"{rej_rate:.1f}%" if rej_rate is not None else "N/A",
    )
