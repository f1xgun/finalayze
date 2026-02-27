"""Portfolio page — equity curve, positions, and performance metrics."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from finalayze.dashboard.api_client import ApiClient


def render(api: ApiClient) -> None:
    """Render the Portfolio page."""
    st.title("Portfolio")

    # Fetch all data
    try:
        portfolio = api.get("/api/v1/portfolio").json()
        perf = api.get("/api/v1/portfolio/performance").json()
        history = api.get("/api/v1/portfolio/history").json()
        positions_data = api.get("/api/v1/portfolio/positions").json()
    except Exception:
        st.error("Cannot reach API server")
        return

    # Summary metrics row
    total_equity = float(portfolio.get("total_equity_usd") or 0.0)
    daily_pnl_usd = float(portfolio.get("daily_pnl_usd") or 0.0)
    daily_pnl_pct = float(portfolio.get("daily_pnl_pct") or 0.0)
    sharpe = perf.get("sharpe_30d")
    max_dd = perf.get("max_drawdown_pct")
    total_cash = float(portfolio.get("total_cash_usd") or 0.0)
    cash_pct = (total_cash / total_equity * 100) if total_equity > 0 else 0.0

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Equity (USD)", f"${total_equity:,.2f}")
    col2.metric("Daily P&L (USD)", f"${daily_pnl_usd:,.2f}", f"{daily_pnl_pct:.2f}%")
    col3.metric("Cash %", f"{cash_pct:.1f}%")
    col4.metric("Sharpe (30d)", f"{sharpe:.2f}" if sharpe is not None else "N/A")
    col5.metric("Max Drawdown", f"{(float(max_dd) if max_dd else 0.0) * 100:.1f}%")

    # Equity curve with drawdown shading
    snapshots = history.get("snapshots", [])
    if snapshots and isinstance(snapshots, list):
        st.subheader("Equity Curve")
        df = pd.DataFrame(snapshots)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        if "market_id" in df.columns and "equity" in df.columns:
            df_pivot = df.pivot_table(index="timestamp", columns="market_id", values="equity")
            st.line_chart(df_pivot)
        elif "equity" in df.columns:
            st.line_chart(df.set_index("timestamp")["equity"])

        if "drawdown_pct" in df.columns:
            st.subheader("Drawdown (%)")
            if "market_id" in df.columns:
                df_dd = df.pivot_table(
                    index="timestamp", columns="market_id", values="drawdown_pct"
                )
                st.area_chart(df_dd)
            else:
                st.area_chart(df.set_index("timestamp")["drawdown_pct"])
    else:
        st.info("No historical data yet — equity curve will appear after the first trading cycle.")

    # Per-market equity table
    markets = portfolio.get("markets", [])
    if markets and isinstance(markets, list):
        st.subheader("By Market")
        mdf = pd.DataFrame(markets)
        st.dataframe(mdf, use_container_width=True)

    # Open positions heatmap
    pos_list = positions_data.get("positions", [])
    if pos_list and isinstance(pos_list, list):
        st.subheader("Open Positions")
        pdf = pd.DataFrame(pos_list)
        if "unrealized_pnl_pct" in pdf.columns:
            st.dataframe(
                pdf.style.background_gradient(
                    subset=["unrealized_pnl_pct"],
                    cmap="RdYlGn",
                ),
                use_container_width=True,
            )
        else:
            st.dataframe(pdf, use_container_width=True)
    else:
        st.info("No open positions.")
