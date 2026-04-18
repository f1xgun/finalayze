"""Portfolio page — equity curve, positions, and performance metrics."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from finalayze.dashboard.api_client import ApiClient

# D-10 threshold: a position is "at risk" when distance_atr < 0.5 (red bucket)
_RED_BUCKET_THRESHOLD = 0.5


def _count_at_risk(positions: list[dict[str, object]] | None) -> int:
    """Count positions in the red ATR bucket (D-04 / D-10 / D-11).

    I-07: Defensively reject bool values for distance_atr. In Python,
    `isinstance(True, (int, float))` is True, so a stray bool would be
    treated as 0 or 1 and misclassified. Exclude bools explicitly.
    """
    if not positions:
        return 0
    count = 0
    for p in positions:
        da = p.get("distance_atr") if isinstance(p, dict) else None
        if isinstance(da, (int, float)) and not isinstance(da, bool) and da < _RED_BUCKET_THRESHOLD:
            count += 1
    return count


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

    # Summary metrics row — detect currency from markets
    markets = portfolio.get("markets", [])
    has_moex = any(m.get("market_id") == "moex" for m in markets) if markets else False
    currency_label = (
        "RUB" if has_moex and not any(m.get("market_id") == "us" for m in markets) else "USD"
    )
    currency_sym = "\u20bd" if currency_label == "RUB" else "$"

    total_equity = float(portfolio.get("total_equity_usd") or 0.0)
    daily_pnl = float(portfolio.get("daily_pnl_usd") or 0.0)
    daily_pnl_pct = float(portfolio.get("daily_pnl_pct") or 0.0)
    sharpe = perf.get("sharpe_30d")
    max_dd = perf.get("max_drawdown_pct")
    total_cash = float(portfolio.get("total_cash_usd") or 0.0)
    cash_pct = (total_cash / total_equity * 100) if total_equity > 0 else 0.0

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric(f"Total Equity ({currency_label})", f"{currency_sym}{total_equity:,.2f}")
    pnl_label = f"{currency_sym}{daily_pnl:,.2f}"
    col2.metric(f"Daily P&L ({currency_label})", pnl_label, f"{daily_pnl_pct:.2f}%")
    col3.metric("Cash %", f"{cash_pct:.1f}%")
    col4.metric("Sharpe (30d)", f"{sharpe:.2f}" if sharpe is not None else "N/A")
    col5.metric("Max Drawdown", f"{(float(max_dd) if max_dd else 0.0) * 100:.1f}%")

    # STOP-04 D-11: mini-badge for positions at risk
    positions_list = positions_data.get("positions", []) or []
    at_risk = _count_at_risk(positions_list)
    total_positions = len(positions_list)
    dot = "\U0001f534" if at_risk > 0 else "\U0001f7e2"  # red / green circle
    col6.metric(f"{dot} Positions at risk", f"{at_risk}/{total_positions}")
    st.page_link("pages/positions.py", label="\u2192 See details")

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

    # Per-market equity table — rename _usd columns to actual currency
    if markets and isinstance(markets, list):
        st.subheader("By Market")
        mdf = pd.DataFrame(markets)
        _col_rename = {
            "equity_usd": f"equity_{currency_label.lower()}",
            "cash_usd": f"cash_{currency_label.lower()}",
            "positions_value_usd": f"positions_value_{currency_label.lower()}",
            "daily_pnl_usd": f"daily_pnl_{currency_label.lower()}",
        }
        mdf = mdf.rename(columns={k: v for k, v in _col_rename.items() if k in mdf.columns})
        st.dataframe(mdf, use_container_width=True)

    # Open positions heatmap — rename _usd columns
    pos_list = positions_data.get("positions", [])
    if pos_list and isinstance(pos_list, list):
        st.subheader("Open Positions")
        pdf = pd.DataFrame(pos_list)
        _pos_rename = {
            "market_value_usd": f"market_value_{currency_label.lower()}",
            "unrealized_pnl_usd": f"unrealized_pnl_{currency_label.lower()}",
        }
        pdf = pdf.rename(columns={k: v for k, v in _pos_rename.items() if k in pdf.columns})
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


# Called by st.navigation/page.run(); app.py guarantees "api" is set at runtime
if (_api := st.session_state.get("api")) is not None:
    render(_api)
