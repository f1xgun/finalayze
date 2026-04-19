"""Portfolio page — equity curve, positions, and performance metrics."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from finalayze.dashboard.api_client import ApiClient

# D-10 threshold: a position is "at risk" when distance_atr < 0.5 (red bucket)
_RED_BUCKET_THRESHOLD = 0.5

# D-13: Plotly subplot stack — mirrors sandbox.py:106-136 exactly
_EQUITY_ROW_HEIGHT = 0.7
_DD_ROW_HEIGHT = 0.3
_SUBPLOT_SPACING = 0.12

# Compact-currency thresholds: render large values as "₽2.61M" instead of
# "₽2,611,415.32" so 6-column metric tiles do not truncate at narrow widths.
_COMPACT_M_THRESHOLD = 1_000_000
_COMPACT_K_THRESHOLD = 10_000

# D-15: period dropdown options. The "All" sentinel maps to a 10-year window —
# generous upper bound that effectively disables the cutoff for live trading
# horizons without forcing the API to special-case "no limit".
_DEFAULT_PERIOD_LABEL = "30d"
_PERIOD_OPTIONS: dict[str, int] = {
    "7d": 7,
    "30d": 30,
    "90d": 90,
    "All": 3650,
}


def _format_compact_currency(value: float, sym: str) -> str:
    """Compact currency format — keeps long RUB values inside narrow metric tiles."""
    abs_v = abs(value)
    if abs_v >= _COMPACT_M_THRESHOLD:
        return f"{sym}{value / _COMPACT_M_THRESHOLD:.2f}M"
    if abs_v >= _COMPACT_K_THRESHOLD:
        return f"{sym}{value / 1_000:.1f}K"
    return f"{sym}{value:,.2f}"


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

    # D-15: period dropdown drives ?days= on /portfolio/history AND /portfolio/performance.
    # Default 30d per Phase 55 D-13 / Phase 56 D-13.
    period_label = st.selectbox(
        "Period",
        options=list(_PERIOD_OPTIONS.keys()),
        index=list(_PERIOD_OPTIONS.keys()).index(_DEFAULT_PERIOD_LABEL),
        key="portfolio_period",
    )
    days = _PERIOD_OPTIONS[period_label]

    # Fetch all data — period dropdown plumbs ?days= into history + performance only
    # (portfolio summary + positions are point-in-time snapshots, no window).
    try:
        portfolio = api.get("/api/v1/portfolio").json()
        perf = api.get("/api/v1/portfolio/performance", params={"days": days}).json()
        history = api.get("/api/v1/portfolio/history", params={"days": days}).json()
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
    col1.metric(
        f"Total Equity ({currency_label})",
        _format_compact_currency(total_equity, currency_sym),
    )
    pnl_label = _format_compact_currency(daily_pnl, currency_sym)
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

    # D-13 / D-14: equity curve + drawdown via Plotly subplot stack — replaces the
    # legacy Streamlit-native chart blocks. One go.Scatter per market_id on the
    # equity row, red filled drawdown area on the bottom row.
    snapshots = history.get("snapshots", [])
    if snapshots and isinstance(snapshots, list):
        st.subheader("Equity Curve & Drawdown")
        df = pd.DataFrame(snapshots)
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            row_heights=[_EQUITY_ROW_HEIGHT, _DD_ROW_HEIGHT],
            vertical_spacing=_SUBPLOT_SPACING,
        )

        # D-14: one Scatter trace per market_id (us, moex, moex_bonds), color-coded.
        if "market_id" in df.columns and "equity" in df.columns:
            df_pivot = df.pivot_table(index="timestamp", columns="market_id", values="equity")
            for market_id in df_pivot.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df_pivot.index,
                        y=df_pivot[market_id],
                        name=str(market_id),
                        mode="lines+markers",
                    ),
                    row=1,
                    col=1,
                )
        elif "equity" in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df["timestamp"],
                    y=df["equity"],
                    name="Equity",
                    mode="lines+markers",
                ),
                row=1,
                col=1,
            )

        # D-13: drawdown row — red filled area below zero (per-market when available).
        # Drawdown values are stored as positive fractions (peak-relative loss); we
        # negate them for visual "below zero line" semantics on the chart.
        if "drawdown_pct" in df.columns:
            if "market_id" in df.columns:
                df_dd = df.pivot_table(
                    index="timestamp",
                    columns="market_id",
                    values="drawdown_pct",
                )
                for market_id in df_dd.columns:
                    dd_pct = df_dd[market_id].apply(
                        lambda v: -float(v) * 100 if v is not None and not pd.isna(v) else 0.0
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=df_dd.index,
                            y=dd_pct,
                            name=f"DD {market_id}",
                            mode="lines+markers",
                            fill="tozeroy",
                            line={"color": "red"},
                            marker={"color": "red", "size": 6},
                            opacity=0.4,
                            showlegend=False,
                        ),
                        row=2,
                        col=1,
                    )
            else:
                dd_pct = df["drawdown_pct"].apply(
                    lambda v: -float(v) * 100 if v is not None and not pd.isna(v) else 0.0
                )
                fig.add_trace(
                    go.Scatter(
                        x=df["timestamp"],
                        y=dd_pct,
                        name="Drawdown %",
                        mode="lines",
                        fill="tozeroy",
                        line={"color": "red"},
                    ),
                    row=2,
                    col=1,
                )

        equity_axis_label = f"Equity ({currency_label})"
        fig.update_yaxes(title_text=equity_axis_label, row=1, col=1)
        fig.update_yaxes(title_text="Drawdown %", row=2, col=1)
        fig.update_layout(
            height=600,
            margin={"t": 30, "b": 30, "l": 60, "r": 20},
            hovermode="x unified",
            hoverdistance=-1,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No historical data yet — equity curve will appear after the first trading cycle.")

    # D-16: second metrics row — extended performance tiles consuming Plan 56-04's
    # PerformanceResponse fields (sortino_30d, win_rate, profit_factor,
    # avg_win_loss_ratio, n_snapshots). Each tile renders "—" when the underlying
    # value is null per Phase 54 D-03 ("null is the signal").
    sortino = perf.get("sortino_30d")
    win_rate = perf.get("win_rate")
    profit_factor = perf.get("profit_factor")
    avg_wl = perf.get("avg_win_loss_ratio")
    n_snap = perf.get("n_snapshots") or perf.get("n_observations") or 0
    n_trades = perf.get("n_paired_trades") or 0

    col_a, col_b, col_c, col_d, col_e = st.columns(5)
    col_a.metric("Sortino (30d)", f"{sortino:.2f}" if sortino is not None else "—")
    col_b.metric(
        "Win rate",
        f"{win_rate * 100:.1f}%" if win_rate is not None else "—",
        delta=None if n_trades == 0 else f"{n_trades} trades",
    )
    col_c.metric(
        "Profit factor",
        f"{profit_factor:.2f}" if profit_factor is not None else "—",
    )
    col_d.metric("Avg win/loss", f"{avg_wl:.2f}" if avg_wl is not None else "—")
    col_e.metric("Snapshots", f"{n_snap}")

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
