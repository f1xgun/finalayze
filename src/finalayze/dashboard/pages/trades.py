"""Trades page -- filterable trade log + TRAD-02 analytics metrics row.

D-13: default period 30 days.
D-14: period dropdown 7d / 30d / 90d / All (shared with Signals page).
TRAD-02: analytics row displays win_rate, avg_win, avg_loss, profit_factor.
TRAD-01 (D-07): null slippage_bps renders as "—".
"""

from __future__ import annotations

import math

import pandas as pd
import streamlit as st

from finalayze.dashboard.api_client import ApiClient

_PERIOD_OPTIONS: dict[str, int | None] = {
    "7d": 7,
    "30d": 30,
    "90d": 90,
    "All": None,
}
_DEFAULT_PERIOD_LABEL = "30d"


def _format_slippage(value: object) -> str:
    """Render `slippage_bps` for the trades DataFrame.

    Returns "—" for None, NaN, or non-numeric; otherwise a 2-decimal float
    (preserves sign). Per D-07 UI convention: null is "—" not "N/A" or "0".

    pandas converts JSON `null` in a numeric column to `float("nan")`, so we
    treat NaN as null too.
    """
    if value is None:
        return "—"
    try:
        f = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return "—"
    if math.isnan(f):
        return "—"
    return f"{f:.2f}"


def render(api: ApiClient) -> None:
    """Render the Trades page."""
    st.title("Trades")

    period_label = st.selectbox(
        "Period",
        list(_PERIOD_OPTIONS.keys()),
        index=list(_PERIOD_OPTIONS.keys()).index(_DEFAULT_PERIOD_LABEL),
    )
    period_days = _PERIOD_OPTIONS[period_label]

    # First st.columns() call -- 3 cols for filter row.
    # Tests MUST mock st.columns with side_effect so the 6-col call below sees a fresh list.
    col1, col2, col3 = st.columns(3)
    market_filter = col1.selectbox("Market", ["all", "us", "moex"])
    symbol_filter = col2.text_input("Symbol (optional)")
    limit = col3.slider("Limit", 10, 500, 100)

    trades_params: dict[str, object] = {"limit": limit}
    analytics_params: dict[str, object] = {}
    if market_filter != "all":
        trades_params["market"] = market_filter
        analytics_params["market"] = market_filter
    if symbol_filter:
        trades_params["symbol"] = symbol_filter
    if period_days is not None:
        analytics_params["period"] = period_days
        # /trades list endpoint may gain ?period in a future plan; forward-compat.
        trades_params["period"] = period_days

    try:
        trades_resp = api.get("/api/v1/trades", params=trades_params).json()
        analytics = api.get("/api/v1/trades/analytics", params=analytics_params).json()
    except Exception:
        st.error("Cannot reach API server")
        return

    trade_list = trades_resp.get("trades", [])
    total = trades_resp.get("total", 0)
    st.caption(f"Showing {len(trade_list)} of {total} trades")

    if trade_list:
        df = pd.DataFrame(trade_list)
        # TRAD-01 D-07: render null slippage as "—"; non-null as "1.23".
        if "slippage_bps" in df.columns:
            df["slippage_bps"] = df["slippage_bps"].apply(_format_slippage)
        st.dataframe(df, use_container_width=True)

        # Scatter over the un-stringified slippage_bps values only (dropna before str-format).
        numeric_slip = pd.DataFrame(trade_list)
        if "slippage_bps" in numeric_slip.columns:
            numeric_slip = numeric_slip.dropna(subset=["slippage_bps"]).copy()
            if not numeric_slip.empty:
                st.subheader("Slippage by Time of Day")
                if "timestamp" in numeric_slip.columns:
                    numeric_slip["timestamp"] = pd.to_datetime(numeric_slip["timestamp"], format="ISO8601")
                st.scatter_chart(
                    numeric_slip,
                    x="timestamp",
                    y="slippage_bps",
                    color="market_id" if "market_id" in numeric_slip.columns else None,
                )
    else:
        st.info("No trades recorded yet.")

    # TRAD-02: extended analytics row.
    # SECOND st.columns() call -- 6 cols for metrics. Test fixtures MUST use side_effect
    # so this call receives a fresh 6-element list distinct from the filter row above.
    st.subheader("Trade Analytics")
    mcols = st.columns(6)
    mcols[0].metric("Total Trades", analytics.get("total_trades", 0))
    wr = analytics.get("win_rate")
    mcols[1].metric("Win Rate", f"{wr * 100:.1f}%" if wr is not None else "—")
    aw = analytics.get("avg_win")
    mcols[2].metric("Avg Win", f"{aw:.2f}" if aw is not None else "—")
    al = analytics.get("avg_loss")
    mcols[3].metric("Avg Loss", f"{al:.2f}" if al is not None else "—")
    pf = analytics.get("profit_factor")
    mcols[4].metric("Profit Factor", f"{pf:.2f}" if pf is not None else "—")
    sl = analytics.get("avg_slippage_bps")
    mcols[5].metric("Avg Slippage bps", f"{sl:.1f}" if sl is not None else "—")


# Called by st.navigation/page.run(); app.py guarantees "api" is in session_state.
if (_api := st.session_state.get("api")) is not None:
    render(_api)
