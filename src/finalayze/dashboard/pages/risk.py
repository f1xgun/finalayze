"""Risk page — circuit breakers, segment exposure, and emergency override."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from finalayze.dashboard.api_client import ApiClient

_LEVEL_BADGE = {
    0: "NORMAL",
    1: "CAUTION",
    2: "HALTED",
    3: "LIQUIDATE",
}
_HTTP_OK = 200


def render(api: ApiClient) -> None:
    """Render the Risk page."""
    st.title("Risk")

    try:
        risk = api.get("/api/v1/risk/status").json()
        exposure = api.get("/api/v1/risk/exposure").json()
    except Exception:
        st.error("Cannot reach API server")
        return

    # Circuit breaker status per market
    st.subheader("Circuit Breakers")
    markets = risk.get("markets", [])
    if markets:
        cols = st.columns(len(markets))
        for col, m in zip(cols, markets, strict=False):
            level = m.get("circuit_breaker_level", 0)
            badge = _LEVEL_BADGE.get(level, "UNKNOWN")
            since = m.get("level_since") or "—"
            col.metric(
                m.get("market_id", "?").upper(),
                badge,
                f"since {since}",
            )
            if level >= 2:  # noqa: PLR2004
                col.error(f"Circuit breaker active: {badge}")
    else:
        st.info("No circuit breaker data available.")

    if risk.get("cross_market_halted"):
        st.error("Cross-market circuit breaker TRIPPED — all markets halted")

    # Per-segment exposure bar chart
    segments = exposure.get("segments", [])
    if segments:
        st.subheader("Exposure by Segment")
        edf = pd.DataFrame(segments)
        if "segment_id" in edf.columns and "pct_of_portfolio" in edf.columns:
            chart_df = edf.set_index("segment_id")[["pct_of_portfolio"]]
            st.bar_chart(chart_df)
        st.dataframe(edf, use_container_width=True)

        total_pct = exposure.get("total_invested_pct", 0.0)
        st.metric("Total Invested", f"{total_pct:.1f}%")
    else:
        st.info("No exposure data available — no open positions.")

    # Emergency override form (requires explicit confirmation)
    st.subheader("Emergency Override")
    st.warning(
        "Use emergency override only in critical situations. "
        "This directly sets the circuit breaker level for a market."
    )
    with st.form("override_form"):
        market_id = st.selectbox("Market", ["us", "moex"])
        level = st.selectbox(
            "Override Level",
            [0, 1, 2, 3],
            format_func=lambda x: f"{x} — {_LEVEL_BADGE.get(x, str(x))}",
        )
        confirm_check = st.checkbox("I confirm this emergency override")
        submitted = st.form_submit_button("Apply Override")

        if submitted:
            if not confirm_check:
                st.error("Please check the confirmation box to proceed.")
            else:
                resp = api.post(
                    "/api/v1/risk/override",
                    json={"market_id": market_id, "level": level},
                )
                if resp.status_code == _HTTP_OK and resp.json().get("applied"):
                    badge = _LEVEL_BADGE.get(level, "?")
                    st.success(f"Override applied: {market_id.upper()} → level {level} ({badge})")
                else:
                    detail = resp.json().get("detail", f"HTTP {resp.status_code}")
                    st.error(f"Override failed: {detail}")
