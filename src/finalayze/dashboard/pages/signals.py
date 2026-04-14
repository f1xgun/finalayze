"""Signals page — strategy performance matrix and recent signals table."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from finalayze.dashboard.api_client import ApiClient


def render(api: ApiClient) -> None:
    """Render the Signals page."""
    st.title("Signals")

    try:
        strategies_raw = api.get("/api/v1/strategies/performance")
        signals_raw = api.get("/api/v1/signals")
    except Exception:
        st.error("Cannot reach API server")
        return

    if signals_raw.status_code == 501:  # noqa: PLR2004
        st.info("Signals endpoint not yet implemented. Data will appear after strategy cycles run.")
        return

    strategies_resp = strategies_raw.json() if strategies_raw.status_code == 200 else {}  # noqa: PLR2004
    signals_resp = signals_raw.json()

    strategies = strategies_resp.get("strategies", [])
    signals = signals_resp.get("signals", [])

    # Strategy performance matrix
    st.subheader("Strategy Performance Matrix")
    if strategies:
        sdf = pd.DataFrame(strategies)
        _strategy_cols = [
            "strategy",
            "market_id",
            "win_rate",
            "profit_factor",
            "trades_today",
            "last_signal_at",
        ]
        display_cols = [c for c in _strategy_cols if c in sdf.columns]
        sdf_display = sdf[display_cols] if display_cols else sdf

        gradient_cols = [c for c in ["win_rate", "profit_factor"] if c in sdf_display.columns]
        if gradient_cols:
            st.dataframe(
                sdf_display.style.background_gradient(
                    subset=gradient_cols,
                    cmap="RdYlGn",
                ),
                use_container_width=True,
            )
        else:
            st.dataframe(sdf_display, use_container_width=True)
    else:
        st.info("No strategy performance data yet.")

    # Recent signals table
    st.subheader("Recent Signals")
    if signals:
        sig_df = pd.DataFrame(signals)
        _signal_cols = [
            "symbol",
            "strategy",
            "market_id",
            "segment_id",
            "direction",
            "confidence",
            "created_at",
        ]
        display_cols = [c for c in _signal_cols if c in sig_df.columns]
        sig_display = sig_df[display_cols] if display_cols else sig_df
        st.dataframe(sig_display, use_container_width=True)
    else:
        st.info("No signals recorded yet.")


# Streamlit multipage auto-discovery requires module-level execution
if not st.session_state.get("authenticated", False):
    st.warning("Please log in on the main page first.")
    st.stop()

render(st.session_state.get("api"))  # type: ignore[arg-type]
