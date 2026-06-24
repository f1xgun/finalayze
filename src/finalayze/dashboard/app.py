"""Streamlit dashboard entry point — auth gate, navigation, and frame.

Run with:
    streamlit run src/finalayze/dashboard/app.py

Requires .streamlit/secrets.toml with:
    password = "your-dashboard-password"
    api_key = "your-api-key"
    api_url = "http://localhost:8000"
"""

from __future__ import annotations

import streamlit as st

from finalayze.dashboard.api_client import ApiClient

st.set_page_config(
    page_title="Finalayze",
    page_icon="F",
    layout="wide",
    initial_sidebar_state="expanded",
)

_PASSWORD = st.secrets.get("password", "")

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    st.title("Finalayze — Login")
    if not _PASSWORD:
        st.error("Password not configured in secrets.toml")
        st.stop()
    pwd = st.text_input("Password", type="password")
    if st.button("Login"):
        if pwd == _PASSWORD:
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("Invalid password")
    st.stop()

# Authenticated — build shared API client (available to all pages via session_state)
_api = ApiClient(
    base_url=st.secrets.get("api_url", "http://localhost:8000"),
    api_key=st.secrets.get("api_key", ""),
)
st.session_state["api"] = _api

# Explicit page navigation (replaces deprecated pages/ directory convention)
page = st.navigation(
    [
        st.Page("pages/system_status.py", title="System Status", icon=":material/monitor_heart:"),
        st.Page("pages/signals.py", title="Signals", icon=":material/ssid_chart:"),
        st.Page("pages/trades.py", title="Trades", icon=":material/swap_horiz:"),
        st.Page("pages/risk.py", title="Risk", icon=":material/shield:"),
        st.Page("pages/portfolio.py", title="Portfolio", icon=":material/account_balance:"),
        st.Page("pages/saa_allocation.py", title="SAA Target", icon=":material/donut_large:"),
        st.Page("pages/rebalance_history.py", title="Rebalances", icon=":material/history:"),
        st.Page("pages/positions.py", title="Positions", icon=":material/track_changes:"),
        st.Page("pages/alerts.py", title="Alerts", icon=":material/notifications:"),
        st.Page("pages/sandbox.py", title="Sandbox", icon=":material/science:"),
        st.Page("pages/experiments_list.py", title="Experiments", icon=":material/biotech:"),
    ],
)

page.run()
