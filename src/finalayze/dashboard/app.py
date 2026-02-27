"""Streamlit dashboard entry point — auth gate and home page.

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

# Authenticated — build shared API client stored in session state
_api = ApiClient(
    base_url=st.secrets.get("api_url", "http://localhost:8000"),
    api_key=st.secrets.get("api_key", ""),
)
st.session_state["api"] = _api

# Home page: system overview summary
st.title("Finalayze — Operator Dashboard")
st.markdown("Use the sidebar to navigate between pages.")

# Quick health summary
try:
    health = _api.get("/api/v1/health").json()
    mode = health.get("mode", "unknown")
    status = health.get("status", "unknown")

    col1, col2 = st.columns(2)
    col1.metric("Mode", mode.upper())
    col2.metric("Status", status.upper())

    components = health.get("components", {})
    if components:
        st.subheader("Component Health")
        comp_cols = st.columns(len(components))
        items = list(components.items())
        for col, (name, comp_status) in zip(comp_cols, items, strict=False):
            icon = "OK" if comp_status == "ok" else "ERR"
            col.metric(name.upper(), f"{icon}: {comp_status}")
except Exception:
    st.warning("Could not reach API — check that the API server is running.")
