"""System Status page — page 1 of the Finalayze operator dashboard."""

from __future__ import annotations

import streamlit as st

from finalayze.dashboard.api_client import ApiClient

_MODE_LABEL = {
    "real": "REAL",
    "test": "TEST",
    "sandbox": "SANDBOX",
    "debug": "DEBUG",
}

_HTTP_OK = 200


def _render_components(components: dict[str, object]) -> None:
    """Render component health columns."""
    comp_cols = st.columns(len(components))
    items = list(components.items())
    for col, (name, comp_status) in zip(comp_cols, items, strict=False):
        icon = "OK" if comp_status == "ok" else "ERROR"
        col.metric(name.upper(), f"{icon}")
        if comp_status != "ok":
            col.error(str(comp_status))


def _render_errors(api: ApiClient) -> None:
    """Render recent error list from the API."""
    st.subheader("Recent Errors")
    try:
        errors = api.get("/api/v1/system/errors").json()
        if not isinstance(errors, list):
            errors = []
    except Exception:
        errors = []

    if not errors:
        st.success("No recent errors")
        return

    for err in errors[:10]:
        ts = err.get("timestamp", "?")
        comp = err.get("component", "?")
        msg = err.get("message", "?")
        header = f"[{ts}] {comp}: {msg}"
        if len(header) > 100:  # noqa: PLR2004
            header = header[:97] + "..."
        with st.expander(header):
            tb = err.get("traceback_excerpt", "")
            if tb:
                st.code(str(tb))
            else:
                st.write("No traceback available")


def render(api: ApiClient) -> None:
    """Render the System Status page."""
    st.title("System Status")

    # Health + mode badge
    try:
        health = api.get("/api/v1/health").json()
    except Exception:
        st.error("Cannot reach API server")
        return

    mode = health.get("mode", "unknown")
    status = health.get("status", "unknown")

    col_mode, col_status = st.columns(2)
    col_mode.metric("Work Mode", _MODE_LABEL.get(str(mode), str(mode).upper()))
    col_status.metric("Overall Status", str(status).upper())

    # Component health table
    st.subheader("Component Health")
    components = health.get("components", {})
    if components and isinstance(components, dict):
        _render_components(components)
    else:
        st.info("No component data.")

    # Recent errors
    _render_errors(api)

    # Mode switcher with two-step confirmation for REAL
    st.subheader("Change Work Mode")
    new_mode = st.selectbox("New mode", ["debug", "sandbox", "test", "real"])
    confirm = ""
    if new_mode == "real":
        st.warning("Switching to REAL mode will execute live trades. Enter the confirm token.")
        confirm = st.text_input("Confirm token (required for REAL mode)")

    if st.button("Apply Mode Change"):
        payload: dict[str, object] = {"mode": new_mode}
        if confirm:
            payload["confirm_token"] = confirm
        resp = api.post("/api/v1/system/mode", json=payload)
        if resp.status_code == _HTTP_OK:
            st.success(f"Mode changed to {new_mode.upper()}")
            st.rerun()
        else:
            detail = resp.json().get("detail", f"HTTP {resp.status_code}")
            st.error(f"Failed: {detail}")
