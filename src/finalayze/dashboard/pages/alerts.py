"""Alerts page — paginated, filterable Telegram-replay dashboard (Phase 57-04).

ALRT-03 D-15..D-18:
- D-15: threaded rendering. anomaly_llm child rows are indented below their
  anomaly_raw parent using the literal ↳ (U+21B3) glyph.
- D-16: filters — alert_type (multi), symbol, priority — wired through
  ApiClient.list_alerts kwargs.
- D-17: child rows indented inline (no separate column) so the chronological
  ordering is preserved while the parent/child link remains visible.
- D-18: pagination via ``page`` + ``page_size`` (default 50, max 200);
  caption shows ``Page N of M``.

The page consumes ``ApiClient.list_alerts(...)`` (also added in this plan).
"""

from __future__ import annotations

import math

import pandas as pd
import streamlit as st

from finalayze.dashboard.api_client import ApiClient

_PAGE_SIZE = 50

# D-16: discrete option lists for the multi-select / select widgets. Keep
# alert_type values aligned to the names emitted by Phase 57-02/03 alerter
# methods (anomaly_raw / anomaly_llm / stop_loss / signal / generic).
_ALERT_TYPES: list[str] = [
    "anomaly_raw",
    "anomaly_llm",
    "stop_loss",
    "signal",
    "generic",
]

# Priority filter values must match the uppercase AlertPriority.name form
# (Phase 57-02 revision Mi5: stored value is the IntEnum .name string).
_PRIORITY_OPTIONS: list[str] = ["All", "CRITICAL", "IMPORTANT", "INFO"]

# Indent prefix for anomaly_llm rows (U+21B3 — DOWNWARDS ARROW WITH TIP RIGHTWARDS).
# The literal glyph (NOT the \u escape) so the dashboard renders the arrow
# directly in the message column.
_CHILD_INDENT = "  ↳ "


def render(api: ApiClient) -> None:
    """Render the Alerts page (paginated, filterable Telegram replay)."""
    st.title("Alerts")

    # ── D-16 filter row ─────────────────────────────────────────────────────
    # Use st.<widget>(...) directly (rather than ``col.<widget>(...)``) so the
    # widgets remain patchable in unit tests via ``patch.object(st, ...)``.
    # The 3-column layout is purely visual; widgets are placed in order.
    st.columns(3)
    selected_types = st.multiselect("Alert type", options=_ALERT_TYPES)
    selected_symbol = st.text_input("Symbol (optional)")
    selected_priority = st.selectbox("Priority", options=_PRIORITY_OPTIONS)

    # ── D-18 pagination ─────────────────────────────────────────────────────
    page = st.number_input("Page", min_value=1, value=1, step=1)

    # Build params for the /api/v1/alerts call. Only include filters when the
    # user has actually selected a value (avoids ``priority=All`` arriving at
    # the API layer where it would not match any stored AlertPriority.name).
    params: dict[str, object] = {"page": int(page), "page_size": _PAGE_SIZE}
    if selected_types:
        params["alert_type"] = selected_types
    if selected_symbol:
        params["symbol"] = selected_symbol
    if selected_priority and selected_priority != "All":
        params["priority"] = selected_priority

    # ── Fetch ────────────────────────────────────────────────────────────────
    try:
        payload = api.get("/api/v1/alerts", params=params).json()
    except Exception:
        st.error("Cannot reach API server")
        return

    alerts: list[dict[str, object]] = (
        payload.get("alerts", []) if isinstance(payload, dict) else []
    )
    total = int(payload.get("total", 0) if isinstance(payload, dict) else 0)
    page_size = int(
        payload.get("page_size", _PAGE_SIZE) if isinstance(payload, dict) else _PAGE_SIZE
    )
    current_page = int(payload.get("page", page) if isinstance(payload, dict) else page)

    # ── D-15/D-17 threaded rendering ────────────────────────────────────────
    # Build display_rows: parent rows first, with each anomaly_llm child
    # immediately following its anomaly_raw parent and indented via ↳.
    display_rows = _build_threaded_rows(alerts)

    # Pagination caption — shown above the table for visibility.
    pages_total = max(1, math.ceil(total / page_size)) if page_size else 1
    st.caption(
        f"Page {current_page} of {pages_total} — showing {len(alerts)} of {total} alerts"
    )

    if not display_rows:
        st.info("No alerts match the current filters.")
        return

    df = pd.DataFrame(display_rows)
    st.dataframe(df, use_container_width=True)


def _build_threaded_rows(
    alerts: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Re-order alerts so anomaly_llm children sit directly below their parent.

    Algorithm: index parents by id, then walk the source list once. For each
    parent (or non-anomaly row), append it. For each child (parent_id set),
    skip it on the first walk and instead emit it inline behind its parent.
    Rows whose parent is NOT visible in this page are appended at the end so
    they're never silently dropped.
    """
    by_id: dict[str, dict[str, object]] = {}
    for a in alerts:
        aid = str(a.get("id", ""))
        if aid:
            by_id[aid] = a

    children_of: dict[str, list[dict[str, object]]] = {}
    parents: list[dict[str, object]] = []
    orphans: list[dict[str, object]] = []
    for a in alerts:
        parent_id = a.get("parent_id")
        if parent_id and str(parent_id) in by_id:
            children_of.setdefault(str(parent_id), []).append(a)
        elif parent_id and str(parent_id) not in by_id:
            # Parent not on this page (filtered out or different page).
            orphans.append(_indent_row(a))
        else:
            parents.append(a)

    out: list[dict[str, object]] = []
    for p in parents:
        out.append(p)
        out.extend(_indent_row(child) for child in children_of.get(str(p.get("id", "")), []))
    out.extend(orphans)
    return out


def _indent_row(row: dict[str, object]) -> dict[str, object]:
    """Return a copy of ``row`` with the message indented under its parent."""
    indented = dict(row)
    msg = indented.get("message", "")
    indented["message"] = f"{_CHILD_INDENT}{msg}"
    return indented


# Called by st.navigation/page.run(); app.py guarantees "api" is set at runtime
if (_api := st.session_state.get("api")) is not None:
    render(_api)
