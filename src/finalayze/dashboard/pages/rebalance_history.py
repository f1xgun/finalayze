"""SAA Rebalance History page (Phase 83) -- read-only view of persisted rebalance runs.

Shows each run's date, mode, reconciliation status, fill rate, and per-leg requested/filled outcomes
from the Phase 82 audit trail. Read-only: no order placement (that is the token-gated CLI,
scripts/run_rebalance.py); real-money go-live is a hard stop.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import streamlit as st

if TYPE_CHECKING:
    from finalayze.dashboard.api_client import ApiClient


def _when(run: dict[str, Any]) -> str:
    """A compact 'YYYY-MM-DD HH:MM:SS' from the run's ISO created_at."""
    return str(run.get("created_at", ""))[:19].replace("T", " ")


def _build_run_rows(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Shape the API runs payload into summary table rows (pure -- the page's testable logic)."""
    return [
        {
            "When": _when(run),
            "As of": run.get("as_of"),
            "Mode": run.get("mode"),
            "Status": run.get("status"),
            "Fill rate": run.get("fill_rate"),
            "Legs": len(run.get("orders", [])),
        }
        for run in data.get("runs", [])
    ]


def render(api: ApiClient) -> None:
    """Render the SAA Rebalance History page."""
    st.title("SAA Rebalance History")

    try:
        data = api.saa_rebalance_runs()
    except Exception:  # connection failure -- never crash the dashboard
        st.error("Cannot reach API server")
        return

    if not data or "runs" not in data:
        st.info(
            "No active SAA portfolio. Create one via "
            "`scripts/create_saa_portfolio.py`, then refresh."
        )
        return

    runs = data["runs"]
    if not runs:
        st.info("No rebalance runs yet. Run `scripts/run_rebalance.py` (sandbox), then refresh.")
        return

    st.caption(f"Portfolio {data.get('portfolio_id', '-')} · {len(runs)} recent run(s)")
    st.dataframe(_build_run_rows(data), use_container_width=True, hide_index=True)

    for run in runs:
        with st.expander(f"{_when(run)} · {run.get('status')} · fill {run.get('fill_rate')}"):
            order_rows = [
                {
                    "Asset class": order.get("asset_class"),
                    "Symbol": order.get("symbol"),
                    "Side": order.get("side"),
                    "Requested": order.get("requested_qty"),
                    "Filled": order.get("filled_qty"),
                    "Status": order.get("status"),
                    "Reason": order.get("reason") or "",
                }
                for order in run.get("orders", [])
            ]
            st.dataframe(order_rows, use_container_width=True, hide_index=True)

    st.caption(
        "Read-only audit trail. Run a rebalance via `scripts/run_rebalance.py` (sandbox); "
        "real-money go-live is a hard stop."
    )


# Streamlit runs this module AS A SCRIPT via st.navigation / page.run() (app.py registers it by
# path), so rendering only happens through a module-level render() call -- mirror every other page
# (Phase 81 CR-01). app.py puts the shared ApiClient in session_state before page.run().
if (_api := st.session_state.get("api")) is not None:
    render(_api)
