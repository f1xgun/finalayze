"""SAA Target allocation page (Phase 81) -- read-only view of the active portfolio's allocation.

Shows the budget, risk profile, regime-tilted target weights, per-leg target notionals
(budget x weight), and the deposit mark. Read-only: no order placement from the web UI (that is the
token-gated CLI, scripts/run_rebalance.py); real-money go-live is a hard stop.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import streamlit as st

if TYPE_CHECKING:
    from finalayze.dashboard.api_client import ApiClient


def _build_leg_rows(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Shape the API ``legs`` payload into table rows (pure -- the page's testable logic)."""
    rows: list[dict[str, Any]] = []
    for asset_class, leg in data.get("legs", {}).items():
        rows.append(
            {
                "Asset class": asset_class,
                "Symbol": leg.get("symbol") or "(deposit - manual)",
                "Target weight": leg.get("weight"),
                "Target notional (RUB)": leg.get("target_notional_rub"),
            }
        )
    return rows


def render(api: ApiClient) -> None:
    """Render the SAA Target page."""
    st.title("SAA Target Allocation")

    try:
        data = api.saa_target_allocation()
    except Exception:  # connection failure -- never crash the dashboard
        st.error("Cannot reach API server")
        return

    if not data or "legs" not in data:
        st.info(
            "No active SAA portfolio. Create one via "
            "`scripts/create_saa_portfolio.py`, then refresh."
        )
        return

    col1, col2, col3 = st.columns(3)
    col1.metric("Risk profile", str(data.get("risk_profile", "-")).title())
    col2.metric("Budget (RUB)", data.get("budget_rub", "-"))
    col3.metric("Deposit value (RUB)", data.get("deposit_current_notional_rub", "-"))
    st.caption(f"As of {data.get('as_of', '-')} · portfolio {data.get('portfolio_id', '-')}")

    st.dataframe(_build_leg_rows(data), use_container_width=True, hide_index=True)
    st.caption(
        "Read-only. Run a rebalance via `scripts/run_rebalance.py` (sandbox preview by default); "
        "real-money go-live is a hard stop."
    )
