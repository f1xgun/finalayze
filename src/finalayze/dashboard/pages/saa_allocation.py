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


def _build_benchmark_rows(cert: dict[str, Any]) -> list[dict[str, Any]]:
    """Shape the cert per-regime stories + full-window row into a table (pure, testable, P87)."""
    rows: list[dict[str, Any]] = [
        {
            "Regime": story.get("unit_label"),
            "Period": f"{story.get('window_start')} -> {story.get('window_end')}",
            "Allocation Sharpe": f"{story.get('allocation_sharpe'):.4f}",
            "Best-naive Sharpe": f"{story.get('best_naive_sharpe'):.4f}",
            "Verdict": story.get("unit_verdict"),
        }
        for story in cert.get("regime_stories", [])
    ]
    rows.append(
        {
            "Regime": "full window",
            "Period": "-",
            "Allocation Sharpe": f"{cert.get('alloc_sharpe_full'):.4f}",
            "Best-naive Sharpe": f"{cert.get('best_naive_sharpe_full'):.4f}",
            "Verdict": cert.get("full_verdict"),
        }
    )
    return rows


def render_cert_decision(api: ApiClient) -> None:
    """Render the binding-cert verdict + deposit-anchor benchmark beside the recommendation (P87).

    Every value comes from the committed cert via the API; HARD_FAIL is shown as HARD_FAIL (never
    softened), and the honest "when do risk assets pay" framing carries no fabricated threshold.
    """
    st.divider()
    st.subheader("Binding verdict -- deposit vs bonds vs equity (honest measurement)")
    try:
        cert = api.saa_cert_decision()
    except Exception:  # connection failure -- never crash the dashboard
        st.error("Cannot reach API server")
        return
    if not cert or "phase_verdict" not in cert:
        st.info(
            "No committed allocation-gate cert. Run `scripts/run_allocation_gate.py --live` to "
            "produce one, then refresh."
        )
        return

    # 1. Verdict banner -- HARD_FAIL red (not softened); a tighten-rescued pass is amber, not green.
    verdict = str(cert.get("phase_verdict", ""))
    headline = str(cert.get("headline", ""))
    if verdict == "HARD_FAIL":
        st.error(headline)
    elif "TIGHTEN" in verdict:
        st.warning(headline)  # PASS_AFTER_TIGHTEN: a rescue, never visually a clean pass
    else:
        st.success(headline)

    # 2. Benchmark comparison -- per-regime rows + the full-window row, numbers from the cert.
    st.dataframe(_build_benchmark_rows(cert), use_container_width=True, hide_index=True)
    if cert.get("high_rate_caveat"):
        st.caption(str(cert["high_rate_caveat"]))  # verbatim honesty caveat from the cert

    # 3. "When do risk assets pay" -- honest-qualitative, no fabricated threshold.
    if cert.get("when_framing"):
        st.info(str(cert["when_framing"]))

    # 4. Escalation + N=1 caveat -- rendered only when present (both sourced from the cert).
    bits: list[str] = []
    if cert.get("escalation"):
        bits.append(f"Escalation: `{cert['escalation']}`")
    if cert.get("n1_caveat"):
        bits.append("N=1: single observed easing cycle -- suggestive, not robust.")
    if bits:
        st.warning(" - ".join(bits))

    # 5. Provenance footer (staleness visible).
    st.caption(
        f"Cert: {cert.get('cert_timestamp', '-')} - sha {str(cert.get('git_sha', ''))[:8]} - "
        f"{cert.get('staleness_days', '?')}d ago"
    )


# Streamlit runs this module AS A SCRIPT via st.navigation / page.run() (app.py registers it by
# path), so rendering only happens through a module-level render() call -- mirror every other page.
# app.py puts the shared ApiClient in session_state before page.run().
if (_api := st.session_state.get("api")) is not None:
    render(_api)
    render_cert_decision(_api)
