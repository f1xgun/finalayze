"""Decision History page -- reverse-chronological list of decided experiments with reasoning."""

from __future__ import annotations

from typing import TYPE_CHECKING

import streamlit as st

from finalayze.core.experiment_manager import ExperimentManager
from finalayze.core.schemas import ExperimentState, ExperimentStatus

if TYPE_CHECKING:
    from finalayze.dashboard.api_client import ApiClient

_TERMINAL_STATUSES = {
    ExperimentStatus.ACCEPTED,
    ExperimentStatus.REJECTED,
    ExperimentStatus.INCONCLUSIVE,
}


def render(api: ApiClient) -> None:  # noqa: ARG001
    """Render the Decision History page."""
    st.title("Decision History")

    mgr = ExperimentManager()
    exp_ids = mgr.list_experiments()

    decided: list[ExperimentState] = []
    for eid in exp_ids:
        try:
            exp = mgr.read_experiment(eid)
        except FileNotFoundError:
            continue
        if exp.status in _TERMINAL_STATUSES:
            decided.append(exp)

    # Sort reverse chronological by created date
    decided.sort(key=lambda e: e.created, reverse=True)

    if not decided:
        st.info("No decisions recorded yet.")
        return

    for exp in decided:
        with st.expander(f"[{exp.status.value.upper()}] {exp.experiment_id} ({exp.created})"):
            st.write(f"**Hypothesis:** {exp.hypothesis}")
            st.write(
                f"**Criteria:** {exp.success_criteria.metric} "
                f"{exp.success_criteria.operator} {exp.success_criteria.threshold}"
            )
            st.write(f"**Verdict:** {exp.verdict}")
            st.write(f"**Reasoning:** {exp.reasoning}")

            if exp.results:
                last_result = exp.results[-1]
                col1, col2 = st.columns(2)
                with col1:
                    sharpe = float(last_result.metrics.get("sharpe", 0))
                    st.metric("Sharpe", f"{sharpe:.4f}")
                with col2:
                    pf = float(last_result.metrics.get("profit_factor", 0))
                    st.metric("Profit Factor", f"{pf:.4f}")
