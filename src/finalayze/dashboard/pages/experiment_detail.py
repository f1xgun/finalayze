"""Experiment Detail page -- deep-dive view with debate context, criteria, charts, comparison."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from finalayze.core.debate_manager import DebateManager
from finalayze.core.experiment_manager import ExperimentManager

if TYPE_CHECKING:
    from finalayze.dashboard.api_client import ApiClient

_STATUS_DISPLAY: dict[str, tuple[str, str]] = {
    "accepted": ("success", "ACCEPTED"),
    "rejected": ("error", "REJECTED"),
    "inconclusive": ("warning", "INCONCLUSIVE"),
    "running": ("info", "RUNNING"),
    "pending": ("info", "PENDING"),
    "completed": ("info", "COMPLETED"),
}

_METRICS_OF_INTEREST = ["sharpe", "profit_factor", "max_drawdown", "win_rate", "total_trades"]


def render(api: ApiClient) -> None:  # noqa: ARG001
    """Render the Experiment Detail page."""
    st.title("Experiment Detail")

    experiment_id = st.query_params.get("experiment_id", "")
    if not experiment_id:
        st.warning("No experiment selected. Go to the Experiment List to pick one.")
        return

    mgr = ExperimentManager()
    try:
        exp = mgr.read_experiment(experiment_id)
    except FileNotFoundError:
        st.error(f"Experiment '{experiment_id}' not found.")
        return

    # -- Status header --
    fn_name, label = _STATUS_DISPLAY.get(exp.status.value, ("info", exp.status.value.upper()))
    getattr(st, fn_name)(f"**{exp.experiment_id}** — {label}")

    # -- Hypothesis --
    st.subheader("Hypothesis")
    st.write(exp.hypothesis)

    # -- Success Criteria --
    st.subheader("Success Criteria")
    st.write(f"**Metric:** {exp.success_criteria.metric}")
    st.write(f"**Threshold:** {exp.success_criteria.operator} {exp.success_criteria.threshold}")

    # -- Debate Context --
    st.subheader("Debate Context")
    if exp.debate_id:
        try:
            dm = DebateManager()
            debate = dm.read_debate(exp.debate_id)
            st.write(f"**Topic:** {debate.topic}")
            st.write(f"**Agents:** {', '.join(debate.agents)}")
            if debate.resolution:
                st.write(f"**Resolution:** {debate.resolution}")
            if debate.arbiter_report:
                st.write("**Arbiter Report:**")
                st.json(debate.arbiter_report.model_dump())
        except FileNotFoundError:
            st.warning(f"Linked debate '{exp.debate_id}' not found.")
    else:
        st.info("No debate linked to this experiment.")

    # -- Backtest Results: A/B/AB comparison chart --
    st.subheader("Backtest Results")
    if not exp.results:
        st.info("No backtest results yet.")
    else:
        fig = go.Figure()
        for result in exp.results:
            vals = [float(result.metrics.get(m, 0)) for m in _METRICS_OF_INTEREST]
            fig.add_trace(go.Bar(name=result.run_name, x=_METRICS_OF_INTEREST, y=vals))
        fig.update_layout(
            barmode="group",
            height=400,
            title="Metric Comparison",
            xaxis_title="Metric",
            yaxis_title="Value",
        )
        st.plotly_chart(fig, use_container_width=True)

        # -- Comparison Table --
        st.subheader("Comparison Table")
        rows: list[dict[str, object]] = []
        for result in exp.results:
            row: dict[str, object] = {
                "Run": result.run_name,
                "Iteration": result.iteration_name,
            }
            for m in _METRICS_OF_INTEREST:
                row[m] = float(result.metrics.get(m, 0))
            rows.append(row)
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True)

    # -- Verdict --
    if exp.verdict:
        st.subheader("Verdict")
        st.write(f"**Decision:** {exp.verdict}")
        st.write(f"**Reasoning:** {exp.reasoning}")

    # -- Preset Overrides --
    if exp.preset_overrides:
        st.subheader("Preset Overrides")
        st.json(exp.preset_overrides)
