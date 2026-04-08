"""Experiments List page -- filterable table of all experiments with status and key metrics."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd
import streamlit as st

from finalayze.core.experiment_manager import ExperimentManager
from finalayze.core.schemas import ExperimentState

if TYPE_CHECKING:
    from finalayze.dashboard.api_client import ApiClient

_HYPOTHESIS_TRUNCATE = 80


def _extract_metric(results: list[Any], key: str) -> float | None:
    """Extract a metric from the last result's metrics dict, or None."""
    if not results:
        return None
    last = results[-1]
    val = last.metrics.get(key)
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _format_criteria(exp: ExperimentState) -> str:
    """Format success criteria as a short string like 'sharpe >= 0.15'."""
    sc = exp.success_criteria
    return f"{sc.metric} {sc.operator} {sc.threshold}"


def _load_experiments(mgr: ExperimentManager) -> list[ExperimentState]:
    """Load all experiments, skipping broken files."""
    experiments: list[ExperimentState] = []
    for eid in mgr.list_experiments():
        try:
            experiments.append(mgr.read_experiment(eid))
        except FileNotFoundError:
            continue
    return experiments


def _build_dataframe(experiments: list[ExperimentState]) -> pd.DataFrame:
    """Build a display DataFrame from experiment states."""
    rows: list[dict[str, Any]] = []
    for exp in experiments:
        hypothesis = exp.hypothesis
        if len(hypothesis) > _HYPOTHESIS_TRUNCATE:
            hypothesis = hypothesis[:_HYPOTHESIS_TRUNCATE] + "..."
        rows.append(
            {
                "ID": exp.experiment_id,
                "Status": exp.status.value,
                "Hypothesis": hypothesis,
                "Created": exp.created,
                "Criteria": _format_criteria(exp),
                "Runs": len(exp.results),
                "Sharpe": _extract_metric(exp.results, "sharpe"),
                "PF": _extract_metric(exp.results, "profit_factor"),
            }
        )
    return pd.DataFrame(rows)


def render(api: ApiClient) -> None:  # noqa: ARG001
    """Render the Experiments List page."""
    st.title("Experiment Lab")

    if st.button("Refresh"):
        st.rerun()

    mgr = ExperimentManager()
    experiments = _load_experiments(mgr)

    if not experiments:
        st.info("No experiments found. Create experiments via ExperimentManager.")
        return

    # -- Filters --
    col1, col2 = st.columns(2)
    with col1:
        unique_statuses = sorted({e.status.value for e in experiments})
        status_filter = st.selectbox("Filter by status", ["All", *unique_statuses])
    with col2:
        search_text = st.text_input("Search hypothesis", "")

    filtered = experiments
    if status_filter != "All":
        filtered = [e for e in filtered if e.status.value == status_filter]
    if search_text:
        lower_search = search_text.lower()
        filtered = [e for e in filtered if lower_search in e.hypothesis.lower()]

    if not filtered:
        st.warning("No experiments match the current filters.")
        return

    # -- DataFrame display --
    df = _build_dataframe(filtered)
    gradient_cols = [c for c in ["Sharpe", "PF"] if c in df.columns and df[c].notna().any()]
    if gradient_cols:
        st.dataframe(
            df.style.background_gradient(subset=gradient_cols, cmap="RdYlGn"),
            use_container_width=True,
        )
    else:
        st.dataframe(df, use_container_width=True)

    # -- Navigation buttons --
    for exp in filtered:
        if st.button(f"View {exp.experiment_id}", key=f"view_{exp.experiment_id}"):
            st.query_params["experiment_id"] = exp.experiment_id
            st.switch_page("pages/experiment_detail.py")
