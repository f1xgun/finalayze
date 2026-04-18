"""Experiment Lab — unified experiment lifecycle dashboard.

Single page with tabs: List, Detail, Decision History.
Navigation via session_state instead of query_params to avoid multipage issues.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from finalayze.core.debate_manager import DebateManager
from finalayze.core.experiment_manager import ExperimentManager
from finalayze.core.schemas import ExperimentState, ExperimentStatus

if TYPE_CHECKING:
    from finalayze.dashboard.api_client import ApiClient

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_STATUS_COLORS: dict[str, str] = {
    "accepted": "#28a745",
    "rejected": "#dc3545",
    "inconclusive": "#ffc107",
    "running": "#17a2b8",
    "pending": "#6c757d",
    "completed": "#007bff",
}

_STATUS_EMOJI: dict[str, str] = {
    "accepted": "✅",
    "rejected": "❌",
    "inconclusive": "⚠️",
    "running": "🔄",
    "pending": "⏳",
    "completed": "📊",
}

_KEY_METRICS = ["sharpe", "profit_factor", "max_drawdown", "win_rate", "total_trades"]

_TERMINAL_STATUSES = {
    ExperimentStatus.ACCEPTED,
    ExperimentStatus.REJECTED,
    ExperimentStatus.INCONCLUSIVE,
}

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def _load_all(mgr: ExperimentManager) -> list[ExperimentState]:
    experiments: list[ExperimentState] = []
    for eid in mgr.list_experiments():
        try:
            experiments.append(mgr.read_experiment(eid))
        except FileNotFoundError:
            continue
    return experiments


def _metric_val(results: list[Any], key: str) -> float | None:
    if not results:
        return None
    val = results[-1].metrics.get(key)
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _fmt_criteria(exp: ExperimentState) -> str:
    sc = exp.success_criteria
    return f"{sc.metric} {sc.operator} {sc.threshold}"


# ---------------------------------------------------------------------------
# Tab 1: List
# ---------------------------------------------------------------------------


def _render_list(experiments: list[ExperimentState]) -> None:
    """Render the experiments list view."""
    # Summary metrics row
    total = len(experiments)
    accepted = sum(1 for e in experiments if e.status == ExperimentStatus.ACCEPTED)
    rejected = sum(1 for e in experiments if e.status == ExperimentStatus.REJECTED)
    pending = sum(1 for e in experiments if e.status == ExperimentStatus.PENDING)
    running = sum(1 for e in experiments if e.status == ExperimentStatus.RUNNING)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total", total)
    m2.metric("Accepted", accepted)
    m3.metric("Rejected", rejected)
    m4.metric("Pending / Running", f"{pending} / {running}")

    st.divider()

    # Filters
    col1, col2 = st.columns([1, 2])
    with col1:
        statuses = sorted({e.status.value for e in experiments})
        status_filter = st.selectbox("Status", ["All", *statuses], key="list_status")
    with col2:
        search = st.text_input("🔍 Search hypothesis", key="list_search")

    filtered = experiments
    if status_filter != "All":
        filtered = [e for e in filtered if e.status.value == status_filter]
    if search:
        q = search.lower()
        filtered = [e for e in filtered if q in e.hypothesis.lower()]

    if not filtered:
        st.info("No experiments match the filters.")
        return

    # Experiment cards
    for exp in filtered:
        emoji = _STATUS_EMOJI.get(exp.status.value, "")
        color = _STATUS_COLORS.get(exp.status.value, "#6c757d")
        sharpe = _metric_val(exp.results, "sharpe")
        pf = _metric_val(exp.results, "profit_factor")

        with st.container(border=True):
            top1, top2, top3 = st.columns([3, 1, 1])
            with top1:
                st.markdown(
                    f"**{exp.experiment_id}** "
                    f"<span style='background-color:{color};color:white;"
                    f"padding:2px 8px;border-radius:4px;font-size:0.8em'>"
                    f"{emoji} {exp.status.value.upper()}</span>",
                    unsafe_allow_html=True,
                )
                st.caption(exp.hypothesis[:120])
            with top2:
                if sharpe is not None:
                    st.metric("Sharpe", f"{sharpe:.3f}")
                else:
                    st.metric("Sharpe", "—")
            with top3:
                if pf is not None:
                    st.metric("PF", f"{pf:.2f}")
                else:
                    st.metric("PF", "—")

            bot1, bot2, bot3 = st.columns([2, 1, 1])
            with bot1:
                st.caption(f"Criteria: {_fmt_criteria(exp)} · Runs: {len(exp.results)}")
            with bot2:
                st.caption(f"Created: {exp.created}")
            with bot3:
                if st.button("View details →", key=f"view_{exp.experiment_id}"):
                    st.session_state["selected_experiment"] = exp.experiment_id
                    st.session_state["experiment_tab"] = "Detail"
                    st.rerun()


# ---------------------------------------------------------------------------
# Tab 2: Detail
# ---------------------------------------------------------------------------


def _render_detail(experiments: list[ExperimentState]) -> None:
    """Render the experiment detail view."""
    exp_ids = [e.experiment_id for e in experiments]

    if not exp_ids:
        st.info("No experiments available.")
        return

    selected = st.session_state.get("selected_experiment", exp_ids[0])
    if selected not in exp_ids:
        selected = exp_ids[0]

    chosen = st.selectbox(
        "Select experiment",
        exp_ids,
        index=exp_ids.index(selected),
        key="detail_select",
    )
    st.session_state["selected_experiment"] = chosen

    exp = next(e for e in experiments if e.experiment_id == chosen)

    # Status badge
    emoji = _STATUS_EMOJI.get(exp.status.value, "")
    color = _STATUS_COLORS.get(exp.status.value, "#6c757d")
    st.markdown(
        f"### {emoji} {exp.experiment_id} "
        f"<span style='background-color:{color};color:white;"
        f"padding:3px 12px;border-radius:6px;font-size:0.7em'>"
        f"{exp.status.value.upper()}</span>",
        unsafe_allow_html=True,
    )

    # Hypothesis & Criteria
    st.markdown(f"> {exp.hypothesis}")
    st.caption(f"**Criteria:** {_fmt_criteria(exp)} · **Created:** {exp.created}")

    st.divider()

    # Two columns: Debate Context + Verdict
    left, right = st.columns(2)

    with left:
        st.markdown("#### 💬 Debate Context")
        if exp.debate_id:
            try:
                dm = DebateManager()
                debate = dm.read_debate(exp.debate_id)
                st.markdown(f"**Topic:** {debate.topic}")
                st.markdown(f"**Agents:** {', '.join(debate.agents)}")
                if debate.resolution:
                    st.markdown(f"**Resolution:** {debate.resolution}")
            except FileNotFoundError:
                st.warning(f"Debate '{exp.debate_id}' not found.")
        else:
            st.caption("No debate linked.")

    with right:
        st.markdown("#### 🏆 Verdict")
        if exp.verdict:
            st.markdown(f"**Decision:** {exp.verdict}")
            st.markdown(f"**Reasoning:** {exp.reasoning}")
        else:
            st.caption("No verdict yet.")

    st.divider()

    # Backtest results
    st.markdown("#### 📈 Backtest Results")
    if not exp.results:
        st.info(
            "No backtest results yet. Run `scripts/run_iteration.py --hypothesis "
            f"{exp.experiment_id}` to generate."
        )
        return

    # A/B/AB comparison chart
    fig = go.Figure()
    available_metrics = set()
    for result in exp.results:
        available_metrics.update(result.metrics.keys())

    chart_metrics = [m for m in _KEY_METRICS if m in available_metrics]
    if not chart_metrics:
        chart_metrics = list(available_metrics)[:6]

    colors = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A"]
    for i, result in enumerate(exp.results):
        vals = [float(result.metrics.get(m, 0)) for m in chart_metrics]
        fig.add_trace(
            go.Bar(
                name=result.run_name,
                x=chart_metrics,
                y=vals,
                marker_color=colors[i % len(colors)],
            )
        )

    fig.update_layout(
        barmode="group",
        height=350,
        margin={"t": 30, "b": 40, "l": 50, "r": 20},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "center", "x": 0.5},
        xaxis_title="",
        yaxis_title="Value",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Comparison table with deltas
    st.markdown("#### Comparison Table")
    rows: list[dict[str, object]] = []
    for result in exp.results:
        row: dict[str, object] = {"Run": result.run_name}
        for m in chart_metrics:
            row[m] = float(result.metrics.get(m, 0))
        rows.append(row)

    df = pd.DataFrame(rows)

    # Add delta row if multiple runs
    if len(exp.results) >= 2:  # noqa: PLR2004
        first = rows[0]
        last = rows[-1]
        delta_row: dict[str, object] = {"Run": "Δ (last - first)"}
        for m in chart_metrics:
            v_first = float(first.get(m, 0))
            v_last = float(last.get(m, 0))
            delta_row[m] = round(v_last - v_first, 4)
        delta_df = pd.DataFrame([delta_row])
        df = pd.concat([df, delta_df], ignore_index=True)

    st.dataframe(df, use_container_width=True)

    # Preset overrides
    if exp.preset_overrides:
        with st.expander("Preset Overrides"):
            st.json(exp.preset_overrides)


# ---------------------------------------------------------------------------
# Tab 3: Decision History
# ---------------------------------------------------------------------------


def _render_history(experiments: list[ExperimentState]) -> None:
    """Render the decision history view."""
    decided = [e for e in experiments if e.status in _TERMINAL_STATUSES]
    decided.sort(key=lambda e: e.created, reverse=True)

    if not decided:
        st.info(
            "No decisions recorded yet. Experiments must reach "
            "ACCEPTED, REJECTED, or INCONCLUSIVE status."
        )
        return

    # Summary
    accepted = sum(1 for e in decided if e.status == ExperimentStatus.ACCEPTED)
    rejected = sum(1 for e in decided if e.status == ExperimentStatus.REJECTED)
    inconclusive = sum(1 for e in decided if e.status == ExperimentStatus.INCONCLUSIVE)

    c1, c2, c3 = st.columns(3)
    c1.metric("✅ Accepted", accepted)
    c2.metric("❌ Rejected", rejected)
    c3.metric("⚠️ Inconclusive", inconclusive)

    st.divider()

    for exp in decided:
        emoji = _STATUS_EMOJI.get(exp.status.value, "")
        color = _STATUS_COLORS.get(exp.status.value, "#6c757d")
        _header = (
            f"{emoji} **{exp.experiment_id}** — "
            f"<span style='color:{color}'>{exp.status.value.upper()}</span> "
            f"({exp.created})"
        )

        with st.expander(
            f"{emoji} {exp.experiment_id} — {exp.status.value.upper()} ({exp.created})"
        ):
            st.markdown(f"**Hypothesis:** {exp.hypothesis}")
            st.markdown(f"**Criteria:** {_fmt_criteria(exp)}")
            st.divider()
            st.markdown(f"**Verdict:** {exp.verdict}")
            st.markdown(f"**Reasoning:** {exp.reasoning}")

            if exp.results:
                st.divider()
                last = exp.results[-1]
                metric_cols = st.columns(min(len(last.metrics), 4))
                for i, (k, v) in enumerate(list(last.metrics.items())[:4]):
                    try:
                        metric_cols[i].metric(k.replace("_", " ").title(), f"{float(v):.4f}")
                    except (ValueError, TypeError):
                        metric_cols[i].metric(k.replace("_", " ").title(), str(v))

            if st.button("View full detail →", key=f"hist_{exp.experiment_id}"):
                st.session_state["selected_experiment"] = exp.experiment_id
                st.session_state["experiment_tab"] = "Detail"
                st.rerun()


# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------


def render(api: ApiClient) -> None:  # noqa: ARG001
    """Render the Experiment Lab page."""
    st.title("🧪 Experiment Lab")
    st.caption("Hypothesis → Backtest → Verdict — full experiment lifecycle")

    if st.button("🔄 Refresh data"):
        st.rerun()

    mgr = ExperimentManager()
    experiments = _load_all(mgr)

    if not experiments:
        st.info(
            "No experiments found.\n\n"
            "Create one with:\n"
            "```python\n"
            "from finalayze.core.experiment_manager import ExperimentManager\n"
            "from finalayze.core.schemas import SuccessCriteria\n\n"
            "mgr = ExperimentManager()\n"
            "mgr.create_experiment('my-experiment', 'My hypothesis', "
            "SuccessCriteria(metric='sharpe', operator='>=', threshold=0.1))\n"
            "```"
        )
        return

    # Navigation — use experiment_tab as source of truth, consume it before widget
    decided_count = sum(1 for e in experiments if e.status in _TERMINAL_STATUSES)
    views = ["List", "Detail", "History"]

    # If a button set experiment_tab, use it and clear
    nav_override = st.session_state.pop("experiment_tab", None)
    if nav_override and nav_override in views:
        st.session_state["experiment_view_radio"] = nav_override

    view = st.radio(
        "View",
        views,
        horizontal=True,
        format_func=lambda v: {
            "List": f"📋 List ({len(experiments)})",
            "Detail": "🔬 Detail",
            "History": f"📜 History ({decided_count})",
        }[v],
        key="experiment_view_radio",
    )

    st.divider()

    if view == "List":
        _render_list(experiments)
    elif view == "Detail":
        _render_detail(experiments)
    else:
        _render_history(experiments)


# Called by st.navigation/page.run(); app.py guarantees "api" is set at runtime
if (_api := st.session_state.get("api")) is not None:
    render(_api)
