"""PERF-02: source-introspection tests for dashboard/pages/portfolio.py.

These tests assert source-file invariants on the rewritten portfolio page that
arrives in Plan 56-05. They use ``inspect.getsource(portfolio.render)`` (with a
fallback to ``inspect.getsource(portfolio)``) to inspect the literal source of
the page module. This mirrors the Phase 54 dashboard test pattern: cheap
checks that lock the rewrite's contract without standing up Streamlit.

The rewrite must:

1. Replace ``st.line_chart`` / ``st.area_chart`` with a Plotly subplot stack
   (``make_subplots`` + ``go.Scatter`` + ``st.plotly_chart``) — D-13.
2. Add a period dropdown (``7d`` / ``30d`` / ``90d`` / ``All``) at the top of
   the page that drives ``?days=`` on /portfolio/history AND
   /portfolio/performance — D-15.
3. Render one ``go.Scatter`` trace per ``market_id`` on the equity row — D-14.
4. Render a second ``st.columns(...)`` row exposing the new
   ``sortino_30d`` / ``win_rate`` / ``profit_factor`` /
   ``avg_win_loss_ratio`` / ``n_snapshots`` tiles produced by Plan 56-04 — D-16.
"""

from __future__ import annotations

import inspect

from finalayze.dashboard.pages import portfolio


def _render_source() -> str:
    """Return the source of ``portfolio.render`` (or the whole module on fallback)."""
    try:
        return inspect.getsource(portfolio.render)
    except (OSError, TypeError):
        return inspect.getsource(portfolio)


def test_uses_plotly_subplots() -> None:
    """D-13: equity-curve section uses Plotly subplots, not Streamlit-native charts."""
    src = _render_source()

    assert "make_subplots" in src, (
        "Expected Plotly subplot stack via `make_subplots(...)` in render; "
        "see sandbox.py:106-136 for the exact precedent."
    )
    assert "plotly_chart" in src, (
        "Expected the Plotly figure to be rendered via st.plotly_chart(...)."
    )
    assert "st.line_chart" not in src, (
        "st.line_chart still present in render — D-13 requires Plotly rewrite."
    )
    assert "st.area_chart" not in src, (
        "st.area_chart still present in render — D-13 requires Plotly rewrite."
    )


def test_period_dropdown_drives_query_params() -> None:
    """D-15: period dropdown (7d / 30d / 90d / All) drives ?days= on both endpoints."""
    src = _render_source()

    assert "st.selectbox" in src or "st.radio" in src, (
        "Expected a period selector widget (st.selectbox or st.radio) at the top "
        "of the Portfolio page."
    )
    # The dropdown value must end up as a `days` query param somewhere — either
    # as `params={"days": ...}` or as a `?days=` literal in the path.
    assert "days" in src, (
        "Expected the period selector to plumb a `days` query param through to "
        "the api_client (params={'days': ...} or ?days=N in the path)."
    )

    for option in ("7d", "30d", "90d", "All"):
        assert option in src, (
            f"Period option {option!r} not found in render source — D-15 requires "
            "the four-option dropdown."
        )


def test_per_market_scatter_traces() -> None:
    """D-14: one go.Scatter trace per market_id on the equity (top) subplot."""
    src = _render_source()

    assert "go.Scatter" in src, (
        "Expected go.Scatter traces in the rewritten chart — Plotly precedent from sandbox.py."
    )
    # Per-market loop indicator — accept any of the natural pivot/iteration
    # patterns the planner is likely to use.
    per_market_loop = (
        "for market_id in" in src
        or "for col in df_pivot.columns" in src
        or "for col in df_dd" in src
        or "for market_id, " in src
    )
    assert per_market_loop, (
        "Expected a per-market loop emitting one Scatter per market_id — "
        "single hardcoded trace is not D-14 compliant."
    )
    assert "row=1, col=1" in src, "Equity traces must land in the top subplot (row=1, col=1)."


def test_extended_metrics_row() -> None:
    """D-16: second metrics row reads PerformanceResponse fields from Plan 56-04."""
    src = _render_source()

    perf_fields = [
        "sortino_30d",
        "win_rate",
        "profit_factor",
        "avg_win_loss_ratio",
        "n_snapshots",
    ]
    found = [field for field in perf_fields if field in src]
    assert len(found) >= 4, (
        f"Expected at least 4 of the new PerformanceResponse fields in the "
        f"second metrics row, found {found}. D-16 requires the new tiles to "
        "consume Plan 56-04's response schema."
    )

    # The existing 6-column metrics row at line 61 stays; the new D-16 row adds
    # a SECOND st.columns(...) call below the chart.
    columns_calls = src.count("st.columns(")
    assert columns_calls >= 2, (
        f"Expected at least 2 st.columns(...) calls in render (existing 6-col "
        f"row + new D-16 row), found {columns_calls}."
    )
