# Phase 35: Experiment Lab UI - Context

**Gathered:** 2026-04-08
**Status:** Ready for planning

<domain>
## Phase Boundary

This phase delivers a Streamlit web interface for the experiment lifecycle: list page, detail page with charts and A/B/AB comparison, and decision history page. Scope: 3 Streamlit pages in the existing dashboard, data loading from ExperimentManager/results files, Plotly visualizations. Does NOT include new REST API endpoints or real-time features.

</domain>

<decisions>
## Implementation Decisions

### Page Layout & Navigation
- New pages in `src/finalayze/dashboard/pages/` — matches existing Streamlit multipage app pattern
- Add "Experiments" section to existing sidebar with 3 sub-pages (List, Detail, History)
- Page files: `experiments_list.py`, `experiment_detail.py`, `decision_history.py`
- Data loading: read from ExperimentManager + results/ files directly (no REST API needed for Streamlit)

### Charts & Visualization
- Chart library: Plotly (already in project for dashboard) — interactive charts
- A/B/AB comparison: side-by-side metric bars + delta table (matches SC-2 "A vs B vs A+B comparison table")
- Status indicators: colored badges — green=ACCEPTED, red=REJECTED, yellow=INCONCLUSIVE, blue=RUNNING, gray=PENDING
- Backtest result charts: equity curves from result JSON, drawdown overlay — same style as existing iteration charts

### Data Loading & Filtering
- List page filtering: status dropdown + text search on hypothesis — minimal but functional
- Detail page navigation: Streamlit `st.query_params` with experiment_id for deep linking
- Decision history: most recent verdict first (reverse chronological)
- No auto-refresh — manual refresh button sufficient for experiment lifecycle timescales

### Claude's Discretion
- Exact column widths and Streamlit layout ratios
- Chart color palette beyond status badges
- Loading state / spinner behavior
- Empty state messaging

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `src/finalayze/dashboard/` — existing Streamlit multipage app with sidebar navigation
- `src/finalayze/dashboard/pages/` — existing page files to follow as pattern
- `src/finalayze/core/experiment_manager.py` — ExperimentManager CRUD (Phase 34)
- `src/finalayze/core/schemas.py` — ExperimentState, ExperimentResult Pydantic models
- Plotly already in dependencies for existing dashboard charts

### Established Patterns
- Streamlit pages use `st.set_page_config()` + sidebar navigation
- Data loaded via manager classes, not REST API
- Charts use Plotly `go.Figure` with consistent styling
- Status displayed with `st.metric()` and `st.dataframe()`

### Integration Points
- `src/finalayze/dashboard/pages/` — new page files added here
- `src/finalayze/dashboard/app.py` or sidebar config — register new pages
- `results/experiments/{id}/` — read result JSON files for charts
- `.planning/experiments/` — read experiment definitions via ExperimentManager

</code_context>

<specifics>
## Specific Ideas

No specific requirements — open to standard Streamlit approaches matching existing dashboard style.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>
