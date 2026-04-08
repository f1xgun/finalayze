# Phase 35: Experiment Lab UI - Research

**Researched:** 2026-04-07
**Domain:** Streamlit multipage dashboard, ExperimentManager integration, Plotly visualization
**Confidence:** HIGH

## Summary

Phase 35 adds three Streamlit pages to the existing operator dashboard: an experiment list page, an experiment detail page with A/B/AB comparison charts, and a decision history page. All data comes from `ExperimentManager` (reads `.planning/experiments/*.md` files with YAML frontmatter) and `DebateManager` (reads `.planning/debates/*.md` files) — no REST API changes needed.

The existing codebase provides a clear, consistent pattern: each page is a Python module in `src/finalayze/dashboard/pages/` exposing a `render(api: ApiClient) -> None` function, plus a smoke test in `tests/unit/test_dashboard_pages.py`. The experiment pages will follow this exact pattern, except that they do NOT use the `ApiClient` for data — they load data directly from `ExperimentManager` and `DebateManager` (these are Layer 0 file managers, safe to call from the dashboard layer).

The Streamlit version in use is **1.54.0** [VERIFIED: pip]. Navigation is the built-in Streamlit multipage convention (files in `pages/` directory auto-discovered). Plotly (`plotly.graph_objects`, `plotly.express`) is already in the dependency tree and used in `sandbox.py` for equity curves and drawdown charts — new pages reuse those same patterns.

**Primary recommendation:** Follow the `sandbox.py` pattern for Plotly subplots; follow the `signals.py` pattern for dataframe display with gradient coloring; load data from `ExperimentManager`/`DebateManager` directly (no API call); keep all pages importable with no Streamlit runtime state dependency (for smoke tests).

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- New pages in `src/finalayze/dashboard/pages/` — matches existing Streamlit multipage app pattern
- Add "Experiments" section to existing sidebar with 3 sub-pages (List, Detail, History)
- Page files: `experiments_list.py`, `experiment_detail.py`, `decision_history.py`
- Data loading: read from ExperimentManager + results/ files directly (no REST API needed for Streamlit)
- Chart library: Plotly (already in project for dashboard) — interactive charts
- A/B/AB comparison: side-by-side metric bars + delta table (matches SC-2 "A vs B vs A+B comparison table")
- Status indicators: colored badges — green=ACCEPTED, red=REJECTED, yellow=INCONCLUSIVE, blue=RUNNING, gray=PENDING
- Backtest result charts: equity curves from result JSON, drawdown overlay — same style as existing iteration charts
- List page filtering: status dropdown + text search on hypothesis — minimal but functional
- Detail page navigation: Streamlit `st.query_params` with experiment_id for deep linking
- Decision history: most recent verdict first (reverse chronological)
- No auto-refresh — manual refresh button sufficient for experiment lifecycle timescales

### Claude's Discretion
- Exact column widths and Streamlit layout ratios
- Chart color palette beyond status badges
- Loading state / spinner behavior
- Empty state messaging

### Deferred Ideas (OUT OF SCOPE)
None — discussion stayed within phase scope.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| UI-EXP-01 | Experiment list page shows all experiments with status, hypothesis summary, and key metrics | `ExperimentManager.list_experiments()` + `read_experiment()` provide all data; `st.dataframe()` with gradient coloring per `signals.py` pattern |
| UI-EXP-02 | Experiment detail page shows debate context, success criteria, backtest results with charts, A vs B vs A+B comparison table | `ExperimentManager.read_experiment()` for criteria+results; `DebateManager.read_debate()` for debate context; `ExperimentResult.metrics` dict drives Plotly bar chart comparison |
| UI-EXP-03 | Decision history page shows accepted/rejected experiments with reasoning, reverse chronological | Filter `ExperimentManager` results by terminal status (ACCEPTED/REJECTED/INCONCLUSIVE); `ExperimentState.reasoning` contains verdict explanation |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| streamlit | 1.54.0 | Multipage web app framework | Already in project, all pages use it |
| plotly | (existing) | Interactive charts | Already used in `sandbox.py` for equity/drawdown charts |
| pandas | (existing) | DataFrame display and manipulation | Used in all existing pages |
| pyyaml | (existing) | Read experiment YAML frontmatter | Used by ExperimentManager |

[VERIFIED: local environment — `uv run python -c "import streamlit; print(streamlit.__version__)"` → 1.54.0]

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| plotly.graph_objects | bundled | `go.Bar`, `go.Scatter`, `make_subplots` | A/B comparison bars, equity curves |
| plotly.express | bundled | Histograms, quick charts | Simpler single-series charts |
| `st.query_params` | 1.30+ | Deep-link to detail page by experiment_id | Navigation from list → detail |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Direct ExperimentManager calls | REST API endpoint | API adds latency + requires server; file reads are instant and match existing dashboard pattern for Streamlit pages that don't need live trading data |
| `st.query_params` | `st.session_state` | `query_params` is bookmarkable/shareable; `session_state` loses state on refresh |

**Installation:** No new packages needed — Streamlit, Plotly, pandas are all in current dependencies.

## Architecture Patterns

### Recommended Project Structure
```
src/finalayze/dashboard/pages/
├── experiments_list.py       # UI-EXP-01: list + filter + status badges
├── experiment_detail.py      # UI-EXP-02: detail + charts + A/B/AB table
├── decision_history.py       # UI-EXP-03: terminal-status experiments, reverse chron
└── (existing pages unchanged)

tests/unit/
├── test_dashboard_pages.py   # Add 3 smoke tests (import + callable render)
```

### Pattern 1: Page Module Structure
**What:** Each page is a module with a top-level `render(api: ApiClient) -> None` function and private helper functions. No module-level Streamlit calls (prevents import errors during testing).
**When to use:** All new pages — matches existing convention.
**Example:**
```python
# Source: src/finalayze/dashboard/pages/signals.py (existing)
from __future__ import annotations
import streamlit as st
from finalayze.dashboard.api_client import ApiClient

def render(api: ApiClient) -> None:
    """Render the Signals page."""
    st.title("Signals")
    # ... data loading and display
```

For experiment pages, the `api` parameter is accepted for signature consistency but experiment data is loaded from `ExperimentManager` directly:
```python
# Source: pattern from experiment_manager.py
from finalayze.core.experiment_manager import ExperimentManager

def render(api: ApiClient) -> None:  # api kept for signature compat
    st.title("Experiments")
    mgr = ExperimentManager()  # reads .planning/experiments/
    exp_ids = mgr.list_experiments()
    experiments = [mgr.read_experiment(eid) for eid in exp_ids]
```

### Pattern 2: Status Badge Display
**What:** Map `ExperimentStatus` enum to `st.success` / `st.error` / `st.warning` / `st.info` calls in list rows, or colored text in dataframes.
**When to use:** List page status column, detail page header, history page filter badges.
**Example:**
```python
# Derived from context decisions + st.success/error/warning pattern in sandbox.py
_STATUS_DISPLAY = {
    "accepted": ("success", "ACCEPTED"),
    "rejected": ("error", "REJECTED"),
    "inconclusive": ("warning", "INCONCLUSIVE"),
    "running": ("info", "RUNNING"),
    "pending": ("info", "PENDING"),
    "completed": ("info", "COMPLETED"),
}

def _status_badge(status: str) -> None:
    fn_name, label = _STATUS_DISPLAY.get(status, ("info", status.upper()))
    getattr(st, fn_name)(label)
```

### Pattern 3: A/B/AB Comparison Chart (Plotly Bar)
**What:** Grouped bar chart showing metric values across experiment runs (A=baseline, B=hypothesis, A+B=combined). Delta table below.
**When to use:** Experiment detail page — the core visual comparison per SC-2.
**Example:**
```python
# Source: pattern from sandbox.py go.Scatter/go.Bar usage
import plotly.graph_objects as go

def _render_comparison(results: list[ExperimentResult]) -> None:
    if not results:
        st.info("No backtest results yet.")
        return
    metrics_of_interest = ["sharpe", "profit_factor", "max_drawdown", "win_rate"]
    fig = go.Figure()
    for result in results:
        metric_vals = [float(result.metrics.get(m, 0)) for m in metrics_of_interest]
        fig.add_trace(go.Bar(name=result.run_name, x=metrics_of_interest, y=metric_vals))
    fig.update_layout(barmode="group", height=400)
    st.plotly_chart(fig, use_container_width=True)
```

### Pattern 4: Deep-link Navigation (st.query_params)
**What:** List page links to detail page via `st.query_params`. Detail page reads `experiment_id` from query params on load.
**When to use:** List → Detail navigation (locked decision).
**Example:**
```python
# Source: Streamlit 1.30+ st.query_params API [ASSUMED - API available in 1.30+, project uses 1.54.0]
# List page: navigate button
if st.button("View Detail", key=f"detail_{exp.experiment_id}"):
    st.query_params["experiment_id"] = exp.experiment_id
    st.switch_page("pages/experiment_detail.py")

# Detail page: read param
experiment_id = st.query_params.get("experiment_id", "")
```

### Pattern 5: Equity Curve Chart (Plotly Subplots)
**What:** Two-row subplot: equity curve top, drawdown bar bottom. Matches `sandbox.py` exactly.
**When to use:** Experiment detail page — backtest result visualization.
**Example:**
```python
# Source: src/finalayze/dashboard/pages/sandbox.py (verified)
from plotly.subplots import make_subplots
fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
    row_heights=[0.7, 0.3], vertical_spacing=0.05)
fig.add_trace(go.Scatter(x=dates, y=equity, name="Equity", mode="lines"), row=1, col=1)
fig.add_trace(go.Bar(x=dates, y=drawdown, name="Drawdown %", marker_color="red"), row=2, col=1)
fig.update_layout(height=500, margin={"t": 30, "b": 30})
st.plotly_chart(fig, use_container_width=True)
```

### Anti-Patterns to Avoid
- **Streamlit calls at module level:** Causes import errors in test smoke tests. All `st.*` calls must be inside the `render()` function or private helpers called from it.
- **Calling `ExperimentManager` with a hardcoded path that doesn't exist:** Both `.planning/experiments/` and `results/experiments/` directories may not exist yet (Phase 34 created them but no experiments may exist). Always handle `FileNotFoundError` and show empty state.
- **Storing ApiClient in session state for experiment pages:** These pages don't use the API — don't introduce API calls that require the server to be running.
- **Parsing ExperimentResult metrics as strings without conversion:** `ExperimentResult.metrics` is `dict[str, Any]` and values may be stored as strings in the YAML (e.g., `"0.1331"`) — always cast to `float()` before arithmetic.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Experiment data parsing | Custom YAML parser | `ExperimentManager.read_experiment()` | Already handles frontmatter, nested model reconstruction, validation |
| Debate context loading | Custom file reader | `DebateManager.read_debate()` + `DebateState` schema | Already implemented in Phase 33 |
| Status badge logic | Custom HTML/CSS | `st.success()`, `st.error()`, `st.warning()`, `st.info()` | Theme-consistent, no CSS injection needed |
| Metric comparison table | Custom HTML table | `st.dataframe()` with pandas DataFrame | Sortable, theme-consistent, already used everywhere |
| Equity curve chart | Custom charting | Copy the `make_subplots` block from `sandbox.py` | Already battle-tested in the project |

**Key insight:** ExperimentManager and DebateManager are complete file-I/O layers from Phase 33/34. The UI phase is purely presentation — don't re-implement data loading.

## Common Pitfalls

### Pitfall 1: Experiment Detail Page With No experiments_dir
**What goes wrong:** `ExperimentManager()` defaults to `.planning/experiments/` — if that dir doesn't exist or has no `.md` files, `list_experiments()` returns `[]`. Detail page with no experiment_id in query params crashes.
**Why it happens:** Phase 34 creates the directory lazily on first use; in a clean install there may be no experiments.
**How to avoid:** Always check `experiment_id = st.query_params.get("experiment_id", "")` and show an informative empty state if blank. Wrap `mgr.read_experiment(eid)` in try/except `FileNotFoundError`.
**Warning signs:** `KeyError` or `FileNotFoundError` in page render.

### Pitfall 2: Debate Context Not Linked
**What goes wrong:** `ExperimentState.debate_id` may be `None` (experiments created without a debate). Calling `DebateManager.read_debate(None)` raises `TypeError`.
**Why it happens:** Debate linkage is optional — `create_experiment(debate_id=None)` is valid.
**How to avoid:** Guard: `if exp.debate_id: debate = dm.read_debate(exp.debate_id)` else show "No debate linked."
**Warning signs:** `TypeError: argument of type 'NoneType' is not iterable` in path construction.

### Pitfall 3: Metrics Dict Values Are Strings
**What goes wrong:** `ExperimentResult.metrics` is typed `dict[str, Any]` and when deserialized from YAML, numeric values may be strings (e.g., `{"sharpe": "0.1331"}`). Plotly bar charts fail with `TypeError` or show zero.
**Why it happens:** YAML round-trips values from `run_iteration.py` summary.json which uses Decimal-serialized strings.
**How to avoid:** Always cast: `float(result.metrics.get("sharpe", 0))` with a default of 0.
**Warning signs:** Flat bars in comparison chart even though metrics exist in the data.

### Pitfall 4: `st.switch_page` Path Format
**What goes wrong:** `st.switch_page("pages/experiment_detail.py")` path is relative to the app root, not the pages directory. Wrong path causes a Streamlit runtime error.
**Why it happens:** Streamlit multipage apps auto-discover pages but `switch_page` needs the exact path relative to the app entrypoint.
**How to avoid:** Use `st.switch_page("src/finalayze/dashboard/pages/experiment_detail.py")` if running from project root, or use the page name as Streamlit displays it. Test navigation manually after implementation.
**Warning signs:** `StreamlitAPIException: Page not found` on button click.

### Pitfall 5: Importing Streamlit at Module Level in Test Context
**What goes wrong:** Any `st.set_page_config()` or `st.session_state` call at module import time causes `StreamlitAPIException` when pytest imports the module.
**Why it happens:** The smoke test pattern (in `test_dashboard_pages.py`) imports the module directly.
**How to avoid:** All `st.*` calls inside `render()` or its private helper functions only. Never at module level.
**Warning signs:** `StreamlitAPIException: set_page_config() can only be called once per app` in pytest output.

## Code Examples

Verified patterns from official sources:

### List Page Structure (experiments_list.py)
```python
# Source: pattern from signals.py + sandbox.py (verified)
from __future__ import annotations
import pandas as pd
import streamlit as st
from finalayze.core.experiment_manager import ExperimentManager
from finalayze.dashboard.api_client import ApiClient

_STATUS_COLORS = {
    "accepted": "st.success",
    "rejected": "st.error",
    "inconclusive": "st.warning",
    "running": "st.info",
    "pending": "st.info",
    "completed": "st.info",
}

def render(api: ApiClient) -> None:
    st.title("Experiments")

    col_refresh, _ = st.columns([1, 5])
    with col_refresh:
        if st.button("Refresh"):
            st.cache_data.clear()

    mgr = ExperimentManager()
    exp_ids = mgr.list_experiments()

    if not exp_ids:
        st.info("No experiments found. Create experiments via ExperimentManager.")
        return

    experiments = []
    for eid in exp_ids:
        try:
            experiments.append(mgr.read_experiment(eid))
        except FileNotFoundError:
            continue

    # Filters
    all_statuses = sorted({e.status for e in experiments})
    status_filter = st.selectbox("Filter by status", ["All"] + all_statuses)
    text_filter = st.text_input("Search hypothesis", "")

    filtered = [
        e for e in experiments
        if (status_filter == "All" or e.status == status_filter)
        and text_filter.lower() in e.hypothesis.lower()
    ]

    # Display as dataframe
    rows = []
    for e in filtered:
        latest_metrics = e.results[-1].metrics if e.results else {}
        rows.append({
            "ID": e.experiment_id,
            "Status": e.status.upper(),
            "Hypothesis": e.hypothesis[:80],
            "Created": e.created,
            "Runs": len(e.results),
            "Sharpe": float(latest_metrics.get("sharpe", 0)) if latest_metrics else None,
            "PF": float(latest_metrics.get("profit_factor", 0)) if latest_metrics else None,
        })
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True)
```

### Decision History Page (decision_history.py)
```python
# Source: pattern from signals.py + ExperimentState schema (verified)
from __future__ import annotations
import streamlit as st
from finalayze.core.experiment_manager import ExperimentManager
from finalayze.core.schemas import ExperimentStatus
from finalayze.dashboard.api_client import ApiClient

_TERMINAL_STATUSES = {ExperimentStatus.ACCEPTED, ExperimentStatus.REJECTED, ExperimentStatus.INCONCLUSIVE}

def render(api: ApiClient) -> None:
    st.title("Decision History")

    mgr = ExperimentManager()
    exp_ids = mgr.list_experiments()

    decided = []
    for eid in exp_ids:
        try:
            exp = mgr.read_experiment(eid)
            if exp.status in _TERMINAL_STATUSES:
                decided.append(exp)
        except FileNotFoundError:
            continue

    # Reverse chronological
    decided.sort(key=lambda e: e.created, reverse=True)

    if not decided:
        st.info("No decisions recorded yet.")
        return

    for exp in decided:
        with st.expander(f"[{exp.status.upper()}] {exp.experiment_id} ({exp.created})"):
            st.write(f"**Hypothesis:** {exp.hypothesis}")
            st.write(f"**Verdict:** {exp.verdict}")
            st.write(f"**Reasoning:** {exp.reasoning}")
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `st.experimental_query_params` | `st.query_params` | Streamlit 1.30 | Stable API for deep linking; `experimental_` prefix removed |
| `st.cache` | `st.cache_data` / `st.cache_resource` | Streamlit 1.18 | Separate caches for data vs. connections |

**Deprecated/outdated:**
- `st.experimental_query_params`: Replaced by `st.query_params` in Streamlit 1.30. Project uses 1.54.0 — use `st.query_params` directly.
- `st.experimental_rerun`: Replaced by `st.rerun()` — already used in `app.py`.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `st.query_params` API and `st.switch_page` are available in Streamlit 1.54.0 | Architecture Patterns (Pattern 4) | Navigation between list and detail would need to use `st.session_state` instead |
| A2 | `ExperimentResult.metrics` dict values may be stored as strings in YAML, matching the `summary.json` string-decimal format | Common Pitfalls (Pitfall 3) | If values are always floats, the cast is safe but unnecessary |
| A3 | `DebateManager.read_debate(debate_id)` is the correct method name and signature | Code Examples | If the method is named differently, debate context loading section needs adjustment |

## Open Questions

1. **Debate body content**
   - What we know: `DebateManager._read_file()` returns `(frontmatter_dict, body_text)`. `DebateState` has `arbiter_report` and `resolution` fields.
   - What's unclear: Does the debate markdown body contain the full agent discussion text, or is that in `AgentOutput` objects stored elsewhere?
   - Recommendation: For the detail page, display `DebateState.resolution` and `DebateState.arbiter_report.to_markdown()` if present. That's sufficient for "debate context."

2. **results/experiments directory**
   - What we know: CONTEXT.md mentions reading from `results/experiments/{id}/` for result JSON files. `ExperimentResult.metrics` is a dict — the flat metrics dict is sufficient for bars/tables.
   - What's unclear: Whether Phase 34 writes equity curve timeseries data to `results/experiments/{id}/` or only the aggregate metrics dict into the YAML frontmatter.
   - Recommendation: Implement the comparison chart from `ExperimentResult.metrics` dict (aggregate metrics only). If equity curve timeseries data doesn't exist in `results/experiments/`, skip the equity subplot and show "No timeseries data" — don't block the phase on this.

## Environment Availability

Step 2.6: The phase is a pure Streamlit page addition — no new external dependencies beyond what's already installed. Streamlit 1.54.0 is confirmed available. Plotly is confirmed available (used in `sandbox.py`).

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| streamlit | All 3 pages | Yes | 1.54.0 | — |
| plotly | Comparison chart, equity curve | Yes | existing | — |
| pandas | DataFrame display | Yes | existing | — |
| ExperimentManager | Data loading | Yes | Phase 34 complete | — |
| DebateManager | Debate context | Yes | Phase 33 complete | — |

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest |
| Config file | pyproject.toml |
| Quick run command | `uv run pytest tests/unit/test_dashboard_pages.py -x` |
| Full suite command | `uv run pytest tests/unit/ -x` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| UI-EXP-01 | `experiments_list.render` is importable and callable | smoke | `uv run pytest tests/unit/test_dashboard_pages.py::test_experiments_list_render_importable -x` | ❌ Wave 0 |
| UI-EXP-02 | `experiment_detail.render` is importable and callable | smoke | `uv run pytest tests/unit/test_dashboard_pages.py::test_experiment_detail_render_importable -x` | ❌ Wave 0 |
| UI-EXP-03 | `decision_history.render` is importable and callable | smoke | `uv run pytest tests/unit/test_dashboard_pages.py::test_decision_history_render_importable -x` | ❌ Wave 0 |

**Note:** The render functions cannot be called in tests without a Streamlit runtime. The importability + `callable()` pattern is the established project standard (see `test_dashboard_pages.py`). Full render testing is manual-only (visual inspection).

### Sampling Rate
- **Per task commit:** `uv run pytest tests/unit/test_dashboard_pages.py -x`
- **Per wave merge:** `uv run pytest tests/unit/ -x`
- **Phase gate:** Full suite green before `/gsd-verify-work`

### Wave 0 Gaps
- [ ] `tests/unit/test_dashboard_pages.py` — add 3 new test functions for `experiments_list`, `experiment_detail`, `decision_history` (file exists, needs 3 new test functions appended)

*(Existing test infrastructure covers the framework — only the 3 new test functions are missing.)*

## Security Domain

These are read-only Streamlit pages with no user input that modifies state. The dashboard already has password-gating in `app.py`. No new attack surface is introduced.

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no — handled by app.py | existing password gate |
| V3 Session Management | no | existing `st.session_state["authenticated"]` |
| V4 Access Control | no | read-only file display |
| V5 Input Validation | yes (minimal) | `st.query_params.get("experiment_id", "")` used as filename stem — `ExperimentState`'s `experiment_id_safe` validator guards against path traversal at the schema level |
| V6 Cryptography | no | — |

**Path traversal note:** `ExperimentManager._experiment_path()` joins the experiments_dir with `{experiment_id}.md`. Since `ExperimentState.experiment_id` enforces `[a-zA-Z0-9_-]+` via `experiment_id_safe` validator, passing `experiment_id` from `st.query_params` through `mgr.read_experiment(experiment_id)` is safe — the schema validation happens before file I/O.

## Sources

### Primary (HIGH confidence)
- Codebase: `src/finalayze/dashboard/pages/` — verified all 6 existing pages and their patterns
- Codebase: `src/finalayze/core/experiment_manager.py` — verified full ExperimentManager API
- Codebase: `src/finalayze/core/schemas.py` lines 533-767 — verified ExperimentState, DebateState, ExperimentResult schemas
- Codebase: `src/finalayze/core/debate_manager.py` — verified DebateManager structure
- Codebase: `tests/unit/test_dashboard_pages.py` — verified smoke test pattern
- Local: `uv run python -c "import streamlit; print(streamlit.__version__)"` → 1.54.0

### Secondary (MEDIUM confidence)
- Streamlit docs: `st.query_params` available since 1.30.0 — project uses 1.54.0, so confirmed available [ASSUMED based on Streamlit changelog knowledge]

### Tertiary (LOW confidence)
- None

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all libraries verified in local environment
- Architecture: HIGH — patterns directly derived from existing codebase files
- Pitfalls: HIGH — derived from actual schema constraints and existing code patterns
- Test patterns: HIGH — verified against existing `test_dashboard_pages.py`

**Research date:** 2026-04-07
**Valid until:** 2026-05-07 (stable Streamlit version, stable codebase patterns)
