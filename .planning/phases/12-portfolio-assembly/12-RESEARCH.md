# Phase 12: Portfolio Assembly - Research

**Researched:** 2026-03-21
**Domain:** Portfolio orchestration, equity curve merging, rebalancing, crisis brake
**Confidence:** HIGH

## Summary

Phase 12 assembles OFZ bond and equity backtest engines into a unified PortfolioBacktestOrchestrator. The codebase already has all building blocks: `BondBacktestEngine` (bond_engine.py) produces `BondBacktestResult` with equity_curve/dates, `BacktestEngine` (engine.py) produces `list[TradeResult]` and `list[PortfolioState]`, and `PortfolioAggregator` (portfolio_aggregator.py) demonstrates curve alignment via forward-fill and combined metric computation. The new orchestrator wraps these, applies 40/60 allocation weighting, monthly rebalancing, and a USDRUB crisis brake.

The key complexity lies in: (1) date alignment between bond and equity curves that may have different trading calendars, (2) USDRUB crisis brake requiring 20-bar lookback on FX data already available via `MarketContext.moex_data.fx_rates`, and (3) walk-forward Sharpe computation on the merged curve using existing `_compute_sharpe_from_snapshots` or the bond walk-forward's `_compute_excess_sharpe_from_equity` pattern.

**Primary recommendation:** Build `PortfolioBacktestOrchestrator` as a thin coordinator that delegates to existing engines, reuses `PortfolioAggregator._align_curves` for date alignment, and adds allocation/rebalancing/crisis-brake logic as a post-processing step on the merged equity curve.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- Location: new file `src/finalayze/backtest/portfolio_orchestrator.py` at backtest layer
- Curve merging: run BondBacktestEngine and BacktestEngine independently, merge equity curves by date alignment and weighted sum (40% OFZ + 60% equity)
- Aggregate metrics: compute Sharpe/DD/PF on the merged portfolio curve using existing PerformanceAnalyzer
- Walk-forward: apply walk-forward validation on the merged portfolio curve (not individual engines)
- Allocation: static 40/60 split via initial capital allocation (OFZ gets 40% of total, equity gets 60%)
- Monthly rebalancing: at each month boundary, compare actual weights to 40/60 target; if drift > 5%, adjust next period's capital allocation
- RUB crisis brake: check USDRUB 20-bar return > 15% -- if triggered, freeze equity allocation, shift new capital to 80/20 OFZ/equity until FX stabilizes
- Crisis brake data: USDRUB from MacroSnapshot/MOEX ISS -- same data already available
- WF window: 12mo train + 6mo test (same as equity WF) applied to merged portfolio curve
- Sharpe measurement: annualized Sharpe on WF test windows averaged across folds
- If Sharpe < 0.10: report achieved Sharpe -- aspirational target, not a hard gate for phase completion
- Bond carry contribution: OFZ carry (6-8% yield) provides base return that should lift blended Sharpe

### Claude's Discretion
- Internal PortfolioBacktestOrchestrator API design (run method signature, result dataclass)
- Test structure and fixture design for portfolio-level tests
- How to handle date alignment gaps between bond and equity curves
- Monthly rebalancing implementation details (exact drift calculation)
- Crisis brake cooldown period (how long to wait before reverting to 40/60)

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| PORT-01 | PortfolioBacktestOrchestrator for joint equity + OFZ backtest with merged equity curve | Existing `BondBacktestEngine` and `BacktestEngine` are independent, can be run separately. `PortfolioAggregator._align_curves` provides forward-fill date alignment. New orchestrator coordinates both and merges curves with weighted sum. |
| PORT-02 | Portfolio allocation 40% OFZ carry + 60% equity with RUB crisis brake (USD/RUB +15% over 20 bars -> freeze equity) | Initial capital split at construction time. Monthly rebalancing via weight drift check. USDRUB data available from `MarketContext.moex_data.fx_rates` as `FXRate` objects. Crisis brake computes 20-bar log return on USDRUB series. |
| PORT-03 | Blended MOEX portfolio walk-forward Sharpe >= +0.10 (combined equity + OFZ) | `run_bond_walk_forward` in `bond_walk_forward.py` provides the pattern: take pre-computed equity curve + dates, slice into WF windows, compute excess Sharpe per fold. Apply same pattern to merged portfolio curve. Aspirational target, not hard gate. |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| Python stdlib (dataclasses, decimal) | 3.12 | Result dataclasses, Decimal arithmetic | Project standard |
| dateutil.relativedelta | existing | Month boundary computation for rebalancing | Already used in walk_forward.py |
| structlog | existing | Structured logging | Project standard |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| PerformanceAnalyzer | existing | Sharpe/DD/PF on merged curve | Aggregate metrics |
| PortfolioAggregator | existing | Forward-fill date alignment reference | Reuse `_align_curves` pattern |
| bond_walk_forward | existing | Walk-forward on pre-computed curves | Pattern for merged WF |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Custom WF on merged curve | Re-run engines per WF fold | Too expensive -- bond + equity engines together would be very slow per fold. Slicing pre-computed curve is correct for "analysis" style WF. |
| PerformanceAnalyzer | Custom Sharpe computation | PerformanceAnalyzer expects `list[PortfolioState]` not raw floats. For merged curve, use `_compute_sharpe_from_snapshots` pattern directly or convert to PortfolioState wrappers. |

**Installation:**
No new dependencies needed. All code uses existing project libraries.

## Architecture Patterns

### Recommended Project Structure
```
src/finalayze/backtest/
    portfolio_orchestrator.py     # NEW: PortfolioBacktestOrchestrator + result dataclass
tests/unit/
    test_portfolio_orchestrator.py  # NEW: portfolio-level orchestration tests
```

### Pattern 1: Engine Independence + Curve Merging
**What:** Run bond and equity engines independently, then merge their equity curves post-hoc.
**When to use:** Always -- the engines have fundamentally different mechanics (bonds use Decimal, equity uses Decimal via PortfolioState).
**Example:**
```python
# Source: existing PortfolioAggregator._align_curves pattern
@dataclass
class PortfolioBacktestResult:
    """Combined bond + equity portfolio result."""
    bond_equity_curve: list[float]
    equity_equity_curve: list[float]
    merged_equity_curve: list[float]
    dates: list[date]
    bond_trades: list[TradeResult]
    equity_trades: list[TradeResult]
    # Aggregate metrics on merged curve
    sharpe: float
    max_drawdown_pct: float
    profit_factor: float
    total_return_pct: float
    # Allocation tracking
    bond_weight_series: list[float]  # actual bond weight per date
    equity_weight_series: list[float]
    crisis_brake_active_dates: list[date]
    # Walk-forward
    wf_sharpe: float  # averaged across WF folds

class PortfolioBacktestOrchestrator:
    def __init__(
        self,
        bond_weight: float = 0.40,
        equity_weight: float = 0.60,
        rebalance_threshold: float = 0.05,  # 5% drift
        crisis_usdrub_threshold: float = 0.15,  # 15% spike
        crisis_usdrub_window: int = 20,  # bars
        crisis_bond_weight: float = 0.80,
    ) -> None: ...

    def run(
        self,
        bond_result: BondBacktestResult,
        equity_trades: list[TradeResult],
        equity_snapshots: list[PortfolioState],
        usdrub_series: list[tuple[date, float]],  # date -> USDRUB rate
        total_capital: float,
    ) -> PortfolioBacktestResult: ...
```

### Pattern 2: Monthly Rebalancing via Weight Drift
**What:** At each month boundary in the merged curve, compute actual bond/equity weights from current values. If drift from target exceeds threshold (5%), adjust weights for next period.
**When to use:** Every date crossing a month boundary.
**Implementation detail:**
```python
def _check_rebalance(
    self,
    current_date: date,
    prev_date: date | None,
    bond_value: float,
    equity_value: float,
) -> tuple[float, float]:
    """Return (bond_scale, equity_scale) for next period.

    If month boundary crossed AND drift > threshold, return adjusted scales.
    Otherwise return (1.0, 1.0) -- no adjustment.
    """
    if prev_date is None or current_date.month == prev_date.month:
        return 1.0, 1.0

    total = bond_value + equity_value
    if total <= 0:
        return 1.0, 1.0

    actual_bond_weight = bond_value / total
    drift = abs(actual_bond_weight - self._bond_weight)

    if drift <= self._rebalance_threshold:
        return 1.0, 1.0

    # Scale curves to restore target weights
    target_bond = total * self._bond_weight
    target_equity = total * self._equity_weight
    bond_scale = target_bond / bond_value if bond_value > 0 else 1.0
    equity_scale = target_equity / equity_value if equity_value > 0 else 1.0
    return bond_scale, equity_scale
```

### Pattern 3: USDRUB Crisis Brake
**What:** Monitor 20-bar USDRUB return. If > 15%, shift allocation to 80/20 OFZ/equity. Revert when 20-bar return drops below threshold.
**When to use:** On each bar of the merged curve.
**Implementation detail:**
```python
def _is_crisis(
    self,
    current_date: date,
    usdrub_lookup: dict[date, float],
    sorted_dates: list[date],
    bar_idx: int,
) -> bool:
    """Check if USDRUB spiked > 15% over 20 bars."""
    if bar_idx < self._crisis_usdrub_window:
        return False

    current_rate = usdrub_lookup.get(current_date)
    lookback_date = sorted_dates[bar_idx - self._crisis_usdrub_window]
    lookback_rate = usdrub_lookup.get(lookback_date)

    if current_rate is None or lookback_rate is None or lookback_rate <= 0:
        return False

    return (current_rate - lookback_rate) / lookback_rate > self._crisis_usdrub_threshold
```

### Pattern 4: Walk-Forward on Merged Curve
**What:** Reuse `bond_walk_forward.generate_wf_windows` to generate windows, then slice the pre-computed merged equity curve (not re-run engines).
**When to use:** After the full merged curve is computed.
**Source:** `bond_walk_forward.py` lines 86-112 -- identical window generation pattern.

### Anti-Patterns to Avoid
- **Re-running engines per WF fold:** Both engines are expensive. The decision says WF applies to the merged curve, not to re-running engines. Use analysis-style WF (slice pre-computed curve).
- **Mixing Decimal and float in curve merging:** Bond engine uses `list[Decimal]` for equity_curve, equity engine uses `list[PortfolioState]` with Decimal equity. Convert to float at the merging boundary.
- **Allocating total capital then splitting:** The engines already received their capital. The orchestrator should scale the output curves by weights, not modify engine capital inputs. (Actually, per CONTEXT.md: "static 40/60 split via initial capital allocation" -- so pass 40% to bond engine, 60% to equity engine. The orchestrator takes pre-computed results.)
- **Complex cooldown logic:** Crisis brake should be simple: check condition each bar, shift weights when true, revert when false. No hysteresis needed for a first implementation.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Date alignment / forward-fill | Custom date join | `PortfolioAggregator._align_curves` pattern | Already handles forward-fill correctly with edge cases |
| WF window generation | Custom window splitter | `bond_walk_forward.generate_wf_windows` | Tested, handles month boundaries correctly |
| Sharpe computation | Custom Sharpe | `_compute_excess_sharpe_from_equity` from bond_walk_forward | Handles edge cases (too few returns, zero std) |
| Max drawdown | Custom DD | Existing pattern in PortfolioAggregator or BondBacktestEngine | Multiple tested implementations available |

**Key insight:** This phase is pure orchestration. Every computational building block exists. The value is in wiring them together correctly with allocation, rebalancing, and crisis brake logic.

## Common Pitfalls

### Pitfall 1: Decimal/Float Boundary
**What goes wrong:** Bond engine produces `list[Decimal]` equity curve, equity engine produces `list[PortfolioState]` with Decimal equity. Mixing Decimal and float causes TypeError.
**Why it happens:** Two engines were designed independently with slightly different return types.
**How to avoid:** Convert both to `list[float]` at the orchestrator boundary before merging. Use `float()` conversion.
**Warning signs:** TypeError in curve addition.

### Pitfall 2: Different Date Ranges
**What goes wrong:** Bond and equity curves may start/end on different dates. MOEX bond calendar and MOEX equity calendar are the same, but data availability may differ.
**Why it happens:** Different symbols have different first-available dates.
**How to avoid:** Use union of dates + forward-fill (existing pattern). The merged curve starts at the first date where both curves have values.
**Warning signs:** Merged curve shorter than expected.

### Pitfall 3: Capital Scaling Confusion
**What goes wrong:** If bond engine receives 400K and equity engine receives 600K, the raw curves already reflect the 40/60 split. Applying weights again would double-count.
**Why it happens:** Misunderstanding whether weights apply at input (capital) or output (curve).
**How to avoid:** Per CONTEXT.md: "static 40/60 split via initial capital allocation." So pass appropriate capital to each engine. The merged curve is the **sum** of the two raw curves (not weighted sum). Rebalancing adjusts by scaling curves at month boundaries.
**Warning signs:** Portfolio total != sum of parts.

### Pitfall 4: USDRUB Data Gaps
**What goes wrong:** USDRUB rates from `fx_rates` may have gaps (weekends, holidays). 20-bar lookback on sparse data gives wrong return.
**Why it happens:** FX rates are not daily MOEX candles -- they come from CBR which publishes on business days.
**How to avoid:** Build USDRUB lookup dict, forward-fill gaps to match merged curve dates. Count 20 **available** data points, not 20 calendar days.
**Warning signs:** Crisis brake never triggers or triggers on noise.

### Pitfall 5: Monthly Rebalancing Creates Discontinuities
**What goes wrong:** Scaling curves at month boundaries creates jumps in the equity curve that inflate Sharpe variance.
**Why it happens:** Rebalancing changes the level of sub-curves, creating artificial returns.
**How to avoid:** Apply rebalancing as a weight adjustment for **future** bars only. At month boundary, compute new weights but don't retroactively adjust past values.
**Warning signs:** Spikes in daily returns at month boundaries.

## Code Examples

### Merging Two Equity Curves (from PortfolioAggregator)
```python
# Source: backtest/portfolio_aggregator.py lines 228-244
@staticmethod
def _align_curves(
    layer_results: list[LayerResult],
    common_dates: list[date],
) -> dict[str, list[float]]:
    """Interpolate each layer's equity to common dates via forward-fill."""
    layer_curves: dict[str, list[float]] = {}
    for lr in layer_results:
        date_to_equity = dict(zip(lr.dates, lr.equity_curve, strict=False))
        curve: list[float] = []
        last_val = lr.equity_curve[0] if lr.equity_curve else 0.0
        for d in common_dates:
            if d in date_to_equity:
                last_val = date_to_equity[d]
            curve.append(last_val)
        layer_curves[lr.layer_id] = curve
    return layer_curves
```

### Excess Sharpe on Equity Curve Slice (from bond_walk_forward)
```python
# Source: backtest/bond_walk_forward.py lines 242-277
def _compute_excess_sharpe_from_equity(
    equity: list[float],
    risk_free_annual_pct: float,
    trading_days_per_year: int = 252,
) -> float:
    daily_returns = [
        equity[i] / equity[i - 1] - 1.0
        for i in range(1, len(equity)) if equity[i - 1] > 0
    ]
    daily_rf = (1 + risk_free_annual_pct / 100) ** (1 / trading_days_per_year) - 1.0
    excess = [r - daily_rf for r in daily_returns]
    mean_excess = statistics.mean(excess)
    std_excess = statistics.stdev(excess)
    if std_excess <= 0:
        return 0.0
    return float(mean_excess / std_excess * math.sqrt(trading_days_per_year))
```

### USDRUB Data Extraction (from run_iteration.py)
```python
# Source: scripts/run_iteration.py lines 679-702
# FX rates are FXRate(timestamp, pair, rate) objects
rub_candles: list[Candle] = []
if moex_data.fx_rates:
    for fx in moex_data.fx_rates:
        if fx.pair == "USDRUB":
            rate_float = float(fx.rate)
            # Build date -> rate lookup for crisis brake
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Separate bond/equity backtests | Joint portfolio backtest | Phase 12 (now) | Enables aggregate risk management |
| No rebalancing | Monthly rebalancing with drift threshold | Phase 12 (now) | Maintains target allocation |
| No crisis brake | USDRUB spike detection | Phase 12 (now) | Defensive in currency crises |

**Existing patterns reused:**
- `PortfolioAggregator` (Phase 4) -- date alignment, curve combining
- `run_bond_walk_forward` (Phase 7) -- WF on pre-computed curves
- `_compute_moex_sizing_data` (Phase 9) -- USDRUB data extraction

## Open Questions

1. **Crisis brake cooldown period**
   - What we know: CONTEXT.md says "shift to 80/20 until FX stabilizes"
   - What's unclear: Definition of "stabilizes" -- is it when 20-bar return drops below 15%, or below some lower threshold (e.g. 10%) to add hysteresis?
   - Recommendation: Simple approach -- revert when 20-bar return drops below 15% (same threshold). No hysteresis for v1. Can add later if crisis brake chatters.

2. **PerformanceAnalyzer vs. raw computation for merged curve**
   - What we know: PerformanceAnalyzer expects `list[PortfolioState]` with Decimal equity. Merged curve is `list[float]`.
   - What's unclear: Whether to wrap floats in PortfolioState or compute metrics directly.
   - Recommendation: Compute Sharpe/DD/PF directly using existing helper functions (`_compute_excess_sharpe_from_equity`, drawdown loops). Wrapping in PortfolioState adds unnecessary conversion.

3. **Walk-forward risk-free rate**
   - What we know: Bond WF uses RUONIA 15%. Equity WF uses 0%.
   - What's unclear: Which to use for the blended portfolio Sharpe.
   - Recommendation: Use RUONIA (15%) since this is a RUB-denominated MOEX portfolio. The Sharpe target (+0.10) is excess-over-RUONIA.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest (existing) |
| Config file | pyproject.toml |
| Quick run command | `uv run pytest tests/unit/test_portfolio_orchestrator.py -x` |
| Full suite command | `uv run pytest tests/unit/ -x --timeout=60` |

### Phase Requirements -> Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| PORT-01 | Orchestrator runs both engines and produces merged curve | unit | `uv run pytest tests/unit/test_portfolio_orchestrator.py::TestPortfolioOrchestrator::test_merged_curve_is_sum -x` | Wave 0 |
| PORT-01 | Merged curve dates are union of bond and equity dates | unit | `uv run pytest tests/unit/test_portfolio_orchestrator.py::TestPortfolioOrchestrator::test_date_alignment -x` | Wave 0 |
| PORT-01 | Aggregate metrics (Sharpe, DD, PF) computed on merged curve | unit | `uv run pytest tests/unit/test_portfolio_orchestrator.py::TestPortfolioOrchestrator::test_aggregate_metrics -x` | Wave 0 |
| PORT-02 | 40/60 allocation via initial capital split | unit | `uv run pytest tests/unit/test_portfolio_orchestrator.py::TestAllocation::test_initial_capital_split -x` | Wave 0 |
| PORT-02 | Monthly rebalancing triggers on 5% drift | unit | `uv run pytest tests/unit/test_portfolio_orchestrator.py::TestRebalancing::test_drift_triggers_rebalance -x` | Wave 0 |
| PORT-02 | No rebalance when drift < 5% | unit | `uv run pytest tests/unit/test_portfolio_orchestrator.py::TestRebalancing::test_no_rebalance_below_threshold -x` | Wave 0 |
| PORT-02 | Crisis brake activates on USDRUB +15% over 20 bars | unit | `uv run pytest tests/unit/test_portfolio_orchestrator.py::TestCrisisBrake::test_crisis_activates -x` | Wave 0 |
| PORT-02 | Crisis brake shifts to 80/20 allocation | unit | `uv run pytest tests/unit/test_portfolio_orchestrator.py::TestCrisisBrake::test_crisis_allocation_shift -x` | Wave 0 |
| PORT-02 | Crisis brake deactivates when FX stabilizes | unit | `uv run pytest tests/unit/test_portfolio_orchestrator.py::TestCrisisBrake::test_crisis_deactivates -x` | Wave 0 |
| PORT-03 | Walk-forward Sharpe computed on merged curve | unit | `uv run pytest tests/unit/test_portfolio_orchestrator.py::TestWalkForward::test_wf_sharpe_on_merged -x` | Wave 0 |
| PORT-03 | WF uses 12mo/6mo windows consistent with equity WF | unit | `uv run pytest tests/unit/test_portfolio_orchestrator.py::TestWalkForward::test_wf_window_params -x` | Wave 0 |

### Sampling Rate
- **Per task commit:** `uv run pytest tests/unit/test_portfolio_orchestrator.py -x`
- **Per wave merge:** `uv run pytest tests/unit/ -x --timeout=120`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/unit/test_portfolio_orchestrator.py` -- covers PORT-01, PORT-02, PORT-03
- [ ] No framework install needed -- pytest already configured

## Sources

### Primary (HIGH confidence)
- `src/finalayze/backtest/bond_engine.py` -- BondBacktestEngine API, BondBacktestResult dataclass
- `src/finalayze/backtest/engine.py` -- BacktestEngine.run() signature, returns tuple[list[TradeResult], list[PortfolioState]]
- `src/finalayze/backtest/portfolio_aggregator.py` -- LayerResult, PortfolioAggregator._align_curves, forward-fill pattern
- `src/finalayze/backtest/performance.py` -- PerformanceAnalyzer.analyze(), Sharpe/DD computation
- `src/finalayze/backtest/walk_forward.py` -- WalkForwardOptimizer, window generation
- `src/finalayze/backtest/bond_walk_forward.py` -- run_bond_walk_forward, excess Sharpe from equity curve
- `scripts/run_iteration.py` -- USDRUB data extraction from MarketContext, segment iteration pattern

### Secondary (MEDIUM confidence)
- `src/finalayze/data/fetchers/cbr.py` -- MacroSnapshot.usdrub field, MacroContextProvider

### Tertiary (LOW confidence)
- None -- all findings verified from codebase inspection

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- all libraries already in use, no new dependencies
- Architecture: HIGH -- patterns directly observed in existing PortfolioAggregator and bond_walk_forward
- Pitfalls: HIGH -- identified from concrete type mismatches and data patterns in codebase

**Research date:** 2026-03-21
**Valid until:** 2026-04-21 (stable domain, no external dependencies)
