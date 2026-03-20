# Phase 9: Strategy Wiring - Research

**Researched:** 2026-03-20
**Domain:** MOEX strategy integration, position sizing pipeline, combiner routing
**Confidence:** HIGH

## Summary

Phase 9 wires four existing but disconnected MOEX components into the live backtest pipeline: DividendGapStrategy with per-symbol hold bars, CBRStrategyWrapper in the combiner, and two new sizing pipeline steps (BrentGateStep, RubOilRegimeStep). All core logic already exists -- the work is integration, not invention.

The codebase is well-structured for this. The `PositionSizingStep` protocol is a clean `adjust(size, context) -> Decimal` interface with six existing implementations to follow. The combiner already has ADX regime routing with `_MOMENTUM_STRATEGIES` and `_MR_STRATEGIES` frozensets, so adding `_EVENT_STRATEGIES` follows the same pattern. The preset YAML files for ru_finance and ru_tech already contain `cbr_calendar` entries; ru_blue_chips and ru_energy need them added.

**Primary recommendation:** Follow existing patterns exactly -- new sizing steps mirror `RegimeStep`, event strategy bypass mirrors reinforcer strategy handling, and per-symbol hold bars use the existing `resolve_max_hold_bars()` dict-based dispatch.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- BrentGateStep: threshold 5000 RUB/bbl, scale=0.5, energy sector only, data from MOEX ISS Brent USD x USDRUB
- RubOilRegimeStep: wraps existing RubOilRegimeSignal, NORMAL=1.0/ELEVATED=0.5/CRISIS=0.25, after RegimeStep before HardCaps, ru_* only
- DividendGap hold bars: yield-based tiers (>8% -> 60, 5-8% -> 40, <5% -> 25), `_EVENT_STRATEGIES` bypass ADX gating, min_combined_confidence floor 0.40 when event fires, default hold bars updated from 15 to 40
- CBRStrategyWrapper: registered in combiner for ru_* segments via preset YAML

### Claude's Discretion
- Internal implementation details for SizingStep interface adaptation
- Test structure and fixture design
- Error handling for missing Brent/USDRUB data (graceful degradation to scale=1.0)

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| STRAT-01 | DividendGapStrategy calendar from expanded YAML, `_EVENT_STRATEGIES` bypass in combiner | Combiner routing pattern found (lines 34-36), preset YAMLs already have dividend_gap enabled, `DEFAULT_STRATEGY_HOLD_BARS` needs update from 15->40 |
| STRAT-02 | CBRStrategyWrapper wired into combiner for CBR rate decisions | CBRStrategyWrapper fully implemented, already in ru_finance/ru_tech presets, missing from ru_blue_chips/ru_energy presets |
| STRAT-03 | RubOilRegimeSignal integrated as RubOilRegimeStep in sizing pipeline | `RubOilRegimeSignal.get_regime()` returns `RegimeState` with position_scale, wrap as `PositionSizingStep.adjust()` |
| STRAT-04 | BrentGateStep in sizing pipeline for energy sector gating | New step, follows `RegimeStep` pattern, needs Brent-in-RUB computation in `SizingContext` |
</phase_requirements>

## Standard Stack

No new libraries needed. All implementation uses existing codebase modules.

### Core
| Module | Purpose | Why Standard |
|--------|---------|--------------|
| `risk/position_sizing_pipeline.py` | `PositionSizingStep` protocol, pipeline chain | All sizing steps use this |
| `strategies/combiner.py` | `StrategyCombiner` with ADX routing | Strategy signal combination |
| `strategies/dividend_gap.py` | `DividendGapStrategy` | Already fully implemented |
| `strategies/cbr_strategy_wrapper.py` | `CBRStrategyWrapper` | Already fully implemented |
| `risk/rub_oil_regime.py` | `RubOilRegimeSignal` correlation logic | Reuse for RubOilRegimeStep |
| `backtest/config.py` | Hold bars and ATR stop defaults | Per-strategy configuration |

### Supporting
| Module | Purpose | When to Use |
|--------|---------|-------------|
| `strategies/presets/*.yaml` | Per-segment strategy weights | Register cbr_calendar in all ru_* |
| `scripts/run_iteration.py` | Strategy setup and pipeline wiring | Integration point for new steps |
| `backtest/engine.py` | Pipeline step chain construction | Insert new steps in correct order |

## Architecture Patterns

### Pattern 1: PositionSizingStep Protocol
**What:** Every sizing adjustment implements `adjust(size: Decimal, context: SizingContext) -> Decimal`
**When to use:** BrentGateStep, RubOilRegimeStep
**Example (from existing RegimeStep):**
```python
class RegimeStep:
    """Scale position by regime_scale with a floor of 0.15."""
    def adjust(self, size: Decimal, context: SizingContext) -> Decimal:
        scale = max(context.regime_scale, _REGIME_FLOOR)
        return (size * scale).quantize(_FOUR_DP, rounding=ROUND_HALF_UP)
```

**Key insight:** Steps receive `SizingContext` (frozen dataclass) and return adjusted `Decimal`. They do NOT mutate context. The pipeline chains: `KellyStep -> VolTargetStep -> RegimeStep -> [new steps] -> MetaLabelStep -> HardCapsStep`.

### Pattern 2: Combiner Strategy Pool Routing
**What:** Frozenset-based strategy classification gates signal generation by ADX regime
**When to use:** Adding `_EVENT_STRATEGIES` bypass
**Existing pattern:**
```python
_MOMENTUM_STRATEGIES = frozenset({"momentum", "dual_momentum"})
_MR_STRATEGIES = frozenset({"mean_reversion", "pairs", "ou_mean_reversion", "rsi2_connors"})
_REINFORCER_STRATEGIES = frozenset({"ml_ensemble"})
```
**New addition:** `_EVENT_STRATEGIES = frozenset({"dividend_gap", "cbr_calendar"})` -- strategies in this set skip ADX gating entirely (lines 349-357 in combiner.py).

### Pattern 3: Preset YAML Registration
**What:** Each strategy must be listed in segment YAML with enabled/weight/params
**When to use:** Adding cbr_calendar to ru_blue_chips and ru_energy
**Current state:**
- ru_blue_chips.yaml: has dividend_gap (weight=0.17), NO cbr_calendar
- ru_energy.yaml: has dividend_gap (weight=0.19), NO cbr_calendar
- ru_finance.yaml: has both (dividend_gap weight=0.15, cbr_calendar weight=0.10)
- ru_tech.yaml: has both (dividend_gap weight=0.20, cbr_calendar weight=0.05)

### Pattern 4: Dict-based Max Hold Bars
**What:** `resolve_max_hold_bars()` supports both `int` and `dict[str, int]` dispatch
**When to use:** Per-symbol hold bars for dividend_gap
**Key insight:** The strategy's internal `_max_hold_bars` already controls exit logic (line 164 in dividend_gap.py). The backtest config `DEFAULT_STRATEGY_HOLD_BARS` is the engine's fallback. For yield-based tiers, the logic goes inside `DividendGapStrategy.__init__` or a helper that sets per-entry hold bars based on gap_pct (which correlates with yield).

### Recommended Pipeline Step Order
```
KellyStep -> VolTargetStep -> RegimeStep -> RubOilRegimeStep -> BrentGateStep -> MetaLabelStep -> HardCapsStep
```
Rationale: RubOilRegimeStep is a broader regime adjustment (all ru_* equities), BrentGateStep is sector-specific (energy only). Both go after the generic RegimeStep and before ML/caps.

### Anti-Patterns to Avoid
- **Adding fields to frozen SizingContext:** The dataclass is `frozen=True`. Instead, pass Brent/RUB data via step constructor (inject at engine init time), not via context fields.
- **Modifying combiner weights for event strategies:** Event strategies should bypass ADX gating but still participate in weighted scoring. Do NOT give them special weight handling.
- **Force-closing dividend positions via engine max_hold_bars:** The strategy itself manages gap closure exits. The engine hold bar limit is a safety net. Set it to 40 (the decision default) but the yield-based tiers happen inside the strategy.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| RUB/oil correlation | Custom correlation computation | `RubOilRegimeSignal` from `rub_oil_regime.py` | Already handles window, Pearson, threshold mapping |
| CBR signal logic | New CBR signal generation | `CBRStrategyWrapper` + `CBRCalendar` | Full contrarian logic implemented |
| Dividend gap detection | New dividend matcher | `DividendGapStrategy` | 3-tier calendar loading, gap tracking, exit logic |
| Position scale mapping | Custom regime->scale mapping | `RegimeState.position_scale` | Already maps NORMAL/ELEVATED/CRISIS to scales |

## Common Pitfalls

### Pitfall 1: SizingContext is Frozen
**What goes wrong:** Trying to add `brent_rub_price` or `rub_oil_regime` fields to SizingContext for new steps.
**Why it happens:** Seems natural to pass data through context, but it's `frozen=True` with `slots=True`.
**How to avoid:** Inject data via step constructor. BrentGateStep takes `brent_rub_price: float` and `segment_id: str` at init time. RubOilRegimeStep takes `RubOilRegimeSignal` instance at init.
**Warning signs:** `dataclasses.FrozenInstanceError` at runtime, or adding new optional fields breaks all existing tests.

### Pitfall 2: Event Strategy Dilution in Combiner
**What goes wrong:** DividendGap fires with confidence 0.65 but combined score drops below 0.50 threshold because other strategies fire SELL or don't fire.
**Why it happens:** Default `normalize_mode="firing"` divides by total firing weight. If dividend_gap (weight=0.17) fires alone, net = 0.65, but if another strategy fires SELL, it dilutes.
**How to avoid:** When event strategy fires, apply min_combined_confidence floor of 0.40 (lower than default 0.50) so the signal passes the threshold check. The `_EVENT_STRATEGIES` frozenset marks these strategies.
**Warning signs:** DividendGap has calendar entries but generates zero trades in backtest.

### Pitfall 3: cbr_calendar Missing from Presets
**What goes wrong:** CBRStrategyWrapper is appended to strategy list but combiner ignores it because the preset YAML doesn't have `cbr_calendar` entry.
**Why it happens:** Combiner iterates `strategies_cfg` from YAML, not the registered strategy objects.
**How to avoid:** Add `cbr_calendar` to ALL four ru_* preset YAMLs (ru_blue_chips and ru_energy are missing it).
**Warning signs:** `_setup_cbr_strategy` returns a valid strategy but combiner never calls `generate_signal` on it.

### Pitfall 4: Dividend Hold Bars Engine vs Strategy
**What goes wrong:** Engine force-closes at 15 bars (old default) even though strategy wants to hold for 60 bars on high-yield dividends.
**Why it happens:** `DEFAULT_STRATEGY_HOLD_BARS["dividend_gap"] = 15` in backtest/config.py is the ENGINE's max hold. The strategy's own `_max_hold_bars` is independent.
**How to avoid:** Update `DEFAULT_STRATEGY_HOLD_BARS["dividend_gap"]` to 60 (max of all tiers). The strategy's yield-based logic will close earlier when appropriate. The engine hold bar is a safety ceiling.
**Warning signs:** All dividend positions closed at exactly 15 bars regardless of yield.

### Pitfall 5: BrentGateStep Applied to Non-Energy
**What goes wrong:** All ru_* positions get Brent-gated, not just energy.
**Why it happens:** Pipeline steps don't have access to segment_id by default.
**How to avoid:** Inject segment_id into BrentGateStep constructor. Step returns `size` unchanged when segment != "ru_energy". Same pattern for RubOilRegimeStep (only ru_*).
**Warning signs:** Positions in ru_finance being scaled by Brent threshold.

## Code Examples

### BrentGateStep (new, follows RegimeStep pattern)
```python
class BrentGateStep:
    """Gate energy sector positions when Brent-in-RUB below threshold."""

    def __init__(
        self,
        brent_rub_price: float,
        segment_id: str,
        threshold: float = 5000.0,
        scale_below: Decimal = Decimal("0.5"),
    ) -> None:
        self._brent_rub = brent_rub_price
        self._segment_id = segment_id
        self._threshold = threshold
        self._scale_below = scale_below

    def adjust(self, size: Decimal, context: SizingContext) -> Decimal:
        if self._segment_id != "ru_energy":
            return size
        if self._brent_rub >= self._threshold:
            return size
        return (size * self._scale_below).quantize(_FOUR_DP, rounding=ROUND_HALF_UP)
```

### RubOilRegimeStep (wraps existing RubOilRegimeSignal)
```python
class RubOilRegimeStep:
    """Scale positions by RUB/oil decorrelation regime."""

    def __init__(
        self,
        regime_signal: RubOilRegimeSignal,
        segment_id: str,
    ) -> None:
        self._regime_signal = regime_signal
        self._segment_id = segment_id

    def adjust(self, size: Decimal, context: SizingContext) -> Decimal:
        if not self._segment_id.startswith("ru_"):
            return size
        # candles/bar_index unused by RubOilRegimeSignal (uses stored data)
        state = self._regime_signal.get_regime([], 0)
        return (size * state.position_scale).quantize(_FOUR_DP, rounding=ROUND_HALF_UP)
```

### Combiner Event Strategy Bypass
```python
# Add to combiner.py constants (line ~37):
_EVENT_STRATEGIES = frozenset({"dividend_gap", "cbr_calendar"})

# In generate_signal(), after ADX regime gating block (lines 349-357):
is_event = strategy_name in _EVENT_STRATEGIES
# Event strategies skip ADX gating:
if not is_event:
    if regime == "trend" and is_mr:
        continue
    if regime == "mr" and is_trend:
        continue
```

### Per-Symbol Hold Bars in DividendGapStrategy
```python
def _yield_hold_bars(self, gap_pct: float) -> int:
    """Map dividend yield (gap %) to max hold bars."""
    if gap_pct >= 8.0:
        return 60
    if gap_pct >= 5.0:
        return 40
    return 25
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Fixed 15-bar hold for dividend_gap | Yield-based tiers (25/40/60) | Phase 9 | High-yield dividends get proper recovery window |
| All strategies subject to ADX routing | Event strategies bypass ADX | Phase 9 | DividendGap/CBR fire regardless of regime |
| No energy sector gating | BrentGateStep at 5000 RUB/bbl | Phase 9 | Energy positions reduced when oil weak |
| RubOilRegimeSignal standalone | Integrated as sizing pipeline step | Phase 9 | Automatic portfolio-wide RUB/oil risk adjustment |

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest (via uv run) |
| Config file | pyproject.toml [tool.pytest.ini_options] |
| Quick run command | `uv run pytest tests/unit/ -x -q` |
| Full suite command | `uv run pytest tests/ --cov -q` |

### Phase Requirements -> Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| STRAT-01a | DividendGap per-symbol hold bars (yield tiers) | unit | `uv run pytest tests/unit/test_dividend_gap.py -x` | Partial (exists but no yield-tier tests) |
| STRAT-01b | `_EVENT_STRATEGIES` bypass ADX in combiner | unit | `uv run pytest tests/unit/test_strategy_combiner.py -x` | No (new tests needed) |
| STRAT-01c | min_combined_confidence floor for event signals | unit | `uv run pytest tests/unit/test_strategy_combiner.py -x` | No (new tests needed) |
| STRAT-02 | CBRStrategyWrapper registered in all ru_* presets | unit | `uv run pytest tests/unit/test_strategy_combiner.py -x` | No (preset validation needed) |
| STRAT-03 | RubOilRegimeStep scales positions by regime | unit | `uv run pytest tests/unit/test_position_sizing_pipeline.py -x` | No (new step tests) |
| STRAT-04 | BrentGateStep gates energy when Brent < threshold | unit | `uv run pytest tests/unit/test_position_sizing_pipeline.py -x` | No (new step tests) |

### Sampling Rate
- **Per task commit:** `uv run pytest tests/unit/ -x -q`
- **Per wave merge:** `uv run pytest tests/ --cov -q`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/unit/test_position_sizing_pipeline.py` -- add BrentGateStep and RubOilRegimeStep tests
- [ ] `tests/unit/test_strategy_combiner.py` -- add _EVENT_STRATEGIES bypass and confidence floor tests
- [ ] `tests/unit/test_dividend_gap.py` -- add yield-based hold bar tier tests

## Open Questions

1. **Brent-in-RUB data availability during backtest**
   - What we know: MOEX ISS provides both Brent USD and USDRUB. `run_iteration.py` already fetches IMOEX candles for regime provider.
   - What's unclear: Whether Brent/USDRUB candles are fetched or need to be added to the iteration script's data loading.
   - Recommendation: Add Brent/USDRUB fetch to `run_iteration.py` (same pattern as IMOEX fetch), compute product, pass to BrentGateStep constructor. Graceful degradation: if fetch fails, skip the step (scale=1.0).

2. **CBR event data for backtests**
   - What we know: `_setup_cbr_strategy` reads from `event_data["cbr"]` which comes from `results/event_data/cbr/decisions.json`.
   - What's unclear: Whether this JSON file has sufficient historical events for meaningful backtest coverage.
   - Recommendation: Verify file exists and has events. If sparse, the strategy will simply generate fewer signals, which is fine for initial wiring.

## Sources

### Primary (HIGH confidence)
- `src/finalayze/risk/position_sizing_pipeline.py` -- SizingStep protocol, pipeline construction
- `src/finalayze/strategies/combiner.py` -- ADX routing, strategy pool constants, signal combination
- `src/finalayze/strategies/dividend_gap.py` -- Full strategy implementation with calendar and gap tracking
- `src/finalayze/strategies/cbr_strategy_wrapper.py` -- CBR wrapper, already BaseStrategy-compliant
- `src/finalayze/risk/rub_oil_regime.py` -- Correlation logic, RegimeState mapping
- `src/finalayze/backtest/config.py` -- Hold bars and ATR stop defaults
- `src/finalayze/backtest/engine.py` -- Pipeline step chain construction (lines 157-164)
- `scripts/run_iteration.py` -- Strategy setup functions (lines 362-568)
- `src/finalayze/strategies/presets/ru_*.yaml` -- All four MOEX segment presets

### Secondary (MEDIUM confidence)
- CONTEXT.md decisions -- User-locked thresholds and architectural choices

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- all modules exist, read directly from source
- Architecture: HIGH -- patterns verified from 6+ existing implementations
- Pitfalls: HIGH -- identified from actual code structure (frozen dataclass, preset-driven combiner, engine hold bars)

**Research date:** 2026-03-20
**Valid until:** 2026-04-20 (stable domain, no external dependency changes expected)
