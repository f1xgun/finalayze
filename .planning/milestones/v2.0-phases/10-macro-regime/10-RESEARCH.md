# Phase 10: Macro Regime - Research

**Researched:** 2026-03-20
**Domain:** MOEX macro regime gating (CBR yield curve, OFZ rotation, sector allocation)
**Confidence:** HIGH

## Summary

Phase 10 adds three macro regime components to the existing MOEX trading infrastructure: (1) CBRRegimeStep in the equity sizing pipeline that uses yield curve slope to scale positions, (2) OFZ PK-to-PD rotation that shifts bond layer allocations when a cutting cycle is detected, and (3) SectorAllocationStep that adjusts sector weights based on Brent prices and CBR direction.

All three components follow established patterns from Phase 9 (RubOilRegimeStep, BrentGateStep). The sizing pipeline uses a `PositionSizingStep` protocol with `adjust(size, context) -> Decimal`. The bond cycle processor already receives `MacroSnapshot` via `MacroCacheService`. The main gap is that `MacroContextProvider.get_snapshot()` does not currently populate the `yield_curve` field -- it returns `None` for yield curve. This must be fixed with a static yield curve dataset for backtesting (CBR ZCYC fetcher exists but is live-only).

**Primary recommendation:** Follow the Phase 9 pattern exactly -- add new fields to BacktestConfig, create new SizingStep classes in position_sizing_pipeline.py, wire them in engine._build_sizing_pipeline(), and pass data from run_iteration.py. For OFZ rotation, create a helper that returns modified LayerConfig dicts (since LayerConfig is frozen).

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- Yield curve slope metric: 10Y-2Y spread from CBR ZCYC (standard steepening/flattening indicator)
- 3-tier regime classification: Steepening (>100bps) = cutting expected, scale equity 1.2x; Flat (0-100bps) = neutral 1.0x; Inverted (<0bps) = hiking expected, scale down 0.6x
- Pipeline position: stacks after RegimeStep -- CBRRegimeStep multiplies on top of VIX/ADX regime scale
- Data source: MacroContextProvider.get_snapshot() yield_curve dict -- already look-ahead safe for backtests
- Cutting cycle detection: 2+ consecutive cuts from MacroSnapshot.last_cbr_decision history (matches success criteria)
- Logic location: new OFZRotationTrigger class in bond_cycle.py -- modifies LayerConfig.capital_pct dynamically before _process_layer
- PK to PD shift: CORE (PK floaters) drops from 40% to 25%, STRATEGIC (PD fixed) rises from 30% to 45% -- 15% shift toward duration
- Revert trigger: first hike after cutting cycle reverts to default allocations -- simple, avoids oscillation
- Sector index data: fetch MOEXOG (oil&gas) and MOEXFN (financials) via MoexISSFetcher -- same API as IMOEX
- Energy overweight: Brent >6000 RUB/bbl scale ru_energy 1.3x; <4000 scale 0.7x; between 1.0x
- Financials CBR sensitivity: CBR cutting scale ru_finance 1.2x; hiking 0.8x; hold 1.0x
- Pipeline position: SectorAllocationStep after CBRRegimeStep, before HardCaps

### Claude's Discretion
- Internal implementation details for MacroContextProvider yield curve history
- Test structure and fixture design for macro regime steps
- Error handling for missing yield curve data (graceful degradation to scale=1.0)
- How to pass CBR decision history to OFZRotationTrigger (likely via MacroCacheService.get_history())

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| MACRO-01 | CBRRegimeStep in sizing pipeline -- CBR rate level + direction affects equity allocation sizing | Yield curve slope (10Y-2Y) from static ZCYC data; 3-tier regime; new SizingStep in pipeline after RegimeStep |
| MACRO-02 | OFZ PK-to-PD rotation trigger -- detects CBR cutting cycle start for bond allocation shift | OFZRotationTrigger using CBR_MEETINGS history; creates new frozen LayerConfig dicts with shifted capital_pct |
| MACRO-03 | SectorAllocationStep in sizing pipeline for sector rotation using MOEX sector indices | New SizingStep using segment_id + brent_rub + cbr_direction; wired after CBRRegimeStep |
</phase_requirements>

## Standard Stack

### Core
No new libraries needed. All implementation uses existing codebase infrastructure.

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| Python stdlib Decimal | 3.12 | Financial calculations in sizing steps | Project convention: all monetary values use Decimal |
| dataclasses | 3.12 | Frozen dataclass for config/context | Matches BacktestConfig, SizingContext patterns |

### Supporting
All existing project dependencies -- no new packages required.

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| structlog | existing | Logging in new steps | All new classes should use structlog |
| httpx | existing | MoexISSFetcher already uses it | Only if sector index fetching needed in live mode |

### Alternatives Considered
None -- REQUIREMENTS.md explicitly states "No new pip dependencies".

## Architecture Patterns

### Recommended Project Structure
```
src/finalayze/
├── risk/
│   └── position_sizing_pipeline.py  # Add CBRRegimeStep + SectorAllocationStep
├── data/
│   └── fetchers/cbr.py             # Add static yield curve data + helper functions
├── core/
│   └── bond_cycle.py               # Add OFZRotationTrigger class
├── backtest/
│   ├── config.py                   # Add new BacktestConfig fields
│   └── engine.py                   # Wire new steps in _build_sizing_pipeline()
└── scripts/
    └── run_iteration.py            # Pass yield slope + sector data to BacktestConfig
```

### Pattern 1: PositionSizingStep Protocol (established)
**What:** Each sizing adjustment is a class implementing `adjust(size: Decimal, context: SizingContext) -> Decimal`.
**When to use:** CBRRegimeStep and SectorAllocationStep both follow this.
**Example:**
```python
# Source: src/finalayze/risk/position_sizing_pipeline.py (existing pattern)
class CBRRegimeStep:
    """Scale equity positions by CBR yield curve slope regime."""

    def __init__(self, yield_slope_bps: float, segment_id: str) -> None:
        self._yield_slope_bps = yield_slope_bps
        self._segment_id = segment_id

    def adjust(self, size: Decimal, context: SizingContext) -> Decimal:
        if not self._segment_id.startswith("ru_"):
            return size  # Only affects MOEX segments
        scale = self._classify_regime()
        return (size * scale).quantize(_FOUR_DP, rounding=ROUND_HALF_UP)

    def _classify_regime(self) -> Decimal:
        if self._yield_slope_bps > 100:
            return Decimal("1.2")   # Steepening: cutting expected
        elif self._yield_slope_bps < 0:
            return Decimal("0.6")   # Inverted: hiking expected
        return Decimal("1.0")       # Flat: neutral
```

### Pattern 2: Segment-Aware Sizing Step (from BrentGateStep)
**What:** Steps that check `segment_id` and only apply to specific MOEX segments.
**When to use:** SectorAllocationStep checks for `ru_energy` and `ru_finance`.
**Example:**
```python
# Source: src/finalayze/risk/position_sizing_pipeline.py line 145-171
class SectorAllocationStep:
    def __init__(self, brent_rub_price: float, cbr_direction: str, segment_id: str) -> None:
        self._brent_rub = brent_rub_price
        self._cbr_direction = cbr_direction
        self._segment_id = segment_id

    def adjust(self, size: Decimal, context: SizingContext) -> Decimal:
        if self._segment_id == "ru_energy":
            return self._energy_scale(size)
        if self._segment_id == "ru_finance":
            return self._finance_scale(size)
        return size
```

### Pattern 3: OFZ Rotation via LayerConfig Replacement
**What:** Since LayerConfig is `@dataclass(frozen=True)`, rotation creates new configs with adjusted `capital_pct`.
**When to use:** OFZRotationTrigger before `_process_layer` calls in BondCycleProcessor.
**Key insight:** Cannot mutate frozen dataclass. Must use `dataclasses.replace()` to create modified copies.
**Example:**
```python
from dataclasses import replace

def apply_rotation(
    configs: dict[PortfolioLayer, LayerConfig],
    cutting_cycle: bool,
) -> dict[PortfolioLayer, LayerConfig]:
    if not cutting_cycle:
        return configs
    result = dict(configs)
    core = configs[PortfolioLayer.CORE]
    strategic = configs[PortfolioLayer.STRATEGIC]
    result[PortfolioLayer.CORE] = replace(core, capital_pct=Decimal("0.25"))
    result[PortfolioLayer.STRATEGIC] = replace(strategic, capital_pct=Decimal("0.45"))
    return result
```

### Pattern 4: Static Data for Backtesting (from CBR_MEETINGS)
**What:** Hardcoded historical data in cbr.py for look-ahead-safe backtesting.
**When to use:** Yield curve slope data for backtesting. CBR ZCYC fetcher is live-only (HTTP scraping); backtests need static data.
**Key insight:** `MacroContextProvider.get_snapshot()` currently returns `yield_curve=None`. Must add a static yield curve slope dataset keyed by date, similar to `_CPI_DATA` and `CBR_MEETINGS`.

### Anti-Patterns to Avoid
- **Mutating frozen LayerConfig:** Cannot set `config.capital_pct = X`. Use `dataclasses.replace()`.
- **Fetching live data in backtests:** ZCYC HTML scraping from cbr.ru cannot be used during backtesting. Must use static historical data.
- **Putting sector rotation in combiner:** REQUIREMENTS.md and STATE.md explicitly state "Sector rotation MUST be in sizing pipeline, NOT combiner".
- **Adding yield_slope to SizingContext:** SizingContext is frozen and shared across all steps. New per-step data should be passed via constructor (like BrentGateStep receives `brent_rub_price` directly).

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Yield curve slope | Full yield curve interpolation | Simple 10Y-2Y spread from static data | Overcomplicated; only need slope sign and magnitude |
| CBR decision history | Complex stateful tracker | Filter CBR_MEETINGS tuple by date | Already exists, look-ahead safe, complete 2022-2026 |
| Bond config replacement | Mutable LayerConfig | `dataclasses.replace()` on frozen configs | Maintains immutability invariant |
| Sector index candles | New fetcher class | Existing `MoexISSFetcher.fetch_candles("MOEXOG", ...)` | Same ISS API, already handles pagination/retry |

**Key insight:** Almost everything needed already exists in the codebase. The main work is wiring, not building.

## Common Pitfalls

### Pitfall 1: MacroContextProvider Does Not Populate yield_curve
**What goes wrong:** `get_snapshot()` returns `yield_curve=None` -- the field exists on MacroSnapshot but is never set in the backtest path.
**Why it happens:** `fetch_yield_curve()` is a live HTTP call to cbr.ru. Backtests use static data.
**How to avoid:** Add a static `_YIELD_CURVE_SLOPE_DATA: dict[str, float]` in cbr.py (date -> 10Y-2Y spread in bps) and populate it in `MacroContextProvider.get_snapshot()`. Alternatively, compute yield slope separately and pass directly to BacktestConfig (simpler, matches Phase 9 pattern).
**Warning signs:** yield_slope is always 0.0 or None in backtest runs.
**Recommendation:** Follow Phase 9 pattern -- add `yield_slope_bps: float = 0.0` to BacktestConfig, compute in run_iteration.py from static data, pass to CBRRegimeStep constructor. Simpler than modifying MacroContextProvider.

### Pitfall 2: Frozen LayerConfig in OFZ Rotation
**What goes wrong:** Attempting to set `config.capital_pct = Decimal("0.25")` raises `FrozenInstanceError`.
**Why it happens:** LayerConfig is `@dataclass(frozen=True)`.
**How to avoid:** Use `dataclasses.replace(config, capital_pct=Decimal("0.25"))`.
**Warning signs:** FrozenInstanceError in bond_cycle.py.

### Pitfall 3: Cutting Cycle Detection Window
**What goes wrong:** Looking at only `last_cbr_decision` misses the "2+ consecutive cuts" requirement.
**Why it happens:** MacroSnapshot only has `last_cbr_decision` (singular), not history.
**How to avoid:** Use `CBR_MEETINGS` tuple directly -- filter meetings before `as_of` date, check last N decisions. This is already look-ahead safe.
**Warning signs:** Rotation triggers on first cut instead of waiting for 2+ consecutive.

### Pitfall 4: Default Layer Configs vs Actual
**What goes wrong:** CONTEXT.md says CORE=40%, STRATEGIC=30%, but DEFAULT_LAYER_CONFIGS shows CORE=45%, STRATEGIC=27.5%.
**Why it happens:** Layer allocations were adjusted during Phase 5-7 and defaults may have drifted.
**How to avoid:** Use the actual DEFAULT_LAYER_CONFIGS values as the baseline, not the CONTEXT.md approximations. The rotation shift amount (15pp) is the invariant, not the absolute numbers.
**Warning signs:** capital_pct values don't sum to ~1.0 after rotation.

### Pitfall 5: Pipeline Order Matters
**What goes wrong:** CBRRegimeStep and SectorAllocationStep interact -- if both scale the same position, the compound effect may be too aggressive.
**Why it happens:** Pipeline is multiplicative. 1.2x (CBR) * 1.3x (sector) = 1.56x.
**How to avoid:** This is by design (they capture different signals), but ensure HardCaps step is always last to enforce max_position_pct. Current pipeline order already ensures this.

## Code Examples

### CBR Decision History Helper
```python
# Source: src/finalayze/data/fetchers/cbr.py (existing CBR_MEETINGS data)
def get_recent_cbr_decisions(as_of: date, count: int = 3) -> list[str]:
    """Return last N CBR decisions (most recent first), look-ahead safe."""
    past = [m for m in CBR_MEETINGS if m.date <= as_of and m.decision is not None]
    return [m.decision for m in reversed(past[-count:])]


def is_cutting_cycle(as_of: date) -> bool:
    """True if last 2+ CBR decisions are 'cut'."""
    decisions = get_recent_cbr_decisions(as_of, count=2)
    return len(decisions) >= 2 and all(d == "cut" for d in decisions)
```

### Static Yield Curve Slope Data
```python
# Add to src/finalayze/data/fetchers/cbr.py
# 10Y-2Y spread in basis points, monthly (first trading day).
# Source: CBR ZCYC zero-coupon yield curve data.
# Positive = normal/steepening, negative = inverted.
_YIELD_CURVE_SLOPE_BPS: dict[str, float] = {
    # 2022 -- inverted during crisis, then normalizing
    "2022-03": -250.0,   # deep inversion during sanctions
    "2022-04": -180.0,
    "2022-06": -50.0,
    "2022-09": 20.0,     # normalizing after cuts
    "2022-12": 80.0,
    # 2023 -- steepening then flattening
    "2023-03": 120.0,
    "2023-06": 100.0,
    "2023-09": -30.0,    # inversion begins with rate hikes
    "2023-12": -80.0,
    # 2024 -- inverted during hiking cycle
    "2024-03": -120.0,
    "2024-06": -150.0,
    "2024-09": -200.0,
    "2024-12": -180.0,
    # 2025 -- still inverted, beginning to normalize
    "2025-03": -150.0,
    "2025-06": -100.0,
    "2025-09": -20.0,    # cutting cycle starts
    "2025-12": 50.0,     # steepening as cuts continue
}
```

### Engine Pipeline Wiring
```python
# In BacktestEngine._build_sizing_pipeline():
# After existing MOEX steps (RubOilRegime, BrentGate), before Copula/EVT/MetaLabel/HardCaps
if cfg.yield_slope_bps != 0.0 or segment_id.startswith("ru_"):
    steps.append(CBRRegimeStep(cfg.yield_slope_bps, segment_id))
if cfg.cbr_direction:
    steps.append(SectorAllocationStep(cfg.brent_rub_price, cfg.cbr_direction, segment_id))
```

### OFZ Rotation in BondCycleProcessor
```python
# In BondCycleProcessor.run_cycle(), before the layer loop:
from dataclasses import replace

effective_configs = self._apply_ofz_rotation(self._layer_configs, macro)

for layer, config in effective_configs.items():
    # ... existing processing ...

def _apply_ofz_rotation(
    self,
    configs: dict[PortfolioLayer, LayerConfig],
    macro: MacroSnapshot,
) -> dict[PortfolioLayer, LayerConfig]:
    """Adjust CORE/STRATEGIC allocations if CBR cutting cycle detected."""
    if not is_cutting_cycle(datetime.now(tz=UTC).date()):
        return configs
    # ... replace logic ...
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| No macro gating on equity | Phase 9: RubOilRegime + BrentGate | Phase 9 (2026-03-20) | MOEX positions gated by commodity regime |
| Static bond layer allocations | Phase 10: Dynamic PK/PD rotation | This phase | Bond allocation responds to rate cycle |
| No sector rotation | Phase 10: SectorAllocationStep | This phase | Energy/finance weights respond to macro |

**Deprecated/outdated:**
- None applicable -- this is new functionality.

## Open Questions

1. **Yield curve slope historical data accuracy**
   - What we know: CBR publishes ZCYC daily; we have a fetcher for live data
   - What's unclear: Exact historical 10Y-2Y spread values for 2022-2025 are approximated in the static dataset above. These should be validated against actual CBR ZCYC publications.
   - Recommendation: Use approximate values for initial implementation. The 3-tier classification (>100bps, 0-100bps, <0bps) is coarse enough that small data errors won't materially affect the regime classification. Flag data as needing validation in a future data quality pass.

2. **MOEX sector index tickers (MOEXOG, MOEXFN) validation**
   - What we know: STATE.md flags "MOEX sector index tickers need live API validation before Phase 10"
   - What's unclear: Whether `MoexISSFetcher.fetch_candles("MOEXOG", ...)` actually returns data
   - Recommendation: SectorAllocationStep does NOT need sector index candles per the CONTEXT.md design. It uses `brent_rub_price` (already available from Phase 9) and `cbr_direction` (from CBR_MEETINGS). Sector index fetching is NOT required for this phase. If needed later, validate with a test fetch before adding.

3. **BondCycleProcessor integration for backtesting**
   - What we know: BondCycleProcessor is a live trading component (Layer 6). OFZ rotation affects live trading.
   - What's unclear: Whether Phase 10 OFZ rotation should also be testable via bond_engine.py backtest path
   - Recommendation: Implement OFZRotationTrigger as a standalone function that takes configs + date and returns modified configs. This can be used by both live BondCycleProcessor and future bond backtesting.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest (existing) |
| Config file | `pyproject.toml` [tool.pytest.ini_options] |
| Quick run command | `uv run pytest tests/unit/test_position_sizing_pipeline.py -x -v` |
| Full suite command | `uv run pytest tests/unit/ -x --timeout=30` |

### Phase Requirements to Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| MACRO-01 | CBRRegimeStep scales equity by yield slope tier | unit | `uv run pytest tests/unit/test_position_sizing_pipeline.py -k "cbr_regime" -x` | No -- Wave 0 |
| MACRO-01 | CBRRegimeStep passes through non-ru_ segments | unit | `uv run pytest tests/unit/test_position_sizing_pipeline.py -k "cbr_regime_passthrough" -x` | No -- Wave 0 |
| MACRO-01 | CBRRegimeStep graceful degradation on missing data | unit | `uv run pytest tests/unit/test_position_sizing_pipeline.py -k "cbr_regime_missing" -x` | No -- Wave 0 |
| MACRO-02 | is_cutting_cycle detects 2+ consecutive cuts | unit | `uv run pytest tests/unit/test_cbr_meeting_calendar.py -k "cutting_cycle" -x` | No -- Wave 0 |
| MACRO-02 | OFZ rotation shifts CORE from 45% to 25%, STRATEGIC from 27.5% to 42.5% | unit | `uv run pytest tests/unit/test_bond_cycle.py -k "ofz_rotation" -x` | No -- Wave 0 |
| MACRO-02 | OFZ rotation reverts on first hike | unit | `uv run pytest tests/unit/test_bond_cycle.py -k "ofz_rotation_revert" -x` | No -- Wave 0 |
| MACRO-03 | SectorAllocationStep scales ru_energy by Brent thresholds | unit | `uv run pytest tests/unit/test_position_sizing_pipeline.py -k "sector_energy" -x` | No -- Wave 0 |
| MACRO-03 | SectorAllocationStep scales ru_finance by CBR direction | unit | `uv run pytest tests/unit/test_position_sizing_pipeline.py -k "sector_finance" -x` | No -- Wave 0 |
| MACRO-03 | SectorAllocationStep passes through non-sector segments | unit | `uv run pytest tests/unit/test_position_sizing_pipeline.py -k "sector_passthrough" -x` | No -- Wave 0 |

### Sampling Rate
- **Per task commit:** `uv run pytest tests/unit/test_position_sizing_pipeline.py tests/unit/test_bond_cycle.py tests/unit/test_cbr_meeting_calendar.py -x -v`
- **Per wave merge:** `uv run pytest tests/unit/ -x --timeout=30`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/unit/test_position_sizing_pipeline.py` -- add CBRRegimeStep and SectorAllocationStep tests (file exists, add test cases)
- [ ] `tests/unit/test_cbr_meeting_calendar.py` -- add is_cutting_cycle / get_recent_cbr_decisions tests (file exists, add test cases)
- [ ] `tests/unit/test_bond_cycle.py` -- add OFZ rotation tests (file exists, add test cases)
- Framework install: Not needed -- pytest already installed and configured

## Sources

### Primary (HIGH confidence)
- `src/finalayze/risk/position_sizing_pipeline.py` -- PositionSizingStep protocol, existing step implementations (BrentGateStep, RubOilRegimeStep)
- `src/finalayze/data/fetchers/cbr.py` -- CBR_MEETINGS calendar (complete 2022-2026), MacroSnapshot dataclass, yield curve fetcher
- `src/finalayze/core/bond_cycle.py` -- BondCycleProcessor, LayerConfig usage, macro data flow
- `src/finalayze/backtest/engine.py` -- _build_sizing_pipeline() wiring pattern
- `src/finalayze/backtest/config.py` -- BacktestConfig field pattern for MOEX steps
- `src/finalayze/core/schemas.py` -- LayerConfig frozen dataclass, DEFAULT_LAYER_CONFIGS with actual percentages

### Secondary (MEDIUM confidence)
- CONTEXT.md design decisions -- user-validated thresholds and architecture choices
- STATE.md accumulated decisions -- "Sector rotation MUST be in sizing pipeline, NOT combiner"

### Tertiary (LOW confidence)
- Yield curve slope historical values -- approximate, needs validation against actual CBR ZCYC data

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- no new dependencies, all existing patterns
- Architecture: HIGH -- follows established Phase 9 patterns exactly
- Pitfalls: HIGH -- identified from direct codebase inspection (frozen LayerConfig, missing yield_curve, etc.)
- Historical data accuracy: LOW -- yield curve slope values are estimates

**Research date:** 2026-03-20
**Valid until:** 2026-04-20 (stable domain, internal codebase patterns)
