# Phase 10: Macro Regime - Context

**Gathered:** 2026-03-20
**Status:** Ready for planning

<domain>
## Phase Boundary

Add macro regime gating to MOEX equity and bond allocation: CBRRegimeStep sizes equity by yield curve slope, OFZ PK-to-PD rotation shifts bond allocation when CBR cutting cycle detected, and SectorAllocationStep adjusts energy/financials weights based on Brent and CBR direction.

</domain>

<decisions>
## Implementation Decisions

### CBRRegimeStep Design
- Yield curve slope metric: 10Y-2Y spread from CBR ZCYC (standard steepening/flattening indicator)
- 3-tier regime classification: Steepening (>100bps) = cutting expected → scale equity 1.2x; Flat (0-100bps) = neutral 1.0x; Inverted (<0bps) = hiking expected → scale down 0.6x
- Pipeline position: stacks after RegimeStep — CBRRegimeStep multiplies on top of VIX/ADX regime scale
- Data source: MacroContextProvider.get_snapshot() yield_curve dict — already look-ahead safe for backtests

### OFZ PK-to-PD Rotation
- Cutting cycle detection: 2+ consecutive cuts from MacroSnapshot.last_cbr_decision history (matches success criteria)
- Logic location: OFZ rotation logic in bond_cycle.py (class or function at Claude's discretion) — modifies LayerConfig.capital_pct dynamically before _process_layer
- PK→PD shift: CORE (PK floaters) drops from 40%→25%, STRATEGIC (PD fixed) rises from 30%→45% — 15% shift toward duration
- Revert trigger: first hike after cutting cycle reverts to default allocations — simple, avoids oscillation

### SectorAllocationStep Design
- Sector index data: fetch MOEXOG (oil&gas) and MOEXFN (financials) via MoexISSFetcher — same API as IMOEX
- Energy overweight: Brent >6000 RUB/bbl → scale ru_energy 1.3x; <4000 → 0.7x; between → 1.0x
- Financials CBR sensitivity: CBR cutting → scale ru_finance 1.2x; hiking → 0.8x; hold → 1.0x
- Pipeline position: after CBRRegimeStep, before HardCaps — sector adjustment after macro regime, before position limits

### Claude's Discretion
- Internal implementation details for MacroContextProvider yield curve history
- Test structure and fixture design for macro regime steps
- Error handling for missing yield curve data (graceful degradation to scale=1.0)
- How to pass CBR decision history to OFZRotationTrigger (likely via MacroCacheService.get_history())

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `MacroSnapshot` dataclass in `data/fetchers/cbr.py` — already has yield_curve dict, last_cbr_decision, key_rate
- `MacroContextProvider` — look-ahead safe, built-in CBR meeting calendar 2022-2026
- `MacroCacheService` in `data/macro_cache.py` — caching + history + DB persistence
- `MoexISSFetcher` in `data/fetchers/moex_iss.py` — can fetch any MOEX index candles
- `BondDurationRotationStrategy` — existing regime classifier (DOVISH/NEUTRAL/HAWKISH)
- `BondCycleProcessor` — already receives MacroSnapshot, passes to strategies
- `LayerConfig` with `capital_pct` field — ready for dynamic adjustment

### Established Patterns
- Position sizing steps: `SizingStep` protocol with `adjust(size, context) -> Decimal`
- Pipeline: Kelly → VolTarget → Regime → RubOilRegime → BrentGate → MetaLabel → HardCaps
- Bond layers: CORE=40% PK, STRATEGIC=30% PD, TACTICAL=20%, SHORT=10%
- Macro data flow: CBRFetcher → MacroContextProvider → MacroCacheService → bond_cycle/strategies

### Integration Points
- `position_sizing_pipeline.py` — add CBRRegimeStep and SectorAllocationStep
- `bond_cycle.py` — add OFZRotationTrigger before _process_layer calls
- `run_iteration.py` — pass yield_curve slope and sector data to BacktestConfig
- `backtest/engine.py` — wire new steps into _build_sizing_pipeline()
- `backtest/config.py` — add new config fields (yield_slope, cbr_decision_history, sector data)

</code_context>

<specifics>
## Specific Ideas

No specific requirements — standard wiring using established patterns.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>
