# Phase 9: Strategy Wiring - Context

**Gathered:** 2026-03-20
**Status:** Ready for planning

<domain>
## Phase Boundary

Wire existing but unconnected MOEX strategies into the backtest pipeline so they generate real trades: DividendGapStrategy with per-symbol hold bars, CBRStrategyWrapper in combiner, BrentGateStep and RubOilRegimeStep in the position sizing pipeline. Establish a positive equity baseline for ru_* segments.

</domain>

<decisions>
## Implementation Decisions

### BrentGateStep Design
- Brent-in-RUB threshold: 5000 RUB/bbl (historical median ~5500, below 5000 signals weak oil)
- Position size reduction below threshold: 50% (scale=0.5), matching RubOilRegime ELEVATED level
- Affects energy sector only (ru_energy segment) — surgical, avoids over-gating
- Data source: compute Brent-in-RUB from MOEX ISS Brent USD × USDRUB via existing fetchers, no new dependencies

### RubOilRegimeStep Design
- Wrap existing RubOilRegimeSignal (RegimeProvider) as a SizingStep — reuse correlation logic, add pipeline interface
- Scale factors: NORMAL=1.0, ELEVATED=0.5, CRISIS=0.25 (matches existing RegimeProvider)
- Pipeline position: after RegimeStep, before HardCaps — stacks with VIX/ADX regime, doesn't replace it
- Applied to ru_* segments only — RUB/oil decorrelation irrelevant for US

### DividendGap Hold Bars & Combiner Dilution
- Per-symbol max_hold_bars based on dividend yield: >8% → 60 bars, 5-8% → 40 bars, <5% → 25 bars
- Add `_EVENT_STRATEGIES` frozenset to combiner — event strategies skip ADX gating entirely, contribute directly to combined score
- min_combined_confidence floor of 0.40 when event strategy fires (matches DividendGap min_confidence)
- Override backtest config default from 15 to 40 bars for dividend_gap

### Claude's Discretion
- Internal implementation details for SizingStep interface adaptation
- Test structure and fixture design
- Error handling for missing Brent/USDRUB data (graceful degradation to scale=1.0)

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `RubOilRegimeSignal` in `src/finalayze/risk/rub_oil_regime.py` — full correlation logic, just needs SizingStep wrapper
- `CBRStrategyWrapper` in `src/finalayze/strategies/cbr_strategy_wrapper.py` — already production-ready
- `DividendGapStrategy` in `src/finalayze/strategies/dividend_gap.py` — fully implemented with 3-tier calendar loading
- `moex_dividends.yaml` expanded to 150+ events in Phase 8 with status field
- `CBRFetcher` in `src/finalayze/data/fetchers/cbr.py` — CBR rate decisions from API
- MOEX ISS fetcher already provides Brent and USDRUB data

### Established Patterns
- Position sizing pipeline: `KellyStep → VolTargetStep → RegimeStep → HardCapsStep` in `risk/position_sizing_pipeline.py`
- Each step implements `SizingStep` protocol with `apply(context: SizingContext) -> SizingContext`
- Strategy registration via preset YAML files in `strategies/presets/`
- ADX routing in combiner uses `_MOMENTUM_STRATEGIES` and `_MR_STRATEGIES` frozensets
- Backtest config defaults in `backtest/config.py` with strategy-specific overrides

### Integration Points
- `run_iteration.py` lines 362-504: strategy setup functions for DividendGap and CBR
- `combiner.py` lines 34-42: strategy pool constants for ADX routing
- `position_sizing_pipeline.py` lines 145-182: pipeline step chain
- `backtest/config.py` lines 22-54: default hold bars and ATR multipliers per strategy

</code_context>

<specifics>
## Specific Ideas

No specific requirements — standard wiring using established patterns.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>
