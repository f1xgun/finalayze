# Phase 47: Cross-Asset Features & Asymmetric Barriers - Context

**Gathered:** 2026-04-14
**Status:** Ready for planning

<domain>
## Phase Boundary

Add Brent crude multi-period return features for ru_energy and introduce per-segment asymmetric triple barrier configuration. Changes span `technical.py` (feature computation), `auto_ml_research.py` (barrier config + routing), and `train_models.py` (parity).

</domain>

<decisions>
## Implementation Decisions

### Brent Return Features
- Add `brent_ret_5d` and `brent_ret_21d` (5-day and 21-day log returns) to feature pipeline
- Compute inside existing `_compute_brent_return_features()` in `technical.py:679` — already has Brent data access and lag logic
- Keep existing `brent_return` (1-bar) for backward compatibility — models may already use it
- Return 0.0 for any individual feature with insufficient history (same fallback pattern as `_compute_commodity_features`)

### Asymmetric Barriers
- ru_energy asymmetry: lower_atr_mult=2.0, upper_atr_mult=1.5 (wider downside for commodity-linked volatility)
- Config via `_SEGMENT_BARRIER_CONFIG` dict in `auto_ml_research.py` mapping segment_id to (upper_mult, lower_mult), fallback to current symmetric defaults
- `train_models.py` also gets asymmetric barriers — same `_SEGMENT_BARRIER_CONFIG` pattern for parity
- Only ru_energy gets custom barriers; other ru_* segments keep symmetric defaults

### Claude's Discretion
- Exact implementation of multi-period return computation (rolling vs point-to-point)
- Whether to add clipping to multi-period returns (existing 1-bar clips to [-0.15, 0.15])
- Test structure and naming

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `_compute_brent_return_features()` at `technical.py:679` — computes 1-bar Brent log return with lag
- `_compute_commodity_features()` at `technical.py:425` — computes `brent_zscore_60d`, same Brent data pattern
- `_EXTERNAL_DATA_LAG_BARS` constant — already applied in existing Brent features
- `_TB_UPPER_ATR_MULT = 2.0` / `_TB_LOWER_ATR_MULT = 2.0` — current symmetric defaults
- `_MOEX_ATR_UPLIFT = 1.2` — existing MOEX uplift factor applied to both mults

### Established Patterns
- MOEX vs US config routing via `_is_moex_segment()` — used for hparams, lookback, fold params
- Feature functions return `dict[str, float]` with 0.0 defaults when data unavailable
- Brent candles available via `moex_data.commodity_candles["BZ=F"]` (fetched by `_fetch_moex_macro_data()`)

### Integration Points
- `auto_ml_research.py:490-491` — barrier mults computed, used at line 511-512 in `build_triple_barrier_dataset()`
- `train_models.py:558-559` — same pattern for production training
- `compute_features()` at `technical.py:711` — calls `_compute_brent_return_features()` and merges result
- `MarketContext.moex_data` — passed to feature computation, carries Brent candles

</code_context>

<specifics>
## Specific Ideas

- Success criteria requires columns named exactly `brent_ret_5d` and `brent_ret_21d`
- Barrier asymmetry logged at run start: barrier parameters already printed, just need to show different upper/lower
- `_SEGMENT_BARRIER_CONFIG` should use segment_id keys (e.g., "ru_energy") not patterns

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 47-cross-asset-features-asymmetric-barriers*
*Context gathered: 2026-04-14 via autonomous smart discuss*
