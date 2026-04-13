# Phase 32: Critical Sandbox Fixes - Context

**Gathered:** 2026-04-07
**Status:** Ready for planning
**Source:** Board of Directors meeting (8 expert agents, 2 rounds)

<domain>
## Phase Boundary

Fix all issues preventing the MOEX sandbox from generating adequate trading signals.
The system currently produces ~1-3% of potential signals due to cascading bugs and
misconfigurations discovered by 8 domain-expert agents.

**Target market: MOEX only.** US market fixes are out of scope for this phase.
All changes must be validated against `ru_*` segments.

</domain>

<decisions>
## Implementation Decisions

### Data Pipeline Fixes (P0)
- `_CANDLE_LOOKBACK` in `src/finalayze/orchestration/trading_loop.py` line 80: change from 60 to 210
- Note: T-Bank daily candle API supports up to 1 year per request, so 210*2=420 calendar days is fine in one call
- Staleness threshold: change `_STALENESS_THRESHOLD_HOURS` from 48.0 to 72.0 (covers weekends). For MOEX New Year holidays (10 days), add calendar-aware check using existing `is_moex_trading_day()` from `src/finalayze/data/moex_calendar.py`

### Sandbox Data Wiring (P1)
- In `scripts/run_sandbox.py` (~line 268): wrap TinkoffFetcher in CachingFetcher from `src/finalayze/data/fetchers/caching.py`
- In `scripts/run_sandbox.py`: pass RateLimiter (4 req/sec) to TinkoffFetcher
- Also check `src/finalayze/main.py` `_build_trading_loop()` for the same wiring

### Safety Defaults (P0)
- Kill switch: add `if self._kill_switch and self._kill_switch.is_killed: raise RuntimeError("Kill switch active")` at top of `TradingLoop.start()`
- Rollout default: in `config/settings.py`, add validator: if `mode == WorkMode.SANDBOX` and `rollout_phase` not explicitly set via env var, default to `RolloutPhase.MINIMAL`

### News Pipeline Activation (P1)
- Enable `event_driven` in MOEX presets: `ru_blue_chips.yaml`, `ru_energy.yaml`, `ru_finance.yaml` — set `enabled: true` (weight already set at 0.15)
- Document LLM setup in README or .env.example: `FINALAYZE_LLM_PROVIDER=openrouter`, `FINALAYZE_LLM_API_KEY=<key>`, default model is `meta-llama/llama-3.1-8b-instruct:free` ($0/day)
- Do NOT enable event_driven for US segments (out of scope, prompt is MOEX-focused)

### Signal Diagnostics (P1)
- Add fields to `CycleLogEntry` in `src/finalayze/core/validation_logger.py`: `signals_dropped_no_bars: int = 0`, `signals_dropped_below_threshold: int = 0`, `signals_dropped_pre_trade: int = 0`
- Increment these counters at appropriate points in `_process_instrument` and `_process_market_cycle`
- Add structlog events at INFO level (not DEBUG) when strategies return None due to insufficient bars

### ML Quality Gate Fixes (P2)
- Fix profit_factor gate in `src/finalayze/ml/training/quality_gates.py`: compute PF from fold predictions (threshold trades at prob > 0.55 for BUY, compare gross profit vs gross loss)
- Fix Brier gate: in walk-forward evaluation (`scripts/train_models.py` `_evaluate_fold_metrics`), fit calibrator on cal_idx, then evaluate Brier on calibrated test_idx probabilities
- These are code fixes only — do NOT retrain models in this phase

### Claude's Discretion
- Exact implementation of calendar-aware staleness (simple weekend check vs full MOEX calendar integration)
- Whether to add staleness fix as a new method or modify existing `_is_candle_stale()`
- CycleLogEntry field naming conventions
- How to document LLM setup (README section vs .env.example comments vs separate doc)

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Trading Loop
- `src/finalayze/orchestration/trading_loop.py` — Main loop, lines 80 (_CANDLE_LOOKBACK), 88 (_STALENESS_THRESHOLD_HOURS), 1552-1624 (signal processing gates)
- `src/finalayze/core/kill_switch.py` — KillSwitch class with is_killed property

### Configuration
- `config/settings.py` — Settings class, rollout_phase field, WorkMode enum
- `src/finalayze/risk/rollout.py` — RolloutPhase enum (MINIMAL/STANDARD/FULL) and limits

### Data Pipeline
- `scripts/run_sandbox.py` — Sandbox bootstrap, TinkoffFetcher creation (~line 268)
- `src/finalayze/main.py` — Docker entry point, `_build_trading_loop()` method
- `src/finalayze/data/fetchers/caching.py` — CachingFetcher wrapper
- `src/finalayze/data/rate_limiter.py` — RateLimiter class
- `src/finalayze/data/moex_calendar.py` — MOEX holiday calendar

### Strategy Presets
- `src/finalayze/strategies/presets/ru_blue_chips.yaml` — event_driven already has weight 0.15
- `src/finalayze/strategies/presets/ru_energy.yaml` — event_driven weight set
- `src/finalayze/strategies/presets/ru_finance.yaml` — event_driven weight set

### Diagnostics
- `src/finalayze/core/validation_logger.py` — CycleLogEntry dataclass

### ML Pipeline
- `src/finalayze/ml/training/quality_gates.py` — FoldMetrics, gate thresholds, profit_factor gate
- `scripts/train_models.py` — `_evaluate_fold_metrics` function (~line 1006)
- `src/finalayze/ml/calibration.py` — EnsembleCalibrator class

</canonical_refs>

<specifics>
## Specific Ideas

### Board Meeting Key Findings (source of requirements)

**Quant Analyst:** _CANDLE_LOOKBACK=60 kills RSI2 Connors (needs 201), dual_momentum (needs 126), OU mean reversion (needs 126). Effective signal rate in sandbox < 1 trade/symbol/month.

**Risk Officer:** Sandbox readiness 7/10. Kill switch not checked on startup. Default rollout=FULL is dangerous. Correlation check is a stub.

**Data Quality Agent:** CachingFetcher exists but unused in sandbox. RateLimiter exists but not passed. Staleness 48h fails Monday mornings. Normalizer can silently drop valid candles.

**News Pipeline Agent:** Entire pipeline is built and wired. Only blocker: FINALAYZE_LLM_API_KEY not set. event_driven enabled only for ru_blue_chips. OpenRouter free tier = $0/day.

**ML Engineer:** ML is functionally dead — all model weights = 0.0, always returns 0.5. profit_factor gate never populated (bug). Brier gate uses uncalibrated probs (bug).

**Live Monitor:** 14 gates in signal path. Most impactful drops: bar count (45% weight), market hours (50% cycles), staleness (Monday), cash reserve (nарастающий).

### MOEX-First Constraint
All fixes target MOEX segments. US-specific issues (yfinance reliability, Alpaca wiring, Finnhub earnings) are explicitly deferred. News pipeline uses Russian-language RSS feeds and Telegram channels.

</specifics>

<deferred>
## Deferred Ideas

- US fetcher registration in sandbox — not needed for MOEX-first
- PEAD strategy re-enablement — needs EarningsCalendarFetcher (new component)
- US impact prompt for event_driven — MOEX prompts are sufficient
- ML model retraining — fix gates first, retrain in separate phase
- ADX threshold tuning (15,35 → 20,30) — requires experiment framework (Phase 34)
- dual_momentum min_confidence tuning (0.65 → 0.50) — requires experiment framework
- Per-layer circuit breaker — architectural change for Phase 33+
- Gap Fill strategy — new strategy for future phase

</deferred>

---

*Phase: 32-critical-sandbox-fixes*
*Context gathered: 2026-04-07 from Board of Directors meeting (8 agents, 2 rounds)*
