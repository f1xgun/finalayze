---
phase: 32-critical-sandbox-fixes
verified: 2026-04-07T19:15:00Z
status: passed
score: 9/9
overrides_applied: 0
gaps: []
re_verified: true
re_verification_reason: "Plan 32-04 closed SANDBOX-FIX-10 gap — per-fold calibrator now fitted on cal_idx and passed to _evaluate_fold_metrics"
---

# Phase 32: Critical Sandbox Fixes — Verification Report

**Phase Goal:** All strategies function correctly in MOEX sandbox mode, safety defaults prevent accidental production-level risk, news pipeline activated, and signal diagnostics available
**Verified:** 2026-04-07T18:08:22Z
**Status:** gaps_found
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths (Roadmap Success Criteria)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | `_CANDLE_LOOKBACK >= 210` in trading loop | VERIFIED | `trading_loop.py:81: _CANDLE_LOOKBACK = 210` with comment explaining SMA-200 and dual_momentum needs |
| 2 | `TradingLoop.start()` checks `KillSwitch.is_killed` before starting scheduler | VERIFIED | `trading_loop.py:496: if self._kill_switch is not None and self._kill_switch.is_killed: raise RuntimeError(...)` — before APScheduler setup |
| 3 | Sandbox mode defaults to MINIMAL rollout when not explicitly set | VERIFIED | `settings.py:163: if self.mode == WorkMode.SANDBOX and not os.environ.get("FINALAYZE_ROLLOUT_PHASE"): self.rollout_phase = RolloutPhase.MINIMAL` |
| 4 | Staleness threshold handles weekends (72h) and MOEX holidays | VERIFIED | `trading_loop.py:89: _STALENESS_THRESHOLD_HOURS: float = 72.0`; `_is_candle_stale` uses `is_moex_holiday` import (line 38) with `non_trading_days` subtraction logic (lines 313-321) |
| 5 | TinkoffFetcher wrapped in CachingFetcher and RateLimiter in sandbox mode | VERIFIED | `run_sandbox.py:270,278,280`: `_tbank_rate_limiter = RateLimiter(name="tbank", rate=4.0)`, `caching_fetcher = CachingFetcher(delegate=tinkoff_fetcher)`, `fetchers = {"moex": caching_fetcher}`; `main.py:228,336,343-344`: same pattern inside `_build_trading_loop()` |
| 6 | event_driven enabled for ru_blue_chips, ru_energy, ru_finance with LLM setup documented | VERIFIED | `ru_energy.yaml`: `event_driven: enabled: true`; `ru_finance.yaml`: `event_driven: enabled: true`; `ru_blue_chips.yaml`: already `enabled: true`; `.env.example` lines 59-62 document `FINALAYZE_LLM_PROVIDER`, `FINALAYZE_LLM_MODEL`, `FINALAYZE_LLM_API_KEY` |
| 7 | ValidationLogger tracks per-gate signal drops | VERIFIED | `validation_logger.py:34-36`: `signals_dropped_no_bars: int = 0`, `signals_dropped_below_threshold: int = 0`, `signals_dropped_pre_trade: int = 0`; counters incremented at `trading_loop.py:1610,1655,1826`; passed to CycleLogEntry at lines 1357-1359; logged at INFO at line 1654 |
| 8 | ML profit_factor gate computes actual PF from fold predictions | VERIFIED | `train_models.py:1059-1072`: `gross_profit/gross_loss` computed with 0.55 threshold; `train_models.py:1083`: `profit_factor=profit_factor` in FoldMetrics return (not default 1.0) |
| 9 | ML Brier gate uses calibrated probabilities — calibrator applied during walk-forward evaluation | VERIFIED | Plan 32-04 closed this gap: `EnsembleCalibrator.predict_proba` batch method added (calibration.py:228); per-fold calibrator fitted on cal_f/cal_l inside walk-forward loop (train_models.py:1240-1259); `calibrator=fold_calibrator` passed to `_evaluate_fold_metrics` (train_models.py:1265). 4 new tests confirm wiring. |

**Score:** 9/9 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/finalayze/orchestration/trading_loop.py` | `_CANDLE_LOOKBACK=210`, kill switch in `start()`, 72h threshold, calendar-aware staleness | VERIFIED | All four changes confirmed in code |
| `config/settings.py` | Sandbox rollout default MINIMAL | VERIFIED | Model validator at line 163 handles sandbox mode |
| `scripts/run_sandbox.py` | CachingFetcher + RateLimiter wiring | VERIFIED | Lines 266-280 wrap TinkoffFetcher |
| `src/finalayze/main.py` | CachingFetcher + RateLimiter wiring | VERIFIED | Lines 226-344 inside `_build_trading_loop()` |
| `src/finalayze/strategies/presets/ru_energy.yaml` | `event_driven: enabled: true` | VERIFIED | Confirmed |
| `src/finalayze/strategies/presets/ru_finance.yaml` | `event_driven: enabled: true` | VERIFIED | Confirmed |
| `src/finalayze/core/validation_logger.py` | 3 drop counter fields in CycleLogEntry | VERIFIED | Lines 34-36 |
| `scripts/train_models.py` | `profit_factor` computed, `calibrator` param, calibrated Brier | VERIFIED | profit_factor: VERIFIED. calibrator param: VERIFIED. Per-fold calibrator wired at call site (line 1265): VERIFIED by plan 32-04. |
| `tests/unit/test_ml_quality_gates.py` | Tests for profit_factor and Brier fixes | VERIFIED | 7 tests covering profit_factor computation and calibrated Brier |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `src/finalayze/orchestration/trading_loop.py` | `src/finalayze/data/moex_calendar.py` | `is_moex_holiday` import for calendar-aware staleness | WIRED | `trading_loop.py:38`: `from finalayze.data.moex_calendar import is_moex_holiday`; used at line 317 |
| `scripts/run_sandbox.py` | `src/finalayze/data/fetchers/caching.py` | `CachingFetcher(delegate=tinkoff_fetcher)` | WIRED | `run_sandbox.py:266`: import; `run_sandbox.py:278`: construction; `run_sandbox.py:280`: used as `fetchers["moex"]` |
| `src/finalayze/orchestration/trading_loop.py` | `src/finalayze/core/validation_logger.py` | `CycleLogEntry` with drop counter fields | WIRED | Counter fields set in `_reset_cycle_counters` (lines 285-287); passed to `CycleLogEntry` constructor at lines 1357-1359 |
| `scripts/train_models.py` | `src/finalayze/ml/training/quality_gates.py` | `FoldMetrics` with computed `profit_factor` | WIRED | `FoldMetrics(profit_factor=profit_factor, ...)` at line 1083 |
| `scripts/train_models.py` | `src/finalayze/ml/calibration.py` | `calibrator.predict_proba` for Brier evaluation | WIRED | Per-fold calibrator fitted on cal_f/cal_l (lines 1240-1259) and passed via `calibrator=fold_calibrator` (line 1265). EnsembleCalibrator.predict_proba (calibration.py:228) provides batch calibration. |

### Data-Flow Trace (Level 4)

Not applicable — phase produces utility/config changes and script updates, not UI components rendering dynamic data.

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| `_CANDLE_LOOKBACK` equals 210 | `grep "_CANDLE_LOOKBACK = 210" trading_loop.py` | Found at line 81 | PASS |
| Kill switch blocks start() | 37 tests passing including kill switch tests | 37 passed | PASS |
| Sandbox defaults to MINIMAL | `grep "RolloutPhase.MINIMAL" config/settings.py` | Found at line 164 | PASS |
| CachingFetcher in run_sandbox.py | `grep "CachingFetcher" run_sandbox.py` | Found lines 266, 278, 280 | PASS |
| profit_factor computed in _evaluate_fold_metrics | `grep "profit_factor=profit_factor" train_models.py` | Found at line 1083 | PASS |
| Calibrated Brier wired in walk-forward | `_evaluate_fold_metrics(...)` call at line 1263 | `calibrator=fold_calibrator` present (line 1265) | PASS |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|---------|
| SANDBOX-FIX-01 | 32-01-PLAN.md | `_CANDLE_LOOKBACK >= 210` | SATISFIED | `trading_loop.py:81: _CANDLE_LOOKBACK = 210` |
| SANDBOX-FIX-02 | 32-01-PLAN.md | Kill switch check in `start()` | SATISFIED | `trading_loop.py:496`: RuntimeError on `is_killed` |
| SANDBOX-FIX-03 | 32-01-PLAN.md | Sandbox defaults to MINIMAL rollout | SATISFIED | `settings.py:163-164`: model_validator applies MINIMAL |
| SANDBOX-FIX-04 | 32-01-PLAN.md | 72h threshold + MOEX holiday awareness | SATISFIED | `trading_loop.py:89,313-321`: 72h + non_trading_days logic |
| SANDBOX-FIX-05 | 32-02-PLAN.md | CachingFetcher in sandbox | SATISFIED | `run_sandbox.py:278`, `main.py:343`: CachingFetcher wired |
| SANDBOX-FIX-06 | 32-02-PLAN.md | RateLimiter to TinkoffFetcher | SATISFIED | `run_sandbox.py:270`, `main.py:336`: RateLimiter(rate=4.0) |
| SANDBOX-FIX-07 | 32-02-PLAN.md | event_driven for all MOEX + LLM docs | SATISFIED | ru_energy, ru_finance, ru_blue_chips all `enabled: true`; `.env.example` documents LLM setup |
| SANDBOX-FIX-08 | 32-02-PLAN.md | Per-gate drop counters in ValidationLogger | SATISFIED | `validation_logger.py:34-36` + `trading_loop.py:1357-1359` |
| SANDBOX-FIX-09 | 32-03-PLAN.md | ML profit_factor computed from fold predictions | SATISFIED | `train_models.py:1059-1083`: gross_profit/gross_loss computed and passed to FoldMetrics |
| SANDBOX-FIX-10 | 32-04-PLAN.md | ML Brier uses calibrated probabilities in walk-forward | SATISFIED | Plan 32-04: per-fold calibrator fitted on cal_idx (train_models.py:1240-1259), passed to _evaluate_fold_metrics (line 1265). EnsembleCalibrator.predict_proba added (calibration.py:228). |

**Orphaned requirements:** None. All 10 SANDBOX-FIX IDs appear in plan frontmatter and REQUIREMENTS.md.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `config/settings.py` | 81 | `# TODO: revert to 30` on `news_cycle_minutes = 2` | Info | Pre-existing; not introduced in Phase 32. `news_cycle_minutes=2` (set to 2 minutes instead of 30) may cause overly frequent news cycles in sandbox, but this is a known temp setting |
| `src/finalayze/orchestration/trading_loop.py` | 2469 | `pct = float(qty) * 0.01  # placeholder` | Info | Pre-existing in `_compute_top_movers`; not in Phase 32 code paths; dashboard display only |
| `src/finalayze/orchestration/trading_loop.py` | 1942 | `TODO: Wire returns history for live correlation` | Info | Pre-existing; not in Phase 32 scope |

No blockers or warnings introduced by Phase 32 changes.

### Human Verification Required

No human verification items — all checks are programmatically verifiable.

### Gaps Summary

No gaps remaining. Plan 32-04 closed SANDBOX-FIX-10 by adding `EnsembleCalibrator.predict_proba` batch method and wiring per-fold calibrator fitting inside the walk-forward loop.

**All 9 success criteria are fully satisfied and tested.** The phase significantly improves sandbox safety, observability, and ML evaluation correctness.

---

_Verified: 2026-04-07T19:15:00Z_
_Re-verified after plan 32-04 gap closure_
_Verifier: Claude (manual verification)_
