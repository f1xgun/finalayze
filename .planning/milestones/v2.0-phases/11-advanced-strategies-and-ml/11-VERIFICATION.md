---
phase: 11-advanced-strategies-and-ml
verified: 2026-03-21T00:00:00Z
status: gaps_found
score: 8/10 must-haves verified
re_verification: false
gaps:
  - truth: "Quality gates pass on 2024-2025 validation data for ru_blue_chips"
    status: failed
    reason: "wf_gate_results.json shows overall_passed=false for ru_blue_chips; accuracy gate pass rate 0.33, brier_score pass rate 0.0, bh_passed=false"
    artifacts:
      - path: "models/ru_blue_chips/wf_gate_results.json"
        issue: "overall_passed=false; best_accuracy=0.44 (below threshold); models force-saved"
    missing:
      - "Literal success criterion 3 requires quality gates passing; reinforcer-only mode is an acceptable mitigation per user context but the criterion as written is not met"
  - truth: "Quality gates pass on 2024-2025 validation data for us_tech (retrained at v3)"
    status: failed
    reason: "wf_gate_results.json for us_tech also shows overall_passed=false; accuracy gate pass rate 0.18, brier_score 0.0, bh_passed=false -- regression from previous us_tech enablement that showed quality gates passing"
    artifacts:
      - path: "models/us_tech/wf_gate_results.json"
        issue: "overall_passed=false; best_accuracy=0.48 on 11 folds; previously quality gates passed for us_tech"
    missing:
      - "us_tech schema-v3 retrain degraded quality gate results; may require retuning or gate threshold adjustment for v3 feature set"
human_verification:
  - test: "Confirm ru_blue_chips ML backtest improvement is acceptable"
    expected: "Sharpe improves from -0.03 baseline to +0.0001 or better; no drawdown regression"
    why_human: "Backtest metrics come from run_iteration output logged externally, not verifiable via file grep; quality gate failure context requires judgment call"
  - test: "Confirm us_tech v3 backtest is not regressed"
    expected: "Sharpe approximately +0.012, PF ~1.16, matching SUMMARY claim"
    why_human: "us_tech quality gate failure is concerning; user must confirm live backtest result still acceptable"
---

# Phase 11: Advanced Strategies and ML — Verification Report

**Phase Goal:** Preferred share arbitrage captures pref/ord spread convergence, and ML ensemble operates on ru_* segments with Russian macro features
**Verified:** 2026-03-21
**Status:** gaps_found (quality gates failed; mitigation via reinforcer-only accepted by user but literal success criterion not met)
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|---------|
| 1 | PairsStrategy generates BUY signals on SBER/SBERP and TATN/TATNP when spread z-score < -2.0 | VERIFIED | `pairs.py:280` guards SELL only; BUY path unblocked; 15 tests pass including `test_allow_short_false_allows_buy` |
| 2 | PairsStrategy suppresses SELL signals when allow_short=False | VERIFIED | `pairs.py:165,280`: `allow_short = bool(params.get("allow_short", True))`; guard at `if direction == SELL and not allow_short: return None` |
| 3 | TATNP instrument has a valid FIGI for Tinkoff data fetching | VERIFIED | `instruments.py:521-525`: `figi="BBG004S68CP5"` present |
| 4 | Cointegration validated on post-2022 data only (cointegration_start="2023-01-01") | VERIFIED | `pairs.py:220-221`: `cutoff = datetime.fromisoformat(cointegration_start)` filters candles before cointegration test; preset has `cointegration_start: "2023-01-01"` |
| 5 | 7 new MOEX ML features (cbr_rate_level, cbr_rate_delta, cbr_direction_cut, cbr_direction_hike, usdrub_return, usdrub_vol, brent_return) computed with 2-bar lag | VERIFIED | `technical.py:562-708`: three new functions; `technical.py:812-830`: all three wired into `compute_features()` all_features dict |
| 6 | FEATURE_SCHEMA_VERSION bumped to 3 | VERIFIED | `loader.py:26`: `FEATURE_SCHEMA_VERSION: int = 3` |
| 7 | Existing 4 MOEX features unchanged | VERIFIED | Functions `_compute_fx_features`, `_compute_commodity_features`, `_compute_macro_features`, `_compute_turnover_features` untouched per grep; still wired at `technical.py:806-810` |
| 8 | ru_blue_chips preset has ml_ensemble enabled with weight=0.10 in reinforcer-only mode | VERIFIED | `ru_blue_chips.yaml:80-83`: `ml_ensemble: enabled: true, weight: 0.10` |
| 9 | ML models exist for ru_blue_chips segment | VERIFIED | `models/ru_blue_chips/`: xgb.pkl, lgbm.pkl, catboost.pkl, calibrator.pkl, meta_learner.pkl, selected_features.json all present |
| 10 | Quality gates pass on 2024-2025 validation data for ru_blue_chips (and us_tech) | FAILED | `models/ru_blue_chips/wf_gate_results.json`: `overall_passed=false`; `models/us_tech/wf_gate_results.json`: `overall_passed=false`. Models force-saved. |

**Score:** 9/10 truths verified (quality gate failure noted; reinforcer-only mitigation applied)

---

## Required Artifacts

### Plan 11-01 (ADV-01)

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/finalayze/strategies/pairs.py` | allow_short parameter gating SELL signals | VERIFIED | Line 165: `allow_short = bool(params.get("allow_short", True))`; line 280: SELL guard |
| `src/finalayze/strategies/presets/ru_blue_chips.yaml` | Pairs config with SBER/SBERP, TATN/TATNP, z_entry=2.0, allow_short=false | VERIFIED | Lines 57-67: pairs enabled, both pairs present, z_entry=2.0, allow_short=false, cointegration_start="2023-01-01" |
| `src/finalayze/markets/instruments.py` | TATNP instrument with FIGI field | VERIFIED | Line 521-525: `figi="BBG004S68CP5"` |

### Plan 11-02 (ADV-02)

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/finalayze/ml/features/technical.py` | `_compute_cbr_features`, `_compute_fx_return_features`, `_compute_brent_return_features` | VERIFIED | All three functions defined at lines 562, 632, 679; wired at lines 812-830 |
| `src/finalayze/ml/loader.py` | `FEATURE_SCHEMA_VERSION = 3` | VERIFIED | Line 26: `FEATURE_SCHEMA_VERSION: int = 3` |
| `tests/unit/test_features_moex.py` | Tests for new MOEX features including cbr_direction_cut | VERIFIED | 46 tests pass; `cbr_direction_cut` tested at lines 346-352 |

### Plan 11-03 (ADV-03)

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/finalayze/strategies/presets/ru_blue_chips.yaml` | ml_ensemble enabled | VERIFIED | Lines 80-83: `ml_ensemble: enabled: true` |
| `models/ru_blue_chips/` | Trained ML models | VERIFIED | xgb.pkl, lgbm.pkl, catboost.pkl, calibrator.pkl, meta_learner.pkl, selected_features.json, wf_gate_results.json all present |
| `models/us_tech/` | Retrained us_tech models at schema v3 | VERIFIED | All model files present; wf_gate_results.json present |

---

## Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `ru_blue_chips.yaml` pairs.allow_short | `pairs.py` SELL suppression | `params.get("allow_short")` read in `generate_signal` → passed to `_compute_signal` | WIRED | Lines 165, 197 pass `allow_short` to `_compute_signal`; guard at line 280 |
| `technical.py` new functions | `compute_features()` return dict | `cbr_features`, `fx_return_features`, `brent_return_features` merged into `all_features` | WIRED | Lines 812-814 call functions; lines 828-830 merge into all_features |
| `loader.py` FEATURE_SCHEMA_VERSION | model load/save | Checked at load time; schema version 2 models rejected | WIRED | Lines 129-138: version mismatch raises error |
| `ru_blue_chips.yaml` ml_ensemble | `ml_strategy.py` MLStrategy | `ml_ensemble.enabled` and `weight` read by `MLStrategy.enabled_segment_ids()` | WIRED | `ml_strategy.py:59-71` reads `ml_ensemble` key from YAML |
| `models/ru_blue_chips/` | `loader.py` | `load_registry` loads models with FEATURE_SCHEMA_VERSION=3 | WIRED | Models saved with schema v3 via `--force-save`; loader version check passes |
| `PairsStrategy` | run scripts / combiner | `run_iteration.py:87,285-301` instantiates PairsStrategy; combiner pools `pairs` in `_MR_STRATEGIES` | WIRED | `run_iteration.py:87`, `combiner.py:36` |

---

## Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|---------|
| ADV-01 | 11-01-PLAN.md | Preferred share arbitrage (SBER/SBERP, TATN/TATNP) via adapted PairsStrategy with Kalman filter | SATISFIED | allow_short implemented; pairs enabled in preset; TATNP FIGI added; cointegration_start filtering |
| ADV-02 | 11-02-PLAN.md | 10 Russian macro ML features (CBR rate/delta/direction, USDRUB return/zscore/vol, Brent return, IMOEX relative, turnover zscore) | SATISFIED | 7 new feature columns added (4 CBR + 2 FX + 1 Brent); combined with 4 existing MOEX features and cross-asset relative_strength_21d for IMOEX relative; schema bumped to v3 |
| ADV-03 | 11-03-PLAN.md | ML ensemble enabled for ru_* segments with macro features, reinforcer-only mode | PARTIALLY SATISFIED | ru_blue_chips ml_ensemble enabled at weight=0.10 (reinforcer-only); models trained and present; quality gates FAILED (force-save used); backtest showed marginal improvement (Sharpe +0.0001 vs -0.03 baseline per SUMMARY) |

No orphaned requirements: ADV-01, ADV-02, ADV-03 are the only IDs mapped to Phase 11 in REQUIREMENTS.md and all three are claimed by plans.

---

## Anti-Patterns Found

| File | Pattern | Severity | Impact |
|------|---------|----------|--------|
| `models/ru_blue_chips/wf_gate_results.json` | `overall_passed: false` — models force-saved bypassing quality gates | WARNING | ML signals for ru_blue_chips may have poor predictive accuracy (best_accuracy=0.44); reinforcer-only mode (weight=0.10) limits downside — ML can only boost, not originate trades |
| `models/us_tech/wf_gate_results.json` | `overall_passed: false` on schema-v3 retrain | WARNING | us_tech previously had passing gates; retrain with 7 new zero-value features may have disrupted model calibration on 11 folds |
| `models/ru_blue_chips/selected_features.json` | Only `brent_return` from new CBR/FX features selected (8 total features) | INFO | Feature selection excluded cbr_rate_level, cbr_direction_*, usdrub_return, usdrub_vol — new features have limited predictive value on 3-fold MOEX window |

---

## Human Verification Required

### 1. Confirm ru_blue_chips ML backtest result

**Test:** Run `uv run python scripts/run_iteration.py --name ml-ru-verify --description "Verify ru_blue_chips ML" --segments ru_blue_chips` and compare metrics to baseline iteration without ML.
**Expected:** Sharpe >= 0.0 (not negative), no material drawdown increase vs baseline.
**Why human:** Backtest metrics are written to external results files; quality gate failure raises concern about whether the marginal Sharpe improvement (+0.0001 per SUMMARY) is stable.

### 2. Confirm us_tech schema-v3 retrain is not regressed

**Test:** Run `uv run python scripts/run_iteration.py --name us-tech-v3-verify --description "us_tech at schema v3" --segments us_tech` and compare to previous us_tech ML results.
**Expected:** Sharpe approximately +0.012, PF ~1.16 per SUMMARY claim.
**Why human:** `wf_gate_results.json` for us_tech shows `overall_passed=false` which contradicts previous enablement where gates passed; human must confirm backtest outcome is still acceptable.

### 3. Confirm quality gate failure is intentionally accepted

**Test:** Review `models/ru_blue_chips/wf_gate_results.json` and `models/us_tech/wf_gate_results.json`.
**Expected:** User explicitly accepts force-save + reinforcer-only as acceptable mitigation for gate failures on both segments.
**Why human:** Success criterion 3 literally states "quality gates passing" which is NOT met. This is a policy decision requiring explicit human sign-off.

---

## Gaps Summary

The phase is structurally complete: all implementation artifacts exist, are substantive, and are properly wired. The sole gap is the literal failure of ROADMAP.md Success Criterion 3: "ML ensemble is enabled for at least one ru_* segment in reinforcer-only mode, **with quality gates passing** on 2024-2025 calm-period validation data."

Quality gates failed for both ru_blue_chips (3 folds, best_accuracy=0.44) and us_tech (11 folds, best_accuracy=0.48 post-v3 retrain). Models were force-saved in both cases.

The user has acknowledged this in the verification prompt and provided the following mitigations:
- Reinforcer-only mode (weight=0.10) means ML can only boost existing strategy signals, never originate trades
- ru_blue_chips backtest showed marginal Sharpe improvement (+0.0001 vs -0.03 baseline)
- us_tech retrained at schema v3 with Sharpe +0.012, PF 1.16

The gap is therefore a literal criterion failure with an accepted mitigation, not a broken implementation. Whether to mark this as passed or gaps_found depends on whether "quality gates passing" is treated as a hard requirement. This report marks it `gaps_found` to be precise, and defers final judgment to human verification.

---

_Verified: 2026-03-21_
_Verifier: Claude (gsd-verifier)_
