---
phase: 2
slug: moex-equity-validation
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-14
---

# Phase 2 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 7.x (via uv run pytest) |
| **Config file** | `pyproject.toml` [tool.pytest.ini_options] |
| **Quick run command** | `uv run pytest tests/unit/test_tune_strategy_params.py tests/unit/test_strategy_combiner.py tests/unit/test_backtest_config.py -x` |
| **Full suite command** | `uv run pytest` |
| **Estimated runtime** | ~60 seconds |

---

## Sampling Rate

- **After every task commit:** Run `uv run pytest tests/unit/test_tune_strategy_params.py tests/unit/test_strategy_combiner.py tests/unit/test_backtest_config.py -x`
- **After every plan wave:** Run `uv run pytest`
- **Before `/gsd:verify-work`:** Full suite must be green + walk-forward OOS Sharpe > 0.1 on 2+ segments
- **Max feedback latency:** 60 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 02-01-01 | 01 | 1 | EQF-02 | unit | `uv run pytest tests/unit/test_tune_strategy_params.py -x` | Exists | ⬜ pending |
| 02-01-02 | 01 | 1 | EQF-02 | unit | `uv run pytest tests/unit/test_moex_preset_validation.py -x` | ❌ W0 | ⬜ pending |
| 02-02-01 | 02 | 2 | EQF-03 | integration | `uv run python scripts/run_strategy_isolation.py --segment ru_blue_chips --all` | N/A (script) | ⬜ pending |
| 02-02-02 | 02 | 2 | EQF-03 | integration | `uv run python scripts/run_iteration.py --segments ru_blue_chips,ru_energy` | N/A (script) | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/unit/test_moex_preset_validation.py` — validate ru_* YAML presets load correctly with all strategies enabled
- [ ] Fix `run_iteration.py` UNIVERSE to include `ru_finance`
- [ ] Fix `tune_strategy_params.py` to use TinkoffFetcher for `ru_*` segments
- [ ] Fix `test_pairs_cointegration.py` to use TinkoffFetcher instead of yfinance

*These infra fixes must be done in Wave 1 before tuning can begin.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Walk-forward OOS Sharpe > 0.1 | EQF-02 | Requires full backtest run with T-Invest data | Run `scripts/run_iteration.py` and check OOS metrics in summary |
| Preset calibration is MOEX-specific | EQF-03 | Subjective review of YAML param values | Compare ru_*.yaml params to us_*.yaml — must differ meaningfully |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 60s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
