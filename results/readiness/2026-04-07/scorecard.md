# Go-Live Readiness Scorecard

**Date:** 2026-04-07
**Score:** 40/100 (NOT READY)
**Previous:** N/A (first assessment)

## Automated Checks

| # | Criterion | Weight | Status | Details |
|---|-----------|--------|--------|---------|
| 1 | Tests green | 15 | FAIL | 1 failed (circuit_breaker_e2e), 4194 total. Coverage 28% < 50% threshold |
| 2 | Lint clean | 5 | FAIL | ~3498 ruff issues (mostly import sorting, formatting) |
| 3 | Mypy clean | 5 | FAIL | 193 type errors (main.py, various modules) |
| 4 | WF Sharpe > 0 (ru_* segments) | 15 | WARN | Latest iteration "us-tech-schema-v3" REJECT, WF Sharpe 0.0115, no ru_* data |
| 5 | Sandbox 5+ trading days | 15 | PASS | 10 sandbox days (2026-02-23 to 2026-03-24) |
| 6 | Max DD < 2.27% | 10 | PASS | 0.00% max drawdown in sandbox |
| 7 | Fill rate > 95% | 5 | PASS | 98.3% (2978/3029 orders filled) |
| 8 | Circuit breakers configured | 5 | PASS | Level 0 max (never triggered). Config exists in settings. |
| 9 | Tinkoff token (real) set | 5 | PASS | FINALAYZE_TINKOFF_TOKEN is SET |
| 10 | Telegram alerts configured | 5 | PASS | Bot token + chat ID both SET |
| 11 | Emergency procedures tested | 5 | PENDING | Not verified (manual check required) |
| 12 | Starting capital verified | 10 | PENDING | Not verified (manual check required) |

## Score Breakdown

| Category | Possible | Earned |
|----------|----------|--------|
| Tests & Quality (1-3) | 25 | 0 |
| Backtest Gates (4) | 15 | 0 |
| Sandbox Metrics (5-8) | 35 | 35 |
| Configuration (9-10) | 10 | 10 |
| Manual Checks (11-12) | 15 | 0 |
| **Total** | **100** | **45** |

> Note: Score adjusted to 40 because criterion 4 (WF Sharpe) is WARN not PASS — latest iteration was REJECT.

## Verdict: NOT READY

Score 40/100 is below the 70 threshold for "ALMOST" and well below 85 for "READY".

## Critical Blockers

1. **Tests failing** (weight: 15) — `test_trading_resumes_after_manual_reset` in circuit breaker E2E
2. **Lint not clean** (weight: 5) — 3498 issues, mostly auto-fixable (import sorting)
3. **Type errors** (weight: 5) — 193 mypy errors in main.py and modules
4. **No positive ru_* backtest** (weight: 15) — Latest iteration REJECT, no ru_* segments tested

## Non-Critical Items

5. **Emergency procedures** — Not tested in sandbox mode yet
6. **Starting capital** — 500K RUB not verified in T-Invest account
7. **Sandbox data stale** — Last cycle 2026-03-24, 14 days ago. Need fresh sandbox run.

## Recommendations

| Priority | Action | Est. Effort | Impact |
|----------|--------|-------------|--------|
| P0 | Fix circuit breaker E2E test | 1-2h | +15 points |
| P0 | Run `ruff check --fix .` to auto-fix lint | 10min | +5 points |
| P1 | Fix mypy errors in main.py | 2-3h | +5 points |
| P1 | Run backtest on ru_* segments with positive Sharpe | 2-4h | +15 points |
| P1 | Restart sandbox trading to get fresh data | 30min | validates metrics |
| P2 | Test emergency /stop command | 15min | +5 points |
| P2 | Verify capital in T-Invest | 5min | +10 points |

## Path to Ready

With P0+P1 fixes: **40 → 75 points (ALMOST)**
With all items: **40 → 100 points (READY)**

Estimated effort to ALMOST: ~1 sprint (1 week)
Estimated effort to READY: ~2 sprints (2 weeks) including fresh sandbox validation
