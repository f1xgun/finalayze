# Pitfalls Research

**Domain:** MOEX ML AutoResearch & Adaptation — adding autonomous ML experiment capabilities with MOEX data support to an existing trading system (v9.0)
**Researched:** 2026-04-13
**Confidence:** HIGH (grounded in codebase inspection of auto_ml_research.py, quality_gates.py, tinkoff_data.py, and known constraints in PROJECT.md and MEMORY.md)

---

## Critical Pitfalls

### Pitfall 1: Calling TinkoffFetcher from a Sync Script Without InstrumentRegistry

**What goes wrong:**
`auto_ml_research.py` is a synchronous script that currently creates `YFinanceFetcher` directly (no dependencies). `TinkoffFetcher` requires `InstrumentRegistry` to perform FIGI lookups before each gRPC fetch call. If `InstrumentRegistry` is not initialized (it reads instrument configs and potentially the DB), every `fetch_candles()` call raises `InstrumentNotFoundError` and silently falls through the `except Exception` handler — returning empty candles for every MOEX symbol. The research loop continues with zero data and produces a "no valid folds" abort with no clear error about the root cause.

**Why it happens:**
The script was built for US markets where `YFinanceFetcher(market_id=market_id)` is the entire instantiation. Swapping the fetcher without understanding the `InstrumentRegistry` dependency creates a silent failure: the `_fetch_us_candles()` function swallows exceptions with `print(f"  Failed to fetch {sym}: {exc}")` rather than raising. With 15 symbols all returning empty, the absence of data looks identical to a network timeout rather than a configuration error.

**How to avoid:**
Create a dedicated `_fetch_moex_candles()` function that: (1) asserts `FINALAYZE_TINKOFF_TOKEN` is set and non-empty before constructing anything, (2) builds a minimal `InstrumentRegistry` from the static YAML instrument map (not from DB — this avoids needing a running database for training scripts), (3) instantiates `TinkoffFetcher` with `sandbox=False` (sandbox has no historical candles), (4) logs a prominent error and exits if candles_by_sym is empty after fetching. Add a guard: if `len(candles_by_sym) == 0`, raise `RuntimeError("No candles fetched — check FINALAYZE_TINKOFF_TOKEN and instrument registry")` rather than returning `None` quietly.

**Warning signs:**
- `fetch_candles()` returns `[]` for known symbols like `SBER` with no exception raised
- All 15 symbols print `"Failed to fetch {sym}: InstrumentNotFoundError"` in sequence
- Script reaches "No data fetched, aborting." after spending 0 seconds on fetching (no network calls made)
- `InstrumentNotFoundError` in logs — registry was not populated

**Phase to address:**
Phase 1 (MOEX data adapter). Write a smoke test that calls the MOEX fetch path with the real token (skipped in CI without token) before any other phase work.

---

### Pitfall 2: Look-Ahead Bias When Joining Macro Features to the Feature Vector

**What goes wrong:**
MOEX macro features — CBR key rate, USDRUB close, IMOEX close, Brent close — are indexed by date. Joining them to the candle feature vector with `merge(on='date', how='left')` gives the macro value for the same trading day as the bar being labeled. At open-of-day decision time, today's USDRUB close does not yet exist; yesterday's is the only available value. This is classic look-ahead bias: the model appears to know where USDRUB closed before the market session ends.

CBR rate is even worse: the CBR announces rate changes at ~13:30 Moscow time. A model trained on `cbr_rate[t]` has seen information that was not available at the open-of-day entry decision for `t`, causing it to pick up rate-announcement effects that cannot be exploited live.

**Why it happens:**
pandas `merge(on='date')` is the natural operation and it "works" — no error, full join, no NaN. The look-ahead is invisible in the merged dataframe. `build_triple_barrier_dataset()` in `labeling.py` accepts `benchmark_candles` and `vix_candles` as pre-aligned lists — any alignment shift must be done by the caller, which is easy to forget when adding new macro series.

**How to avoid:**
Always `shift(1)` all macro series before joining: `macro_df['cbr_rate'] = macro_df['cbr_rate'].shift(1)`. Do this inside the macro feature builder, not at the call site. Add an explicit assertion in the feature builder: after computing features for bar at timestamp `t`, verify that the CBR rate used was announced before `t`. Write a unit test: feed a synthetic macro series with a known event at day `d`, verify that the feature vector for day `d` uses the value from day `d-1`, not day `d`.

**Warning signs:**
- Walk-forward in-sample accuracy substantially higher than out-of-sample (look-ahead inflates in-sample)
- Macro feature importance ranks `cbr_rate` or `usdrub_close` at the top — plausible but suspicious without the shift test
- Backtest profit factor > 2.5 on MOEX but sandbox shows profit factor near 1.0

**Phase to address:**
Phase 2 (MOEX macro features). Add the shift test as a mandatory unit test before merging macro feature code.

---

### Pitfall 3: Walk-Forward Folds Produce Zero or One Valid Folds for MOEX

**What goes wrong:**
`generate_folds()` uses `_WF_TRAIN_MONTHS=12`, `_WF_CAL_MONTHS=2`, `_WF_TEST_MONTHS=4`, `_WF_STEP_MONTHS=3`, `_PURGE_GAP=100` (days). One fold requires approximately `12*30 + 100 + 2*30 + 100 + 4*30 = 740` days of data. With `_MOEX_LOOKBACK_DAYS=730`, the first fold barely fits (if at all) and the step of 3 months immediately pushes the second fold beyond the end of data. Result: 0 or 1 folds.

With 1 fold, `evaluate_walk_forward()` computes `passes/1 = 1.0` for any gate that passes on that single fold — trivially meeting the `min_passing_folds_ratio=0.60` threshold. The model is declared "overall passed" on essentially zero cross-validation, producing confident but meaningless verdicts.

**Why it happens:**
The fold constants were tuned for US data (5 years = ~60 months, producing 10–12 folds). Nobody adjusted them for the 24-month MOEX window. The code path `if not folds: print("No valid folds, aborting.")` correctly aborts on 0 folds, but silently proceeds with 1 fold — the dangerous case.

**How to avoid:**
Define MOEX-specific fold constants: `MOEX_WF_TRAIN_MONTHS=8`, `MOEX_WF_CAL_MONTHS=1`, `MOEX_WF_TEST_MONTHS=3`, `MOEX_WF_STEP_MONTHS=2`, `MOEX_PURGE_GAP=21` (21 trading days ~= 1 calendar month). This yields 3–4 folds on 730 days. Parameterize `generate_folds()` to accept these via a `FoldConfig` dataclass instead of module-level globals. Add a guard: `if len(folds) < 3: raise RuntimeError(f"Insufficient folds ({len(folds)}) for robust walk-forward validation")`.

**Warning signs:**
- `generate_folds()` returns `[1 fold]` for any `ru_*` segment
- `evaluate_walk_forward()` logs `n_folds=1` with `overall_passed=True`
- Walk-forward Sharpe computed from a single 3-month test window (extremely high variance)
- Experiment score of ~0.85 for a MOEX segment that the quality team knows should score ~0.55

**Phase to address:**
Phase 1 (MOEX data adapter) to establish what volume is available, and Phase 3 (adaptive quality gates) to tune fold constants for that volume.

---

### Pitfall 4: Adaptive Quality Gate Relaxation Enabling Degenerate Predictors

**What goes wrong:**
`check_accuracy_gate()` caps the threshold at `0.55` for `n_eff < 20`. On a MOEX test fold with 60 samples and `avg_hold_bars=4.0`, `n_eff = 60/4 = 15`. A model that predicts all-BUY on a 60%-positive class achieves 0.60 accuracy and clears the capped 0.55 gate — not because it has skill but because it learned class imbalance. `check_class_balance_gate()` exists to catch this (`_MIN_CLASS_RATIO=0.30`), but gate ordering matters: if accuracy is evaluated before class balance, the result can be misleadingly `gate_pass_rates = {accuracy: 1.0, class_balance: 0.0}` with `overall_passed=False` — correct outcome but confusing diagnostic.

The deeper issue: when the cap at 0.55 is further relaxed to handle even smaller MOEX datasets, the gap between "model has skill" and "model predicts majority class" narrows to zero. The quality gate system can no longer distinguish.

**Why it happens:**
Adaptive gate relaxation is the correct direction for small samples. But the existing implementation relaxes the accuracy gate without adding a compensating signal-quality check. The Brier gate with `_dynamic_brier_threshold` (floor at 0.15, tighter for small `n_eff`) is the intended compensator — but a degenerate all-BUY predictor also achieves a low Brier score on an imbalanced class (if positive class probability is ~0.6, always predicting 1 gives Brier of `(1-0.6)^2 * 0.4 + (0-0.6)^2 * 0.6 ≈ 0.22`, which passes the 0.25 cap but fails the 0.15 floor for `n_eff=15`). So the Brier gate does protect — verify this path is actually firing.

**How to avoid:**
Do not raise `_MAX_ACCURACY_THRESHOLD` beyond 0.57 for any `n_eff` value. Add a fold-level variance check as a new gate: if accuracy standard deviation across folds exceeds 0.15, flag as high-variance and require all 4+ folds to be present before declaring PASS. Add an explicit unit test: feed an all-BUY predictor on a 60%-positive class, verify class_balance gate fails, verify overall_passed=False regardless of accuracy value.

**Warning signs:**
- `buy_ratio` consistently above 0.75 across folds (degenerate predictor signature)
- Sensitivity near 1.0, specificity near 0.0
- `gate_pass_rates = {accuracy: 1.0, class_balance: 0.0, ...}` with `overall_passed=False` — correct but check that class_balance fires reliably
- `avg_profit_factor` near 1.0 despite high `avg_accuracy` (degenerate predictor has no edge on profitable trades specifically)

**Phase to address:**
Phase 3 (adaptive quality gates).

---

### Pitfall 5: ExperimentManager Integration Creating ID Conflicts Across Parallel Segment Runs

**What goes wrong:**
`auto_ml_research.py` generates experiment config names like `ablate-rsi`, `hp-xgb_max_depth=3`, `random-1-n8`. If the research loop creates an `ExperimentManager` markdown file for each of these 100+ configs (one `.md` per config), running the loop for two segments concurrently — e.g., `ru_blue_chips` and `us_tech` — produces ID collisions: both try to write `.planning/experiments/ablate-rsi.md`. `ExperimentManager._write_file()` uses `path.write_text()` (non-atomic), so concurrent writes corrupt the YAML frontmatter.

Even without concurrency, re-running the same segment wipes previous results because the experiment ID is not namespaced by segment or run timestamp.

**Why it happens:**
The research loop's experiment naming convention was designed for internal logging, not for the ExperimentManager's flat file namespace. The ExperimentManager expects unique, stable experiment IDs per hypothesis — the research loop generates hundreds of disposable config names per run.

**How to avoid:**
Map the two concepts to their correct abstraction levels. Use `ExperimentManager` for the top-level research hypothesis (one experiment per `run_research_loop()` invocation): ID = `{segment}_{strategy}_{YYYYMMDD_HHMM}`. Use the existing JSONL log (`results/experiments/{segment}_experiment_log.jsonl`) for per-config sub-results. Only promote the single best result (winner of the internal competition) to an ExperimentManager hypothesis with a meaningful description. This keeps ExperimentManager's namespace clean and human-readable.

**Warning signs:**
- `.planning/experiments/` contains files named `ablate-rsi.md` or `hp-xgb_max_depth=3.md` with no segment context
- Concurrent runs produce corrupt YAML frontmatter (YAML parse error when loading experiment state)
- `experiment_manager.list_experiments()` returns experiments whose IDs do not correspond to meaningful hypotheses

**Phase to address:**
Phase 4 (ExperimentManager integration). Define the ID schema before writing any integration code.

---

### Pitfall 6: Ensemble Weight Optimization Overfitting to Small MOEX Validation Sets

**What goes wrong:**
Optimizing XGBoost/LightGBM/CatBoost ensemble weights via grid search or Optuna on a MOEX validation set of 30–80 samples finds spurious optima. A weight vector [0.7, 0.2, 0.1] that scores well on one 3-month window will perform like random weighting on the next window because the optimization signal is dominated by noise at this sample size. The parameter space has 2 degrees of freedom (3 weights summing to 1), and with 30 samples and a noisy binary target, the expected improvement over equal weights is statistically near zero — but in-sample observed improvement can be 5–8%.

**Why it happens:**
Weight optimization is conceptually attractive: each model has different strengths. But the validation set is too small to distinguish model-specific skill from random correlation with the class distribution of that window. This is the same overfitting problem as hyperparameter search on small datasets, but disguised as "ensemble calibration."

**How to avoid:**
Default to equal weights (1/3 each) for MOEX segments unless the optimization is validated across 4+ independent folds. Use bootstrap confidence intervals: if the 95% CI on any weight includes 1/3, do not deviate from equal weighting. Constrain the weight search space so no single model weight exceeds 0.5 (simplex constraint). Log the weight-optimization gain separately so it can be audited: if optimized weights show accuracy improvement < 0.01 on OOS folds, revert to equal weights.

**Warning signs:**
- Optimized weights differ dramatically fold-to-fold: [0.8,0.1,0.1] then [0.1,0.8,0.1]
- Average optimized-weight accuracy lower than average equal-weight accuracy across held-out folds
- Score improvement from weight optimization > 3% on MOEX (suspicious at this sample size — would be extraordinary if real)

**Phase to address:**
Phase 5 (ensemble weight optimization strategy).

---

### Pitfall 7: Cross-Segment Feature Transfer Breaking on MOEX Distribution Shift

**What goes wrong:**
Feature names are shared across segments (same keys in `dict[str, float]`), creating an illusion of compatibility. A feature importance ranking trained on `us_tech` (AAPL, MSFT, NVDA) reflects US market dynamics: Amihud illiquidity rank, momentum z-scores, and VIX-based volatility regime. Transferring the same selected feature list to `ru_blue_chips` (SBER, LKOH, GAZP) creates a feature set that is identical in name but different in statistical properties:
- MOEX Amihud values are orders of magnitude larger (thinner market)
- MOEX momentum decays faster due to thinner order books
- Calendar features have Russian holidays (not NYSE calendar) — the existing `moex_calendar.trading_days_gap` handles this, but features using `timedelta` directly do not
- VIX is replaced by IMOEX volatility regime on MOEX — the US VIX feature has no MOEX analog

**Why it happens:**
The feature pipeline computes whatever is in `compute_features()` regardless of market. Cross-segment transfer naively uses "same feature name = same feature meaning." The distribution shift is only visible at model evaluation time when MOEX accuracy is below 0.52.

**How to avoid:**
Before applying US-selected features to MOEX, run a distribution shift check: compute Jensen-Shannon divergence between US and MOEX feature distributions for each selected feature. Flag features with JS divergence > 0.3 as "potentially invalid transfers" — log them and consider excluding. When adding MOEX macro features (CBR rate, USDRUB, IMOEX), give them priority over US-derived features: if the feature budget is 10, fill 4 slots with MOEX macro features first before drawing from US-derived feature rankings. Do not transfer hyperparameter configurations without re-tuning: MOEX's noisier labels benefit from shallower trees (max_depth 3–4 vs 5–6 for US).

**Warning signs:**
- Walk-forward accuracy on MOEX with US-selected features consistently below 0.52 (noise floor)
- Feature importance on MOEX shows VIX-based features with near-zero importance (VIX irrelevant for MOEX)
- MOEX macro features added in Phase 2 appear at the bottom of importance rankings (wrong feature set inherited from US selection)

**Phase to address:**
Phase 6 (cross-segment transfer strategy).

---

### Pitfall 8: Brent Fetch via yfinance Causing Non-Deterministic Training

**What goes wrong:**
`PROJECT.md` §Data Sources confirms Brent crude (`BZ=F`) is fetched via yfinance. If MOEX macro features include Brent as a predictor and it is fetched at training time via yfinance, the feature values depend on yfinance availability at that moment. In CI environments without network access, yfinance returns empty data — Brent feature is NaN or missing — and training silently proceeds on incomplete features. Two training runs on the same code at different times produce different models because Brent data differs.

**Why it happens:**
yfinance is an unofficial API with no SLA, rate limits, and schema changes without notice. It is already excluded from MOEX candle fetching by policy, but Brent as a CME commodity has no T-Invest equivalent. The macro feature builder may use a conditional fallback — which hides the failure mode in production.

**How to avoid:**
Cache Brent candles to disk using `CachingFetcher` with a 24-hour TTL immediately after fetching. Store fixture Brent data in `tests/fixtures/brent_candles.json` for use in CI. Make the Brent fetch a graceful degradation: if yfinance fails, use the cached file; if no cache exists, use a neutral value (zero-mean, unit-variance normalized to 0.5) and log a `WARNING`. Never fail training because Brent is unavailable — degrade, not crash.

**Warning signs:**
- `pytest` passes locally but fails in CI with `KeyError: 'brent_return'` or empty Brent series
- Two training runs on identical code produce different feature importance rankings (Brent present vs absent)
- MOEX macro feature importances vary dramatically between runs

**Phase to address:**
Phase 2 (MOEX macro features). Write the Brent fixture and caching before writing macro feature code — not after.

---

### Pitfall 9: T-Bank Sandbox Endpoint Has No Historical Candles

**What goes wrong:**
`TinkoffFetcher` defaults to `sandbox=True` for development safety. The sandbox gRPC endpoint (`sandbox-invest-public-api.tbank.ru:443`) is a paper-trading environment — it has no historical candle data. Calling `fetch_candles(symbol, start, end)` on the sandbox returns an empty list for any historical date range. The research script fetches zero candles, the dataset is empty, and the script aborts with "Empty dataset, aborting." — with no indication that the issue is the sandbox flag, not the symbol or date range.

**Why it happens:**
The sandbox-vs-live distinction is correct for the live trading loop (sandbox prevents accidental real orders). For historical data training scripts, sandbox=True is always wrong. The default is set in `TinkoffFetcher.__init__` for safety, but training scripts must override it explicitly.

**How to avoid:**
In `_fetch_moex_candles()`, always construct `TinkoffFetcher` with `sandbox=False`. Add a startup assertion: `assert not fetcher._sandbox, "TinkoffFetcher for training must use sandbox=False (historical data not available in sandbox)"`. Document this in the function docstring and in the script's `--help` output. In tests, use a mock fetcher that returns pre-seeded candles rather than connecting to any endpoint.

**Warning signs:**
- All symbols return 0 candles despite valid token and correct symbol names
- No gRPC error logs — the call succeeds but returns empty (sandbox has no data, not an error)
- Script aborts at "Empty dataset, aborting." on first run but not after changing to `sandbox=False`

**Phase to address:**
Phase 1 (MOEX data adapter). This is a one-line fix but a common trap for anyone unfamiliar with the dual-endpoint design.

---

### Pitfall 10: Combinatorial Feature Engineering Explosion on Small Datasets

**What goes wrong:**
One of the v9.0 research strategies is "automatic feature engineering" — generating candidate features by combining existing ones (ratios, products, lags, differences). On a 730-day MOEX dataset with 10 base features, generating all pairwise ratios and lagged versions yields hundreds of candidate features. Feature selection then runs on a dataset where `n_samples` (730) << `n_features` (300+), making any selection method statistically unreliable. The selected features are almost certainly overfit to the specific 730-day window.

**Why it happens:**
Combinatorial feature generation works well when `n_samples >> n_features` (e.g., US tech with 1825 days and 15 features → ratio ~120:1). On MOEX, the ratio inverts: 730 days and 300 candidate features → ratio ~2.4:1. Standard feature selection methods (mutual information, LASSO) cannot distinguish signal from noise at this ratio.

**How to avoid:**
Cap the candidate feature set hard: no more than `n_samples / 20` candidate features before selection. For MOEX with 730 days, the cap is ~36 features. Do not generate all pairwise combinations — generate only domain-motivated combinations (e.g., `return / volatility`, `momentum / liquidity`), not statistical fishing. Run a permutation test on any generated feature: if the feature's mutual information with the label is not significantly above the mutual information after label shuffling (p > 0.05), discard it before selection.

**Warning signs:**
- Feature selection step takes > 5 minutes (hundreds of candidates being evaluated)
- Selected features include `feature_a_times_feature_b` type combinations without domain motivation
- Ablation experiments show no consistent winner — every subset scores within noise of every other subset

**Phase to address:**
Phase 6 (automatic feature engineering strategy). This phase needs the constraint built in from the start, not discovered after running a 300-feature experiment.

---

## Technical Debt Patterns

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| Reuse US fold constants for MOEX | No code change needed | 0–1 valid folds, misleadingly high gate pass rates | Never — constants must be parameterized |
| Skip `InstrumentRegistry` init, use mock in production scripts | Fast to code | Missing FIGI → all fetches fail silently in production | Never in production; test mocks only |
| Use `shift(0)` (no shift) for macro series join | Simplest pandas join | Look-ahead bias corrupts all trained models | Never |
| Raise accuracy cap to 0.60 for MOEX | Models pass gates more easily | Enables degenerate predictors on 50–80 sample folds | Never |
| Store all 100+ experiment configs as ExperimentManager files | Full traceability | Namespace pollution, file conflicts, slow list_experiments() | Never — promote only the research run winner |
| Use `sandbox=True` for historical data fetches | Safe default | Returns empty data, training aborts silently | Never for training scripts |
| Generate all pairwise feature combinations | Explores more space | n_features > n_samples → selection is noise | Never without permutation test and hard cap |
| Equal weights always for ensemble (skip optimization) | No overfitting risk | May leave 2–5% accuracy on the table | Acceptable for MVP when n_folds < 4 |

---

## Integration Gotchas

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| TinkoffFetcher in sync script | Instantiating without `InstrumentRegistry` | Build minimal registry from static YAML map; assert token present before constructing |
| TinkoffFetcher gRPC | Forgetting `GRPC_DNS_RESOLVER=native` before SDK import | `tinkoff_data.py` sets it at module level — import that module before any `t_tech.invest` usage |
| TinkoffFetcher endpoint | Using default SDK target (`invest-public-api.tinkoff.ru`) that no longer resolves | Always pass `target=_TBANK_GRPC_TARGET` from `tinkoff_data.py` |
| TinkoffFetcher for training | Using `sandbox=True` (default) | Explicitly set `sandbox=False`; sandbox endpoint has no historical candles |
| CBR rate as macro feature | Fetching rate for "today" and using as same-day feature | `shift(1)`: CBR announces at ~13:30 Moscow time, unavailable at open-of-day |
| MOEX ISS IMOEX | Using same-day IMOEX close as open-of-day feature | `shift(1)`: yesterday's close is the only available value at today's open |
| Brent via yfinance | No caching, fails offline/CI | `CachingFetcher` with 24h TTL; fixture data in tests |
| ExperimentManager | One file per internal research config (100+ files) | One experiment per research loop run; use JSONL for sub-results |
| Feature engineering | All pairwise combinations without cap | Hard cap at `n_samples / 20`; permutation test before selection |

---

## Performance Traps

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| Fetching all MOEX candles in single gRPC call | `grpc_timeout` exceeded; fetch returns empty | Batch by 30-day windows (T-Invest API limit); retry on `UNAVAILABLE` | Any symbol with > 1 year of daily candles |
| Re-training 3 models per fold per experiment without data caching | 2–4 hour runtime for one MOEX segment research run | Pre-cache feature matrices to disk after first build; add `--skip-data-fetch` flag | Any run beyond 20 experiment configs |
| `select_features_efficient()` called independently per fold | Feature sets differ fold-to-fold, adding noise to fold comparison | Fix feature set after first baseline fold; reuse across subsequent folds of same experiment | Every multi-fold experiment — already partially avoided by `config.feature_subset` |
| Combinatorial feature generation without cap | Feature selection takes > 5 min; hundreds of pointless candidates | Hard cap at `n_samples / 20` candidates; generate domain-motivated pairs only | Any MOEX segment with < 1000 samples |

---

## Security Mistakes

| Mistake | Risk | Prevention |
|---------|------|------------|
| Logging `FINALAYZE_TINKOFF_TOKEN` in experiment JSONL | Token in plaintext file that may be committed to git | Never include env vars in `ExperimentResult`; scrub config dict before logging |
| Storing trained model pickle files in `results/experiments/` | Pickle deserialization vulnerability if results dir is shared | Keep model files in `models/<segment>/` only; reference by path in experiment log |
| ExperimentManager writing verdicts without snapshot_sha | Stale claim applied to wrong codebase version | Reuse `snapshot_sha` mechanism from `AgentOrchestrator` when integrating |

---

## "Looks Done But Isn't" Checklist

- [ ] **MOEX data adapter:** `_fetch_moex_candles()` returns non-empty candles for `SBER` — verify by running against live API, not just a unit test with a mock
- [ ] **Sandbox=False for training:** `TinkoffFetcher` instantiated with `sandbox=False` in training path — check the constructor call, not just docs
- [ ] **Macro features shift:** `cbr_rate` in feature vector uses yesterday's rate, not today's — inspect first 3 rows of feature matrix manually for a known CBR announcement date
- [ ] **Walk-forward folds count:** `generate_folds()` for `ru_blue_chips` returns >= 3 folds — log fold count before proceeding; abort if < 3
- [ ] **Degenerate predictor rejection:** All-BUY predictor fails `class_balance` gate before accuracy is evaluated — add unit test for this exact scenario
- [ ] **ExperimentManager ID uniqueness:** Running the script twice for the same segment produces experiment IDs with different timestamps — no file collisions, no overwrites
- [ ] **Ensemble weight confidence interval:** Weight optimization uses OOS folds for fitting — inspect weight fitting code for any use of training data
- [ ] **Cross-segment distribution shift:** JS divergence check executed and logged before applying US feature list to MOEX — look for `feature_distribution_shift` log entry
- [ ] **Brent fixture in CI:** `pytest` passes without network access using fixture Brent candles — run with `FINALAYZE_TINKOFF_TOKEN=` unset to simulate CI

---

## Recovery Strategies

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| Look-ahead bias in macro features | HIGH | Fix shift; retrain all MOEX models from scratch; invalidate all saved `ru_*` model files in `models/` |
| TinkoffFetcher returns empty candles | LOW | Check token present, `sandbox=False`, registry initialized; add debug logging to `_run_async`; test manually |
| Zero valid folds for MOEX | MEDIUM | Adjust fold constants per MOEX-specific values; may need to reduce `_PURGE_GAP` proportionally to MOEX trading days (~250/yr) |
| ExperimentManager file conflicts | LOW | Delete conflicting `.md` files; add segment prefix + timestamp to all IDs going forward |
| Ensemble weights overfit | MEDIUM | Reset to equal weights; re-run quality gate evaluation; document that weight optimization requires >= 4 folds |
| Cross-segment transfer fails | MEDIUM | Discard US feature list; run MOEX-native feature selection from scratch on MOEX data only |
| Combinatorial feature explosion | LOW | Delete generated candidates; cap at `n_samples / 20`; add permutation test before selection |
| Brent non-determinism in CI | LOW | Add fixture file; wrap Brent fetch in `CachingFetcher` with 24h TTL |

---

## Pitfall-to-Phase Mapping

| Pitfall | Prevention Phase | Verification |
|---------|------------------|--------------|
| TinkoffFetcher sync integration (Pitfall 1) | Phase 1: MOEX data adapter | pytest with real or mock fetcher returns candles for 3+ MOEX symbols; 0-candle result fails loudly |
| Sandbox endpoint for training (Pitfall 9) | Phase 1: MOEX data adapter | `sandbox=False` assertion in training code; unit test verifies correct endpoint used |
| Look-ahead bias in macro features (Pitfall 2) | Phase 2: MOEX macro features | Unit test: synthetic macro with known lag, assert feature uses t-1 value |
| Brent yfinance non-determinism (Pitfall 8) | Phase 2: MOEX macro features | pytest passes without network access using fixture data |
| Zero valid folds for MOEX (Pitfall 3) | Phase 1 + Phase 3 | `generate_folds()` returns >= 3 folds for `ru_blue_chips`; script aborts if fewer |
| Degenerate predictor bypass (Pitfall 4) | Phase 3: Adaptive quality gates | Unit test: all-BUY predictor on 60%-positive class fails class_balance gate; overall_passed=False |
| ExperimentManager ID conflicts (Pitfall 5) | Phase 4: ExperimentManager integration | Two parallel segment runs produce non-overlapping `.md` files |
| Ensemble weight overfitting (Pitfall 6) | Phase 5: Ensemble weight optimization | CI check: equal weights used when n_folds < 4; weight CI includes 1/3 → no deviation |
| Cross-segment distribution shift (Pitfall 7) | Phase 6: Cross-segment transfer | Distribution shift log present; US-only features excluded from MOEX feature set |
| Combinatorial feature explosion (Pitfall 10) | Phase 6: Automatic feature engineering | Feature candidate count logged; always <= n_samples/20 before selection |

---

## Sources

- Codebase inspection: `scripts/auto_ml_research.py` (sync data fetch pattern, fold generation constants, experiment config naming)
- Codebase inspection: `src/finalayze/ml/training/quality_gates.py` (accuracy cap at 0.55, `_SMALL_SAMPLE_CUTOFF=20`, Brier dynamic threshold, gate evaluation order)
- Codebase inspection: `src/finalayze/data/fetchers/tinkoff_data.py` (gRPC event loop pattern, sandbox vs live endpoint, `GRPC_DNS_RESOLVER=native`)
- Codebase inspection: `src/finalayze/core/experiment_manager.py` (file-based CRUD, flat namespace, `write_text()` non-atomicity)
- Codebase inspection: `src/finalayze/ml/features/technical.py` (feature naming, `_MIN_CANDLES=80`, `MoexMarketData` usage)
- `.planning/PROJECT.md` §Data Sources (Brent via yfinance, CBR XML, MOEX ISS), §Known Issues (ML quality gates fail for small MOEX datasets), §Constraints
- `MEMORY.md` (gRPC C-ares resolver fix, SDK target override, sandbox vs live endpoint distinction, `FINALAYZE_TINKOFF_TOKEN` env var)
- Known constraint: T-Invest sandbox endpoint has no historical candles (verified in prior v6.0 development)
- Known constraint: MOEX history ~730 days vs US ~1825 days (documented in `auto_ml_research.py` constants `_MOEX_LOOKBACK_DAYS=730`)
- Known constraint: `_MOEX_MAX_FEATURES=10` vs `_US_MAX_FEATURES=15` (documented in `auto_ml_research.py`)

---
*Pitfalls research for: MOEX ML AutoResearch & Adaptation (v9.0 milestone)*
*Researched: 2026-04-13*
