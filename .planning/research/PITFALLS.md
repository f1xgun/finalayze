# Pitfalls Research

**Domain:** Adding multi-agent orchestration, conflict detection, and auto-apply loops to a live MOEX trading system (v8.0)
**Researched:** 2026-04-12
**Confidence:** HIGH (codebase analysis of 480+ files, prior phase research docs, risk module inspection)

---

## Critical Pitfalls

### Pitfall 1: Auto-Apply Bypasses the 11-Check Pre-Trade Pipeline

**What goes wrong:**
The experiment verdict auto-apply path (ACCEPT → strategy toggle or parameter change) modifies preset YAML files directly and reloads config. If auto-apply skips the normal `PreTradeChecker` evaluation, a parameter set that passes backtesting (on historical data) can produce positions that fail the 11 live pre-trade checks: max exposure, drawdown limits, PDT rule, correlation limits, parameter freshness, etc. The system might load a new config that, on the next strategy cycle, generates a signal that the pre-trade checker would block — but only if the checker runs. If auto-apply also restarts or reloads the combiner mid-cycle, the checker is bypassed for the in-flight cycle.

**Why it happens:**
Auto-apply is designed as a write operation on YAML presets. The strategy combiner re-reads presets on the next cycle. But nothing in the auto-apply path enforces that the new parameters produce compliant pre-trade results. A backtest environment does not run `PreTradeChecker` — it uses a simulated broker and positions are reset per walk-forward window. The live pre-trade checker tracks cumulative drawdown, position history, and rolling PDT trades — none of which exist in backtest context.

**How to avoid:**
- Auto-apply MUST gate on circuit breaker state before writing any preset: if `CircuitLevel != NORMAL`, reject the apply and log `auto_apply_blocked_by_circuit_breaker`.
- After writing the YAML, do NOT reload the combiner mid-cycle. Apply takes effect at the next cycle boundary only (APScheduler's `misfire_grace_time` already provides this naturally).
- Add a dry-run pre-trade check as part of verdict execution: generate a synthetic signal with the new parameters and run it through `PreTradeChecker`. If it fails, mark the experiment `INCONCLUSIVE` rather than applying.
- Log all auto-apply events to the Telegram alerter with the changed parameters, so a human can observe and revert if needed.

**Warning signs:**
- Experiment verdict is ACCEPT but circuit breaker immediately trips on next cycle.
- PreTradeChecker starts failing checks it never failed before, shortly after an auto-apply event.
- Rapid CAUTION → HALTED escalation within 2 cycles of parameter application.

**Phase to address:**
Phase implementing auto-apply (the v8.0 "auto-apply loop" phase). Circuit-breaker gate must be the FIRST check in `apply_verdict()`.

---

### Pitfall 2: Conflict Detector Triggers Continuous Debate Storms

**What goes wrong:**
The conflict detector compares multi-agent outputs for contradictions. If the comparison is too sensitive (e.g., any disagreement on a float metric triggers a conflict), and multiple agents run in each cycle, the system generates dozens of debates per day. Each debate spawns an experiment, each experiment runs 3 backtests (A/B/AB), and each backtest is 12+ months of walk-forward data. The system becomes permanently occupied running backtests and cannot process live trading decisions. Worse: if debate escalation triggers auto-apply, and two simultaneous debates both ACCEPT conflicting parameter changes, the second write overwrites the first within the same cycle.

**Why it happens:**
Debates are file-based and stateless from the trading loop's perspective. Nothing prevents two agents from both creating debate files about the same parameter at the same time (e.g., both `quant-analyst` and `risk-officer` disagree about the ADX threshold). Without debouncing or deduplication on the topic, the system can stack up 10+ debates on the same strategy parameter within one day, each spawning its own A/B/AB backtest triple.

**How to avoid:**
- Implement topic-level debate deduplication: before creating a debate file, scan `.planning/debates/` for any open or escalated debate on the same topic slug. If one exists and is < 7 days old, do not create a new debate — link to the existing one.
- Rate-limit conflict detection: only trigger a debate if the same contradiction is observed in at least 2 consecutive agent cycles (not a single observation). One-off disagreements should be logged but not escalated.
- Backtest runs from experiment verdicts must be queued, not parallelized. A single queue with `max_concurrent=1` prevents backtest pile-up.
- Define a minimum confidence delta for conflict: only flag as contradiction if agent confidence values differ by > 0.15 (not just any disagreement).

**Warning signs:**
- `.planning/debates/` directory contains > 10 files with `status: open`.
- `results/experiments/` directory growing faster than 3 entries/day.
- Strategy cycles start taking longer than 5 minutes (backtest processes consuming CPU).

**Phase to address:**
Phase implementing the conflict detector. Debouncing and deduplication logic belongs in the detector, not the debate manager.

---

### Pitfall 3: File-Based Preset Write Races with Live Strategy Cycle

**What goes wrong:**
The strategy combiner reads preset YAML files on first call per segment (cached in memory). Auto-apply writes a new preset YAML. If the write happens mid-cycle while the combiner has already read the old preset for some instruments but not yet for others, the cycle runs with two different parameter sets for the same segment — half the instruments use old parameters, half use new. This is particularly dangerous for `min_combined_confidence` and strategy weights, which gate trade entry.

**Why it happens:**
`StrategyCombiner` uses a module-level or instance-level cache that is populated lazily. `auto_apply_verdict()` writes the YAML file directly (via `Path.write_text()`). There is no lock or versioning between the file write and the combiner's cache. The trading loop's APScheduler cycle and the auto-apply event (triggered by experiment verdict) are independent — they do not share a lock.

**How to avoid:**
- Auto-apply MUST write to a staging file (e.g., `presets/ru_blue_chips.yaml.pending`) and only rename to the live path at a cycle boundary.
- Add a cycle-start hook in `TradingLoop._strategy_cycle_impl()` that checks for `.pending` files and renames them before the combiner processes any instrument. This keeps the rename atomic within a cycle.
- The combiner cache should have a `reload_segment(segment_id)` method. Call it explicitly after the rename, before processing any instruments for that segment.
- Use `os.replace()` (atomic on Linux) not `Path.write_text()` for the final rename — ensures no partial reads.

**Warning signs:**
- Trade log shows instruments in the same segment with different `min_combined_confidence` values in the same cycle.
- Combiner logs cache hits with an old timestamp shortly after a preset write.
- A BUY signal fires with old parameters, and the new parameters would have blocked it.

**Phase to address:**
Phase implementing auto-apply. Staging-file-plus-atomic-rename must be the write pattern, enforced in the `apply_verdict()` function.

---

### Pitfall 4: Experiment ACCEPT on Backtest Sharpe Does Not Translate to Live Performance

**What goes wrong:**
The experiment success criteria use backtest metrics (WF Sharpe, profit factor, max drawdown) as the verdict gate. An experiment that shows WF Sharpe improvement from 0.08 to 0.14 on walk-forward data gets ACCEPTED and auto-applied. But backtests use end-of-bar prices, no slippage for strategy parameter changes (parameters are static across the walk-forward window), and no execution latency. In live MOEX trading, the same parameters may not translate to the same improvement — especially for `ou_mean_reversion` (stationary-parameter strategies) and `pairs` (cointegration-based), where the walk-forward parameter freshness check (`_MAX_PARAM_AGE_BARS = 5`) means parameters fitted on old data may be stale by the time they're applied live.

**Why it happens:**
Backtest and live environments differ in: (1) execution slippage (MOEX has 0.1%-0.3% bid-ask for mid-cap stocks), (2) parameter fitting (OU and pairs strategies fit parameters on in-sample data, which isn't available in the live environment), (3) walk-forward Sharpe is computed on 6-month test windows — insufficient to measure regime sensitivity. The experiment registry compares backtest metrics to a fixed threshold, not to a live baseline.

**How to avoid:**
- Add a mandatory sandbox validation gate before auto-apply: any ACCEPTED experiment must run for ≥3 sandbox trading days before applying to live parameters. The auto-apply pipeline should be: ACCEPT → sandbox\_monitoring → sandbox\_score\_gate → live\_apply.
- Success criteria thresholds should be conservative: WF Sharpe improvement must exceed 0.05 (not 0.01) to avoid applying noise as signal.
- For `ou_mean_reversion` and `pairs`, auto-apply is blocked by default. Mark these strategies with `auto_apply_blocked: true` in the experiment definition schema, requiring manual override.
- Log the live sandbox performance delta against the experiment's projected improvement. If delta > 50% mismatch after 5 trading days, flag for human review rather than keeping the auto-applied parameters.

**Warning signs:**
- Auto-applied parameters produce lower WF Sharpe in the next backtest iteration than the pre-apply baseline.
- Sandbox position fill rates drop after auto-apply (market impact from parameter changes).
- Strategy fires more frequently than expected (weight changes not accounting for regime filter impact).

**Phase to address:**
Phase implementing auto-apply verdict execution. The sandbox-gate requirement must be in the `apply_verdict()` workflow, not deferred to a later phase.

---

### Pitfall 5: Arbiter Agent Claims Create Circular Dependency on Codebase State

**What goes wrong:**
The arbiter verifies `FileLineSource` claims by checking that a file path and line number contain the stated excerpt. If auto-apply has modified a preset YAML (which changes strategy parameters), and a subsequent debate references the OLD parameter values in a `FileLineSource` claim (citing the old preset line), the arbiter marks the claim CONTRADICTED (because the file now shows new values). This is a false contradiction — the claim was valid at debate creation time but invalid at arbiter evaluation time. The false CONTRADICTED verdict triggers another experiment, which reverses the auto-apply, which triggers another debate, creating a loop.

**Why it happens:**
Debates are created and arbitrated asynchronously. The gap between debate creation and arbiter execution can span hours or days. In that window, auto-apply may have already changed the referenced file. The arbiter has no concept of "file state at debate creation time" — it always checks the current file state.

**How to avoid:**
- Add a `snapshot_sha` field to `FileLineSource`: when a debate claim is created, record the git SHA or file content hash at that line. The arbiter compares the current file state against the snapshot, not an expected value. If the file has changed (SHA differs), mark the claim `UNTESTABLE` (not CONTRADICTED) with a note that the referenced code has been modified since the claim was made.
- Auto-apply must create a git commit (or at minimum write a changelog entry to `.planning/debates/changelog.md`) with the applied change, so the arbiter can detect that a file change was intentional and corresponds to a completed experiment.
- Implement a "debate freeze" period: after auto-apply, block new debates on the same topic for 48 hours. This prevents immediate re-debate of freshly applied parameters.

**Warning signs:**
- Arbiter CONTRADICTED verdicts spike after auto-apply events.
- Debates reference file lines that no longer exist (line number drift after preset rewrite).
- `experiment_id` values in debates point to other experiments that have already been ACCEPTED — circular experiment chains.

**Phase to address:**
Phase implementing arbiter integration with auto-apply. The `snapshot_sha` field belongs in Phase 33 schemas; the changelog write belongs in the auto-apply phase.

---

### Pitfall 6: Strategy Toggle Auto-Apply Leaves Open Positions Unmanaged

**What goes wrong:**
Auto-apply can toggle a strategy from enabled to disabled (REJECT verdict on a strategy enablement experiment, or ACCEPT verdict on a strategy disablement experiment). If the strategy being disabled (`event_driven`, `dual_momentum`, etc.) currently has open positions that it generated signals for, disabling the strategy does not close those positions. The positions remain open but now have no strategy actively managing their exit. Stop-loss still applies (ATR-based, independent of strategy), but the strategy's exit signals (SELL on momentum reversal, mean-reversion target hit, etc.) will no longer fire.

**Why it happens:**
Strategy toggling in the preset YAML changes whether the combiner includes that strategy in signal generation. It does not affect `_stop_states`, `_entry_prices`, or `_cycle_exited_symbols` in `TradingLoop`. These dictionaries track ALL open positions regardless of which strategy generated them. A disabled strategy is simply absent from signal generation — its positions coast on ATR stop-loss only.

**How to avoid:**
- Before disabling a strategy via auto-apply, query `TradingLoop.get_positions()` for positions tagged with that strategy (this requires strategy tagging on position entry — `_entry_strategy: dict[str, str]`).
- If any positions exist that were opened by the strategy being disabled, block the disable and mark the experiment `INCONCLUSIVE` with reason "open positions must be closed first".
- Alternatively: apply the disable at the next DAILY RESET boundary (not mid-cycle), so the daily reset closes or re-evaluates all positions before the strategy goes dark.

**Warning signs:**
- Strategy is marked disabled in preset YAML but `_stop_states` still contains symbols it would have managed.
- Position exits are delayed after a strategy toggle (ATR-based exits fire much later than strategy-based exits).
- Profit factor drops on segments where a strategy was disabled without position cleanup.

**Phase to address:**
Phase implementing auto-apply for strategy toggles. Position-ownership tagging should be added to `TradingLoop._entry_prices` or a new `_entry_strategy` dict.

---

## Technical Debt Patterns

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| Skip sandbox gate for "safe" parameter changes (weight adjustment < 0.05) | Faster auto-apply cycle | Small weight changes compound — multiple small changes over weeks drift the portfolio far from backtested parameters | Never — always require sandbox gate |
| Use file modification timestamp as conflict dedup key instead of topic hash | Simple implementation | Two debates on the same topic within 1 second both survive dedup, race condition | Never — use topic slug hash |
| Hard-code success criteria thresholds in Python instead of per-experiment YAML | One less moving part | Thresholds cannot be adjusted per strategy or market regime without code changes | Only if thresholds never differ across strategy types |
| Apply verdict immediately (not at cycle boundary) to reduce latency | Parameters available sooner | Races with in-flight cycle reading old parameters | Never for live trading — cycle boundary is mandatory |
| Use subprocess for interaction test backtest runs instead of in-process | Isolation, no state leakage | Subprocess startup overhead: each backtest takes 30+ minutes already, adding 2-3s startup is fine, but subprocess makes it harder to mock in tests | Acceptable for production, use in-process mocks for tests |
| Store debate claims as plain text (not Pydantic models) to simplify agent prompt | Agent produces simpler output | Unvalidated claims reach the arbiter, which must handle malformed input gracefully — increases arbiter complexity | Never — schema validation at ingestion is cheaper than arbiter error handling |

---

## Integration Gotchas

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| File-based experiment registry + live preset YAML | Writing experiment result and preset YAML in same `apply_verdict()` call without checking if the experiment file still exists | Always read-then-write: `ExperimentManager.read_experiment()` first, check `status == ACCEPTED`, THEN write preset. Prevents stale experiment refs from triggering duplicate applies |
| Arbiter agent + `history.jsonl` metric lookup | Comparing float values with `==` (metric value `1.2900001` does not equal claimed `1.29`) | Use `abs(actual - claimed) <= 0.01` tolerance as documented in Phase 33 research. Store claimed value with 4 decimal places |
| Conflict detector + existing agent outputs (freeform markdown) | Trying to detect conflicts in freeform agent text with regex | Only compare agents that emit structured `AgentOutput` with typed `Claim` objects. Freeform text agents must be opted-in explicitly, not included automatically |
| Auto-apply + `StrategyCombiner` preset cache | Combiner caches preset on first use per segment (instance variable). Auto-apply writes new YAML. Combiner never sees new file until restart | Call `combiner.invalidate_segment_cache(segment_id)` after successful preset write. Requires adding this method to StrategyCombiner |
| Experiment backtest + `results/iterations/history.jsonl` | Experiment runs appending to `history.jsonl` pollute the `backtest-iteration` skill's iteration history, making trend tracking noisy | Tag experiment entries with `"tags": ["experiment", experiment_id]`. The `iteration-history` skill must filter out `experiment`-tagged entries by default |
| Debate persistence + git history | Auto-apply writes preset YAML but does not commit, so debate claims citing old git SHA become UNTESTABLE after the working tree changes | Auto-apply should write a changelog entry (not a git commit) that records what changed, when, and why. Full git commits are not appropriate for automated agent actions |

---

## Performance Traps

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| Running all 3 interaction test backtests sequentially without segment filtering | Interaction test takes 3-4 hours per run (3 backtests × 12 segments × 30min each) | Pass `--segments <relevant_segment>` to interaction test runner. Most experiments affect 1-2 segments | Immediately — a 3-segment A/B/AB test already takes ~3 hours |
| Conflict detector running on every agent invocation (no cooldown) | Agent invocations slow from <1s to 5s+ as detector reads all debate files | Conflict detector reads debate file directory once per agent session, not per claim. Cache directory listing with 60s TTL | When `.planning/debates/` accumulates > 50 files |
| Arbiter running `ast-index rebuild` before every claim verification | Arbiter runs take 30+ minutes instead of 30s | `ast-index rebuild` only when codebase changes (detect via `git status --porcelain`). Otherwise use existing index | Immediately if rebuild is in every arbiter invocation |
| Experiment registry scanning all experiment files for `get_by_debate()` | Reverse lookup (debate → experiment) takes O(N) file reads | Add a reverse index: on experiment create, append to `.planning/experiments/index.json` with `{debate_id: experiment_id}` mapping | When > 20 experiments exist |
| Auto-apply triggering a full `run_iteration.py` for backtest validation pre-apply | Adds 30+ minutes to every auto-apply event | Separate the backtest-validate step (run by experiment runner) from the apply step (triggered by verdict). Apply should consume pre-existing results, not re-run backtests | Immediately if validation backtest is inline in apply |

---

## Security Mistakes

| Mistake | Risk | Prevention |
|---------|------|------------|
| Writing auto-applied preset YAML content derived from experiment YAML frontmatter without sanitizing `preset_overrides` | Path traversal or YAML injection if `experiment_id` or `preset_overrides` keys contain `../` or YAML anchors | Validate `experiment_id` with `[a-zA-Z0-9_-]` pattern at experiment creation. Use `yaml.safe_load()` for all YAML reads. Validate that `preset_overrides` keys match known segment IDs before applying |
| Arbiter agent executes `ast-index` with a `FileLineSource.path` that contains shell metacharacters | Shell injection if arbiter uses `subprocess.run(f"ast-index outline {path}", shell=True)` | Always pass arguments as a list to `subprocess.run()`: `["ast-index", "outline", path]`. Never use `shell=True`. Validate that `path` starts with `src/` or `tests/` before passing to subprocess |
| Auto-apply modifies strategy parameters that affect position sizing limits | Malicious or buggy experiment verdict could set `weight: 10.0` (10x normal weight), bypassing Kelly sizing limits | Validate all auto-applied numeric parameters against schema bounds before writing: `weight` must be in `[0.0, 1.0]`, `min_combined_confidence` must be in `[0.30, 0.60]`. Reject any value outside predefined safe ranges |
| Debate file YAML frontmatter allows arbitrary `experiment_id` strings that map to file paths | A crafted `experiment_id` of `../../some_other_file` could cause `ExperimentManager._experiment_path()` to write outside `.planning/experiments/` | Strip and validate `experiment_id` to `[a-zA-Z0-9_-]` only. Use `Path(self._base_dir / f"{experiment_id}.md").resolve()` and assert it starts with `self._base_dir.resolve()` |

---

## UX Pitfalls

| Pitfall | User Impact | Better Approach |
|---------|-------------|-----------------|
| Experiment verdict shown as "ACCEPTED" in dashboard but auto-apply is still pending (sandbox gate not cleared) | User believes parameters are live; audits the system; finds old parameters | Add explicit `PENDING_APPLY` status between ACCEPTED and applied. Dashboard shows "Accepted — awaiting sandbox validation (Day 2/3)" |
| Debate files accumulate without resolution; dashboard shows 30+ "open" debates | User loses trust in the debate system — debates are created but nothing comes of them | Add a "stale debate" alert: any debate with `status: open` and `arbiter_report: null` older than 48 hours triggers a Telegram warning |
| Auto-apply event not surfaced in Telegram alerts | User not notified that strategy parameters changed autonomously during live trading | Every auto-apply event MUST fire a Telegram alert with: old parameters, new parameters, experiment ID, and a "REVERT" command option |
| Experiment Lab UI shows experiment history mixed with backtest iterations | User cannot distinguish "experiment ran as hypothesis test" from "regular optimization run" | Filter experiment-tagged entries from the main iteration history view. Show experiments only in the Experiment Lab tab |

---

## "Looks Done But Isn't" Checklist

- [ ] **Conflict detection**: Often missing the "same-topic debounce" check — verify that creating two debates on the same topic within 24 hours creates only one file, not two
- [ ] **Auto-apply circuit breaker gate**: Often implemented but only checked at verdict time, not at apply time — verify that a circuit breaker trip BETWEEN verdict and apply still blocks the apply
- [ ] **Preset cache invalidation**: Often missing after auto-apply — verify that the combiner uses new parameters on the NEXT cycle, not 2+ cycles later
- [ ] **Position ownership tagging**: Required for strategy-disable auto-apply — verify that `_entry_strategy` tracking exists before the disable path is implemented
- [ ] **Telegram auto-apply alerts**: Often added to the happy path but not to the rejection/rollback path — verify alerts fire for BOTH successful apply and blocked apply events
- [ ] **Sandbox gate completion**: Auto-apply may mark experiment as "applied" before the sandbox gate period completes — verify that `status: applied` is only set after all sandbox gate checks pass, not when the YAML write completes
- [ ] **Experiment-tagged history.jsonl entries**: Often experiment runs are indistinguishable from regular runs — verify `tags: [experiment, {experiment_id}]` is present on all experiment iteration entries
- [ ] **Atomic preset rename**: Often implemented as `write_text()` (non-atomic) — verify that the staging-file-plus-rename pattern is used, not direct overwrite

---

## Recovery Strategies

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| Auto-apply applied bad parameters to live preset | HIGH | 1. Send `/stop` via Telegram bot to halt trading. 2. Manually restore preset YAML from git history (`git show HEAD:src/finalayze/strategies/presets/ru_blue_chips.yaml`). 3. Restart TradingLoop. 4. Mark experiment as REJECTED manually. 5. Review circuit breaker state before resuming |
| Debate storm created 20+ experiments from one conflict | MEDIUM | 1. Set all `status: open` debate files to `status: closed` manually. 2. Delete pending experiment files with no backtest results. 3. Tighten conflict detector confidence delta threshold. 4. Restart experiment queue |
| Preset file partially written (truncated during crash) | MEDIUM | 1. Check `.planning/experiments/` for `.pending` files. 2. Compare file size against git baseline: `git diff --stat HEAD`. 3. Restore from git or from the most recent `history.jsonl` entry that preceded the crash |
| Strategy disabled while positions open | HIGH | 1. Do not re-enable the strategy (creates new signals that may contradict open position direction). 2. Wait for ATR stop-loss to close positions naturally. 3. If position is deeply negative, use Telegram `/stop` + manual close via broker UI. 4. After positions clear, re-enable strategy if appropriate |
| Arbiter creates false CONTRADICTED verdict (due to file change after claim creation) | LOW | 1. Mark debate as `status: closed` with note "arbiter evaluated stale code state". 2. If experiment was triggered, set experiment to `status: inconclusive`. 3. Add `snapshot_sha` to the claim schema for future debates |
| Circular experiment chain (experiment reverses auto-apply, triggers new experiment) | HIGH | 1. Manually set all chained experiments to `status: inconclusive`. 2. Remove `.pending` preset files. 3. Set circuit breaker to manual hold. 4. Add topic freeze in conflict detector for the contested parameter |

---

## Pitfall-to-Phase Mapping

| Pitfall | Prevention Phase | Verification |
|---------|------------------|--------------|
| Auto-apply bypasses pre-trade pipeline (Pitfall 1) | Auto-apply phase: `apply_verdict()` implementation | Test: set circuit breaker to HALTED, trigger auto-apply verdict, verify no preset file written |
| Debate storm from oversensitive conflict detection (Pitfall 2) | Conflict detector phase | Test: inject 10 conflicting agent outputs for same topic, verify only 1 debate file created with debouncing |
| File-write race with live cycle (Pitfall 3) | Auto-apply phase: staging-file-plus-rename pattern | Test: write preset mid-cycle and verify combiner uses old params for current cycle, new params for next |
| Backtest ACCEPT doesn't translate to live (Pitfall 4) | Auto-apply phase: sandbox gate requirement | Test: sandbox gate must be required field in apply workflow; missing gate = apply blocked |
| Arbiter false CONTRADICTED after code change (Pitfall 5) | Debate schema phase (add `snapshot_sha`) and auto-apply changelog phase | Test: modify a preset, then run arbiter on a pre-change claim, verify `UNTESTABLE` not `CONTRADICTED` |
| Strategy toggle leaves unmanaged positions (Pitfall 6) | Auto-apply phase for toggles + position ownership tagging | Test: open position with strategy X, disable strategy X via auto-apply, verify apply is blocked until position closes |

---

## Sources

- Codebase analysis: `src/finalayze/risk/circuit_breaker.py` — CircuitLevel states, sticky escalation rules
- Codebase analysis: `src/finalayze/risk/pre_trade_check.py` — 11-check pipeline, `_HALTING_LEVELS`
- Codebase analysis: `src/finalayze/orchestration/trading_loop.py` — `_stop_states`, `_entry_prices`, APScheduler cycle structure
- Codebase analysis: `src/finalayze/strategies/combiner.py` — preset cache loading, segment-per-instance caching
- Codebase analysis: `src/finalayze/strategies/presets/ru_blue_chips.yaml` — preset YAML structure
- Phase 33 research: `.planning/phases/33-structured-debate-protocol/33-RESEARCH.md` — debate schema pitfalls (line drift, float tolerance, orphan debates)
- Phase 34 research: `.planning/phases/34-experiment-registry-runner/34-RESEARCH.md` — preset override mechanics, verdict computation, interaction test race conditions
- `.planning/PROJECT.md` — v8.0 requirements, constraints (500K-2.5M RUB capital, 10% max drawdown hard limit)
- `.planning/research/PITFALLS.md` (v6.0) — prior stability pitfalls; pattern for this document

---
*Pitfalls research for: multi-agent orchestration + conflict detection + auto-apply loop on live MOEX trading system*
*Researched: 2026-04-12*
