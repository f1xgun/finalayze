# Feature Research

**Domain:** Multi-agent orchestration & autonomous decision loop for algorithmic trading
**Researched:** 2026-04-12
**Confidence:** HIGH (grounded in existing codebase + industry research)

---

## Context: What Already Exists (v7.0 baseline)

Before categorizing features, the existing infrastructure must be understood to avoid
re-building what is already there:

| Component | Status | Location |
|-----------|--------|----------|
| `AgentOutput` schema (claims, recommendation, timestamp) | BUILT | `core/schemas.py` |
| `Claim` schema with `ClaimSource` (file/metric discriminator) | BUILT | `core/schemas.py` |
| `FactCheckReport`, `ClaimVerdict`, `ClaimCheckResult` | BUILT | `core/schemas.py` |
| `DebateState`, `DebateStatus` (open/resolved/escalated) | BUILT | `core/schemas.py` |
| `DebateManager` (CRUD, add_agent_position, add_arbiter_report, escalate) | BUILT | `core/debate_manager.py` |
| `ExperimentManager` (create, link_result, record_verdict, list, get_by_debate) | BUILT | `core/experiment_manager.py` |
| `ExperimentState`, `SuccessCriteria`, `ExperimentResult`, `ExperimentStatus` | BUILT | `core/schemas.py` |
| `_compute_verdict()` with INCONCLUSIVE 10% band | BUILT | `core/experiment_manager.py` |
| Experiment Lab Streamlit UI (list/detail/history tabs, Plotly charts) | BUILT | `dashboard/pages/experiments_list.py` |
| 18 Claude Code sub-agents (quant-analyst, risk-officer, ml-engineer, etc.) | BUILT | `.claude/agents/` |
| Weekly deep-dive skill (parallel sub-agents + deliberation rounds) | BUILT | `.claude/skills/weekly-deep-dive.md` |
| Backtest engine with `--hypothesis` and `--run-name` flags | BUILT | `backtest/` |
| Debate→experiment bidirectional link (escalate_debate, debate_id on ExperimentState) | BUILT | both managers |

**The gap:** None of the above components are wired to each other in a running pipeline.
Agents (weekly-deep-dive, daily-review) produce unstructured markdown. `AgentOutput` is
never emitted by any agent skill. Conflict detection does not exist. Debate/experiment
lifecycle is manually invoked only. Auto-apply on verdict is not implemented.

---

## Feature Landscape

### Table Stakes (Users Expect These)

These are the minimum features for v8.0 to deliver on its stated goal: "wire debate/experiment
infrastructure into live agent workflows so agents emit structured claims, conflicts
auto-trigger debates, and experiment verdicts auto-apply."

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| Agents emit `AgentOutput` with structured `Claim` objects | Foundational contract -- all downstream features require it. Currently agents produce unstructured markdown, making the debate/experiment infrastructure unreachable. | MEDIUM | Wire into weekly-deep-dive and daily-review skills. `Claim` must include source references (`MetricSource` from history.jsonl or `FileLineSource` from YAML presets). Ungrounded claims are worse than markdown -- false precision. |
| Conflict detector comparing multi-agent outputs for contradictions | The only reason to run multiple agents in parallel is to surface disagreements. Without detection, parallel agents are just expensive parallel summarizers. | HIGH | Compare `AgentOutput` objects from the same review round. Identify claims with the same topic but opposing direction or contradicting magnitude. Must handle same claim stated in different words (semantic, not string equality). See complexity notes below. |
| Arbiter auto-triggers on detected conflict | Manual arbiter invocation breaks the autonomous loop. Without auto-trigger, conflicts sit unresolved in files indefinitely. | HIGH | When `ConflictDetector` fires: create debate via `DebateManager.create_debate()`, add agent positions, run arbiter agent, record `FactCheckReport` via `add_arbiter_report()`, escalate to experiment if warranted via `escalate_debate()`. |
| Full orchestration: disagreement -> debate -> experiment -> backtest -> verdict | The complete v8.0 value proposition. All sub-components exist; the work is the coordinator that sequences them. | HIGH | Orchestrator calls: detect -> create debate -> arbiter runs -> escalate -> trigger backtest (`--hypothesis`, `--run-name`) -> `record_verdict()`. Must be resumable: file-based state in DebateManager/ExperimentManager already provides persistence. |
| Auto-apply on ACCEPT: parameter changes or strategy toggles | Without auto-apply, `ExperimentStatus.ACCEPTED` verdicts sit in files with zero effect on the live system. The loop is not autonomous if humans must manually apply results. | HIGH | Read `preset_overrides` from `ExperimentState`, write to strategy YAML presets. Snapshot current YAML to `.planning/param-history/` before overwrite. Log apply action to history.jsonl with `source: experiment_id`. Never apply on INCONCLUSIVE or REJECTED. |
| INCONCLUSIVE routing -> Telegram alert | The 10% inconclusive band already exists in `_compute_verdict()`. Without routing INCONCLUSIVE to human review, these results stall the pipeline silently. | LOW | On `ExperimentStatus.INCONCLUSIVE`: emit Telegram alert to human with experiment_id and verdict reasoning. Add to a pending review queue. Do not auto-apply. |

### Differentiators (Beyond Standard Multi-Agent Patterns)

These features go beyond the TradingAgents-style debate pattern and are specific to
Finalayze's needs as a live trading system managing real capital (500K-2.5M RUB).

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Claim source traceability (file:line or metric:value mandatory) | Makes arbitration fact-checkable against actual data, not LLM opinion. Without this, a `FactCheckReport` is just one LLM judging another LLM with no ground truth. | LOW | Schema already supports it. Gap: agents must populate sources. `MetricSource` ties to history.jsonl (Sharpe, PF, DD). `FileLineSource` references strategy YAML paths. Enforcement: `Claim.source` is already required by the Pydantic schema. |
| Conflict severity scoring | Not all contradictions are equal. A 2% Sharpe disagreement between agents is less critical than one agent recommending disabling a strategy while another recommends increasing its weight. | MEDIUM | Score based on: magnitude of disagreement, risk category of topic, confidence delta between claims. High-severity conflicts escalate to full debate. Low-severity get noted in weekly summary only. Reduces noise from trivial disagreements. |
| Rollback safety gate on auto-apply | Auto-apply to live trading parameters without rollback is a production risk. The system must be able to recover from a bad auto-apply before it causes drawdown. | MEDIUM | Before writing preset overrides: snapshot current YAML to `.planning/param-history/YYYYMMDD-HH.yaml`. After apply: on next backtest iteration, if Sharpe or PF degrades beyond a threshold, auto-revert and Telegram alert. One revert attempt only to avoid oscillation. |
| Debate->experiment UI link in Experiment Lab | Enables auditing why a parameter changed: which agents disagreed, which experiment ran, what verdict was reached. | LOW | The bidirectional link exists in data (DebateState.experiment_id, ExperimentState.debate_id). Gap: Experiment Lab UI does not surface it. Add clickable link to debate file from experiment detail view. |

### Anti-Features (Commonly Requested, Often Problematic)

| Feature | Why Requested | Why Problematic | Alternative |
|---------|---------------|-----------------|-------------|
| Full LLM-to-LLM natural language debate for every daily analysis cycle | "Agents should argue like humans" -- rich deliberation sounds thorough | Massively expensive at daily frequency ($5-10/week for weekly review is acceptable; $5-10/day is not). Latency adds 10-60s to daily cycle. LLMs without structured grounding hallucinate evidence in debates. | Reserve natural language deliberation for the weekly-deep-dive skill where it already works well. Daily cycles emit structured `AgentOutput` only. |
| Real-time conflict detection during live trading | "Catch conflicts as they happen" -- sounds like maximum responsiveness | MOEX trading cycles are sub-minute. Conflict detection adds 10-60s latency. False positives could delay trade execution during market hours. Agents generating signals in real-time are not the same agents doing strategic analysis. | Run conflict detection on the post-cycle analysis batch (weekly-deep-dive, daily-review), not inline with the trading loop. |
| Automatically hold cash when agents conflict | "Conflict = uncertainty = don't trade" -- sounds risk-conservative | Destroys the trading edge. Agents will frequently disagree on signal magnitude while agreeing on direction. The system would hold cash most of the time. | Use the existing circuit breaker and confidence threshold system. Agent conflicts feed the experiment queue, not the signal combiner. |
| Agent consensus voting to override backtest threshold | "If all agents agree, the backtest threshold can be lowered" -- appeals to consensus | Undermines empirical validation. LLM consensus is a prior; backtest metrics are evidence. Priors do not override evidence, especially not for live capital deployment. | Agent consensus can increase experiment priority but cannot bypass `SuccessCriteria`. All `ACCEPTED` verdicts require passing the backtest threshold. |
| Version-controlled parameter history in the database | "Full DB auditability of every parameter change" -- sounds rigorous | Over-engineered for current scale. Adds migration, schema, ORM model, and query layer for a feature that file-based storage handles adequately at 500K-2.5M RUB capital scale. | `preset_overrides` in `ExperimentState` plus YAML backup snapshots provide adequate auditability. |
| Full gRPC async migration for experiment runner | "The experiment trigger should use fully async gRPC" -- sounds architecturally pure | The backtest engine runs as a subprocess/script, not an async service. Converting it to gRPC is a multi-week rewrite with zero benefit for v8.0. | Trigger backtest via subprocess call (already done with `--hypothesis` and `--run-name` flags). Record results by reading history.jsonl after completion. |

---

## Feature Dependencies

```
[Agents emit AgentOutput]
    └──required-by──> [Conflict Detector]
                          └──required-by──> [Arbiter auto-trigger]
                                                └──required-by──> [Full orchestration pipeline]
                                                                      └──required-by──> [Auto-apply on ACCEPT]
                                                                      └──required-by──> [INCONCLUSIVE -> Telegram]

[Claim source traceability] ──enhances──> [Agents emit AgentOutput]
    └──enables──> [Arbiter auto-trigger] (fact-checking requires grounded sources)

[Conflict severity scoring] ──enhances──> [Conflict Detector]
    └──feeds-into──> [Full orchestration pipeline] (low-severity skips debate, high-severity escalates)

[Rollback safety gate] ──requires──> [Auto-apply on ACCEPT]
    (cannot rollback what was never applied)

[Debate->experiment UI link] ──requires──> bidirectional link (already built)
    └──gap: surface in Streamlit only
```

### Dependency Notes

- **AgentOutput with sources is the enabler:** Without structured output from agents, the
  conflict detector has nothing to compare, the arbiter has nothing to fact-check, and the
  orchestration pipeline cannot start. This must be Phase 1.

- **Conflict detection precedes arbiter:** The arbiter makes an LLM API call. It should only
  run when the conflict detector identifies a real contradiction, not on every analysis round.

- **INCONCLUSIVE and Auto-apply are mutually exclusive paths:** The `_compute_verdict()` logic
  already separates these. The gap is routing each path to the correct downstream action
  (apply vs alert), not the verdict computation itself.

- **Rollback gate depends on auto-apply existing:** It makes no sense to implement rollback
  before auto-apply is working. Rollback is a v8.x hardening feature, not a v8.0 blocker.

---

## MVP Definition

### Launch With (v8.0)

- [ ] Agents emit `AgentOutput` with structured `Claim` objects and mandatory source references --
      wire into weekly-deep-dive and daily-review skills. This is the prerequisite for all else.
- [ ] Conflict detector: compare `AgentOutput` objects from the same review round, identify
      contradicting claims, emit a structured conflict report.
- [ ] Arbiter auto-triggers on detected conflict: creates debate via `DebateManager`,
      runs arbiter agent, records `FactCheckReport`, escalates to experiment.
- [ ] Full orchestration: detect -> debate -> experiment -> backtest trigger -> `record_verdict()`.
      State persistence via existing file-based managers (no new DB tables needed).
- [ ] Auto-apply on ACCEPT: read `preset_overrides`, write to YAML preset with backup snapshot.
- [ ] INCONCLUSIVE routing: Telegram alert to human, no auto-apply.
- [ ] Claim source traceability: enforce populated sources in agent output (schema already
      requires it; agent skill implementations must comply).

### Add After Validation (v8.x)

Features to add once the core loop has run in sandbox for 2-4 weeks:

- [ ] Conflict severity scoring -- add after observing real conflict patterns and tuning the
      detector. Implement once false-positive rate is measurable.
- [ ] Rollback safety gate -- add once auto-apply has demonstrated it makes changes that need
      reverting. Premature rollback logic adds complexity with no benefit yet.
- [ ] Debate->experiment UI link in Experiment Lab -- low-effort, add opportunistically.

### Future Consideration (v9+)

- [ ] Automated daily-review conflict detection (currently weekly only) -- defer until weekly
      loop is validated and false-positive rate is known.
- [ ] Agent performance tracking (which agents' claims are VERIFIED vs CONTRADICTED most often).
- [ ] Multi-experiment interaction testing at orchestration level (beyond manual A/B/AB).

---

## Feature Prioritization Matrix

| Feature | User Value | Implementation Cost | Priority |
|---------|------------|---------------------|----------|
| Agents emit AgentOutput | HIGH (enabler) | MEDIUM | P1 |
| Claim source traceability | HIGH (safety) | LOW | P1 |
| Conflict detector | HIGH | HIGH | P1 |
| Arbiter auto-trigger | HIGH | HIGH | P1 |
| Full orchestration pipeline | HIGH | HIGH | P1 |
| Auto-apply on ACCEPT | HIGH | MEDIUM | P1 |
| INCONCLUSIVE -> Telegram routing | HIGH (safety) | LOW | P1 |
| Rollback safety gate | MEDIUM (safety) | MEDIUM | P2 |
| Conflict severity scoring | MEDIUM | MEDIUM | P2 |
| Debate->experiment UI link | LOW | LOW | P3 |

**Priority key:**
- P1: Must have for v8.0
- P2: Add when core loop proves out
- P3: Nice to have, defer to v8.x

---

## Complexity Notes (for Roadmap Phase Sizing)

### Conflict Detection is the Hardest Problem

Comparing two LLM-generated `AgentOutput` objects for semantic contradiction is non-trivial:
- String equality fails (same claim, different wording = false negative)
- Pure embedding similarity misses directional opposition ("strong buy signal" vs "weak sell
  signal" have similar embeddings but opposite recommendations)
- Cosine similarity cannot distinguish "Sharpe 0.52" from "Sharpe 0.48" as contradictory

**Practical approach:** Constrain `Claim.statement` vocabulary in agent skill instructions.
Require claims to follow a structured template: `[METRIC] [DIRECTION] [MAGNITUDE]` where
DIRECTION is one of: INCREASE/DECREASE/STABLE/ENABLE/DISABLE. This enables deterministic
contradiction detection (same METRIC, opposite DIRECTION) without LLM involvement for
common cases. Reserve LLM-based semantic comparison for edge cases where structured parsing
fails.

**TradingAgents precedent:** The TradingAgents framework (arxiv:2412.20138) uses structured
documents for information transfer and natural language dialogue only for debates within the
Researcher Team. This hybrid validates the approach: structure first, narrative second.

### Auto-Apply is a Trust Boundary

Writing to strategy YAML presets affects live capital (500K-2.5M RUB). Implementation must:
1. Validate that `preset_overrides` keys exist in the target YAML before writing (no key injection)
2. Snapshot current YAML to `.planning/param-history/` before overwrite (point-in-time backup)
3. Log the apply action to `results/iterations/history.jsonl` with `source: experiment_id`
4. Never apply on INCONCLUSIVE or REJECTED (this logic already exists in `_compute_verdict()`)
5. Validate the resulting YAML parses correctly before committing (avoid corrupted presets)

### Orchestration Pipeline is Coordination Logic, Not New Infrastructure

The sub-components (DebateManager, ExperimentManager, backtest engine with --hypothesis flag)
already work and are tested. The v8.0 work is writing the coordinator that sequences them.
This is medium-complexity Python, not infrastructure work. The state machine is:

```
IDLE -> CONFLICT_DETECTED -> DEBATE_OPEN -> ARBITER_RAN -> EXPERIMENT_CREATED
     -> BACKTEST_TRIGGERED -> VERDICT_RECORDED -> (APPLIED | HUMAN_REVIEW | REJECTED)
```

Each transition writes to a file (debate or experiment markdown). Crashes are recoverable
by reading the current file state and resuming from the last completed transition.

---

## Sources

- Codebase review: `src/finalayze/core/schemas.py` (lines 563-757), `core/debate_manager.py`,
  `core/experiment_manager.py` -- confirmed existing infrastructure
- `.planning/PROJECT.md` -- v8.0 requirements confirmed (active requirements section)
- `.claude/skills/weekly-deep-dive.md` -- existing agent emission pattern (unstructured markdown)
- TradingAgents framework (arxiv:2412.20138 v7, 2025): bull/bear researcher debate pattern,
  structured documents + natural language dialogue combination validated at ICML 2025
- Cogent: multi-agent orchestration failure playbook (semantic hashing for loop detection,
  escape sequences on conflict detection)
- Microsoft Azure Architecture Center: AI agent design patterns for enterprise orchestration
- Multi-agent systems orchestration research (arxiv:2601.13671v1): coordination patterns,
  centralized/hierarchical models

---
*Feature research for: v8.0 Agent Integration & Autonomous Decision Loop*
*Researched: 2026-04-12*
