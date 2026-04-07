---
title: Agent Consensus Drift Problem & Scientific Method Solution
date: 2026-04-07
context: Board meeting with 4 expert agents revealed contradictions in multi-agent recommendations
---

# Problem: Agent Consensus Drift

## Observed Behavior

When multiple domain-expert agents (quant-analyst, risk-officer, portfolio-strategist, etc.)
analyze the same system and then discuss findings:

1. Each agent finds valid information within their domain
2. When findings conflict, agents "negotiate" freely
3. The negotiated result often drifts to a **third option** that may be worse than either original
4. No mechanism exists to validate whether the consensus is better than the individual proposals
5. Agents don't cite specific sources (file:line, metric values) — assertions are unverifiable

## Root Cause

- Agent definitions (`.claude/agents/*.md`) are plain markdown with no structured output requirements
- No "evidence protocol" — agents can make claims without citing code or data
- No arbiter role — conflicts resolved by social negotiation, not empirical testing
- No interaction testing — proposals validated in isolation, not in combination

## Agreed Solution: Scientific Method for Agent Decisions

### Structured Debate Protocol (all decisions)

Each agent must provide:
1. **Claim** — specific, testable assertion
2. **Source** — file path, line number, metric value, or external reference
3. **Prediction** — "if we do X, metric Y will change by Z"
4. **Risk** — what could go wrong

If agents agree → proceed.
If agents conflict → escalate to experiment.

### Experiment Framework (decisions affecting metrics)

- **Hypothesis** defined before experiment with success criteria
- **Isolated test** — each proposal tested separately via backtest
- **Combination test** — proposals tested together (interaction effects)
- **Sandbox validation** — winning proposal validated in live sandbox
- **Multiple sandbox instances** — for parallel A/B testing of configs

### Experiment Lab UI (Streamlit)

Full lifecycle visibility:
- Why experiment started (hypothesis + context from debate)
- Pre-defined success criteria (metrics + thresholds)
- Current status and time remaining
- Results vs expectations
- Final decision and reasoning

### Architecture decisions (non-measurable)

For pure code organization decisions: implement and move on.
For architectural decisions that affect metrics: treat as experiment.

## Key Insight

Two good solutions individually can conflict when combined (interaction effects).
Must test A, B, AND A+B — like pharmaceutical interaction testing.
