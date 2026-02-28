# Agents Configuration Design

**Date:** 2026-02-28
**Status:** Approved

## Overview

Two-tier Claude Code sub-agent system in `.claude/agents/*.md`.

## Tier 1: Domain Expert Agents (4)

Invoked for analysis, audit, and high-level design review. Cross-cut the
entire codebase. Produce structured audit reports, create GitHub issues,
and make targeted code fixes.

| Agent | Focus |
|---|---|
| `quant-analyst` | Strategy math, signal quality, backtest validity, Sharpe/drawdown |
| `risk-officer` | Position sizing, circuit breakers, stop-losses, pre-trade checks |
| `ml-engineer` | Feature engineering, model architecture, training pipeline, overfitting |
| `systems-architect` | Layer violations, async correctness, data flow, latency |

## Tier 2: Module Agents (12)

Per-source-module implementers. Used as the implementer role in
`subagent-driven-development`. Know their module's files, layer rules,
test patterns, and coding conventions.

| Agent | Module | Layer |
|---|---|---|
| `core-agent` | `core/` | L0 |
| `config-agent` | `config/` | L1 |
| `data-agent` | `data/` | L2 |
| `markets-agent` | `markets/` | L2 |
| `analysis-agent` | `analysis/` | L3 |
| `ml-agent` | `ml/` | L3 |
| `strategies-agent` | `strategies/` | L4 |
| `risk-agent` | `risk/` | L4 |
| `execution-agent` | `execution/` | L5 |
| `backtest-agent` | `backtest/` | L2–L4 (special: no upward imports) |
| `api-agent` | `api/`, `dashboard/` | L6 |
| `infra-agent` | `docker/`, `alembic/`, `pyproject.toml`, CI | infra |

## WORKFLOW Integration

Domain experts:
- Brainstorm phase: before finalising any strategy/risk/ML/architecture design
- Quarterly: full parallel audit of all 4 experts

Module agents:
- `subagent-driven-development`: controller identifies affected module → dispatches module agent as implementer

New WORKFLOW section: **§8 Agent Dispatch Rules**.
