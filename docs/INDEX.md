# Finalayze Documentation Index

This is the master index of all project documentation. Start with the
[CLAUDE.md](../CLAUDE.md) file in the project root for a quick orientation.

## Architecture

| Document | Description |
|---|---|
| [architecture/OVERVIEW.md](architecture/OVERVIEW.md) | High-level system architecture, component map |
| [architecture/DEPENDENCY_LAYERS.md](architecture/DEPENDENCY_LAYERS.md) | Import layering rules, allowed dependencies |
| [architecture/DATA_FLOW.md](architecture/DATA_FLOW.md) | Event flow, data pipeline diagrams |
| [architecture/DECISIONS.md](architecture/DECISIONS.md) | Architecture Decision Records (ADRs) |

## Design

| Document | Description |
|---|---|
| [design/MARKETS.md](design/MARKETS.md) | Multi-market design: US (Alpaca) and MOEX (Tinkoff) |
| [design/SEGMENTS.md](design/SEGMENTS.md) | Stock segment system (9 segments) |
| [design/STRATEGIES.md](design/STRATEGIES.md) | Trading strategies: 8 strategies + ADX routing + combiner |
| [design/RISK.md](design/RISK.md) | Risk management rules and position sizing |
| [design/NEWS_PIPELINE.md](design/NEWS_PIPELINE.md) | News ingestion, LLM analysis, sentiment scoring |
| [design/ML_PIPELINE.md](design/ML_PIPELINE.md) | ML ensemble: XGBoost + LightGBM + LSTM, feature engineering |

## API

| Document | Description |
|---|---|
| [api/ENDPOINTS.md](api/ENDPOINTS.md) | REST API contract, endpoint reference |

## Quality

| Document | Description |
|---|---|
| [quality/GRADES.md](quality/GRADES.md) | Quality grades per module domain |
| [quality/GAPS.md](quality/GAPS.md) | Technical debt tracker |
| [quality/TEST_STRATEGY.md](quality/TEST_STRATEGY.md) | Testing approach per module |

## Operations

| Document | Description |
|---|---|
| [operations/DEPLOYMENT.md](operations/DEPLOYMENT.md) | Production Docker deployment procedures |
| [operations/MONITORING.md](operations/MONITORING.md) | Prometheus + Alertmanager monitoring setup |
| [operations/RUNBOOK.md](operations/RUNBOOK.md) | Operational runbook for incidents |

## Plans & Roadmap

| Document | Description |
|---|---|
| [plans/ROADMAP.md](plans/ROADMAP.md) | Phase overview with current status (Phase 0-5) |
| [plans/PHASE_1.md](plans/PHASE_1.md) | Phase 1 detailed execution plan |

### Phase 5 Plans (2026-03-01 →)

| Document | Description |
|---|---|
| [plans/2026-03-04-sprint4-design.md](plans/2026-03-04-sprint4-design.md) | Sprint 4 architecture design |
| [plans/2026-03-04-sprint4-plan.md](plans/2026-03-04-sprint4-plan.md) | Sprint 4 execution plan |
| [plans/2026-03-05-week1-bug-fixes.md](plans/2026-03-05-week1-bug-fixes.md) | Week 1 bug fix plan |
| [plans/2026-03-06-week3-design.md](plans/2026-03-06-week3-design.md) | Week 3 structural design |
| [plans/2026-03-07-week4-design.md](plans/2026-03-07-week4-design.md) | Week 4 optimization design |
| [plans/2026-03-07-week4-plan.md](plans/2026-03-07-week4-plan.md) | Week 4 execution plan |
| [plans/2026-03-07-ml-training-plan.md](plans/2026-03-07-ml-training-plan.md) | ML training pipeline plan |
| [plans/2026-03-07-ml-ensemble-fix-plan.md](plans/2026-03-07-ml-ensemble-fix-plan.md) | ML ensemble signal quality fixes |
| [plans/2026-03-08-ml-feature-expansion-plan.md](plans/2026-03-08-ml-feature-expansion-plan.md) | 16 new ML features (cross-asset, regime, calendar) |
| [plans/2026-03-08-ml-improvement-plan.md](plans/2026-03-08-ml-improvement-plan.md) | ML accuracy improvement plan |

### Earlier Phase Plans

| Document | Description |
|---|---|
| [plans/2026-03-01-improvement-plan.md](plans/2026-03-01-improvement-plan.md) | Initial improvement plan |
| [plans/2026-03-01-phase6a-risk-compliance.md](plans/2026-03-01-phase6a-risk-compliance.md) | Risk compliance review |
| [plans/2026-03-01-phase6b-strategy-quality.md](plans/2026-03-01-phase6b-strategy-quality.md) | Strategy quality improvements |
| [plans/2026-03-01-phase6c-ml-pipeline.md](plans/2026-03-01-phase6c-ml-pipeline.md) | ML pipeline improvements |
| [plans/2026-03-01-phase6d-architecture-security.md](plans/2026-03-01-phase6d-architecture-security.md) | Architecture & security review |
| [plans/2026-03-02-iteration-tracker-design.md](plans/2026-03-02-iteration-tracker-design.md) | Iteration tracker design |
| [plans/2026-03-04-mvp-launch-plan.md](plans/2026-03-04-mvp-launch-plan.md) | MVP launch plan |

## Evaluations

| Document | Description |
|---|---|
| [evaluations/2026-03-01-eval.md](evaluations/2026-03-01-eval.md) | Initial evaluation |
| [evaluations/2026-03-01-batch-eval.md](evaluations/2026-03-01-batch-eval.md) | Batch backtest evaluation |
| [evaluations/2026-03-01-phase7-post-fix-eval.md](evaluations/2026-03-01-phase7-post-fix-eval.md) | Post-fix evaluation |

## Research

| Document | Description |
|---|---|
| [research/2026-03-02-deep-strategy-research.md](research/2026-03-02-deep-strategy-research.md) | Deep strategy research |
| [research/2026-03-03-trading-research.md](research/2026-03-03-trading-research.md) | Trading system research |
| [research/2026-03-07-ml-deep-research.md](research/2026-03-07-ml-deep-research.md) | ML pipeline deep research |
| [research/ml-trading-system.md](research/ml-trading-system.md) | ML trading system reference |

## Expert Reviews

| Document | Description |
|---|---|
| [reviews/2026-03-02-architecture-review.md](reviews/2026-03-02-architecture-review.md) | Systems architect review |
| [reviews/2026-03-02-ml-review.md](reviews/2026-03-02-ml-review.md) | ML engineer review |
| [reviews/2026-03-02-quant-review.md](reviews/2026-03-02-quant-review.md) | Quant analyst review |
| [reviews/2026-03-02-risk-review.md](reviews/2026-03-02-risk-review.md) | Risk officer review |

## Root-Level Documents

| Document | Description |
|---|---|
| [../CLAUDE.md](../CLAUDE.md) | Agent entry point, quick reference |
| [../WORKFLOW.md](../WORKFLOW.md) | Development process conventions |
| [../CHANGELOG.md](../CHANGELOG.md) | Project changelog |
| [../README.md](../README.md) | Repository README |
