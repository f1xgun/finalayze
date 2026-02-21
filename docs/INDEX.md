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
| [design/SEGMENTS.md](design/SEGMENTS.md) | Stock segment system (8 segments) |
| [design/STRATEGIES.md](design/STRATEGIES.md) | Trading strategies: momentum, mean reversion, event-driven, pairs |
| [design/RISK.md](design/RISK.md) | Risk management rules and position sizing |
| [design/NEWS_PIPELINE.md](design/NEWS_PIPELINE.md) | News ingestion, LLM analysis, sentiment scoring |
| [design/ML_PIPELINE.md](design/ML_PIPELINE.md) | ML ensemble: XGBoost + LightGBM + LSTM |

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
| [operations/RUNBOOK.md](operations/RUNBOOK.md) | Operational runbook for incidents |
| [operations/DEPLOYMENT.md](operations/DEPLOYMENT.md) | Deployment procedures |
| [operations/MONITORING.md](operations/MONITORING.md) | Monitoring and alerting setup |

## Plans

| Document | Description |
|---|---|
| [plans/ROADMAP.md](plans/ROADMAP.md) | Phase overview with current status |
| [plans/PHASE_1.md](plans/PHASE_1.md) | Phase 1 detailed execution plan |

## Root-Level Documents

| Document | Description |
|---|---|
| [../CLAUDE.md](../CLAUDE.md) | Agent entry point, quick reference |
| [../WORKFLOW.md](../WORKFLOW.md) | Development process conventions |
| [../CHANGELOG.md](../CHANGELOG.md) | Project changelog |
| [../README.md](../README.md) | Repository README |
