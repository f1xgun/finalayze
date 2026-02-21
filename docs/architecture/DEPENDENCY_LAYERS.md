# Dependency Layers

## Rule

Imports must flow **downward only**. A module at Layer N may import from
Layer N or any Layer M where M < N. Importing upward (from a higher layer
number) is strictly forbidden.

## Layer Definitions

### Layer 0: Types & Schemas

**Modules:** `core/schemas.py`, `core/exceptions.py`, `core/enums.py`, `core/types.py`

- Pure data definitions: Pydantic models, enums, type aliases, custom exceptions.
- Zero external dependencies beyond Pydantic and the standard library.
- No I/O, no side effects, no imports from any other project layer.

### Layer 1: Configuration

**Modules:** `config/settings.py`, `config/modes.py`, `config/segments.py`

- Reads environment variables and YAML files.
- Provides typed settings objects (Pydantic `BaseSettings`).
- May import from: Layer 0.

### Layer 2: Data / Repository

**Modules:** `data/`, `markets/`

- Database repositories (SQLAlchemy models, CRUD operations).
- Market data adapters (Alpaca, Tinkoff Invest connectors).
- Instrument registry, segment definitions, currency conversion.
- May import from: Layers 0, 1.

### Layer 3: Analysis / ML

**Modules:** `analysis/`, `ml/`

- News sentiment analysis (LLM integration).
- Technical indicator computation (pandas-ta).
- Feature engineering and ML model inference (XGBoost, LightGBM, LSTM).
- May import from: Layers 0, 1, 2.

### Layer 4: Strategy / Risk

**Modules:** `strategies/`, `risk/`

- Trading strategy implementations (momentum, mean reversion, event-driven, pairs).
- Signal combination and weighting.
- Risk management: position sizing, drawdown limits, exposure caps.
- May import from: Layers 0, 1, 2, 3.

### Layer 5: Execution

**Modules:** `execution/`

- Order routing and lifecycle management.
- Broker abstraction (Alpaca, Tinkoff, simulated).
- Fill reconciliation, slippage tracking.
- May import from: Layers 0, 1, 2, 3, 4.

### Layer 6: API / Dashboard

**Modules:** `api/`, `dashboard/`

- FastAPI REST endpoints.
- Streamlit dashboard.
- WebSocket feeds.
- May import from: Layers 0, 1, 2, 3, 4, 5.

## Dependency Matrix

| Source \ Target | L0 | L1 | L2 | L3 | L4 | L5 | L6 |
|---|---|---|---|---|---|---|---|
| **L0: Core** | -- | | | | | | |
| **L1: Config** | ok | -- | | | | | |
| **L2: Data** | ok | ok | -- | | | | |
| **L3: Analysis/ML** | ok | ok | ok | -- | | | |
| **L4: Strategy/Risk** | ok | ok | ok | ok | -- | | |
| **L5: Execution** | ok | ok | ok | ok | ok | -- | |
| **L6: API/Dashboard** | ok | ok | ok | ok | ok | ok | -- |

`ok` = allowed import direction. Empty cells = **forbidden**.

## Enforcement

- **Static analysis:** A custom ruff rule or `import-linter` configuration
  will enforce these boundaries in CI.
- **Code review:** Every PR must be checked for layer violations (see
  [WORKFLOW.md](../../WORKFLOW.md) review checklist).
- **Tests:** An architectural test in `tests/test_architecture.py` can
  programmatically verify no upward imports exist.

## Common Pitfalls

1. **Circular imports between strategies/ and ml/**: Both are at different
   layers (L3 and L4). Strategies may import ML, but ML must not import
   strategy logic. If ML needs strategy context, pass it via schemas (L0).

2. **Config importing data models**: Config (L1) must not import SQLAlchemy
   models from data (L2). Use Pydantic schemas from core (L0) instead.

3. **Core importing anything**: Layer 0 has zero project imports. If you need
   a utility in core that depends on config, it belongs in a higher layer.
