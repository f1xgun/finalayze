# Architecture Decision Records (ADRs)

This document tracks significant architecture decisions made during development.

## Template

Use the following template when adding a new ADR:

---

### ADR-XXXX: [Title]

**Date:** YYYY-MM-DD
**Status:** Proposed | Accepted | Deprecated | Superseded by ADR-XXXX

**Context:**
What is the issue that we are seeing that is motivating this decision?

**Decision:**
What is the change that we are proposing and/or doing?

**Consequences:**
What becomes easier or more difficult to do because of this change?

---

## Decisions

### ADR-0001: Use uv as package manager

**Date:** 2026-02-21
**Status:** Accepted

**Context:**
The project needs a fast, reliable Python package manager that supports lockfiles
and reproducible builds. Alternatives considered: pip + pip-tools, poetry, pdm.

**Decision:**
Use uv for dependency management. The lockfile (`uv.lock`) is committed to
version control.

**Consequences:**
- Fast installs and resolution (10-100x faster than pip).
- Single tool for venv management and package installation.
- Team members must install uv (not just pip).

---

### ADR-0002: Layered architecture with strict import rules

**Date:** 2026-02-21
**Status:** Accepted

**Context:**
As the system grows across markets, strategies, and ML models, unconstrained
imports will create a tangled dependency graph that is hard to test and maintain.

**Decision:**
Adopt a 7-layer architecture (Layer 0 through Layer 6) with strictly downward
imports only. See [DEPENDENCY_LAYERS.md](DEPENDENCY_LAYERS.md).

**Consequences:**
- Clear boundaries make individual layers testable in isolation.
- New contributors can understand the system by reading one layer at a time.
- Some boilerplate is required to pass data across layers via schemas.

---

### ADR-0003: Four work modes (debug, sandbox, test, real)

**Date:** 2026-02-21
**Status:** Accepted

**Context:**
A trading system must never accidentally execute real trades during development
or testing. We need clear separation between development, paper trading, testing,
and live trading environments.

**Decision:**
Implement four work modes controlled via configuration. Every component that
performs I/O or has side effects must check the active mode and behave accordingly.

**Consequences:**
- Safety: real-money trading requires explicit mode activation.
- Complexity: every broker, data source, and execution path has mode-specific behavior.
- Testing: `test` mode uses historical replay, enabling deterministic tests.

---

<!-- Add new ADRs above this line -->
