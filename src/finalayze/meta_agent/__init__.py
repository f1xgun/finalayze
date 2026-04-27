"""Phase 58 Autonomous Meta-Agent (Layer 6).

Cron-driven monitor that snapshots system health from REST endpoints,
classifies it deterministically, and persists every decision to the
agent_decisions hypertable. See ``.planning/phases/58-autonomous-meta-agent/``
for SPEC, CONTEXT, PATTERNS, and RESEARCH artefacts.

Public API surface is intentionally narrow at this stage; modules are
loaded lazily by the orchestrator (``runner.py``) and the FastAPI router
(``api/v1/meta_agent.py``).
"""

from __future__ import annotations

__all__: list[str] = []
