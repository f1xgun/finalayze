"""Meta-agent exception hierarchy (Phase 58, Layer 6).

All exceptions inherit from ``FinalayzeError`` (CONVENTIONS §Exceptions).
Class names end in ``Error`` per ruff N818.

These exceptions surface from the meta-agent spawn path:
  - ``MetaAgentDeniedPathError`` — pre-spawn validator (58-04 owns the validator;
    Plan 58-03 declares the type so the killswitch + executor can import-from).
  - ``MetaAgentSpawnCapExceededError`` — daily spawn cap reached.
"""

from __future__ import annotations

from finalayze.core.exceptions import FinalayzeError


class MetaAgentDeniedPathError(FinalayzeError):
    """Pre-spawn validator rejected a fix prompt that referenced a denied path.

    Raised by the FIX-spawn pre-spawn validator (Plan 58-04) when the prompt
    references one of ``src/finalayze/risk/``, ``src/finalayze/execution/``,
    ``src/finalayze/core/`` (SPEC §Requirement 7 hard-deny list).
    """


class MetaAgentSpawnCapExceededError(FinalayzeError):
    """Daily spawn cap reached; further spawns rejected for the UTC day.

    Raised when ``meta_agent_max_spawns_per_day`` (INVESTIGATE) or
    ``meta_agent_max_fix_spawns_per_day`` (FIX) rows already exist with
    ``status IN ('spawned','completed','failed')`` for the current
    ``date_trunc('day', NOW() AT TIME ZONE 'UTC')`` window (SPEC AC #11).
    """
