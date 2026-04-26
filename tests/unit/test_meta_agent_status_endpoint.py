"""Tests for the GET /api/v1/meta-agent/status endpoint and the
``meta_agent_*`` Settings cluster (Phase 58-01, META-04 / META-08 surface).

Initial test (Task 58-01-01) covers only the Settings field defaults.
The status-endpoint tests live further down (Task 58-01-11) and assume
the FastAPI router exists.
"""

from __future__ import annotations

import pytest

# Module-level constants (PLR2004 — no magic numbers in tests).
_DEFAULT_INTERVAL_MIN = 30
_DEFAULT_TG_CAP = 12
_DEFAULT_SPAWN_CAP = 10
_DEFAULT_FIX_CAP = 2


def test_settings_exposes_meta_agent_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """A freshly constructed ``Settings()`` exposes all six meta_agent_* defaults
    (SPEC §Constraints line 117) — safety defaults: enabled=False, dry_run=True.

    The local .env may override LLM_PROVIDER with the operator's headless
    sentinel (`claude_code_headless`) which is not in the project's Literal —
    explicitly set a valid value via monkeypatch so this test exercises only
    the meta_agent_* surface.
    """
    monkeypatch.setenv("FINALAYZE_LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("FINALAYZE_MODE", "debug")

    from config.settings import Settings

    s = Settings()
    assert s.meta_agent_enabled is False
    assert s.meta_agent_dry_run is True
    assert s.meta_agent_interval_minutes == _DEFAULT_INTERVAL_MIN
    assert s.meta_agent_max_telegram_alerts_per_day == _DEFAULT_TG_CAP
    assert s.meta_agent_max_spawns_per_day == _DEFAULT_SPAWN_CAP
    assert s.meta_agent_max_fix_spawns_per_day == _DEFAULT_FIX_CAP
