"""Tests for meta_agent.runner.MetaAgentRunner (Phase 58-01, META-04).

Dry-run path: snapshot → classify → persist; NO Telegram, NO subprocess
spawns. SPEC §Acceptance Criterion #7.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

# Module-level constants (PLR2004).
_NUM_TICKS = 5
_NOW_BEFORE = datetime(2026, 4, 26, 11, 59, tzinfo=UTC)


@pytest.mark.asyncio
async def test_run_one_tick_dry_run_writes_one_decision_no_side_effects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """run_one_tick() in dry-run writes one row, no Telegram, no spawn."""
    from finalayze.meta_agent import runner as runner_module
    from finalayze.meta_agent.classifier import Severity
    from finalayze.meta_agent.runner import MetaAgentRunner
    from finalayze.meta_agent.snapshot import PositionsSummary, Snapshot

    settings = MagicMock()
    settings.meta_agent_dry_run = True
    settings.meta_agent_enabled = True
    settings.api_key = "test-key"

    fake_snapshot = Snapshot(
        timestamp=datetime.now(UTC),
        alerts_last_hour=[],
        drawdown_pct=0.0,
        equity_persist_failures=0,
        ml_signal_error_rate=None,
        positions_summary=PositionsSummary(raw={"positions": []}),
        raw={},
    )

    async def _fake_build_snapshot(_client, *, now):
        return fake_snapshot

    monkeypatch.setattr(runner_module, "build_snapshot", _fake_build_snapshot)
    monkeypatch.setattr(
        runner_module, "classify", lambda _snap: Severity.HEALTHY,
    )

    persistence = MagicMock()
    persistence.persist_decision = MagicMock()
    executor = MagicMock()
    executor.execute = AsyncMock()

    # Provide a fake http client factory that returns a context-manager-able
    # client we never actually call (build_snapshot is monkeypatched).
    fake_client = AsyncMock()
    fake_client.aclose = AsyncMock()
    runner = MetaAgentRunner(
        settings=settings,
        persistence=persistence,
        executor=executor,
        http_client_factory=lambda: fake_client,
    )

    await runner.run_one_tick()

    # Persisted exactly one decision.
    assert persistence.persist_decision.call_count == 1
    call_kwargs = persistence.persist_decision.call_args.kwargs
    assert call_kwargs["dry_run"] is True
    assert call_kwargs["status"] == "queued"

    # Executor NOT called (dry-run gate).
    executor.execute.assert_not_called()

    # last_run_ts populated.
    assert runner._last_run_ts is not None
    assert runner._last_run_ts >= _NOW_BEFORE
