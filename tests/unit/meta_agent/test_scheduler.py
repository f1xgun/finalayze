"""Tests for meta_agent.scheduler (Phase 58-01, META-04 wiring).

  - register_meta_agent_job: only adds the job when settings.meta_agent_enabled
  - _meta_agent_cycle: sync wrapper that defers to asyncio.run_coroutine_threadsafe
"""

from __future__ import annotations

from unittest.mock import MagicMock

# PLR2004 — module-level constants.
_INTERVAL_MIN = 30


def test_register_meta_agent_job_only_when_enabled() -> None:
    """When meta_agent_enabled=False the scheduler.add_job MUST NOT be called.

    SPEC §Acceptance Criterion #6: meta_agent_enabled=False → no
    APScheduler job registered.
    """
    from finalayze.meta_agent.scheduler import register_meta_agent_job

    scheduler = MagicMock()
    settings = MagicMock()
    settings.meta_agent_enabled = False
    settings.meta_agent_interval_minutes = _INTERVAL_MIN
    runner = MagicMock()

    result = register_meta_agent_job(
        scheduler, settings=settings, runner=runner, async_loop=None,
    )

    assert result is False
    scheduler.add_job.assert_not_called()


def test_register_meta_agent_job_adds_interval_job_when_enabled() -> None:
    """When enabled, register_meta_agent_job adds an IntervalTrigger job."""
    from apscheduler.triggers.interval import IntervalTrigger

    from finalayze.meta_agent.scheduler import register_meta_agent_job

    scheduler = MagicMock()
    settings = MagicMock()
    settings.meta_agent_enabled = True
    settings.meta_agent_interval_minutes = _INTERVAL_MIN
    runner = MagicMock()

    result = register_meta_agent_job(
        scheduler, settings=settings, runner=runner, async_loop=None,
    )

    assert result is True
    assert scheduler.add_job.call_count == 1
    call = scheduler.add_job.call_args
    # First positional: the sync callback. Second positional / kwarg: trigger.
    # We assert the kwargs explicitly — the cycle callable is opaque.
    kwargs = call.kwargs
    assert kwargs["id"] == "meta_agent"
    assert kwargs["replace_existing"] is True
    assert kwargs["coalesce"] is True
    assert kwargs["max_instances"] == 1
    # Trigger may be passed as second positional or as 'trigger' kwarg.
    args = call.args
    trigger = kwargs.get("trigger") if "trigger" in kwargs else args[1]
    assert isinstance(trigger, IntervalTrigger)


def test_meta_agent_cycle_returns_when_async_loop_is_none() -> None:
    """The sync wrapper short-circuits silently when no async loop is wired."""
    from finalayze.meta_agent.scheduler import _make_cycle_callback

    runner = MagicMock()
    cb = _make_cycle_callback(runner=runner, async_loop=None)
    cb()  # no raise — that's the contract
    runner.run_one_tick.assert_not_called()


def test_meta_agent_cycle_dispatches_run_one_tick_to_async_loop(
    monkeypatch,
) -> None:
    """When given a loop, the sync wrapper schedules runner.run_one_tick()
    via asyncio.run_coroutine_threadsafe (mirrors _portfolio_review_cycle
    at trading_loop.py:1615)."""
    import asyncio

    from finalayze.meta_agent import scheduler as scheduler_module
    from finalayze.meta_agent.scheduler import _make_cycle_callback

    calls: list[tuple] = []

    def _fake_run_coro_ts(coro, loop):
        calls.append((coro, loop))
        # close the coroutine so we don't leak a never-awaited warning
        coro.close()
        return MagicMock()

    monkeypatch.setattr(asyncio, "run_coroutine_threadsafe", _fake_run_coro_ts)

    runner = MagicMock()

    async def _fake_tick():
        return None

    runner.run_one_tick = _fake_tick

    fake_loop = MagicMock()
    fake_loop.is_closed.return_value = False
    cb = _make_cycle_callback(runner=runner, async_loop=fake_loop)
    cb()

    assert len(calls) == 1
    assert calls[0][1] is fake_loop
    # Suppress unused-import lint if any
    assert scheduler_module is not None
