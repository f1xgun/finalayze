"""APScheduler glue for the meta-agent (Phase 58-01, META-04 wiring).

Single public entry point: ``register_meta_agent_job(scheduler, settings,
runner, async_loop) -> bool``. Returns True when a job was added; False
when ``settings.meta_agent_enabled`` is False (SPEC §Acceptance Criterion #6).

The cron callback is a sync wrapper that defers to
``asyncio.run_coroutine_threadsafe`` because ``BackgroundScheduler`` does
not await coroutines. Direct analog: ``_portfolio_review_cycle`` at
``orchestration/trading_loop.py:1615``. Misfire grace 60s and
``coalesce=True`` mirror the Phase 57 portfolio review job kwargs.

This module does NOT modify ``trading_loop.py`` — wiring into the live
scheduler lands as a small follow-up task in 58-02 along with the
executor injection point. Existing ``tests/unit/test_trading_loop.py``
fixtures stay green by default (``meta_agent_enabled=False``).
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Callable

import structlog
from apscheduler.triggers.interval import IntervalTrigger

if TYPE_CHECKING:
    from apscheduler.schedulers.background import BackgroundScheduler

    from config.settings import Settings
    from finalayze.meta_agent.runner import MetaAgentRunner

_log = structlog.get_logger()

# Misfire grace — 60 s gives APScheduler room to recover from drift on the
# 30-min cadence without firing duplicates. RESEARCH §4.2 justification.
_MISFIRE_GRACE_SECONDS = 60


def _make_cycle_callback(
    *,
    runner: MetaAgentRunner | Any,
    async_loop: asyncio.AbstractEventLoop | None,
) -> Callable[[], None]:
    """Return a sync callable suitable for ``BackgroundScheduler.add_job``.

    Mirrors the ``_portfolio_review_cycle`` pattern at
    ``trading_loop.py:1615``: schedule the async tick onto the application's
    event loop without blocking the scheduler thread.
    """

    def _cycle() -> None:
        if async_loop is None or async_loop.is_closed():
            _log.info("meta_agent_skipped", reason="no async loop")
            return
        coro = runner.run_one_tick()
        # Fire-and-forget: do NOT call .result() — that would block the
        # scheduler thread for up to 30 minutes.
        asyncio.run_coroutine_threadsafe(coro, async_loop)

    return _cycle


def register_meta_agent_job(
    scheduler: BackgroundScheduler | Any,
    *,
    settings: Settings | Any,
    runner: MetaAgentRunner | Any,
    async_loop: asyncio.AbstractEventLoop | None,
) -> bool:
    """Register the meta-agent cron job on the supplied scheduler.

    Returns True when a job was added, False when not enabled (SPEC §AC #6).
    """
    if not getattr(settings, "meta_agent_enabled", False):
        _log.info("meta_agent_job_skipped", reason="disabled")
        return False

    interval_minutes = settings.meta_agent_interval_minutes
    callback = _make_cycle_callback(runner=runner, async_loop=async_loop)

    scheduler.add_job(
        callback,
        IntervalTrigger(minutes=interval_minutes),
        id="meta_agent",
        replace_existing=True,
        coalesce=True,
        max_instances=1,
        misfire_grace_time=_MISFIRE_GRACE_SECONDS,
    )
    _log.info(
        "meta_agent_scheduled",
        interval_minutes=interval_minutes,
        dry_run=settings.meta_agent_dry_run,
    )
    return True
