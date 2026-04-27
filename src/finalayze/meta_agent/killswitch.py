"""Killswitch for the meta-agent (Phase 58-05, META-08, SPEC AC #15).

SPEC §Requirement 8 mandates a single-action killswitch with two trigger
paths converging on the same primitive:

  1. ``POST /api/v1/meta-agent/disable`` — REST endpoint (Plan 58-05 Task 04).
  2. ``FINALAYZE_META_AGENT_ENABLED=false`` env-var flip — 1 s background
     poller (Plan 58-05 Task 03; RESEARCH §10.2).

Both call:
  a. ``abort_all_inflight()`` — iterates ``meta_agent.spawner._inflight_handles``
     and signals each subprocess via the SIGTERM → 3 s grace → SIGKILL
     sequence implemented in ``spawner._terminate_process_group``.
  b. ``remove_job()`` — removes the ``meta_agent`` APScheduler job.

Total wall-clock budget: ≤ 5 s (SPEC line 75). The ``_terminate_process_group``
helper has a 3 s grace + 1 s SIGKILL reap = 4 s nominal, comfortably
under the SPEC ceiling. Multiple spawns are aborted in parallel via
``asyncio.gather`` so wall-clock is bounded by the slowest (NOT the sum).

Layer: 6. Imports:
  - ``meta_agent.spawner`` (L6) — for ``_inflight_handles`` registry.
  - ``apscheduler.jobstores.base`` (third-party) — for ``JobLookupError``.
  - ``config.settings.get_settings`` (L1) — for the env-var poller.

Concurrency: poller task stored on the instance (``self._poller_task``)
to satisfy RUF006. Lifecycle owned by ``TradingLoop.start()/stop()``
(Plan 58-05 Task 06).
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import structlog

from finalayze.meta_agent import spawner as _spawner

if TYPE_CHECKING:
    from collections.abc import Callable

    from apscheduler.schedulers.background import BackgroundScheduler

    from config.settings import Settings

_log = structlog.get_logger()

# ── Module-level constants (PLR2004; magic-number-free in source) ────────────
_KILLSWITCH_CEILING_S = 5.0  # SPEC line 75 — wall-clock ceiling for abort
_DEFAULT_SIGTERM_GRACE_S = 3.0  # SPEC §Requirement 8 — 3 s SIGTERM grace
_DEFAULT_SIGKILL_REAP_S = 1.0  # bounded SIGKILL reap (kernel guarantees this)
_POLL_INTERVAL_S = 1.0  # RESEARCH §10.2 — 1 s env-var poll cadence


class Killswitch:
    """Two-trigger meta-agent killswitch.

    Constructor injects:
      - ``scheduler``: the ``BackgroundScheduler`` whose ``meta_agent`` job
        we will remove on disable.
      - ``settings_provider``: a callable returning the live ``Settings``
        object (used by the env-var poller to detect transitions).
      - ``sigterm_grace_s`` / ``sigkill_reap_s``: optional overrides for
        tests that want to keep wall-clock under one second.

    Lifecycle:
      - ``start()`` launches the env-var poller task (Plan 58-05 Task 03).
      - ``stop()`` cancels the poller cleanly.
      - ``abort_all_inflight()`` is the abort primitive used by both REST
        and env-var paths.
      - ``remove_job()`` is idempotent on missing job.
    """

    def __init__(
        self,
        *,
        scheduler: BackgroundScheduler | Any,
        settings_provider: Callable[[], Settings | Any],
        sigterm_grace_s: float = _DEFAULT_SIGTERM_GRACE_S,
        sigkill_reap_s: float = _DEFAULT_SIGKILL_REAP_S,
    ) -> None:
        self._scheduler = scheduler
        self._settings_provider = settings_provider
        self._sigterm_grace_s = sigterm_grace_s
        self._sigkill_reap_s = sigkill_reap_s
        # RUF006: poller handle stored on the instance.
        self._poller_task: asyncio.Task[None] | None = None
        self._stopping: asyncio.Event = asyncio.Event()

    # ── Abort primitive ──────────────────────────────────────────────────

    async def abort_all_inflight(self) -> int:
        """Signal every entry in ``spawner._inflight_handles`` via SIGTERM
        → 3 s grace → SIGKILL. Returns the number of handles aborted.

        Spawns are aborted in parallel (``asyncio.gather``) so wall-clock
        is bounded by the slowest abort (≤ ``sigterm_grace_s + sigkill_reap_s``)
        — NOT the sum across all spawns.

        Never raises. Each per-spawn failure is logged but does not abort
        the loop; the operator sees a partial-abort count and a structlog
        warning.
        """
        # Snapshot the registry so concurrent mutations during the gather
        # do not affect this iteration. ``_inflight_handles`` is a dict
        # keyed by decision_id; we copy the values (the Process handles).
        snapshot = list(_spawner._inflight_handles.items())
        if not snapshot:
            _log.info("meta_agent_killswitch_abort_no_inflight")
            return 0

        _log.warning(
            "meta_agent_killswitch_abort_started",
            inflight_count=len(snapshot),
        )

        coros = [
            _spawner._terminate_process_group(
                proc,
                grace_s=self._sigterm_grace_s,
                kill_s=self._sigkill_reap_s,
            )
            for _decision_id, proc in snapshot
        ]
        results = await asyncio.gather(*coros, return_exceptions=True)

        successes = sum(1 for r in results if not isinstance(r, BaseException))
        failures = len(results) - successes
        if failures > 0:
            _log.warning(
                "meta_agent_killswitch_abort_partial",
                successes=successes,
                failures=failures,
            )
        _log.warning(
            "meta_agent_killswitch_abort_completed",
            aborted=successes,
        )
        return successes

    # ── Scheduler hook ───────────────────────────────────────────────────

    def remove_job(self) -> bool:
        """Remove the ``meta_agent`` APScheduler job. Idempotent.

        Returns True when the job was present and removed; False when
        the job was not registered (``JobLookupError`` from APScheduler).
        Other exceptions are logged but suppressed — the killswitch path
        must never raise.
        """
        from apscheduler.jobstores.base import JobLookupError  # noqa: PLC0415

        try:
            self._scheduler.remove_job("meta_agent")
        except JobLookupError:
            _log.info("meta_agent_killswitch_job_not_registered")
            return False
        except Exception:
            _log.warning("meta_agent_killswitch_remove_job_failed", exc_info=True)
            return False
        _log.warning("meta_agent_killswitch_job_removed")
        return True

    # ── Env-var poller (Task 58-05-03) — placeholder until Task 03 ───────

    async def start(self) -> None:
        """Launch the env-var poller task. Implementation lands in Task 03."""

    async def stop(self) -> None:
        """Cancel the env-var poller task cleanly. Implementation lands in Task 03."""
