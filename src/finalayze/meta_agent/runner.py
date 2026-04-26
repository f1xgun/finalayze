"""MetaAgentRunner — orchestrates one cycle of the meta-agent (Phase 58-01).

Responsibilities:
  1. Build a Snapshot via REST self-call (D-01).
  2. Classify deterministically (SPEC §Requirement 2).
  3. Persist a MetaAgentDecisionModel row via the fire-and-forget envelope.
  4. Dry-run gate: short-circuit BEFORE any executor invocation when
     ``settings.meta_agent_dry_run`` is True (SPEC §Requirement 4).
  5. Expose ``status_snapshot()`` for the GET /api/v1/meta-agent/status
     endpoint (Task 58-01-11).

Executor wiring lands in Plan 58-02; this plan accepts ``executor=None``
default and never invokes it (the dry-run path is the only path here).

RUF006: any ``asyncio.create_task`` call stores the handle on
``self._tasks`` set; current implementation does not spawn tasks but the
attribute is reserved for plan 58-02.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import httpx
import structlog

from finalayze.meta_agent.classifier import Severity, classify
from finalayze.meta_agent.snapshot import build_snapshot

if TYPE_CHECKING:
    import asyncio
    from collections.abc import Callable

    from config.settings import Settings

    from finalayze.meta_agent.snapshot import Snapshot

_log = structlog.get_logger()

# Default REST self-call base URL (D-01). Single-host deployment — can be
# overridden via the http_client_factory.
_DEFAULT_BASE_URL = "http://127.0.0.1:8000"
_DEFAULT_TIMEOUT_SECONDS = 10.0


def _default_http_client_factory(*, base_url: str, api_key: str) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        base_url=base_url,
        headers={"X-API-Key": api_key},
        timeout=_DEFAULT_TIMEOUT_SECONDS,
    )


class MetaAgentRunner:
    """One-tick orchestrator. Owned by ``meta_agent.scheduler``.

    Constructor injects ``persistence`` (TradingPersistence-shaped) and
    optional ``executor`` (none in 58-01). The ``http_client_factory``
    defaults to the REST self-call shape but tests inject a fake.
    """

    def __init__(
        self,
        *,
        settings: Settings,
        persistence: Any,
        executor: Any = None,
        http_client_factory: Callable[[], httpx.AsyncClient] | None = None,
        base_url: str = _DEFAULT_BASE_URL,
    ) -> None:
        self._settings = settings
        self._persistence = persistence
        self._executor = executor
        self._base_url = base_url
        self._client_factory = http_client_factory
        self._last_run_ts: datetime | None = None
        # RUF006 reservation — any future asyncio.create_task() call stores
        # its handle here so the lifetime is tied to this instance.
        self._tasks: set[asyncio.Task[Any]] = set()

    # ── public ────────────────────────────────────────────────────────────

    async def run_one_tick(self) -> None:
        """Execute one cycle. NEVER raises — failures are logged and the
        runner returns so the scheduler tick is non-fatal.
        """
        tick_start = datetime.now(UTC)
        self._last_run_ts = tick_start
        _log.info(
            "meta_agent_tick_started",
            dry_run=self._settings.meta_agent_dry_run,
        )

        try:
            snapshot = await self._collect_snapshot(now=tick_start)
        except Exception:
            _log.warning("meta_agent_snapshot_failed", exc_info=True)
            return

        try:
            severity = classify(snapshot)
        except Exception:
            _log.warning("meta_agent_classify_failed", exc_info=True)
            return

        rationale = self._derive_rationale(snapshot, severity)
        summary = f"meta_agent severity={severity.value}"
        decision_id = uuid.uuid4()

        _log.info(
            "meta_agent_classify_completed",
            severity_key=severity.value,
            decision_id=str(decision_id),
            dry_run=self._settings.meta_agent_dry_run,
        )

        # Fire-and-forget persist (PERSIST-05 envelope on the persistence side).
        try:
            self._persistence.persist_decision(
                decision_id=decision_id,
                timestamp=tick_start,
                severity=severity.value,
                summary=summary,
                rationale=rationale,
                actions=[],
                dry_run=self._settings.meta_agent_dry_run,
                status="queued",
                decision_metadata=None,
                parent_decision_id=None,
            )
        except Exception:
            _log.warning("meta_agent_persist_failed", exc_info=True)

        # Dry-run gate — first decision after persist (D-04 / SPEC line 47).
        if self._settings.meta_agent_dry_run:
            return

        # Non-dry-run path lands in Plan 58-02 via the executor.
        if self._executor is None:
            _log.info("meta_agent_executor_missing", severity_key=severity.value)
            return
        try:
            await self._executor.execute(
                decision_id=decision_id,
                severity=severity,
                snapshot=snapshot,
            )
        except Exception:
            _log.warning(
                "meta_agent_executor_failed",
                severity_key=severity.value,
                decision_id=str(decision_id),
                exc_info=True,
            )

    def status_snapshot(self) -> dict[str, Any]:
        """Return the data shape consumed by GET /api/v1/meta-agent/status.

        Task 58-01-11 wraps this in ``MetaAgentStatus``. The
        ``inflight_spawns`` registry lands in Plan 58-05; for now we
        report zeroes.
        """
        return {
            "enabled": self._settings.meta_agent_enabled,
            "dry_run": self._settings.meta_agent_dry_run,
            "last_run_ts": self._last_run_ts,
            "scheduler_active": self._settings.meta_agent_enabled,
            "inflight_spawns": {"investigate": 0, "fix": 0},
        }

    # ── helpers ───────────────────────────────────────────────────────────

    async def _collect_snapshot(self, *, now: datetime) -> Snapshot:
        if self._client_factory is not None:
            client = self._client_factory()
            try:
                return await build_snapshot(client, now=now)
            finally:
                aclose = getattr(client, "aclose", None)
                if aclose is not None:
                    await aclose()
        # Default REST self-call.
        async with _default_http_client_factory(
            base_url=self._base_url,
            api_key=self._settings.api_key,
        ) as client:
            return await build_snapshot(client, now=now)

    @staticmethod
    def _derive_rationale(snapshot: Snapshot, severity: Severity) -> str:
        """One-line rationale stub. The full LLM-generated text lands in
        plan 58-02 / 58-03; this keeps the column NOT NULL constraint
        satisfied during dry-run."""
        if (
            snapshot.alerts_last_hour is None
            and snapshot.drawdown_pct is None
            and snapshot.positions_summary is None
        ):
            return "snapshot_unusable"
        return f"rule-derived severity={severity.value}"
