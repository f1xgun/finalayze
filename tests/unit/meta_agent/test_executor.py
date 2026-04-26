"""Tests for meta_agent.executor.ActionExecutor (Phase 58-02, META-05).

Covers SPEC §Acceptance Criteria #8 + #9:
  - Persist envelope helpers (persist_decision, update_decision_status).
  - Dry-run short-circuit on the FIRST line of execute().
  - HEALTHY → no Telegram (severity-below-threshold gate).
  - WATCH/INVESTIGATE/FIX → send Telegram, stamp metadata, status='sent'.
  - Daily cap enforcement (UTC-day boundary).
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

# Module-level constants (PLR2004).
_FAKE_ALERT_UUID = uuid.UUID("f00dbeef-1234-4abc-8def-0123456789ab")
_FAKE_DECISION_ID = uuid.UUID("deadbeef-0000-4000-8000-000000000001")
_FAKE_TS = datetime(2026, 4, 26, 12, 0, tzinfo=UTC)
_FAKE_COUNT = 7
_CAP_HIGH = 100
_CAP_TWO = 2
_NUM_THIRD = 3


# ─────────────────────────────────────────────────────────────────────────────
# Task 58-02-01: persist_decision + update_decision_status fire-and-forget
# ─────────────────────────────────────────────────────────────────────────────


def test_persistence_helpers_use_fire_and_forget_envelope(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """SPEC AC #8 foundation: TradingPersistence.persist_decision and
    update_decision_status exist, never raise when _db_url is None, and
    log db_persist_skipped (PERSIST-05 envelope, mirrors persist_alert).
    """
    import logging

    from finalayze.orchestration.db_persistence import TradingPersistence

    persistence = TradingPersistence(db_url=None, async_loop=None)

    # When _db_url is None, both helpers must NOT raise.
    caplog.set_level(logging.DEBUG)
    persistence.persist_decision(
        decision_id=_FAKE_DECISION_ID,
        timestamp=_FAKE_TS,
        severity="HEALTHY",
        summary="s",
        rationale="r",
        actions=[],
        dry_run=True,
        decision_metadata=None,
        parent_decision_id=None,
        status="queued",
    )
    persistence.update_decision_status(
        decision_id=_FAKE_DECISION_ID,
        timestamp=_FAKE_TS,
        status="sent",
        outcome=None,
    )

    # Both should have logged db_persist_skipped via structlog.
    msgs = [rec.message for rec in caplog.records]
    skipped = [m for m in msgs if "db_persist_skipped" in m]
    assert len(skipped) >= 2, f"expected >=2 db_persist_skipped log lines, got {msgs!r}"
