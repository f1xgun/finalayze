"""Integration test for /portfolio/positions/{symbol}/stop-history endpoint.

Seeds two stop_loss_events rows and verifies they come back via the API
ordered by timestamp ascending (STOP-03 contract).

Uses ``asyncio.run()`` for the one-shot seed coroutine (I-04 resolution):
the deprecated ``get_event_loop`` + ``run_until_complete`` pairing is avoided
because Python 3.12 deprecates that combination and this project's pytest
configuration may escalate DeprecationWarnings to errors.
"""

from __future__ import annotations

import asyncio
import os

import pytest

pytestmark = pytest.mark.integration


def test_stop_history_returns_events_ordered_by_timestamp() -> None:
    """Seed two rows at different timestamps; endpoint returns them ascending."""
    if not os.environ.get("FINALAYZE_DATABASE_URL"):
        pytest.skip("FINALAYZE_DATABASE_URL not set")

    from datetime import UTC, datetime, timedelta  # noqa: PLC0415
    from decimal import Decimal  # noqa: PLC0415

    from config.settings import Settings  # noqa: PLC0415
    from fastapi.testclient import TestClient  # noqa: PLC0415

    from finalayze.core.db import get_async_session_factory  # noqa: PLC0415
    from finalayze.core.models import StopLossEventModel  # noqa: PLC0415
    from finalayze.main import create_app  # noqa: PLC0415

    async def _seed() -> None:
        factory = get_async_session_factory()
        async with factory() as session:
            base = datetime.now(UTC) - timedelta(hours=2)
            for i, et in enumerate(("entry", "snapshot")):
                session.add(
                    StopLossEventModel(
                        timestamp=base + timedelta(minutes=i * 15),
                        symbol="TEST_SL_HIST",
                        market_id="moex",
                        event_type=et,
                        entry_price=Decimal(100),
                        current_stop=Decimal(95) + Decimal(str(i)),
                        highest_price=Decimal(100) + Decimal(str(i)),
                        atr_value=Decimal("2.5"),
                        activation_atr=Decimal("1.0"),
                        trail_atr=Decimal("1.5"),
                        trail_activated=False,
                        current_price=Decimal(100) + Decimal(str(i)),
                    )
                )
            await session.commit()

    # I-04: asyncio.run() is used here instead of the deprecated
    # get_event_loop + run_until_complete combination. asyncio.run() creates
    # and closes a fresh loop, which is safe in Python 3.12 even when no
    # outer loop exists (the deprecated form warns in 3.12 and the project
    # pytest config may escalate DeprecationWarnings to errors).
    asyncio.run(_seed())

    client = TestClient(create_app())
    resp = client.get(
        "/api/v1/portfolio/positions/TEST_SL_HIST/stop-history?days=1",
        headers={"X-API-Key": Settings().api_key},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["symbol"] == "TEST_SL_HIST"
    events = body["events"]
    assert len(events) == 2
    # Ordered by timestamp ascending
    assert events[0]["event_type"] == "entry"
    assert events[1]["event_type"] == "snapshot"
    assert events[0]["timestamp"] < events[1]["timestamp"]
