"""get_position now reuses the list endpoint + filters (audit 2026-06-28).

Previously GET /portfolio/positions/{symbol} was a permanent stub that always
404'd. It now returns the matching PositionDetail from the (well-tested) list
endpoint, or 404 when the symbol is not open.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

from finalayze.api.v1 import portfolio as pmod


async def test_get_position_returns_matching_symbol(monkeypatch) -> None:
    wanted = MagicMock()
    wanted.symbol = "SBER"
    other = MagicMock()
    other.symbol = "GAZP"

    async def _fake_get_positions(_req: object) -> object:
        return SimpleNamespace(positions=[other, wanted])

    monkeypatch.setattr(pmod, "get_positions", _fake_get_positions)

    result = await pmod.get_position("SBER", MagicMock())
    assert result is wanted


async def test_get_position_404_when_symbol_absent(monkeypatch) -> None:
    async def _fake_get_positions(_req: object) -> object:
        return SimpleNamespace(positions=[])

    monkeypatch.setattr(pmod, "get_positions", _fake_get_positions)

    with pytest.raises(HTTPException) as exc:
        await pmod.get_position("MISSING", MagicMock())
    assert exc.value.status_code == 404
