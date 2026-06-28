"""Tests for the live geopolitical-risk monitor aggregation (no DB — fake reader)."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from finalayze.analysis.geopolitical_risk import GeoRiskLevel
from finalayze.data.sentiment_store import SentimentRow
from finalayze.orchestration.geo_risk_monitor import aggregate_inputs, assess_live


class _FakeReader:
    """Returns canned rolling rows per ticker."""

    def __init__(self, by_ticker: dict[str, list[SentimentRow]]) -> None:
        self._by_ticker = by_ticker

    async def get_rolling(
        self, ticker: str, *, window: str = "7d", market_id: str = "moex"
    ) -> list[SentimentRow]:
        return self._by_ticker.get(ticker, [])


def _row(score: float | None, count: int) -> SentimentRow:
    return SentimentRow(
        bucket=datetime(2026, 6, 28, tzinfo=UTC), avg_score=score, article_count=count
    )


@pytest.mark.asyncio
async def test_aggregate_is_article_count_weighted() -> None:
    reader = _FakeReader(
        {
            "SBER": [_row(-1.0, 90)],  # heavy bearish, high volume
            "GAZP": [_row(0.0, 10)],  # neutral, low volume
        }
    )
    inputs = await aggregate_inputs(reader, ["SBER", "GAZP"])
    # weighted mean = (-1*90 + 0*10)/100 = -0.9
    assert inputs.mean_sentiment == pytest.approx(-0.9)
    assert inputs.article_volume == 100


@pytest.mark.asyncio
async def test_assess_live_flags_high_on_bearish_market() -> None:
    reader = _FakeReader({"SBER": [_row(-0.85, 150)]})
    a = await assess_live(reader, ["SBER"], sanctions_event_count=4)
    assert a.level is GeoRiskLevel.HIGH


@pytest.mark.asyncio
async def test_empty_store_is_normal_no_data() -> None:
    a = await assess_live(_FakeReader({}), ["SBER", "GAZP"])
    assert a.level is GeoRiskLevel.NORMAL
    assert a.recommended_equity_trim_pct == 0


@pytest.mark.asyncio
async def test_none_scores_ignored_but_count_volume() -> None:
    reader = _FakeReader({"SBER": [_row(None, 30), _row(-0.5, 20)]})
    inputs = await aggregate_inputs(reader, ["SBER"])
    # only the scored bucket feeds the mean; both feed volume
    assert inputs.mean_sentiment == pytest.approx(-0.5)
    assert inputs.article_volume == 50
