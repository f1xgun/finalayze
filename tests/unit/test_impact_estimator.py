"""Unit tests for ImpactEstimator."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from finalayze.analysis.event_classifier import EventType
from finalayze.analysis.impact_estimator import ImpactEstimator
from finalayze.core.schemas import NewsArticle, SentimentResult

_SENTIMENT = SentimentResult(sentiment=0.8, confidence=0.9, reasoning="positive")
_ALL_SEGMENTS = [
    "us_tech",
    "us_healthcare",
    "us_finance",
    "us_broad",
    "ru_blue_chips",
    "ru_energy",
    "ru_tech",
    "ru_finance",
]

_PRIMARY_WEIGHT = 1.0
_SECONDARY_WEIGHT = 0.3
_OIL_SECONDARY_WEIGHT = 0.5
_MACRO_WEIGHT = 0.5
_GEOPOLITICAL_WEIGHT = 0.3

_PUB_AT = datetime(2024, 1, 1, tzinfo=UTC)


def _make_article(scope: str | None = "global") -> NewsArticle:
    return NewsArticle(
        id=uuid4(),
        source="test",
        title="Test Article",
        content="Test content",
        url="https://example.com",
        language="en",
        published_at=_PUB_AT,
        scope=scope,
    )


class TestImpactEstimatorGlobalScope:
    def test_global_event_affects_all_segments(self) -> None:
        estimator = ImpactEstimator()
        article = _make_article(scope="global")
        impacts = estimator.estimate(
            article=article,
            event=EventType.MACRO,
            sentiment=_SENTIMENT,
            active_segments=_ALL_SEGMENTS,
        )
        segment_ids = {i.segment_id for i in impacts}
        assert segment_ids == set(_ALL_SEGMENTS)

    def test_global_event_all_weights_are_primary(self) -> None:
        estimator = ImpactEstimator()
        article = _make_article(scope="global")
        impacts = estimator.estimate(
            article=article,
            event=EventType.MACRO,
            sentiment=_SENTIMENT,
            active_segments=_ALL_SEGMENTS,
        )
        assert all(i.weight == _PRIMARY_WEIGHT for i in impacts)


class TestImpactEstimatorUsScope:
    def test_us_event_affects_only_us_segments(self) -> None:
        estimator = ImpactEstimator()
        article = _make_article(scope="us")
        impacts = estimator.estimate(
            article=article,
            event=EventType.EARNINGS,
            sentiment=_SENTIMENT,
            active_segments=_ALL_SEGMENTS,
        )
        segment_ids = {i.segment_id for i in impacts}
        assert all(s.startswith("us_") for s in segment_ids)
        assert not any(s.startswith("ru_") for s in segment_ids)


class TestImpactEstimatorRussiaScope:
    def test_russia_event_affects_only_ru_segments(self) -> None:
        estimator = ImpactEstimator()
        article = _make_article(scope="russia")
        impacts = estimator.estimate(
            article=article,
            event=EventType.CBR_RATE,
            sentiment=_SENTIMENT,
            active_segments=_ALL_SEGMENTS,
        )
        segment_ids = {i.segment_id for i in impacts}
        assert all(s.startswith("ru_") for s in segment_ids)
        assert not any(s.startswith("us_") for s in segment_ids)


class TestImpactEstimatorEventRouting:
    def test_oil_price_affects_ru_energy_primary(self) -> None:
        estimator = ImpactEstimator()
        article = _make_article(scope="sector")
        impacts = estimator.estimate(
            article=article,
            event=EventType.OIL_PRICE,
            sentiment=_SENTIMENT,
            active_segments=_ALL_SEGMENTS,
        )
        ru_energy = next((i for i in impacts if i.segment_id == "ru_energy"), None)
        assert ru_energy is not None
        assert ru_energy.weight == _PRIMARY_WEIGHT

    def test_oil_price_ru_blue_chips_secondary_weight_is_0_5(self) -> None:
        estimator = ImpactEstimator()
        article = _make_article(scope="sector")
        impacts = estimator.estimate(
            article=article,
            event=EventType.OIL_PRICE,
            sentiment=_SENTIMENT,
            active_segments=_ALL_SEGMENTS,
        )
        ru_blue_chips = next((i for i in impacts if i.segment_id == "ru_blue_chips"), None)
        assert ru_blue_chips is not None
        assert ru_blue_chips.weight == pytest.approx(_OIL_SECONDARY_WEIGHT)

    def test_cbr_rate_ru_blue_chips_secondary_weight_is_0_3(self) -> None:
        estimator = ImpactEstimator()
        article = _make_article(scope="sector")
        impacts = estimator.estimate(
            article=article,
            event=EventType.CBR_RATE,
            sentiment=_SENTIMENT,
            active_segments=_ALL_SEGMENTS,
        )
        ru_blue_chips = next((i for i in impacts if i.segment_id == "ru_blue_chips"), None)
        assert ru_blue_chips is not None
        assert ru_blue_chips.weight == pytest.approx(_SECONDARY_WEIGHT)

    def test_fda_affects_us_healthcare_primary(self) -> None:
        estimator = ImpactEstimator()
        article = _make_article(scope="sector")
        impacts = estimator.estimate(
            article=article,
            event=EventType.FDA,
            sentiment=_SENTIMENT,
            active_segments=_ALL_SEGMENTS,
        )
        healthcare = next((i for i in impacts if i.segment_id == "us_healthcare"), None)
        assert healthcare is not None
        assert healthcare.weight == _PRIMARY_WEIGHT

    def test_cbr_rate_affects_ru_finance_primary(self) -> None:
        estimator = ImpactEstimator()
        article = _make_article(scope="sector")
        impacts = estimator.estimate(
            article=article,
            event=EventType.CBR_RATE,
            sentiment=_SENTIMENT,
            active_segments=_ALL_SEGMENTS,
        )
        finance = next((i for i in impacts if i.segment_id == "ru_finance"), None)
        assert finance is not None
        assert finance.weight == _PRIMARY_WEIGHT

    def test_macro_affects_all_segments_at_half_weight(self) -> None:
        estimator = ImpactEstimator()
        article = _make_article(scope="sector")
        impacts = estimator.estimate(
            article=article,
            event=EventType.MACRO,
            sentiment=_SENTIMENT,
            active_segments=_ALL_SEGMENTS,
        )
        segment_ids = {i.segment_id for i in impacts}
        assert segment_ids == set(_ALL_SEGMENTS)
        assert all(i.weight == pytest.approx(_MACRO_WEIGHT) for i in impacts)

    def test_geopolitical_affects_all_segments_at_0_3_weight(self) -> None:
        estimator = ImpactEstimator()
        article = _make_article(scope="sector")
        impacts = estimator.estimate(
            article=article,
            event=EventType.GEOPOLITICAL,
            sentiment=_SENTIMENT,
            active_segments=_ALL_SEGMENTS,
        )
        segment_ids = {i.segment_id for i in impacts}
        assert segment_ids == set(_ALL_SEGMENTS)
        assert all(i.weight == pytest.approx(_GEOPOLITICAL_WEIGHT) for i in impacts)

    def test_sentiment_propagated_to_impacts(self) -> None:
        estimator = ImpactEstimator()
        article = _make_article(scope="global")
        impacts = estimator.estimate(
            article=article,
            event=EventType.MACRO,
            sentiment=_SENTIMENT,
            active_segments=["us_tech"],
        )
        assert impacts[0].sentiment == pytest.approx(0.8)

    def test_none_scope_returns_empty(self) -> None:
        estimator = ImpactEstimator()
        article = _make_article(scope=None)
        impacts = estimator.estimate(
            article=article,
            event=EventType.MACRO,
            sentiment=_SENTIMENT,
            active_segments=_ALL_SEGMENTS,
        )
        assert impacts == []

    def test_segment_impact_is_pydantic_model(self) -> None:
        estimator = ImpactEstimator()
        article = _make_article(scope="global")
        impacts = estimator.estimate(
            article=article,
            event=EventType.MACRO,
            sentiment=_SENTIMENT,
            active_segments=["us_tech"],
        )
        assert len(impacts) == 1
        # Pydantic models have model_dump
        assert hasattr(impacts[0], "model_dump")
        assert isinstance(impacts[0].model_dump(), dict)
