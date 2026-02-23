"""Unit tests for ImpactEstimator."""

from __future__ import annotations

import pytest

from finalayze.analysis.event_classifier import EventType
from finalayze.analysis.impact_estimator import ImpactEstimator
from finalayze.core.schemas import SentimentResult

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


class TestImpactEstimatorGlobalScope:
    def test_global_event_affects_all_segments(self) -> None:
        estimator = ImpactEstimator()
        impacts = estimator.estimate(
            scope="global",
            event=EventType.MACRO,
            sentiment=_SENTIMENT,
            active_segments=_ALL_SEGMENTS,
        )
        segment_ids = {i.segment_id for i in impacts}
        assert segment_ids == set(_ALL_SEGMENTS)

    def test_global_event_all_weights_are_primary(self) -> None:
        estimator = ImpactEstimator()
        impacts = estimator.estimate(
            scope="global",
            event=EventType.MACRO,
            sentiment=_SENTIMENT,
            active_segments=_ALL_SEGMENTS,
        )
        assert all(i.weight == _PRIMARY_WEIGHT for i in impacts)


class TestImpactEstimatorUsScope:
    def test_us_event_affects_only_us_segments(self) -> None:
        estimator = ImpactEstimator()
        impacts = estimator.estimate(
            scope="us",
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
        impacts = estimator.estimate(
            scope="russia",
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
        impacts = estimator.estimate(
            scope="sector",
            event=EventType.OIL_PRICE,
            sentiment=_SENTIMENT,
            active_segments=_ALL_SEGMENTS,
        )
        ru_energy = next((i for i in impacts if i.segment_id == "ru_energy"), None)
        assert ru_energy is not None
        assert ru_energy.weight == _PRIMARY_WEIGHT

    def test_fda_affects_us_healthcare_primary(self) -> None:
        estimator = ImpactEstimator()
        impacts = estimator.estimate(
            scope="sector",
            event=EventType.FDA,
            sentiment=_SENTIMENT,
            active_segments=_ALL_SEGMENTS,
        )
        healthcare = next((i for i in impacts if i.segment_id == "us_healthcare"), None)
        assert healthcare is not None
        assert healthcare.weight == _PRIMARY_WEIGHT

    def test_cbr_rate_affects_ru_finance_primary(self) -> None:
        estimator = ImpactEstimator()
        impacts = estimator.estimate(
            scope="sector",
            event=EventType.CBR_RATE,
            sentiment=_SENTIMENT,
            active_segments=_ALL_SEGMENTS,
        )
        finance = next((i for i in impacts if i.segment_id == "ru_finance"), None)
        assert finance is not None
        assert finance.weight == _PRIMARY_WEIGHT

    def test_sentiment_propagated_to_impacts(self) -> None:
        estimator = ImpactEstimator()
        impacts = estimator.estimate(
            scope="global",
            event=EventType.MACRO,
            sentiment=_SENTIMENT,
            active_segments=["us_tech"],
        )
        assert impacts[0].sentiment == pytest.approx(0.8)
