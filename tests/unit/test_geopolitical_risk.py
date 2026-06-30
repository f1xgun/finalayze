"""Tests for the geopolitical-risk overlay core (forward-only risk-awareness signal).

The overlay is an ADVISORY signal (it informs; it does NOT auto-trade) built on
the live news/sentiment pipeline. It cannot be backtested (no historical
point-in-time sentiment panel), so it carries an explicit disclaimer and the pure
mapping logic is the only thing that is unit-tested here.
"""

from __future__ import annotations

from decimal import Decimal

from finalayze.analysis.geopolitical_risk import (
    GeoRiskInputs,
    GeoRiskLevel,
    assess_geopolitical_risk,
)

_ZERO = Decimal(0)


def test_calm_market_is_normal_no_trim() -> None:
    a = assess_geopolitical_risk(GeoRiskInputs(mean_sentiment=0.2, article_volume=5))
    assert a.level is GeoRiskLevel.NORMAL
    assert a.recommended_equity_trim_pct == _ZERO
    assert a.recommended_fx_hedge_pct == _ZERO  # no ZO rotation when calm
    assert a.disclaimer  # always present


def test_severe_negative_sentiment_with_sanctions_is_high() -> None:
    a = assess_geopolitical_risk(
        GeoRiskInputs(
            mean_sentiment=-0.8,
            article_volume=120,
            sanctions_event_count=5,
            geopolitical_event_count=4,
        )
    )
    assert a.level is GeoRiskLevel.HIGH
    assert a.recommended_equity_trim_pct == Decimal("0.50")
    assert a.recommended_fx_hedge_pct == Decimal("0.03")  # full PROBATION toe-hold
    assert any("ZO" in d for d in a.drivers)  # surfaces the FX-hedge rotation
    assert a.drivers  # explains why


def test_moderate_stress_is_elevated() -> None:
    a = assess_geopolitical_risk(
        GeoRiskInputs(mean_sentiment=-0.45, article_volume=40, sanctions_event_count=1)
    )
    assert a.level is GeoRiskLevel.ELEVATED
    assert a.recommended_equity_trim_pct == Decimal("0.25")
    assert a.recommended_fx_hedge_pct == Decimal("0.015")  # half toe-hold at elevated


def test_score_is_monotonic_in_badness() -> None:
    mild = assess_geopolitical_risk(GeoRiskInputs(mean_sentiment=-0.1, article_volume=10))
    worse = assess_geopolitical_risk(
        GeoRiskInputs(mean_sentiment=-0.6, article_volume=80, sanctions_event_count=3)
    )
    assert worse.score >= mild.score


def test_positive_sentiment_volume_does_not_raise_risk() -> None:
    # high news volume but POSITIVE sentiment must not flag risk (volume only
    # counts as a risk amplifier when sentiment is negative)
    a = assess_geopolitical_risk(GeoRiskInputs(mean_sentiment=0.5, article_volume=200))
    assert a.level is GeoRiskLevel.NORMAL


def test_disclaimer_states_forward_only_and_advisory() -> None:
    a = assess_geopolitical_risk(GeoRiskInputs(mean_sentiment=-0.9, article_volume=100))
    low = a.disclaimer.lower()
    assert "backtest" in low  # explicitly says not backtested
    assert "advisor" in low or "inform" in low  # advisory, not auto-trading
