"""Geopolitical-risk overlay core (Layer 3) — forward-only risk-awareness signal.

WHY THIS EXISTS (and what it is NOT): the active-equity experiment
(docs/research/active_equity_sleeve_experiment.md) confirmed the MOEX equity
drawdown was driven by geopolitical/sanctions shocks -- but the catastrophic core
(the 2022-02-24 -26% invasion gap + the 27-day trading halt) was structurally
UN-CATCHABLE, and a news overlay CANNOT be honestly backtested (no historical
point-in-time sentiment panel exists). So this is deliberately NOT a backtested
trading edge.

It is an ADVISORY risk-awareness signal: it maps the LIVE news/sentiment state
into a geopolitical-risk LEVEL and a *recommended* equity trim toward the
deposit/OFZ anchor. It INFORMS (alert + dashboard); it does NOT auto-trade. Real
money stays behind the operator hard-stop.

This module is the PURE mapping brain (testable, no I/O). The live aggregation
(reading the sentiment store across the active universe) and the alert/API
surfaces wire into it.

See docs/architecture/DEPENDENCY_LAYERS.md for layering rules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from enum import StrEnum

# ── Pre-registered bands & weights (transparent, NOT fitted to any backtest) ──
_ELEVATED_THRESHOLD = 0.33
_HIGH_THRESHOLD = 0.66
_SENTIMENT_WEIGHT = 0.5
_EVENT_WEIGHT = 0.4
_VOLUME_WEIGHT = 0.1
# A sanctions headline weighs 2x a generic geopolitical one; this many weighted
# events in the window saturates the event component.
_SANCTIONS_EVENT_WEIGHT = 2.0
_EVENT_SATURATION = 5.0
# Article count that counts as "elevated news intensity" (amplifier only when
# sentiment is already negative — a loud-but-positive tape is not risk).
_VOLUME_SATURATION = 50.0

_DISCLAIMER = (
    "Forward-only risk-awareness signal from the live news pipeline. NOT "
    "backtested and NOT a proven edge — no historical point-in-time sentiment "
    "data exists to validate it. Advisory only: it informs a recommended equity "
    "trim AND a small rotation into a replacement-bond (ZO) FX-linked toe-hold "
    "(the one structurally-sound ruble-devaluation hedge the instrument-integration "
    "program found — FX-linked + uncorrelated, but its crash payoff is unproven, so "
    "it is capped at the PROBATION toe-hold) rather than trimming only into ruble "
    "deposit/OFZ. It does NOT auto-trade. Real-money changes stay behind the "
    "operator hard-stop."
)

# The replacement-bond (ZO) FX-linked toe-hold to rotate INTO on stress — the one hedge that
# survived the instrument-integration gate (PROBATION). Capped at the gate's PROBATION toe-hold
# (3%); it is FX-linked + uncorrelated but its tail payoff is structurally argued, not measured.
_PROBATION_FX_CAP = Decimal("0.03")
_ELEVATED_FX_HEDGE = Decimal("0.015")  # half the toe-hold at elevated risk


class GeoRiskLevel(StrEnum):
    """Geopolitical-risk band driving the advisory equity trim."""

    NORMAL = "normal"
    ELEVATED = "elevated"
    HIGH = "high"


@dataclass(frozen=True)
class GeoRiskInputs:
    """Aggregated live-news inputs (computed across the active MOEX universe).

    ``mean_sentiment`` is the article-count-weighted average sentiment in [-1, 1]
    (negative = bearish). ``article_volume`` is the recent article count (news
    intensity). The event counts come from the LLM ``EventClassifier``
    (GEOPOLITICAL / SANCTIONS categories) over the same window.
    """

    mean_sentiment: float
    article_volume: int
    sanctions_event_count: int = 0
    geopolitical_event_count: int = 0


@dataclass(frozen=True)
class GeoRiskAssessment:
    """The advisory verdict: a level, a 0..1 score, the recommended trim + ZO rotation, and why."""

    level: GeoRiskLevel
    score: float
    recommended_equity_trim_pct: Decimal
    recommended_fx_hedge_pct: Decimal
    drivers: list[str] = field(default_factory=list)
    disclaimer: str = _DISCLAIMER


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _level_for_score(score: float) -> GeoRiskLevel:
    if score >= _HIGH_THRESHOLD:
        return GeoRiskLevel.HIGH
    if score >= _ELEVATED_THRESHOLD:
        return GeoRiskLevel.ELEVATED
    return GeoRiskLevel.NORMAL


def _trim_for_level(level: GeoRiskLevel) -> Decimal:
    return {
        GeoRiskLevel.NORMAL: Decimal(0),
        GeoRiskLevel.ELEVATED: Decimal("0.25"),
        GeoRiskLevel.HIGH: Decimal("0.50"),
    }[level]


def _fx_hedge_for_level(level: GeoRiskLevel) -> Decimal:
    """The ZO FX-linked toe-hold to rotate INTO at each risk level (capped at the PROBATION 3%)."""
    return {
        GeoRiskLevel.NORMAL: Decimal(0),
        GeoRiskLevel.ELEVATED: _ELEVATED_FX_HEDGE,
        GeoRiskLevel.HIGH: _PROBATION_FX_CAP,
    }[level]


def assess_geopolitical_risk(inputs: GeoRiskInputs) -> GeoRiskAssessment:
    """Map live-news inputs to an advisory geopolitical-risk assessment.

    Composite score in [0, 1] (higher = more risk):
      0.5 * negative-sentiment + 0.4 * sanctions/geo-event intensity
      + 0.1 * news-volume (volume counts ONLY when sentiment is negative).
    The bands and weights are pre-registered and transparent — this is a risk
    heuristic, not a fitted model.
    """
    sentiment_component = _clamp01(-inputs.mean_sentiment)  # only bearish adds risk
    weighted_events = (
        _SANCTIONS_EVENT_WEIGHT * inputs.sanctions_event_count + inputs.geopolitical_event_count
    )
    event_component = _clamp01(weighted_events / _EVENT_SATURATION)
    volume_component = (
        _clamp01(inputs.article_volume / _VOLUME_SATURATION) if inputs.mean_sentiment < 0 else 0.0
    )

    score = (
        _SENTIMENT_WEIGHT * sentiment_component
        + _EVENT_WEIGHT * event_component
        + _VOLUME_WEIGHT * volume_component
    )
    level = _level_for_score(score)

    drivers: list[str] = []
    if sentiment_component > 0:
        drivers.append(f"bearish market sentiment {inputs.mean_sentiment:+.2f}")
    if inputs.sanctions_event_count:
        drivers.append(f"{inputs.sanctions_event_count} sanctions event(s) in window")
    if inputs.geopolitical_event_count:
        drivers.append(f"{inputs.geopolitical_event_count} geopolitical event(s) in window")
    if volume_component > 0 and inputs.article_volume >= _VOLUME_SATURATION:
        drivers.append(f"elevated news volume ({inputs.article_volume} articles)")

    fx_hedge = _fx_hedge_for_level(level)
    if fx_hedge > 0:
        drivers.append(
            f"rotate up to {fx_hedge:.1%} into a ZO FX-linked hedge (vs trimming only to RUB)"
        )

    return GeoRiskAssessment(
        level=level,
        score=score,
        recommended_equity_trim_pct=_trim_for_level(level),
        recommended_fx_hedge_pct=fx_hedge,
        drivers=drivers,
    )
