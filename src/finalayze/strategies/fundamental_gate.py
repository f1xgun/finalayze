"""Rule-based ``earnings_yield`` fundamental gate (INTG-02, D-01, Layer 4).

A standalone decision filter that consumes ``compute_fundamental_features`` DIRECTLY
(Layer 4 -> Layer 3, a permitted downward import) and reads ONLY the ``earnings_yield``
key. It is deliberately NOT routed through ``finalayze.ml.features.technical.compute_features``
or the ML ensemble (D-01: rule-based, no ML dependency; every MOEX preset has
``ml_ensemble.enabled = false``). Because it never edits ``compute_features``, it does
NOT bump ``FEATURE_SCHEMA_VERSION`` (Assumption A5).

Look-ahead safety is inherited, not re-implemented: the gate delegates the ``as_of <= D``
filter to ``compute_fundamental_features._filter_as_of``. A snapshot dated after the bar
date ``D`` therefore cannot influence the verdict at ``D``. Missing data degrades
gracefully — ``compute_fundamental_features`` returns the all-``0.0`` ``_DEFAULT`` (never
NaN, never raises), which maps to a neutral passthrough verdict here.

BACKFILL-01 / D-03 scope note: T-Bank ``get_asset_fundamentals`` is point-in-time only
(no date range — see ``tests/unit/test_lookahead_phase60_fundamental.py``), so a pure
fundamental snapshot is CONSTANT across a backtest window and is therefore measured
LIVE-FORWARD only — it is NOT the MEAS-01 causal lever (SUE/CPI are; Assumption A3).
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING

from finalayze.ml.features.fundamental import compute_fundamental_features

if TYPE_CHECKING:
    from datetime import datetime

    from finalayze.core.schemas import MoexMarketData

# Neutral verdict scale: no position adjustment (the gate is "off" / passthrough).
NEUTRAL_SCALE = Decimal("1.0")

# Boost applied when a symbol is cheap (earnings_yield >= threshold). Modest, tunable
# (D-04) — a confidence/sizing nudge, not a dominant lever; never zeroes a position.
BOOST_SCALE = Decimal("1.2")

# Default cheapness threshold on earnings_yield (= 1 / P/E). 0.08 ~= a P/E of 12.5;
# below that the valuation is not "cheap" enough to boost. Tunable per D-04.
DEFAULT_EARNINGS_YIELD_THRESHOLD = 0.08


@dataclass(frozen=True)
class FundamentalGateVerdict:
    """Typed rule-gate verdict usable as a confidence/sizing gate.

    ``passed`` is True when the symbol cleared the cheapness threshold; ``scale`` is the
    multiplier to apply (``BOOST_SCALE`` on pass, ``NEUTRAL_SCALE`` otherwise — never a
    cut, so a missing/expensive fundamental never harms an otherwise-valid signal).
    ``earnings_yield`` is the raw feature read (always defined, never NaN).
    """

    passed: bool
    scale: Decimal
    earnings_yield: float


def earnings_yield_gate(
    moex_data: MoexMarketData | None,
    *,
    as_of: datetime | None = None,
    threshold: float = DEFAULT_EARNINGS_YIELD_THRESHOLD,
) -> FundamentalGateVerdict:
    """Rule-based ``earnings_yield`` gate (INTG-02, D-01).

    Reads ``compute_fundamental_features(moex_data, as_of=as_of)['earnings_yield']``
    DIRECTLY (no ``compute_features``/ML path) and returns a :class:`FundamentalGateVerdict`:

    - ``earnings_yield >= threshold`` (cheap) -> ``passed=True``, ``scale=BOOST_SCALE``.
    - ``earnings_yield <  threshold`` (or missing) -> ``passed=False``, ``scale=NEUTRAL_SCALE``.

    Look-ahead: the ``as_of <= D`` filter is enforced inside ``compute_fundamental_features``
    (``_filter_as_of``), so a snapshot dated after ``as_of`` cannot influence the verdict.
    Degrades gracefully: no usable snapshot yields the all-``0.0`` default -> neutral
    passthrough (no raise, never NaN).
    """
    features = compute_fundamental_features(moex_data, as_of=as_of)
    earnings_yield = features["earnings_yield"]
    if earnings_yield >= threshold:
        return FundamentalGateVerdict(passed=True, scale=BOOST_SCALE, earnings_yield=earnings_yield)
    return FundamentalGateVerdict(passed=False, scale=NEUTRAL_SCALE, earnings_yield=earnings_yield)
