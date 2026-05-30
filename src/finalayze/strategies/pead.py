"""Post-Earnings Announcement Drift (PEAD) strategy (Layer 4).

Stocks drift in the direction of earnings surprises for 60-90 days
post-announcement. This effect is stronger in emerging markets and
mid-caps due to lower institutional coverage.

Strategy logic:
- Register earnings surprises via add_earnings_surprise().
- After announcement: BUY if sue_score > positive_threshold, SELL if < negative_threshold.
- Signals remain active for drift_window_bars after announcement.
- Confidence scales with |sue_score| magnitude.
- Both US and MOEX markets supported.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from collections.abc import Sequence

import structlog

from finalayze.core.schemas import Candle, Signal, SignalDirection
from finalayze.strategies.base import BaseStrategy

logger = structlog.get_logger(__name__)

# Confidence scaling constants
_CONFIDENCE_BASE = 0.35
_CONFIDENCE_SCALE = 0.10  # per unit of |sue_score| above threshold
_MAX_CONFIDENCE = 0.90


@dataclass(frozen=True, slots=True)
class EarningsSurprise:
    """A single earnings surprise event for a symbol.

    ``is_proxy`` flags a surprise whose ``expected_eps`` is a trend/prior-year
    proxy derived from an ``eps_ttm`` time-series rather than a true analyst
    consensus. MOEX has no consensus feed (D-01 / RESEARCH Q1), so
    :func:`compute_sue_proxy` always sets ``is_proxy=True``. Backtest
    attribution MUST discount a proxy surprise so it is not over-credited as a
    real consensus beat/miss.
    """

    symbol: str
    announcement_date: datetime
    sue_score: float  # Standardized Unexpected Earnings
    actual_eps: float
    expected_eps: float
    is_proxy: bool = False  # True when expected_eps is a trend proxy, not consensus


# Prior-year same-period lookup window: an eps_ttm point within this many days
# of (D - 365d) counts as the "prior-year same-period" baseline.
_PRIOR_YEAR_DAYS = 365
_PRIOR_YEAR_TOLERANCE_DAYS = 60
_MIN_HISTORY_FOR_STD = 2


def compute_sue_proxy(
    symbol: str,
    announcement_date: datetime,
    eps_history: Sequence[tuple[datetime, float]],
) -> EarningsSurprise:
    """Build a LABELLED, point-in-time SUE proxy from an ``eps_ttm`` series.

    The proxy is NOT a true consensus surprise: ``get_asset_reports`` carries no
    actual EPS (RESEARCH Q1), so ``actual`` comes from the ``eps_ttm``
    fundamental time-series and ``expected`` from the prior-year same-period
    ``eps_ttm`` (falling back to a rolling-trend mean). The returned
    ``EarningsSurprise`` always has ``is_proxy=True`` so backtest attribution
    stays honest (D-01).

    Point-in-time (RESEARCH Assumption A4): ``announcement_date`` IS the as-of
    cutoff D. The history is first filtered to entries dated ``<= D`` so a
    future ``report_date`` can never leak into a SUE computed as-of D.

    Degenerate input (empty / single point / zero dispersion) yields a guarded
    ``sue_score == 0.0`` fallback — never a raise, never NaN/inf.
    """
    # A4 look-ahead guard: drop any entry dated AFTER the as-of cutoff D.
    # A future report_date must never contribute to an as-of-D SUE.
    usable = sorted(
        ((dt, v) for (dt, v) in eps_history if dt <= announcement_date),
        key=lambda p: p[0],
    )

    if not usable:
        return EarningsSurprise(
            symbol=symbol,
            announcement_date=announcement_date,
            sue_score=0.0,
            actual_eps=0.0,
            expected_eps=0.0,
            is_proxy=True,
        )

    # actual = latest eps_ttm at/<= D (the as-of point).
    actual_dt, actual = usable[-1]

    # expected = prior-year same-period eps_ttm (nearest point to D - 365d
    # within tolerance); fall back to the rolling-trend mean of prior points,
    # finally to the actual itself (zero surprise) when no history precedes D.
    expected = _prior_year_expected(usable, actual_dt)
    if expected is None:
        expected = actual

    # historical surprises = series of (eps_i - prior_year(eps_i)) over usable.
    surprises = _historical_surprises(usable)

    std = statistics.stdev(surprises) if len(surprises) >= _MIN_HISTORY_FOR_STD else 0.0

    sue_score = (actual - expected) / std if std > 0.0 else 0.0

    return EarningsSurprise(
        symbol=symbol,
        announcement_date=announcement_date,
        sue_score=sue_score,
        actual_eps=actual,
        expected_eps=expected,
        is_proxy=True,
    )


def _prior_year_expected(
    usable: Sequence[tuple[datetime, float]],
    as_of_dt: datetime,
) -> float | None:
    """Prior-year same-period eps_ttm, or rolling-trend fallback.

    ``usable`` is already filtered to date <= D and sorted ascending. Returns
    ``None`` when NO eps point precedes ``as_of_dt`` (no prior-year partner and
    no prior trend) — callers decide the terminal fallback. Only points strictly
    before ``as_of_dt`` are considered, so the as-of point never expects itself.
    """
    target = as_of_dt - timedelta(days=_PRIOR_YEAR_DAYS)
    best: tuple[float, float] | None = None  # (abs_days_diff, value)
    prior: list[float] = []
    for dt, v in usable:
        if dt >= as_of_dt:
            continue
        prior.append(v)
        diff_days = abs((dt - target).days)
        if diff_days <= _PRIOR_YEAR_TOLERANCE_DAYS and (best is None or diff_days < best[0]):
            best = (float(diff_days), v)
    if best is not None:
        return best[1]
    # Fallback: rolling-trend mean of all prior points (excludes the as-of point).
    if prior:
        return statistics.fmean(prior)
    # No point precedes as_of_dt -> no prior-year partner.
    return None


def _historical_surprises(usable: Sequence[tuple[datetime, float]]) -> list[float]:
    """(eps_i - prior_year(eps_i)) for each point WITH a genuine prior partner.

    Points with no preceding eps datum (e.g. the oldest entry) contribute no
    surprise — they have no prior-year baseline to be measured against.
    """
    surprises: list[float] = []
    for dt, v in usable:
        prior = _prior_year_expected(usable, dt)
        if prior is None:
            continue
        surprises.append(v - prior)
    return surprises


class PEADStrategy(BaseStrategy):
    """Post-Earnings Announcement Drift strategy.

    Generates BUY signals for positive earnings surprises and SELL signals
    for negative surprises. Signals are active for a configurable drift
    window after the earnings announcement date.
    """

    _SUPPORTED_SEGMENTS: ClassVar[list[str]] = [
        "us_tech",
        "us_broad",
        "us_healthcare",
        "us_finance",
        "ru_blue_chips",
        "ru_energy",
        "ru_finance",
        "ru_tech",
    ]

    def __init__(
        self,
        positive_threshold: float = 1.0,
        negative_threshold: float = -1.0,
        drift_window_bars: int = 60,
        min_confidence: float = 0.35,
    ) -> None:
        self._positive_threshold = positive_threshold
        self._negative_threshold = negative_threshold
        self._drift_window_bars = drift_window_bars
        self._min_confidence = min_confidence

        # symbol -> list of earnings surprises
        self._surprises: dict[str, list[EarningsSurprise]] = {}

    @property
    def name(self) -> str:
        return "pead"

    def supported_segments(self) -> list[str]:
        return list(self._SUPPORTED_SEGMENTS)

    def get_parameters(self, segment_id: str) -> dict[str, object]:  # noqa: ARG002
        return {
            "positive_threshold": self._positive_threshold,
            "negative_threshold": self._negative_threshold,
            "drift_window_bars": self._drift_window_bars,
            "min_confidence": self._min_confidence,
        }

    def add_earnings_surprise(self, surprise: EarningsSurprise) -> None:
        """Register an earnings surprise event."""
        self._surprises.setdefault(surprise.symbol, []).append(surprise)

    def reset(self) -> None:
        """Clear all state between backtest runs."""
        self._surprises.clear()

    def generate_signal(
        self,
        symbol: str,
        candles: list[Candle],
        segment_id: str,
        sentiment_score: float = 0.0,  # noqa: ARG002
        has_open_position: bool = False,  # noqa: ARG002
    ) -> Signal | None:
        """Generate PEAD signal based on earnings surprise data.

        Checks all registered surprises for the symbol and generates a signal
        if the current candle falls within the drift window of any surprise
        that exceeds the threshold.
        """
        if not candles:
            return None

        surprises = self._surprises.get(symbol)
        if not surprises:
            return None

        current_candle = candles[-1]
        current_date = current_candle.timestamp
        market_id = current_candle.market_id

        # Find the most recent applicable surprise
        best_surprise: EarningsSurprise | None = None
        best_bars_since: int | None = None

        for surprise in surprises:
            # Current candle must be on or after announcement date
            if current_date.date() < surprise.announcement_date.date():
                continue

            # Count bars since announcement
            bars_since = sum(
                1 for c in candles if c.timestamp.date() > surprise.announcement_date.date()
            )

            # Check drift window (signal active for drift_window_bars after announcement)
            if bars_since > self._drift_window_bars:
                continue

            # Pick the most recent surprise
            if best_surprise is None or (
                surprise.announcement_date > best_surprise.announcement_date
            ):
                best_surprise = surprise
                best_bars_since = bars_since

        if best_surprise is None or best_bars_since is None:
            return None

        sue = best_surprise.sue_score

        # Determine direction based on thresholds
        direction: SignalDirection | None = None
        if sue > self._positive_threshold:
            direction = SignalDirection.BUY
        elif sue < self._negative_threshold:
            direction = SignalDirection.SELL
        else:
            return None

        # Compute confidence scaled by |sue_score| magnitude
        excess = abs(sue) - abs(
            self._positive_threshold
            if direction == SignalDirection.BUY
            else self._negative_threshold
        )
        confidence = min(_MAX_CONFIDENCE, _CONFIDENCE_BASE + excess * _CONFIDENCE_SCALE)

        if confidence < self._min_confidence:
            logger.debug(
                "pead: below min_confidence",
                symbol=symbol,
                confidence=confidence,
                min_confidence=self._min_confidence,
            )
            return None

        return Signal(
            strategy_name=self.name,
            symbol=symbol,
            market_id=market_id,
            segment_id=segment_id,
            direction=direction,
            confidence=confidence,
            strategy_payload={
                "sue_score": best_surprise.sue_score,
                "actual_eps": best_surprise.actual_eps,
                "expected_eps": best_surprise.expected_eps,
                "bars_since_announcement": float(best_bars_since),
            },
            reasoning=(
                f"PEAD: SUE={sue:.2f} "
                f"({'positive' if direction == SignalDirection.BUY else 'negative'} surprise), "
                f"{best_bars_since} bars post-announcement "
                f"(window={self._drift_window_bars})"
            ),
        )
