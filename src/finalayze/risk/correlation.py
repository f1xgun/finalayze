"""Correlation-aware position sizing (Layer 4).

Provides pairwise correlation computation for portfolio symbols and a
CorrelationStep for the PositionSizingPipeline that scales positions
down when correlated with existing holdings.
"""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal
from itertools import combinations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from finalayze.core.schemas import Candle
    from finalayze.risk.position_sizing_pipeline import SizingContext

_FOUR_DP = Decimal("0.0001")
_CORR_SCALE_LOWER = Decimal("0.30")
_CORR_SCALE_UPPER = Decimal("1.0")
_MAX_CORRELATED_POSITIONS = 3
_DEFAULT_CORRELATION_THRESHOLD = 0.7
_MIN_SYMBOLS_FOR_PAIRS = 2
_DENOM_EPSILON = 1e-15


def compute_correlation_matrix(
    candle_sets: dict[str, list[Candle]],
    window: int = 60,
) -> dict[tuple[str, str], float]:
    """Compute pairwise correlation of daily returns over window.

    Returns dict mapping (symbol_a, symbol_b) -> correlation coefficient.
    Only includes pairs where both have >= window+1 candles (to get window returns).
    Keys are ordered alphabetically: (min(a,b), max(a,b)).
    """
    if len(candle_sets) < _MIN_SYMBOLS_FOR_PAIRS:
        return {}

    # Pre-compute returns for each symbol
    returns_by_symbol: dict[str, list[float]] = {}
    for symbol, candles in candle_sets.items():
        if len(candles) < window + 1:
            continue
        recent = candles[-(window + 1) :]
        rets: list[float] = []
        for i in range(1, len(recent)):
            prev_close = float(recent[i - 1].close)
            if prev_close == 0:
                rets.append(0.0)
            else:
                rets.append((float(recent[i].close) - prev_close) / prev_close)
        returns_by_symbol[symbol] = rets

    # Compute pairwise Pearson correlation
    result: dict[tuple[str, str], float] = {}
    for sym_a, sym_b in combinations(sorted(returns_by_symbol.keys()), 2):
        rets_a = returns_by_symbol[sym_a]
        rets_b = returns_by_symbol[sym_b]
        n = min(len(rets_a), len(rets_b))
        if n == 0:
            continue

        ra = rets_a[-n:]
        rb = rets_b[-n:]

        mean_a = sum(ra) / n
        mean_b = sum(rb) / n

        cov = sum((a - mean_a) * (b - mean_b) for a, b in zip(ra, rb, strict=True)) / n
        var_a = sum((a - mean_a) ** 2 for a in ra) / n
        var_b = sum((b - mean_b) ** 2 for b in rb) / n

        denom = (var_a * var_b) ** 0.5
        corr = 0.0 if denom < _DENOM_EPSILON else cov / denom

        result[(sym_a, sym_b)] = corr

    return result


def _lookup_correlation(
    sym_a: str,
    sym_b: str,
    correlations: dict[tuple[str, str], float],
) -> float:
    """Look up correlation for a pair, checking both key orderings."""
    if (sym_a, sym_b) in correlations:
        return correlations[(sym_a, sym_b)]
    if (sym_b, sym_a) in correlations:
        return correlations[(sym_b, sym_a)]
    return 0.0


def compute_avg_correlation(
    symbol: str,
    open_positions: list[str],
    correlations: dict[tuple[str, str], float],
) -> float:
    """Average correlation of symbol with open positions.

    Returns 0.0 if open_positions is empty. Missing pairs are treated as 0.
    """
    if not open_positions:
        return 0.0

    total = sum(_lookup_correlation(symbol, pos, correlations) for pos in open_positions)
    return total / len(open_positions)


def count_correlated_positions(
    symbol: str,
    open_positions: list[str],
    correlations: dict[tuple[str, str], float],
    threshold: float = _DEFAULT_CORRELATION_THRESHOLD,
) -> int:
    """Count how many open positions are correlated > threshold with symbol."""
    return sum(
        1 for pos in open_positions if _lookup_correlation(symbol, pos, correlations) > threshold
    )


class CorrelationStep:
    """Pipeline step that scales position based on correlation with portfolio.

    scale = 1 - correlation_scale, bounded [0.30, 1.0].
    """

    def adjust(self, size: Decimal, context: SizingContext) -> Decimal:
        """Scale position size by inverse correlation factor."""
        raw_scale = Decimal(1) - context.correlation_scale
        scale = max(_CORR_SCALE_LOWER, min(_CORR_SCALE_UPPER, raw_scale))
        return (size * scale).quantize(_FOUR_DP, rounding=ROUND_HALF_UP)
