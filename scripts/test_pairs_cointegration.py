"""Test MOEX pairs for cointegration with post-2022 data.

Fetches daily close prices from Tinkoff Invest API for MOEX-listed stocks and runs
three statistical tests for each configured pair:
  1. Engle-Granger cointegration test (statsmodels.tsa.stattools.coint)
  2. Half-life of mean reversion (OLS on spread changes vs lagged spread)
  3. Hurst exponent of the spread (rescaled range analysis)

Decision criteria:
  - KEEP if: p-value < 0.05 AND half-life < 30 days AND Hurst < 0.5
  - DISABLE if: p-value > 0.05 OR half-life > 30 days OR Hurst >= 0.5

Usage:
    FINALAYZE_TINKOFF_TOKEN=... uv run python scripts/test_pairs_cointegration.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from statsmodels.tsa.stattools import coint

load_dotenv()

# Ensure project root is on sys.path for config imports
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

PAIRS: list[tuple[str, str]] = [
    ("SBER", "VTBR"),   # Sberbank vs VTB Bank
    ("GAZP", "LKOH"),   # Gazprom vs Lukoil
    ("SBER", "GMKN"),   # Sberbank vs Norilsk Nickel
    ("MGNT", "FIVE"),   # Magnit vs X5 Retail (retail)
    ("NLMK", "CHMF"),   # NLMK vs Severstal (steel)
    ("ALRS", "PLZL"),   # Alrosa vs Polyus (mining)
    ("ROSN", "TATN"),   # Rosneft vs Tatneft (energy)
    ("SBER", "TCSG"),   # Sberbank vs T-Bank (banks)
]

START_DATE = "2022-01-01"
END_DATE = "2025-12-31"

# Decision thresholds
P_VALUE_THRESHOLD = 0.05
HALF_LIFE_MAX_DAYS = 30
HURST_THRESHOLD = 0.5

# Numerical constants
_EPSILON = 1e-12
_MIN_RS_LAGS = 3
_MIN_OBSERVATIONS = 30
_INF_DISPLAY_THRESHOLD = 1000
_STABILITY_THRESHOLD = 0.5

# Sub-period analysis windows
SUB_PERIODS: list[tuple[str, str, str]] = [
    ("2022-H1", "2022-01-01", "2022-06-30"),
    ("2022-H2", "2022-07-01", "2022-12-31"),
    ("2023", "2023-01-01", "2023-12-31"),
    ("2024", "2024-01-01", "2024-12-31"),
    ("2025", "2025-01-01", "2025-12-31"),
    ("Full", "2022-01-01", "2025-12-31"),
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class PairResult:
    """Results of cointegration analysis for a single pair."""

    pair_name: str
    period: str
    n_observations: int
    p_value: float
    half_life: float
    hurst: float
    verdict: str  # "KEEP" or "DISABLE"
    reason: str


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------


def compute_half_life(spread: np.ndarray) -> float:
    """Compute half-life of mean reversion via OLS.

    Regresses delta_spread on lagged spread:
        delta_spread_t = beta * spread_{t-1} + epsilon_t
    Half-life = -log(2) / beta

    Returns inf if beta >= 0 (non-mean-reverting).
    """
    spread_lag = spread[:-1]
    delta_spread = np.diff(spread)

    # Remove mean to improve numerical stability
    spread_lag_dm = spread_lag - spread_lag.mean()

    if np.std(spread_lag_dm) < _EPSILON:
        return float("inf")

    # OLS: beta = cov(delta, lag) / var(lag)
    beta = float(np.cov(delta_spread, spread_lag_dm)[0, 1] / np.var(spread_lag_dm, ddof=1))

    if beta >= 0:
        return float("inf")

    return float(-np.log(2) / beta)


def compute_hurst_exponent(series: np.ndarray, max_lag: int | None = None) -> float:
    """Compute Hurst exponent via rescaled range (R/S) analysis.

    H < 0.5: mean-reverting (anti-persistent)
    H = 0.5: random walk (Brownian motion)
    H > 0.5: trending (persistent)

    Args:
        series: 1D time series array.
        max_lag: Maximum lag for R/S calculation. Defaults to len(series)//4.

    Returns:
        Estimated Hurst exponent.
    """
    n = len(series)
    if max_lag is None:
        max_lag = max(n // 4, 10)

    min_lag = 4
    if max_lag < min_lag + 2:
        return 0.5  # insufficient data

    lags = range(min_lag, max_lag + 1)
    rs_values: list[float] = []
    valid_lags: list[int] = []

    for lag in lags:
        rs_avg = _compute_rs_for_lag(series, n, lag)
        if rs_avg is not None:
            rs_values.append(rs_avg)
            valid_lags.append(lag)

    if len(valid_lags) < _MIN_RS_LAGS:
        return 0.5  # insufficient data for regression

    log_lags = np.log(np.array(valid_lags, dtype=float))
    log_rs = np.log(np.array(rs_values, dtype=float))

    # OLS fit: log(R/S) = H * log(lag) + c
    coeffs = np.polyfit(log_lags, log_rs, 1)
    return float(coeffs[0])


def _compute_rs_for_lag(series: np.ndarray, n: int, lag: int) -> float | None:
    """Compute average R/S statistic for a given lag size."""
    n_sub = n // lag
    if n_sub < 1:
        return None

    rs_sum = 0.0
    count = 0
    for i in range(n_sub):
        sub = series[i * lag : (i + 1) * lag]
        mean_sub = np.mean(sub)
        deviations = np.cumsum(sub - mean_sub)
        r = float(np.max(deviations) - np.min(deviations))
        s = float(np.std(sub, ddof=1))
        if s > _EPSILON:
            rs_sum += r / s
            count += 1

    if count > 0:
        return rs_sum / count
    return None


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _format_half_life(hl: float, width: int = 7) -> str:
    """Format half-life value, showing 'inf' for very large values."""
    if hl < _INF_DISPLAY_THRESHOLD:
        return f"{hl:{width}.1f}"
    return "inf".rjust(width)


# ---------------------------------------------------------------------------
# Data fetching via TinkoffFetcher
# ---------------------------------------------------------------------------


def fetch_prices(symbols: list[str], start: str, end: str) -> pd.DataFrame:
    """Fetch daily close prices from Tinkoff Invest API.

    Args:
        symbols: List of MOEX tickers (e.g., ["SBER", "VTBR"]).
        start: Start date string (YYYY-MM-DD).
        end: End date string (YYYY-MM-DD).

    Returns:
        DataFrame with columns = symbols and index = date.
    """
    from finalayze.data.fetchers.tinkoff_data import TinkoffFetcher  # noqa: PLC0415
    from finalayze.markets.instruments import build_default_registry  # noqa: PLC0415

    token = os.environ.get("FINALAYZE_TINKOFF_TOKEN", "")
    if not token:
        print("ERROR: FINALAYZE_TINKOFF_TOKEN not set. Cannot fetch MOEX data.")
        return pd.DataFrame()

    registry = build_default_registry()
    fetcher = TinkoffFetcher(token=token, registry=registry, sandbox=False)

    start_dt = datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=UTC)
    end_dt = datetime.strptime(end, "%Y-%m-%d").replace(tzinfo=UTC)

    print(f"  Fetching {symbols} from {start} to {end} via TinkoffFetcher...")

    all_series: dict[str, pd.Series] = {}
    for symbol in symbols:
        try:
            candles = fetcher.fetch_candles(symbol, start_dt, end_dt)
            if candles:
                dates = [c.timestamp.date() for c in candles]
                closes = [float(c.close) for c in candles]
                series = pd.Series(closes, index=pd.DatetimeIndex(dates), name=symbol)
                all_series[symbol] = series
                print(f"    {symbol}: {len(candles)} bars")
            else:
                print(f"    {symbol}: no data")
        except Exception as e:
            print(f"    {symbol}: fetch failed ({e})")

    if not all_series:
        return pd.DataFrame()

    df = pd.DataFrame(all_series)
    return df.dropna()


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------


def analyze_pair(
    prices: pd.DataFrame,
    sym_a: str,
    sym_b: str,
    period_name: str,
) -> PairResult:
    """Run cointegration analysis for a single pair and period.

    Args:
        prices: DataFrame with close prices for both symbols.
        sym_a: First symbol (MOEX ticker).
        sym_b: Second symbol (MOEX ticker).
        period_name: Label for the analysis period.

    Returns:
        PairResult with test statistics and verdict.
    """
    pair_name = f"{sym_a}/{sym_b}"

    if sym_a not in prices.columns or sym_b not in prices.columns:
        return PairResult(
            pair_name=pair_name,
            period=period_name,
            n_observations=0,
            p_value=1.0,
            half_life=float("inf"),
            hurst=0.5,
            verdict="DISABLE",
            reason="Missing data for one or both symbols",
        )

    series_a = prices[sym_a].dropna()
    series_b = prices[sym_b].dropna()

    # Align on common dates
    common_idx = series_a.index.intersection(series_b.index)
    if len(common_idx) < _MIN_OBSERVATIONS:
        return PairResult(
            pair_name=pair_name,
            period=period_name,
            n_observations=len(common_idx),
            p_value=1.0,
            half_life=float("inf"),
            hurst=0.5,
            verdict="DISABLE",
            reason=f"Insufficient data ({len(common_idx)} obs, need >= {_MIN_OBSERVATIONS})",
        )

    a = np.log(series_a.loc[common_idx].values.astype(float))
    b = np.log(series_b.loc[common_idx].values.astype(float))

    # 1. Engle-Granger cointegration test
    _, p_value, _ = coint(a, b)
    p_value = float(p_value)

    # 2. Compute OLS beta and spread
    cov_matrix = np.cov(a, b)
    beta = float(cov_matrix[0, 1] / cov_matrix[1, 1])
    spread = a - beta * b

    # 3. Half-life of mean reversion
    half_life = compute_half_life(spread)

    # 4. Hurst exponent of the spread
    hurst = compute_hurst_exponent(spread)

    # 5. Decision
    verdict, reason = _make_decision(p_value, half_life, hurst)

    return PairResult(
        pair_name=pair_name,
        period=period_name,
        n_observations=len(common_idx),
        p_value=p_value,
        half_life=half_life,
        hurst=hurst,
        verdict=verdict,
        reason=reason,
    )


def _make_decision(p_value: float, half_life: float, hurst: float) -> tuple[str, str]:
    """Apply decision criteria and return (verdict, reason)."""
    reasons = []
    if p_value > P_VALUE_THRESHOLD:
        reasons.append(f"p-value={p_value:.4f} > {P_VALUE_THRESHOLD}")
    if half_life > HALF_LIFE_MAX_DAYS:
        hl_str = _format_half_life(half_life).strip()
        reasons.append(f"half-life={hl_str} > {HALF_LIFE_MAX_DAYS}")
    if hurst >= HURST_THRESHOLD:
        reasons.append(f"hurst={hurst:.3f} >= {HURST_THRESHOLD}")

    if reasons:
        return "DISABLE", "; ".join(reasons)
    return "KEEP", "All criteria met"


# ---------------------------------------------------------------------------
# Report sections (split from main for ruff complexity limits)
# ---------------------------------------------------------------------------


def _run_pair_analysis(
    all_prices: pd.DataFrame,
) -> list[PairResult]:
    """Run analysis for each pair across all sub-periods."""
    all_results: list[PairResult] = []

    for sym_a, sym_b in PAIRS:
        print(f"--- Analyzing {sym_a}/{sym_b} ---")

        for period_name, p_start, p_end in SUB_PERIODS:
            mask = (all_prices.index >= p_start) & (all_prices.index <= p_end)
            period_prices = all_prices.loc[mask]

            result = analyze_pair(period_prices, sym_a, sym_b, period_name)
            all_results.append(result)

            hl_str = _format_half_life(result.half_life)
            print(
                f"  {period_name:8s} | N={result.n_observations:4d} | "
                f"p={result.p_value:.4f} | HL={hl_str} | "
                f"H={result.hurst:.3f} | {result.verdict:7s} | {result.reason}"
            )

        print()

    return all_results


def _print_summary(full_results: list[PairResult]) -> None:
    """Print the summary table for the full analysis period."""
    print("=" * 80)
    print("SUMMARY (Full Period: 2022-2025)")
    print("=" * 80)
    print()
    header = (
        f"{'Pair':>12s} | {'N':>5s} | {'p-value':>8s} | "
        f"{'Half-life':>10s} | {'Hurst':>6s} | {'Verdict':>8s}"
    )
    print(header)
    print("-" * 72)

    for r in full_results:
        hl_str = _format_half_life(r.half_life, width=10)
        print(
            f"{r.pair_name:>12s} | {r.n_observations:5d} | {r.p_value:8.4f} | "
            f"{hl_str} | {r.hurst:6.3f} | {r.verdict:>8s}"
        )

    print()
    print("=" * 80)
    print("DECISION CRITERIA")
    print("=" * 80)
    print(f"  Cointegration p-value < {P_VALUE_THRESHOLD}")
    print(f"  Half-life of mean reversion < {HALF_LIFE_MAX_DAYS} days")
    print(f"  Hurst exponent < {HURST_THRESHOLD} (mean-reverting)")
    print()


def _print_recommendations(
    full_results: list[PairResult],
) -> list[str]:
    """Print keep/disable recommendations. Returns list of disabled pair names."""
    print("=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    print()

    disable_pairs: list[str] = []

    for r in full_results:
        if r.verdict == "KEEP":
            print(f"  [KEEP]    {r.pair_name}: {r.reason}")
        else:
            disable_pairs.append(r.pair_name)
            print(f"  [DISABLE] {r.pair_name}: {r.reason}")

    print()
    return disable_pairs


def _print_stability(all_results: list[PairResult]) -> None:
    """Print regime stability analysis across sub-periods."""
    print("=" * 80)
    print("REGIME STABILITY ANALYSIS")
    print("=" * 80)
    print()
    print("Checking whether cointegration is stable across sub-periods")
    print("(unstable pairs are risky even if full-period test passes):")
    print()

    for sym_a, sym_b in PAIRS:
        pair_name = f"{sym_a}/{sym_b}"
        pair_results = [r for r in all_results if r.pair_name == pair_name and r.period != "Full"]

        passes = sum(1 for r in pair_results if r.verdict == "KEEP")
        total = len(pair_results)

        print(f"  {pair_name}:")
        for r in pair_results:
            marker = "pass" if r.verdict == "KEEP" else "FAIL"
            hl_str = f"{r.half_life:.1f}"
            print(
                f"    {r.period:8s}: {marker} (p={r.p_value:.4f}, HL={hl_str}d, H={r.hurst:.3f})"
            )

        stability = passes / total if total > 0 else 0
        print(f"    Stability: {passes}/{total} sub-periods pass ({stability:.0%})")
        if stability < _STABILITY_THRESHOLD:
            print("    WARNING: Unstable cointegration -- pair may not be reliable")
        print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run cointegration analysis for all MOEX pairs."""
    print("=" * 80)
    print("MOEX Pairs Cointegration Analysis (Post-2022)")
    print(f"Date range: {START_DATE} to {END_DATE}")
    print("=" * 80)
    print()

    # Collect all unique symbols
    all_symbols: set[str] = set()
    for sym_a, sym_b in PAIRS:
        all_symbols.add(sym_a)
        all_symbols.add(sym_b)

    sorted_symbols = sorted(all_symbols)
    print(f"Symbols: {sorted_symbols}")
    print()

    # Fetch full-period data once via TinkoffFetcher
    print("Fetching price data from Tinkoff Invest API...")
    all_prices = fetch_prices(sorted_symbols, START_DATE, END_DATE)

    if all_prices.empty:
        print("ERROR: No price data returned from Tinkoff API.")
        print("Check FINALAYZE_TINKOFF_TOKEN and network connectivity.")
        print()
        print("Recommendation: DISABLE all MOEX pairs until validated")
        print("with working Tinkoff API access.")
        sys.exit(1)

    print(f"Fetched {len(all_prices)} trading days of data.")
    print(f"Date range: {all_prices.index[0]} to {all_prices.index[-1]}")
    print(f"Columns available: {list(all_prices.columns)}")
    print()

    # Run analysis
    all_results = _run_pair_analysis(all_prices)
    full_results = [r for r in all_results if r.period == "Full"]

    # Print reports
    _print_summary(full_results)
    disable_pairs = _print_recommendations(full_results)
    _print_stability(all_results)

    print()
    now = datetime.now(tz=UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"Analysis completed at {now}")


if __name__ == "__main__":
    main()
