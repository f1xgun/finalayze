"""Cross-asset feature computation (Layer 3)."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

# Cross-asset feature constants
_CROSS_ASSET_LOOKBACK = 63
_RELATIVE_STRENGTH_WINDOW = 21
_VOL_FLOOR = 0.01
_DEFAULT_BETA = 1.0
_DEFAULT_CORR = 0.5


def compute_cross_asset_features(
    close_s: pd.Series,
    benchmark_close_s: pd.Series | None,
) -> dict[str, float]:
    """Compute cross-asset features: relative strength, beta, correlation, excess momentum.

    All features compare the stock to a benchmark (e.g., SPY).
    When benchmark_close_s is None or has insufficient data, returns domain-aware defaults.
    No look-ahead bias: uses only past data via rolling windows.
    """
    defaults = {
        "relative_strength_21d": 0.0,
        "rolling_beta_63d": _DEFAULT_BETA,
        "rolling_corr_63d": _DEFAULT_CORR,
        "excess_momentum_score": 0.0,
    }

    if benchmark_close_s is None or len(benchmark_close_s) < _RELATIVE_STRENGTH_WINDOW:
        return defaults

    # Align lengths: use the shorter of the two series (from the end)
    min_len = min(len(close_s), len(benchmark_close_s))
    stock_close = close_s.iloc[-min_len:].reset_index(drop=True)
    bench_close = benchmark_close_s.iloc[-min_len:].reset_index(drop=True)

    stock_returns = stock_close.pct_change()
    bench_returns = bench_close.pct_change()

    # --- relative_strength_21d: stock 21d return minus benchmark 21d return ---
    relative_strength = 0.0
    if min_len >= _RELATIVE_STRENGTH_WINDOW + 1:
        stock_ret_21d = float(
            stock_close.iloc[-1] / stock_close.iloc[-_RELATIVE_STRENGTH_WINDOW - 1] - 1,
        )
        bench_ret_21d = float(
            bench_close.iloc[-1] / bench_close.iloc[-_RELATIVE_STRENGTH_WINDOW - 1] - 1,
        )
        relative_strength = stock_ret_21d - bench_ret_21d

    # --- rolling_beta_63d: cov(stock, bench) / var(bench) over 63d window ---
    rolling_beta = _DEFAULT_BETA
    if min_len >= _CROSS_ASSET_LOOKBACK + 1:
        cov = stock_returns.rolling(_CROSS_ASSET_LOOKBACK, min_periods=_CROSS_ASSET_LOOKBACK).cov(
            bench_returns,
        )
        var = bench_returns.rolling(
            _CROSS_ASSET_LOOKBACK,
            min_periods=_CROSS_ASSET_LOOKBACK,
        ).var()
        last_cov = float(cov.iloc[-1])
        last_var = float(var.iloc[-1])
        if last_var > 0 and math.isfinite(last_cov) and math.isfinite(last_var):
            rolling_beta = last_cov / last_var
        if not math.isfinite(rolling_beta):
            rolling_beta = _DEFAULT_BETA

    # --- rolling_corr_63d: rolling correlation over 63d window ---
    rolling_corr = _DEFAULT_CORR
    if min_len >= _CROSS_ASSET_LOOKBACK + 1:
        corr = stock_returns.rolling(
            _CROSS_ASSET_LOOKBACK,
            min_periods=_CROSS_ASSET_LOOKBACK,
        ).corr(bench_returns)
        last_corr = float(corr.iloc[-1])
        if math.isfinite(last_corr):
            rolling_corr = last_corr

    # --- excess_momentum_score: (stock_ret_63d - bench_ret_63d) / max(stock_vol_63d, floor) ---
    excess_momentum = 0.0
    if min_len >= _CROSS_ASSET_LOOKBACK + 1:
        stock_ret_63d = float(
            stock_close.iloc[-1] / stock_close.iloc[-_CROSS_ASSET_LOOKBACK - 1] - 1,
        )
        bench_ret_63d = float(
            bench_close.iloc[-1] / bench_close.iloc[-_CROSS_ASSET_LOOKBACK - 1] - 1,
        )
        stock_vol_63d = float(
            stock_returns.iloc[-_CROSS_ASSET_LOOKBACK:].std(),
        )
        denom = max(stock_vol_63d, _VOL_FLOOR) if math.isfinite(stock_vol_63d) else _VOL_FLOOR
        excess_momentum = (stock_ret_63d - bench_ret_63d) / denom
        if not math.isfinite(excess_momentum):
            excess_momentum = 0.0

    return {
        "relative_strength_21d": relative_strength,
        "rolling_beta_63d": rolling_beta,
        "rolling_corr_63d": rolling_corr,
        "excess_momentum_score": excess_momentum,
    }
