"""Ichimoku Cloud computation helper (Layer 4).

Provides a pure-computation function for Ichimoku Cloud indicators:
tenkan-sen, kijun-sen, senkou span A/B, cloud thickness, and
bullish/bearish classification.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class IchimokuResult:
    """Result of Ichimoku Cloud computation for the most recent bar."""

    tenkan: float  # Tenkan-sen (conversion line)
    kijun: float  # Kijun-sen (base line)
    senkou_a: float  # Senkou Span A (leading span A)
    senkou_b: float  # Senkou Span B (leading span B)
    cloud_thickness: float  # abs(senkou_a - senkou_b)
    is_bullish: bool  # price above cloud AND tenkan > kijun
    is_bearish: bool  # price below cloud AND tenkan < kijun


def _midpoint(highs: list[float], lows: list[float], period: int) -> float:
    """Compute (highest high + lowest low) / 2 over the last ``period`` bars."""
    h_slice = highs[-period:]
    l_slice = lows[-period:]
    return (max(h_slice) + min(l_slice)) / 2.0


def compute_ichimoku(
    highs: list[float],
    lows: list[float],
    closes: list[float],
    tenkan_period: int = 9,
    kijun_period: int = 26,
) -> IchimokuResult | None:
    """Compute Ichimoku Cloud indicators for the most recent bar.

    Returns ``None`` when there is insufficient data (fewer bars than
    ``kijun_period``).

    Args:
        highs: High prices, oldest first.
        lows: Low prices, oldest first.
        closes: Close prices, oldest first.
        tenkan_period: Look-back for Tenkan-sen (default 9).
        kijun_period: Look-back for Kijun-sen (default 26).

    Returns:
        An ``IchimokuResult`` or ``None`` if data is insufficient.
    """
    min_bars = max(tenkan_period, kijun_period)
    if len(highs) < min_bars or len(lows) < min_bars or len(closes) < min_bars:
        return None

    tenkan = _midpoint(highs, lows, tenkan_period)
    kijun = _midpoint(highs, lows, kijun_period)

    # Senkou Span A = (tenkan + kijun) / 2  (normally projected 26 periods ahead,
    # but we use the current value for trend classification)
    senkou_a = (tenkan + kijun) / 2.0

    # Senkou Span B = midpoint over 2 * kijun_period (52 bars default).
    # If insufficient data for 52 bars, fall back to kijun_period.
    senkou_b_period = 2 * kijun_period
    if len(highs) >= senkou_b_period:
        senkou_b = _midpoint(highs, lows, senkou_b_period)
    else:
        senkou_b = _midpoint(highs, lows, kijun_period)

    cloud_thickness = abs(senkou_a - senkou_b)
    current_close = closes[-1]
    cloud_top = max(senkou_a, senkou_b)
    cloud_bottom = min(senkou_a, senkou_b)

    is_bullish = current_close > cloud_top and tenkan > kijun
    is_bearish = current_close < cloud_bottom and tenkan < kijun

    return IchimokuResult(
        tenkan=tenkan,
        kijun=kijun,
        senkou_a=senkou_a,
        senkou_b=senkou_b,
        cloud_thickness=cloud_thickness,
        is_bullish=is_bullish,
        is_bearish=is_bearish,
    )
