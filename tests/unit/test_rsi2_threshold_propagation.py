"""S7.2 — RSI(2) Connors confidence must scale against the configured threshold.

Audit #18 reported hardcoded thresholds in RSI2/Connors. Most knobs already
flow through the YAML preset (rsi_period / rsi_buy_threshold /
rsi_sell_threshold / sma_trend_period / min_confidence). The remaining
defect is inside the confidence functions ``_compute_buy_confidence`` and
``_compute_sell_confidence``: they compare RSI(2) against the literal
``10.0`` / ``90.0`` instead of the threshold the preset actually used to
gate the signal. Result: a preset that loosens the trigger to e.g.
``rsi_buy_threshold=30`` *fires* at RSI(2)=20, but the hardcoded formula
returns a negative raw → clamps to 0.0 → below ``min_confidence`` →
signal dropped silently.

Contract:
  S7.2-01: _compute_buy_confidence(rsi2=20, threshold=30) > 0.2
           (under hardcoded 10.0 this is 0.0 — clamped).
  S7.2-02: _compute_sell_confidence(rsi2=80, threshold=70) > 0.2.
  S7.2-03: default boundaries (10/90) round-trip unchanged.
  S7.2-04: thresholds are accepted as method arguments (signature
           takes ``threshold`` so the strategy can wire the preset value in).
"""

from __future__ import annotations

import inspect

import pytest

from finalayze.strategies.rsi2_connors import RSI2ConnorsStrategy

# Tunable confidence boundary used by the helpers: a raw value above the
# 0.2 floor means the formula did NOT collapse to the clamp.
_FLOOR = 0.2


@pytest.fixture
def strategy() -> RSI2ConnorsStrategy:
    return RSI2ConnorsStrategy()


# ─── S7.2-01 ────────────────────────────────────────────────────────────────
def test_buy_confidence_uses_threshold_argument(strategy: RSI2ConnorsStrategy) -> None:
    """With threshold=30, RSI=20 must produce confidence above the floor."""
    conf = strategy._compute_buy_confidence(rsi2=20.0, threshold=30.0)
    expected = (30.0 - 20.0) / 30.0 * 0.8 + 0.2  # 0.467
    assert conf == pytest.approx(expected, abs=1e-6)
    assert conf > _FLOOR, (
        "Under loose threshold (30) and moderate RSI (20) the confidence "
        f"must exceed the {_FLOOR} floor; hardcoded 10 would clamp it to 0.0"
    )


# ─── S7.2-02 ────────────────────────────────────────────────────────────────
def test_sell_confidence_uses_threshold_argument(strategy: RSI2ConnorsStrategy) -> None:
    """Symmetric SELL case: threshold=70, RSI=80 should produce positive confidence."""
    conf = strategy._compute_sell_confidence(rsi2=80.0, threshold=70.0)
    # New formula: (rsi2 - threshold) / (100 - threshold) * 0.8 + 0.2
    expected = (80.0 - 70.0) / (100.0 - 70.0) * 0.8 + 0.2  # 0.467
    assert conf == pytest.approx(expected, abs=1e-6)
    assert conf > _FLOOR


# ─── S7.2-03 ────────────────────────────────────────────────────────────────
def test_default_thresholds_unchanged(strategy: RSI2ConnorsStrategy) -> None:
    """Regression guard: the default (10/90) path returns the same value as before."""
    # BUY at rsi2=2 with threshold=10: legacy formula → (10-2)/10*0.8+0.2 = 0.84
    assert strategy._compute_buy_confidence(rsi2=2.0, threshold=10.0) == pytest.approx(0.84)
    # SELL at rsi2=98 with threshold=90: legacy formula → (98-90)/10*0.8+0.2 = 0.84
    assert strategy._compute_sell_confidence(rsi2=98.0, threshold=90.0) == pytest.approx(0.84)


# ─── S7.2-04 ────────────────────────────────────────────────────────────────
def test_confidence_helpers_accept_threshold_kw(strategy: RSI2ConnorsStrategy) -> None:
    """Signature must expose a ``threshold`` parameter (not hardcoded)."""
    sig_buy = inspect.signature(strategy._compute_buy_confidence)
    sig_sell = inspect.signature(strategy._compute_sell_confidence)
    assert "threshold" in sig_buy.parameters, (
        "_compute_buy_confidence must accept ``threshold`` so the strategy "
        "can wire the preset's rsi_buy_threshold in"
    )
    assert "threshold" in sig_sell.parameters, (
        "_compute_sell_confidence must accept ``threshold`` for rsi_sell_threshold"
    )
