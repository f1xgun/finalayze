"""Dynamic quality gates based on effective sample size (AFML Ch. 7).

Provides functions for computing effective sample size and dynamic
thresholds for accuracy and Brier score quality gates.
"""

from __future__ import annotations

import math

# --- Dynamic quality gates (AFML Ch.7) ---
# Binomial test parameters for accuracy threshold
_Z_ALPHA_95 = 1.645  # z-score for 95% confidence
_Z_ALPHA_99 = 1.96  # z-score for 99% confidence (used when confidence != 0.95)
_MAX_ACCURACY_THRESHOLD = 0.75  # cap to prevent impossible thresholds
_MIN_N_EFF_FOR_NORMAL = 5  # below this, use conservative fallback
_TINY_SAMPLE_ACCURACY = 0.90  # near-impossible for tiny samples
_TINY_SAMPLE_BRIER = 0.15  # strict Brier for tiny samples
_BRIER_COIN_FLIP = 0.25  # Brier score for random 50/50 predictions
_BRIER_REFERENCE_N_EFF = 100  # reference n_eff for Brier improvement scaling
_BRIER_IMPROVEMENT_RATE = 0.05  # max improvement at reference n_eff
_MIN_BRIER_THRESHOLD = 0.15  # floor for Brier threshold


def compute_n_eff(n_samples: int, avg_hold_bars: float) -> int:
    """Effective sample size accounting for label overlap.

    Per AFML Ch.7: n_eff = n_samples / avg_hold_bars.
    With 20-bar hold and 1-bar step, ~95% of labels overlap,
    so n_eff is roughly n/20.
    """
    if avg_hold_bars <= 1:
        return n_samples
    return max(1, int(n_samples / avg_hold_bars))


def compute_accuracy_threshold(n_eff: int, confidence: float = 0.95) -> float:
    """Dynamic accuracy gate based on effective sample size.

    Uses binomial test: threshold = 0.5 + z_alpha / (2 * sqrt(n_eff)).
    Larger n_eff -> lower threshold (easier to pass with more data).
    Smaller n_eff -> higher threshold (need stronger signal to be significant).
    """
    z_alpha = _Z_ALPHA_95 if confidence == 0.95 else _Z_ALPHA_99  # noqa: PLR2004
    if n_eff < _MIN_N_EFF_FOR_NORMAL:
        return _TINY_SAMPLE_ACCURACY  # Near-impossible for tiny samples
    threshold = 0.5 + z_alpha / (2 * math.sqrt(n_eff))
    return min(threshold, _MAX_ACCURACY_THRESHOLD)  # Cap at 0.75


def compute_brier_threshold(n_eff: int) -> float:
    """Dynamic Brier score gate.

    Baseline Brier for coin-flip = 0.25.  With small n_eff we demand a
    very low Brier (strict) because we need strong evidence.  As n_eff
    grows, even a modest improvement is significant, so the threshold
    relaxes toward 0.25.

    threshold = min(0.25, 0.15 + 0.05 * sqrt(n_eff / 100))
    """
    if n_eff < _MIN_N_EFF_FOR_NORMAL:
        return _TINY_SAMPLE_BRIER
    relaxation = _BRIER_IMPROVEMENT_RATE * math.sqrt(n_eff) / math.sqrt(_BRIER_REFERENCE_N_EFF)
    return min(_BRIER_COIN_FLIP, _MIN_BRIER_THRESHOLD + relaxation)
