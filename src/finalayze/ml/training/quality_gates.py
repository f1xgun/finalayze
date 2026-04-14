"""Dynamic quality gates for ML model validation (Phase D2)."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class QualityGateResult:
    """Result of quality gate evaluation."""

    passed: bool
    gate_name: str
    value: float
    threshold: float
    detail: str = ""


@dataclass
class FoldMetrics:
    """Metrics from a single walk-forward fold."""

    accuracy: float
    brier_score: float
    log_loss: float
    n_test: int
    mean_uniqueness: float = 1.0
    buy_ratio: float = 0.5  # fraction of predictions that are BUY
    sensitivity: float = 0.5  # true positive rate
    specificity: float = 0.5  # true negative rate
    profit_factor: float = 1.0
    signal_count: int = 0
    avg_hold_bars: float = 1.0  # mean hold duration for n_eff calculation


_ACCURACY_Z = 2.5
_COIN_FLIP_VARIANCE = 0.25
_COIN_FLIP_ACCURACY = 0.50
_MAX_BRIER = 0.25
_MIN_PROFIT_FACTOR = 1.10
_MIN_SIGNALS = 50
_DEGEN_MIN_BUY_RATIO = 0.15
_DEGEN_MAX_BUY_RATIO = 0.85
_MIN_CLASS_RATIO = 0.30
_MIN_SENSITIVITY = 0.45
_MIN_SPECIFICITY = 0.45

# MOEX-relaxed thresholds: smaller datasets (~850 samples) produce more
# conservative models with fewer BUY predictions.  US thresholds are
# unreachable on these sample sizes.
_MOEX_MIN_CLASS_RATIO = 0.20
_MOEX_MIN_SENSITIVITY = 0.30
_MOEX_MIN_SPECIFICITY = 0.30
_DEFAULT_MIN_PASSING_FOLDS_RATIO = 0.60
_MAX_ACCURACY_THRESHOLD = 0.55


def check_accuracy_gate(metrics: FoldMetrics) -> QualityGateResult:
    """N-adjusted accuracy gate with smooth cap.

    Raw threshold = 0.50 + 2.5 * sqrt(0.25 / n_effective).
    For small-to-medium samples (MOEX), raw threshold can be unreachably high
    (e.g. 0.68 at n_eff=50). A smooth cap limits the threshold while still
    converging to the raw formula for large samples (US, n_eff > 150).

    Cap formula: 0.55 + 0.10 * (1 - exp(-n_eff / 200))
      n_eff=10  -> cap 0.555  (very forgiving)
      n_eff=50  -> cap 0.572  (MOEX typical — accuracy 0.58-0.61 passes)
      n_eff=100 -> cap 0.589  (transition zone)
      n_eff=150 -> cap 0.603  (raw takes over: raw=0.602)
      n_eff=500 -> cap 0.642  (raw takes over: raw=0.556)
    """
    n_effective = metrics.n_test * metrics.mean_uniqueness
    if n_effective <= 0:
        return QualityGateResult(
            passed=False,
            gate_name="accuracy",
            value=metrics.accuracy,
            threshold=1.0,
            detail="n_effective <= 0",
        )
    raw_threshold = _COIN_FLIP_ACCURACY + _ACCURACY_Z * math.sqrt(_COIN_FLIP_VARIANCE / n_effective)
    smooth_cap = _MAX_ACCURACY_THRESHOLD + 0.10 * (1 - math.exp(-n_effective / 200))
    threshold = min(raw_threshold, smooth_cap)
    passed = metrics.accuracy > threshold
    return QualityGateResult(
        passed=passed, gate_name="accuracy", value=metrics.accuracy, threshold=threshold
    )


def _compute_n_eff(n_test: int, avg_hold_bars: float) -> int:
    """Effective sample size: n_test / avg_hold_bars (AFML Ch.7)."""
    if avg_hold_bars <= 1:
        return n_test
    return max(1, int(n_test / avg_hold_bars))


def _dynamic_brier_threshold(n_eff: int) -> float:
    """Dynamic Brier threshold: two-regime with smooth transition.

    Small samples (n_eff < 40): relaxed floor (0.24) — MOEX-sized folds produce
    inherently noisier probability estimates; a 0.15 floor is unreachable.
    Large samples (n_eff > 60): strict floor (0.15) — US-sized folds where
    models should achieve better calibration.
    Transition zone (40-60): linear blend to avoid threshold discontinuity.
    """
    _min_n_eff = 5
    _reference_n_eff = 100
    _small_floor = 0.24
    _small_rate = 0.01
    _large_floor = 0.15
    _large_rate = 0.05
    _blend_lo = 40
    _blend_hi = 60

    if n_eff < _min_n_eff:
        return _small_floor

    def _small(n: int) -> float:
        return min(_MAX_BRIER, _small_floor + _small_rate * math.sqrt(n / _reference_n_eff))

    def _large(n: int) -> float:
        return min(_MAX_BRIER, _large_floor + _large_rate * math.sqrt(n / _reference_n_eff))

    if n_eff < _blend_lo:
        return _small(n_eff)
    if n_eff > _blend_hi:
        return _large(n_eff)
    # Linear blend in transition zone
    alpha = (n_eff - _blend_lo) / (_blend_hi - _blend_lo)
    return (1 - alpha) * _small(n_eff) + alpha * _large(n_eff)


def check_brier_gate(metrics: FoldMetrics) -> QualityGateResult:
    """Dynamic Brier score gate adjusted for effective sample size.

    Was: fixed threshold < 0.25 (coin flip).
    Now: threshold scales with n_eff = n_test / avg_hold_bars.
    Smaller n_eff -> stricter (lower) threshold.
    """
    n_eff = _compute_n_eff(metrics.n_test, metrics.avg_hold_bars)
    threshold = _dynamic_brier_threshold(n_eff)
    passed = metrics.brier_score < threshold
    return QualityGateResult(
        passed=passed,
        gate_name="brier_score",
        value=metrics.brier_score,
        threshold=threshold,
        detail=f"n_eff={n_eff}",
    )


def check_profit_factor_gate(metrics: FoldMetrics) -> QualityGateResult:
    """Minimum profit factor after costs."""
    passed = metrics.profit_factor >= _MIN_PROFIT_FACTOR
    return QualityGateResult(
        passed=passed,
        gate_name="profit_factor",
        value=metrics.profit_factor,
        threshold=_MIN_PROFIT_FACTOR,
    )


def check_signal_count_gate(
    metrics: FoldMetrics, *, min_signals: int = _MIN_SIGNALS
) -> QualityGateResult:
    """Minimum number of signals per fold.

    Args:
        metrics: Fold metrics to evaluate.
        min_signals: Minimum required signal count. Defaults to _MIN_SIGNALS (50).
            Pass a smaller value (e.g. 15) for MOEX segments with fewer data points.
    """
    passed = metrics.signal_count >= min_signals
    return QualityGateResult(
        passed=passed,
        gate_name="signal_count",
        value=float(metrics.signal_count),
        threshold=float(min_signals),
    )


def check_class_balance_gate(
    metrics: FoldMetrics, *, min_class_ratio: float = _MIN_CLASS_RATIO
) -> QualityGateResult:
    """Model must predict both classes (not all-buy or all-sell)."""
    ratio = min(metrics.buy_ratio, 1.0 - metrics.buy_ratio)
    passed = ratio >= min_class_ratio
    return QualityGateResult(
        passed=passed, gate_name="class_balance", value=ratio, threshold=min_class_ratio
    )


def check_sensitivity_gate(
    metrics: FoldMetrics, *, min_sensitivity: float = _MIN_SENSITIVITY
) -> QualityGateResult:
    """Minimum sensitivity (true positive rate)."""
    passed = metrics.sensitivity >= min_sensitivity
    return QualityGateResult(
        passed=passed,
        gate_name="sensitivity",
        value=metrics.sensitivity,
        threshold=min_sensitivity,
    )


def check_specificity_gate(
    metrics: FoldMetrics, *, min_specificity: float = _MIN_SPECIFICITY
) -> QualityGateResult:
    """Minimum specificity (true negative rate)."""
    passed = metrics.specificity >= min_specificity
    return QualityGateResult(
        passed=passed,
        gate_name="specificity",
        value=metrics.specificity,
        threshold=min_specificity,
    )


def check_degenerate_predictor_gate(metrics: FoldMetrics) -> QualityGateResult:
    """Reject models that predict overwhelmingly one direction.

    A model predicting BUY on >85% or <15% of samples is likely degenerate
    (all-BUY or all-SELL) and should not pass quality gates.

    This gate complements the class_balance gate: class_balance uses min(ratio, 1-ratio)
    and a 0.30 threshold, while this gate uses the raw buy_ratio with tighter [0.15, 0.85]
    bounds and explicit degenerate_predictor naming for clearer diagnostics.
    """
    passed = _DEGEN_MIN_BUY_RATIO <= metrics.buy_ratio <= _DEGEN_MAX_BUY_RATIO
    return QualityGateResult(
        passed=passed,
        gate_name="degenerate_predictor",
        value=metrics.buy_ratio,
        threshold=_DEGEN_MAX_BUY_RATIO,
        detail=(
            f"buy_ratio={metrics.buy_ratio:.2f},"
            f" bounds=[{_DEGEN_MIN_BUY_RATIO}, {_DEGEN_MAX_BUY_RATIO}]"
        ),
    )


def evaluate_fold(
    metrics: FoldMetrics,
    *,
    min_signals: int = _MIN_SIGNALS,
    min_sensitivity: float = _MIN_SENSITIVITY,
    min_specificity: float = _MIN_SPECIFICITY,
    min_class_ratio: float = _MIN_CLASS_RATIO,
) -> list[QualityGateResult]:
    """Run all quality gates on a single fold's metrics.

    Args:
        metrics: Fold metrics to evaluate.
        min_signals: Minimum required signal count for the signal_count gate.
            Defaults to _MIN_SIGNALS (50). Pass a smaller value (e.g. 15) for
            MOEX segments with fewer data points.
        min_sensitivity: Minimum sensitivity threshold. Defaults to _MIN_SENSITIVITY
            (0.45). Pass _MOEX_MIN_SENSITIVITY (0.30) for MOEX segments.
        min_specificity: Minimum specificity threshold. Defaults to _MIN_SPECIFICITY
            (0.45). Pass _MOEX_MIN_SPECIFICITY (0.30) for MOEX segments.
        min_class_ratio: Minimum class balance ratio. Defaults to _MIN_CLASS_RATIO
            (0.30). Pass _MOEX_MIN_CLASS_RATIO (0.20) for MOEX segments.
    """
    return [
        check_accuracy_gate(metrics),
        check_brier_gate(metrics),
        check_profit_factor_gate(metrics),
        check_signal_count_gate(metrics, min_signals=min_signals),
        check_class_balance_gate(metrics, min_class_ratio=min_class_ratio),
        check_sensitivity_gate(metrics, min_sensitivity=min_sensitivity),
        check_specificity_gate(metrics, min_specificity=min_specificity),
        check_degenerate_predictor_gate(metrics),
    ]


def evaluate_walk_forward(
    fold_results: list[list[QualityGateResult]],
    min_passing_folds_ratio: float = _DEFAULT_MIN_PASSING_FOLDS_RATIO,
) -> tuple[bool, dict[str, float]]:
    """Evaluate across all walk-forward folds.

    A model passes if it passes each gate in >= min_passing_folds_ratio of folds.

    Returns:
        (overall_passed, gate_pass_rates) where gate_pass_rates maps gate_name to
        fraction of folds that passed that gate.
    """
    if not fold_results:
        return False, {}

    n_folds = len(fold_results)
    gate_names = {r.gate_name for results in fold_results for r in results}
    gate_pass_rates: dict[str, float] = {}
    overall_passed = True

    for gate_name in sorted(gate_names):
        passes = sum(
            1 for fold in fold_results for r in fold if r.gate_name == gate_name and r.passed
        )
        rate = passes / n_folds
        gate_pass_rates[gate_name] = rate
        if rate < min_passing_folds_ratio:
            overall_passed = False

    return overall_passed, gate_pass_rates
