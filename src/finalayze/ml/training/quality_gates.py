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
_MIN_CLASS_RATIO = 0.30
_MIN_SENSITIVITY = 0.45
_MIN_SPECIFICITY = 0.45
_DEFAULT_MIN_PASSING_FOLDS_RATIO = 0.60


def check_accuracy_gate(metrics: FoldMetrics) -> QualityGateResult:
    """N-adjusted accuracy gate.

    threshold = 0.50 + 2.5 * sqrt(0.25 / n_effective)
    where n_effective = n_test * mean_uniqueness

    This accounts for sample size and overlapping labels.
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
    threshold = _COIN_FLIP_ACCURACY + _ACCURACY_Z * math.sqrt(_COIN_FLIP_VARIANCE / n_effective)
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
    """Dynamic Brier threshold: stricter for small n_eff, relaxes toward 0.25.

    threshold = min(0.25, 0.15 + 0.05 * sqrt(n_eff / 100))
    """
    _min_n_eff = 5
    _min_brier = 0.15
    _improvement_rate = 0.05
    _reference_n_eff = 100
    if n_eff < _min_n_eff:
        return _min_brier
    relaxation = _improvement_rate * math.sqrt(n_eff) / math.sqrt(_reference_n_eff)
    return min(_MAX_BRIER, _min_brier + relaxation)


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


def check_signal_count_gate(metrics: FoldMetrics) -> QualityGateResult:
    """Minimum number of signals per fold."""
    passed = metrics.signal_count >= _MIN_SIGNALS
    return QualityGateResult(
        passed=passed,
        gate_name="signal_count",
        value=float(metrics.signal_count),
        threshold=float(_MIN_SIGNALS),
    )


def check_class_balance_gate(metrics: FoldMetrics) -> QualityGateResult:
    """Model must predict both classes (not all-buy or all-sell)."""
    ratio = min(metrics.buy_ratio, 1.0 - metrics.buy_ratio)
    passed = ratio >= _MIN_CLASS_RATIO
    return QualityGateResult(
        passed=passed, gate_name="class_balance", value=ratio, threshold=_MIN_CLASS_RATIO
    )


def check_sensitivity_gate(metrics: FoldMetrics) -> QualityGateResult:
    """Minimum sensitivity (true positive rate)."""
    passed = metrics.sensitivity >= _MIN_SENSITIVITY
    return QualityGateResult(
        passed=passed,
        gate_name="sensitivity",
        value=metrics.sensitivity,
        threshold=_MIN_SENSITIVITY,
    )


def check_specificity_gate(metrics: FoldMetrics) -> QualityGateResult:
    """Minimum specificity (true negative rate)."""
    passed = metrics.specificity >= _MIN_SPECIFICITY
    return QualityGateResult(
        passed=passed,
        gate_name="specificity",
        value=metrics.specificity,
        threshold=_MIN_SPECIFICITY,
    )


def evaluate_fold(metrics: FoldMetrics) -> list[QualityGateResult]:
    """Run all quality gates on a single fold's metrics."""
    return [
        check_accuracy_gate(metrics),
        check_brier_gate(metrics),
        check_profit_factor_gate(metrics),
        check_signal_count_gate(metrics),
        check_class_balance_gate(metrics),
        check_sensitivity_gate(metrics),
        check_specificity_gate(metrics),
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
