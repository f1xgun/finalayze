"""Tests for ML quality gates (Phase D2)."""

from __future__ import annotations

import math

from finalayze.ml.training.quality_gates import (
    FoldMetrics,
    check_accuracy_gate,
    check_brier_gate,
    check_class_balance_gate,
    check_profit_factor_gate,
    check_sensitivity_gate,
    check_signal_count_gate,
    check_specificity_gate,
    evaluate_fold,
    evaluate_walk_forward,
)

_LARGE_SAMPLE = 1000
_SMALL_SAMPLE = 50
_SEVEN_GATES = 7


def _make_metrics(**kwargs: object) -> FoldMetrics:
    """Create FoldMetrics with sensible defaults, overriding with kwargs."""
    defaults: dict[str, object] = {
        "accuracy": 0.55,
        "brier_score": 0.20,
        "log_loss": 0.60,
        "n_test": _LARGE_SAMPLE,
        "mean_uniqueness": 1.0,
        "buy_ratio": 0.50,
        "sensitivity": 0.50,
        "specificity": 0.50,
        "profit_factor": 1.20,
        "signal_count": 100,
    }
    defaults.update(kwargs)
    return FoldMetrics(**defaults)  # type: ignore[arg-type]


# --- Accuracy gate ---


class TestAccuracyGate:
    def test_large_sample_passes(self) -> None:
        """With n_test=1000, threshold ~ 0.5396; accuracy 0.60 passes."""
        metrics = _make_metrics(accuracy=0.60, n_test=_LARGE_SAMPLE)
        result = check_accuracy_gate(metrics)
        expected_threshold = 0.50 + 2.5 * math.sqrt(0.25 / _LARGE_SAMPLE)
        assert result.passed is True
        assert result.gate_name == "accuracy"
        assert abs(result.threshold - expected_threshold) < 1e-9

    def test_small_sample_fails(self) -> None:
        """With n_test=50, threshold ~ 0.6768; accuracy 0.60 fails."""
        metrics = _make_metrics(accuracy=0.60, n_test=_SMALL_SAMPLE)
        result = check_accuracy_gate(metrics)
        expected_threshold = 0.50 + 2.5 * math.sqrt(0.25 / _SMALL_SAMPLE)
        assert result.passed is False
        assert result.threshold > 0.60
        assert abs(result.threshold - expected_threshold) < 1e-9

    def test_zero_n_effective_fails(self) -> None:
        """n_test=0 yields n_effective=0, gate fails."""
        metrics = _make_metrics(accuracy=0.99, n_test=0)
        result = check_accuracy_gate(metrics)
        assert result.passed is False
        assert result.detail == "n_effective <= 0"
        assert result.threshold == 1.0


# --- Brier gate ---


class TestBrierGate:
    def test_pass(self) -> None:
        metrics = _make_metrics(brier_score=0.20)
        result = check_brier_gate(metrics)
        assert result.passed is True
        assert result.value == 0.20  # noqa: PLR2004

    def test_fail(self) -> None:
        metrics = _make_metrics(brier_score=0.30)
        result = check_brier_gate(metrics)
        assert result.passed is False
        assert result.value == 0.30  # noqa: PLR2004

    def test_dynamic_threshold_with_high_overlap(self) -> None:
        """With avg_hold_bars=20, n_eff is much smaller, so Brier threshold is stricter."""
        _overlapping_hold_bars = 20.0
        metrics_overlap = _make_metrics(
            brier_score=0.22,
            n_test=_LARGE_SAMPLE,
            avg_hold_bars=_overlapping_hold_bars,
        )
        # n_eff = 1000 / 20 = 50 -> threshold ~ 0.185
        result_overlap = check_brier_gate(metrics_overlap)
        assert result_overlap.passed is False  # 0.22 > ~0.185

        metrics_no_overlap = _make_metrics(
            brier_score=0.22,
            n_test=_LARGE_SAMPLE,
            avg_hold_bars=1.0,
        )
        # n_eff = 1000 -> threshold = 0.25
        result_no_overlap = check_brier_gate(metrics_no_overlap)
        assert result_no_overlap.passed is True  # 0.22 < 0.25

    def test_dynamic_threshold_stricter_for_small_n_eff(self) -> None:
        """Smaller n_eff should give a lower (stricter) Brier threshold."""
        _high_overlap = 40.0
        _low_overlap = 2.0
        small_n_eff_metrics = _make_metrics(n_test=_LARGE_SAMPLE, avg_hold_bars=_high_overlap)
        large_n_eff_metrics = _make_metrics(n_test=_LARGE_SAMPLE, avg_hold_bars=_low_overlap)
        small_result = check_brier_gate(small_n_eff_metrics)
        large_result = check_brier_gate(large_n_eff_metrics)
        assert small_result.threshold < large_result.threshold


# --- Profit factor gate ---


class TestProfitFactorGate:
    def test_pass(self) -> None:
        metrics = _make_metrics(profit_factor=1.20)
        result = check_profit_factor_gate(metrics)
        assert result.passed is True

    def test_fail(self) -> None:
        metrics = _make_metrics(profit_factor=0.90)
        result = check_profit_factor_gate(metrics)
        assert result.passed is False


# --- Signal count gate ---


class TestSignalCountGate:
    def test_pass(self) -> None:
        _passing_signal_count = 100
        metrics = _make_metrics(signal_count=_passing_signal_count)
        result = check_signal_count_gate(metrics)
        assert result.passed is True
        assert result.value == float(_passing_signal_count)

    def test_fail(self) -> None:
        _failing_signal_count = 30
        metrics = _make_metrics(signal_count=_failing_signal_count)
        result = check_signal_count_gate(metrics)
        assert result.passed is False


# --- Class balance gate ---


class TestClassBalanceGate:
    def test_pass(self) -> None:
        """buy_ratio=0.55 -> min(0.55, 0.45)=0.45 >= 0.30 passes."""
        metrics = _make_metrics(buy_ratio=0.55)
        result = check_class_balance_gate(metrics)
        assert result.passed is True
        _expected_min_ratio = 0.45
        assert abs(result.value - _expected_min_ratio) < 1e-9

    def test_fail(self) -> None:
        """buy_ratio=0.95 -> min(0.95, 0.05)=0.05 < 0.30 fails."""
        metrics = _make_metrics(buy_ratio=0.95)
        result = check_class_balance_gate(metrics)
        assert result.passed is False
        _expected_min_ratio = 0.05
        assert abs(result.value - _expected_min_ratio) < 1e-9


# --- Sensitivity gate ---


class TestSensitivityGate:
    def test_pass(self) -> None:
        metrics = _make_metrics(sensitivity=0.50)
        result = check_sensitivity_gate(metrics)
        assert result.passed is True

    def test_fail(self) -> None:
        metrics = _make_metrics(sensitivity=0.30)
        result = check_sensitivity_gate(metrics)
        assert result.passed is False


# --- Specificity gate ---


class TestSpecificityGate:
    def test_pass(self) -> None:
        metrics = _make_metrics(specificity=0.50)
        result = check_specificity_gate(metrics)
        assert result.passed is True

    def test_fail(self) -> None:
        metrics = _make_metrics(specificity=0.30)
        result = check_specificity_gate(metrics)
        assert result.passed is False


# --- evaluate_fold ---


class TestEvaluateFold:
    def test_returns_all_gates(self) -> None:
        metrics = _make_metrics()
        results = evaluate_fold(metrics)
        assert len(results) == _SEVEN_GATES
        gate_names = {r.gate_name for r in results}
        assert gate_names == {
            "accuracy",
            "brier_score",
            "profit_factor",
            "signal_count",
            "class_balance",
            "sensitivity",
            "specificity",
        }


# --- evaluate_walk_forward ---


class TestEvaluateWalkForward:
    def test_all_pass(self) -> None:
        """All folds pass all gates -> overall passes."""
        good_metrics = _make_metrics(
            accuracy=0.70,
            brier_score=0.15,
            profit_factor=1.50,
            signal_count=200,
            buy_ratio=0.50,
            sensitivity=0.60,
            specificity=0.60,
        )
        _n_folds = 3
        fold_results = [evaluate_fold(good_metrics) for _ in range(_n_folds)]
        overall, rates = evaluate_walk_forward(fold_results)
        assert overall is True
        for rate in rates.values():
            assert rate == 1.0

    def test_some_fail_still_passes(self) -> None:
        """1 of 3 folds fails accuracy -> rate=0.67 > 0.60 -> passes."""
        good = _make_metrics(accuracy=0.70, n_test=_LARGE_SAMPLE)
        bad = _make_metrics(accuracy=0.40, n_test=_LARGE_SAMPLE)
        fold_results = [
            evaluate_fold(good),
            evaluate_fold(good),
            evaluate_fold(bad),
        ]
        overall, rates = evaluate_walk_forward(fold_results)
        _expected_pass_rate = 2.0 / 3.0
        assert abs(rates["accuracy"] - _expected_pass_rate) < 1e-9
        assert overall is True

    def test_too_many_fail(self) -> None:
        """2 of 3 folds fail accuracy -> rate=0.33 < 0.60 -> overall fails."""
        good = _make_metrics(accuracy=0.70, n_test=_LARGE_SAMPLE)
        bad = _make_metrics(accuracy=0.40, n_test=_LARGE_SAMPLE)
        fold_results = [
            evaluate_fold(good),
            evaluate_fold(bad),
            evaluate_fold(bad),
        ]
        overall, rates = evaluate_walk_forward(fold_results)
        _expected_fail_rate = 1.0 / 3.0
        assert abs(rates["accuracy"] - _expected_fail_rate) < 1e-9
        assert overall is False

    def test_empty_fold_results(self) -> None:
        """Empty fold_results -> (False, {})."""
        overall, rates = evaluate_walk_forward([])
        assert overall is False
        assert rates == {}
