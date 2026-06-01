"""Regression guard codifying the 62-04 WF-gate reporting artifact (MLDIAG-02).

The Phase-70 H2 audit (70-RESEARCH.md §H2) resolved the 62-04 discrepancy
(``ru_finance`` stdout ``p=0.0162 [PASS]`` vs file ``overall_passed:false``) as a
**reporting artifact of two parallel, independently-computed statistics — NOT a
logic bug**:

1. **Binding gate** — ``finalayze.ml.training.quality_gates.evaluate_walk_forward``
   writes ``overall_passed`` to ``wf_gate_results.json``. It is a per-gate
   fold-ratio gate over 8 sub-gates: for each sub-gate ``rate = passes / n_folds``;
   if ANY ``rate < MOEX_MIN_PASSING_FOLDS_RATIO`` (0.34 = 1/3 of 3 folds),
   ``overall_passed`` is ``False``. ``legit_pass`` downstream is strictly
   ``overall_passed is True and not force_saved`` — it NEVER reads the BH stdout.

2. **Non-binding synthetic-binomtest BH** — ``scripts/training/walk_forward.py``
   ``apply_bh_across_segments`` (:557) FABRICATES trials via
   ``n_correct = int(acc * n_folds * 100)`` (:581), runs
   ``binomtest(n_correct, n_folds*100, 0.5, "greater")`` (:583), applies BH
   (:586), and ``print``s ``{seg}: p=… [PASS/FAIL]`` to stdout (:591). It writes
   only ``bh_passed`` and NEVER touches ``overall_passed``.

These two statistics measure different things and CAN legitimately diverge on the
SAME 3-fold inputs: the binding fold-ratio gate fails while the high-accuracy
synthetic binomtest passes BH. That divergence IS the 62-04 explanation, codified
here as a permanent regression guard — no runtime retrain repro is required.

HONESTY GUARDRAIL (CONTEXT D-05): this test documents the unsound
``int(acc * n_folds * 100)`` fabrication as **cleanup-only**. The synthetic path is
NON-binding, so "fixing" it changes nothing about enablement; recalibrating it (or
``MOEX_MIN_PASSING_FOLDS_RATIO`` / ``BH_FDR``) to FLIP a verdict would be
tune-to-pass / curve-fit and is explicitly NOT done. The fabrication is recorded
honestly, never recalibrated-to-pass. This test modifies NO production gate code.
"""

from __future__ import annotations

import sys
from pathlib import Path

from scipy.stats import binomtest

from finalayze.ml.training.quality_gates import (
    FoldMetrics,
    QualityGateResult,
    evaluate_fold,
    evaluate_walk_forward,
)

# Ensure scripts/ and project root are importable (config/ lives at project root,
# not under src/). Mirrors tests/unit/test_auto_ml_research_moex.py:23-26.
_PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from scripts.training.walk_forward import (  # noqa: E402
    BH_FDR,
    apply_bh_correction,
)

# Named constants — no magic numbers (ruff PLR2004).
_N_FOLDS = 3
_HIGH_ACC = 0.55  # the synthetic-binomtest input; > 0.50 null -> low p-value
_LOW_ACC = 0.40  # below the N-adjusted accuracy threshold (~0.5396 at n=1000)
_PASS_ACC = 0.70  # comfortably above the N-adjusted accuracy threshold
_MIN_RATIO = 0.34  # MOEX_MIN_PASSING_FOLDS_RATIO: 1/3 of 3 folds
_LARGE_N = 1000  # n_test where the accuracy gate threshold is well-defined
# Brier: default _MAX_BRIER=0.25 at avg_hold_bars=1.0 -> 0.15 passes, 0.30 fails.
_PASS_BRIER = 0.15
_FAIL_BRIER = 0.30
# Profit factor: _MIN_PROFIT_FACTOR=1.10 -> 1.50 passes, 0.90 fails.
_PASS_PF = 1.50
_FAIL_PF = 0.90
# The fabricated trial multiplier from walk_forward.py:581-582 (int(acc*n_folds*100)).
_TRIALS_PER_FOLD = 100
_COIN_FLIP_NULL = 0.5


def _make_metrics(**kwargs: object) -> FoldMetrics:
    """Build FoldMetrics with all-passing defaults, overriding via kwargs.

    Defaults keep every NON-targeted sub-gate (signal_count, class_balance,
    sensitivity, specificity, degenerate_predictor) passing in every fold, so the
    only sub-gates that can fail are accuracy / brier_score / profit_factor — the
    exact three that cleared only 1/3 folds in the 62-04 ``ru_blue_chips`` /
    ``ru_finance`` artifacts.
    """
    defaults: dict[str, object] = {
        "accuracy": _PASS_ACC,
        "brier_score": _PASS_BRIER,
        "log_loss": 0.60,
        "n_test": _LARGE_N,
        "mean_uniqueness": 1.0,
        "buy_ratio": 0.50,
        "sensitivity": 0.60,
        "specificity": 0.60,
        "profit_factor": _PASS_PF,
        "signal_count": 200,
    }
    defaults.update(kwargs)
    return FoldMetrics(**defaults)  # type: ignore[arg-type]


def _build_one_of_three_folds() -> list[list[QualityGateResult]]:
    """Build 3 folds where accuracy / brier / profit_factor each pass exactly 1/3.

    Fold 0: accuracy passes, brier + profit_factor fail.
    Fold 1: brier passes, accuracy + profit_factor fail.
    Fold 2: profit_factor passes, accuracy + brier fail.

    => each of the three sub-gates clears 1/3 = 33.3% < MOEX_MIN_PASSING_FOLDS_RATIO
    (0.34), so the binding evaluate_walk_forward returns overall_passed=False. This
    mirrors the 62-04 ru_blue_chips file (accuracy 0.333 / brier 0.333 / pf 0.667).
    """
    fold0 = _make_metrics(accuracy=_PASS_ACC, brier_score=_FAIL_BRIER, profit_factor=_FAIL_PF)
    fold1 = _make_metrics(accuracy=_LOW_ACC, brier_score=_PASS_BRIER, profit_factor=_FAIL_PF)
    fold2 = _make_metrics(accuracy=_LOW_ACC, brier_score=_FAIL_BRIER, profit_factor=_PASS_PF)
    return [evaluate_fold(fold0), evaluate_fold(fold1), evaluate_fold(fold2)]


def _synthetic_bh_pass(acc: float, n_folds: int) -> bool:
    """Replicate the NON-binding synthetic-binomtest BH path EXACTLY.

    Verbatim from scripts/training/walk_forward.py:581-586:
        n_correct = int(acc * n_folds * 100)   # fabricated trials (UNSOUND)
        n_total   = n_folds * 100
        p         = binomtest(n_correct, n_total, 0.5, "greater").pvalue
        passes    = apply_bh_correction([p], fdr=BH_FDR)

    The ``int(acc * n_folds * 100)`` fabrication is statistically unsound (it
    invents ``n_folds*100`` Bernoulli trials that never happened). It is NON-binding
    and flagged as cleanup-only (Plan 04 findings doc) — NEVER recalibrated-to-pass
    per D-05. Replicated here only to prove the divergence, not to "fix" it.
    """
    n_correct = int(acc * n_folds * _TRIALS_PER_FOLD)
    n_total = n_folds * _TRIALS_PER_FOLD
    p = float(binomtest(n_correct, n_total, _COIN_FLIP_NULL, alternative="greater").pvalue)
    passes = apply_bh_correction([p], fdr=BH_FDR)
    return passes[0]


class TestGateDiscrepancy:
    """Codifies the 62-04 reporting artifact: binding gate vs synthetic-BH divergence."""

    def test_binding_gate_and_synthetic_bh_can_diverge(self) -> None:
        """On IDENTICAL 3-fold inputs the two parallel statistics CAN disagree.

        This IS the 62-04 explanation as a regression guard: the binding
        fold-ratio gate returns overall_passed=False (each of accuracy / brier /
        profit_factor clears only 1/3 = 33.3% < MOEX_MIN_PASSING_FOLDS_RATIO=0.34)
        while the non-binding synthetic-binomtest BH returns PASS at high accuracy.
        The stdout ``[PASS]`` (62-04 ru_finance p=0.0162) is informational noise;
        the verdict correctly keys off ``overall_passed`` (an ARTIFACT, not a bug).
        """
        fold_results = _build_one_of_three_folds()

        # (1) Binding statistic: the gate that writes overall_passed.
        overall_passed, gate_pass_rates = evaluate_walk_forward(
            fold_results, min_passing_folds_ratio=_MIN_RATIO
        )
        assert overall_passed is False
        # The three targeted sub-gates each clear only 1/3 of folds.
        one_of_three = 1.0 / _N_FOLDS
        for gate_name in ("accuracy", "brier_score", "profit_factor"):
            assert abs(gate_pass_rates[gate_name] - one_of_three) < 1e-9
            assert gate_pass_rates[gate_name] < _MIN_RATIO

        # (2) Non-binding synthetic statistic on the SAME high-accuracy input.
        bh_pass = _synthetic_bh_pass(_HIGH_ACC, _N_FOLDS)
        assert bh_pass is True

        # (3) The divergence, asserted simultaneously — the 62-04 artifact.
        assert overall_passed is False and bh_pass is True

    def test_legit_pass_keys_off_overall_passed_only(self) -> None:
        """The binding key is overall_passed; the synthetic BH stdout is noise.

        Even when the synthetic-binomtest BH would print ``[PASS]`` (high accuracy
        -> p < BH_FDR), the enablement verdict (legit_pass) is
        ``overall_passed is True and not force_saved``. Here overall_passed is
        False, so the segment stays DISC regardless of the BH stdout line — there
        is no path by which the informational ``[PASS]`` enables a model.
        """
        fold_results = _build_one_of_three_folds()
        overall_passed, _ = evaluate_walk_forward(fold_results, min_passing_folds_ratio=_MIN_RATIO)
        force_saved = False  # this spike NEVER force-saves (D-05)

        bh_pass = _synthetic_bh_pass(_HIGH_ACC, _N_FOLDS)
        legit_pass = overall_passed is True and not force_saved

        # The synthetic BH says PASS, but the binding verdict is DISC.
        assert bh_pass is True
        assert legit_pass is False
