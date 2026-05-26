"""Walk-forward validation training pipeline.

Implements walk-forward cross-validation with calendar-date splitting,
BH multiple testing correction, and quality gate enforcement.
"""

from __future__ import annotations

import json
from datetime import timedelta
from typing import TYPE_CHECKING

from scripts.training.calibration import fit_and_save_meta_learner
from scripts.training.data_loader import is_moex_segment
from scripts.training.dataset_builder import (
    _WINDOW_SIZE,
    LABEL_MODE_TRIPLE_BARRIER,
    build_dataset_with_timestamps,
    compute_uniqueness_from_hold_bars,
)
from scripts.training.model_trainer import (
    FEAT_SEL_EFFICIENT,
    get_catboost_depth,
    get_max_features,
    get_xgboost_max_depth,
    select_features,
)
from scripts.training.quality import compute_n_eff
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

from finalayze.ml.models.catboost_model import CatBoostModel
from finalayze.ml.models.lightgbm_model import LightGBMModel
from finalayze.ml.models.xgboost_model import XGBoostModel
from finalayze.ml.training.quality_gates import FoldMetrics
from finalayze.ml.training.sample_weights import compute_decay_weights

if TYPE_CHECKING:
    from datetime import datetime
    from pathlib import Path

    from config.settings import Settings

    from finalayze.core.schemas import MarketContext

# Walk-forward parameters (D1)
WF_TRAIN_MONTHS = 12
WF_CAL_MONTHS = 2
WF_TEST_MONTHS = 4
WF_STEP_MONTHS = 3

# MOEX walk-forward: shorter windows to fit within 2-year lookback
MOEX_WF_TRAIN_MONTHS = 8
MOEX_WF_CAL_MONTHS = 1
MOEX_WF_TEST_MONTHS = 3
MOEX_WF_STEP_MONTHS = 2

# S2.2: WF purge gap is measured in CALENDAR DAYS (timedelta below), distinct
# from PURGE_GAP_BARS in dataset_builder.py which is a sample-index offset.
# Safety: max label horizon is TB_MAX_HOLD = 20 bars ≈ 30 calendar days on
# daily candles; both values below leave a margin above that.
MOEX_PURGE_GAP_DAYS = 40
_US_PURGE_GAP_DAYS = 80
# Back-compat alias for tests that imported MOEX_PURGE_GAP.
MOEX_PURGE_GAP = MOEX_PURGE_GAP_DAYS

# BH correction (D3)
BH_FDR = 0.10

# MOEX walk-forward has only 3 folds; 60% ratio means 2/3 must pass which is
# statistically harsher than 60% with 11 US folds. 34% (1/3) is reasonable.
MOEX_MIN_PASSING_FOLDS_RATIO = 0.34


def generate_walk_forward_folds(
    timestamps: list[datetime],
    segment_id: str | None = None,
) -> list[tuple[list[int], list[int], list[int]]]:
    """Generate walk-forward fold indices split by calendar date (D4).

    Each fold has: train indices, calibration indices, test indices.
    Purge gaps are applied between splits to prevent label leakage.
    MOEX segments use shorter windows to fit within 2-year lookback.

    Returns list of (train_idx, cal_idx, test_idx) tuples.
    """
    if not timestamps:
        return []

    is_moex = segment_id is not None and is_moex_segment(segment_id)
    train_months = MOEX_WF_TRAIN_MONTHS if is_moex else WF_TRAIN_MONTHS
    cal_months = MOEX_WF_CAL_MONTHS if is_moex else WF_CAL_MONTHS
    test_months = MOEX_WF_TEST_MONTHS if is_moex else WF_TEST_MONTHS
    step_months = MOEX_WF_STEP_MONTHS if is_moex else WF_STEP_MONTHS
    purge_gap = MOEX_PURGE_GAP_DAYS if is_moex else _US_PURGE_GAP_DAYS

    start_date = timestamps[0]
    end_date = timestamps[-1]

    folds: list[tuple[list[int], list[int], list[int]]] = []
    fold_start = start_date

    while True:
        train_end = fold_start + timedelta(days=train_months * 30)
        purge1_end = train_end + timedelta(days=purge_gap)
        cal_end = purge1_end + timedelta(days=cal_months * 30)
        purge2_end = cal_end + timedelta(days=purge_gap)
        test_end = purge2_end + timedelta(days=test_months * 30)

        if test_end > end_date + timedelta(days=1):
            break

        # Calendar-date split (D4): indices by date range, not row index
        train_idx = [i for i, ts in enumerate(timestamps) if fold_start <= ts < train_end]
        cal_idx = [i for i, ts in enumerate(timestamps) if purge1_end <= ts < cal_end]
        test_idx = [i for i, ts in enumerate(timestamps) if purge2_end <= ts < test_end]

        if train_idx and test_idx:
            folds.append((train_idx, cal_idx, test_idx))

        fold_start += timedelta(days=step_months * 30)

    return folds


def apply_bh_correction(
    p_values: list[float],
    fdr: float = BH_FDR,
) -> list[bool]:
    """Apply Benjamini-Hochberg FDR correction (D3).

    Returns a list of booleans: True if the model passes at that index.
    """
    if not p_values:
        return []

    n = len(p_values)
    # Sort p-values with original indices
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    results = [False] * n

    for rank, (orig_idx, pval) in enumerate(indexed, start=1):
        threshold = (rank / n) * fdr
        if pval <= threshold:
            results[orig_idx] = True
        else:
            # Once we fail, all higher p-values also fail
            break

    return results


def evaluate_fold_metrics(
    models: list[XGBoostModel | LightGBMModel | CatBoostModel],
    test_features: list[dict[str, float]],
    test_labels: list[int],
    mean_uniqueness: float = 1.0,
    avg_hold_bars: float = 1.0,
    calibrator: object | None = None,
) -> FoldMetrics:
    """Evaluate models on a test fold and compute FoldMetrics for quality gates."""
    probas_all: list[float] = []
    for feat in test_features:
        probs = []
        for m in models:
            trained = getattr(m, "_trained", None) or getattr(m, "_model", None)
            if trained is None:
                continue
            try:
                probs.append(m.predict_proba(feat))
            except Exception:
                continue
        probas_all.append(sum(probs) / len(probs) if probs else 0.5)

    preds = [round(p) for p in probas_all]
    n_test = len(test_labels)
    n_pos = sum(test_labels)
    n_neg = n_test - n_pos

    acc = float(accuracy_score(test_labels, preds)) if n_test > 0 else 0.5

    # Compute Brier score: use calibrated probabilities if calibrator available
    if calibrator is not None and hasattr(calibrator, "predict_proba"):
        try:
            import numpy as np_  # noqa: PLC0415

            calibrated_probas = calibrator.predict_proba(
                np_.array(probas_all, dtype=np_.float64)
            ).tolist()
            brier = float(brier_score_loss(test_labels, calibrated_probas)) if n_test > 0 else 0.25
        except Exception:
            brier = float(brier_score_loss(test_labels, probas_all)) if n_test > 0 else 0.25
    else:
        brier = float(brier_score_loss(test_labels, probas_all)) if n_test > 0 else 0.25

    # Sensitivity / specificity
    tp = sum(1 for p, y in zip(preds, test_labels, strict=True) if p == 1 and y == 1)
    tn = sum(1 for p, y in zip(preds, test_labels, strict=True) if p == 0 and y == 0)
    sensitivity = tp / n_pos if n_pos > 0 else 0.0
    specificity = tn / n_neg if n_neg > 0 else 0.0

    buy_count = sum(preds)
    buy_ratio = buy_count / n_test if n_test > 0 else 0.5

    # Compute profit factor from BUY predictions vs actual labels
    _pf_threshold = 0.55
    gross_profit = 0.0
    gross_loss = 0.0
    for prob, label in zip(probas_all, test_labels, strict=True):
        if prob >= _pf_threshold:  # model predicts BUY
            if label == 1:
                gross_profit += 1.0
            else:
                gross_loss += 1.0
    profit_factor = (
        gross_profit / gross_loss if gross_loss > 0 else (2.0 if gross_profit > 0 else 1.0)
    )

    return FoldMetrics(
        accuracy=acc,
        brier_score=brier,
        log_loss=float(log_loss(test_labels, probas_all, labels=[0, 1])) if n_test > 0 else 1.0,
        n_test=n_test,
        mean_uniqueness=mean_uniqueness,
        buy_ratio=buy_ratio,
        sensitivity=sensitivity,
        specificity=specificity,
        profit_factor=profit_factor,
        signal_count=n_test,
        avg_hold_bars=avg_hold_bars,
    )


def train_walk_forward(  # noqa: PLR0912, PLR0915
    segment_id: str,
    symbols: list[str],
    output_dir: Path,
    settings: Settings | None = None,
    label_mode: str = LABEL_MODE_TRIPLE_BARRIER,
    *,
    excess_returns: bool = False,
    force_save: bool = False,
    seq_bootstrap: bool = True,
    market_context: MarketContext | None = None,
    feat_sel_mode: str = FEAT_SEL_EFFICIENT,
) -> dict[str, float] | None:
    """Train models using walk-forward validation (D1).

    Aligned with backtest walk-forward: 12mo train, 2mo cal, 4mo test, 3mo step.
    Returns per-gate pass rates, or None if insufficient data.

    If quality gates fail and force_save is False, models are NOT saved (only
    gate results are persisted for diagnostics). Use force_save=True to override.
    When market_context is provided, ambient MOEX/cross-asset data is sliced per
    training window to prevent look-ahead bias.
    """
    import numpy as _np  # noqa: PLC0415
    from config.settings import Settings as _Settings  # noqa: PLC0415

    from finalayze.ml.training.quality_gates import (  # noqa: PLC0415
        evaluate_fold,
        evaluate_walk_forward,
    )
    from finalayze.ml.training.sample_weights import sequential_bootstrap  # noqa: PLC0415

    if settings is None:
        settings = _Settings()

    print(f"\n[{segment_id}] Walk-forward training (label_mode={label_mode})...")
    features, labels, barrier_weights, hold_bars, timestamps = build_dataset_with_timestamps(
        segment_id,
        symbols,
        settings,
        label_mode,
        excess_returns=excess_returns,
        market_context=market_context,
    )
    if not features:
        print(f"[{segment_id}] No samples -- skipping.")
        return None

    folds = generate_walk_forward_folds(timestamps, segment_id=segment_id)
    if not folds:
        is_moex = is_moex_segment(segment_id)
        min_months = (
            (MOEX_WF_TRAIN_MONTHS + MOEX_WF_CAL_MONTHS + MOEX_WF_TEST_MONTHS)
            if is_moex
            else (WF_TRAIN_MONTHS + WF_CAL_MONTHS + WF_TEST_MONTHS)
        )
        print(f"[{segment_id}] No valid WF folds (need {min_months}+ months of data).")
        return None

    print(f"[{segment_id}] {len(folds)} walk-forward folds")

    # S2.1 (Phase-46 stability): select features ONCE on the union of all
    # fold training indices, not per-fold. Per-fold selection makes the
    # final saved feature list represent only the last fold and introduces
    # spurious model churn. Selection still excludes every test row → no
    # look-ahead. Equivalent fix already lives in auto_ml_research.py:787-805.
    import pandas as pd  # noqa: PLC0415

    union_train_indices: set[int] = set()
    for _train_idx, _cal_idx, _test_idx in folds:
        union_train_indices.update(_train_idx)
    sorted_train_indices = sorted(union_train_indices)
    union_train_df = pd.DataFrame([features[i] for i in sorted_train_indices])
    union_train_series = pd.Series([labels[i] for i in sorted_train_indices])
    max_feats = get_max_features(segment_id)
    selected = select_features(union_train_df, union_train_series, max_feats, mode=feat_sel_mode)
    print(
        f"[{segment_id}] feature_selection_stable: "
        f"{len(selected) if selected else 0} features from {len(sorted_train_indices)} "
        f"union-of-train rows"
    )

    all_fold_results = []
    last_acc = 0.0
    best_models: list[XGBoostModel | LightGBMModel | CatBoostModel] | None = None
    best_selected_features: list[str] | None = selected or None
    best_test_f: list[dict[str, float]] = []
    best_test_l: list[int] = []
    best_train_l: list[int] = []

    for fold_idx, (train_idx, cal_idx, test_idx) in enumerate(folds):
        train_f = [features[i] for i in train_idx]
        train_l = [labels[i] for i in train_idx]
        cal_f = [features[i] for i in cal_idx]
        cal_l = [labels[i] for i in cal_idx]
        test_f = [features[i] for i in test_idx]
        test_l = [labels[i] for i in test_idx]

        if len(train_f) < _WINDOW_SIZE:
            print(f"[{segment_id}] Fold {fold_idx}: too few train ({len(train_f)}), skip.")
            continue

        if selected:
            train_f = [{k: row[k] for k in selected} for row in train_f]
            cal_f = [{k: row[k] for k in selected} for row in cal_f]
            test_f = [{k: row[k] for k in selected} for row in test_f]

        # Sample weights
        decay_w = compute_decay_weights(len(train_f))
        train_hb: list[int] = []
        if hold_bars is not None:
            train_hb = [hold_bars[i] for i in train_idx if i < len(hold_bars)]
            if train_hb:
                uniq = compute_uniqueness_from_hold_bars(train_hb)
                u_mean = float(uniq.mean()) if len(uniq) > 0 else 1.0
                uniq = uniq / u_mean if u_mean > 0 else uniq
            else:
                uniq = _np.ones(len(train_f), dtype=_np.float64)
        else:
            uniq = _np.ones(len(train_f), dtype=_np.float64)

        if barrier_weights is not None:
            bw_idx = [i for i in train_idx if i < len(barrier_weights)]
            train_bw = _np.array([barrier_weights[i] for i in bw_idx])
            dampened = _np.sqrt(_np.abs(train_bw))
            bw_mean = float(dampened.mean()) if len(dampened) > 0 else 1.0
            norm_bw = dampened / bw_mean if bw_mean > 0 else dampened
        else:
            norm_bw = _np.ones(len(train_f), dtype=_np.float64)

        sw = decay_w * uniq[: len(decay_w)] * norm_bw[: len(decay_w)]

        # Sequential bootstrapping: debias overlapping labels (AFML Ch. 4)
        if seq_bootstrap and hold_bars is not None and train_hb:
            sb_starts = _np.arange(len(train_f), dtype=_np.int64)
            sb_holds = _np.array(train_hb[: len(train_f)], dtype=_np.int64)
            sb_n = len(train_f)
            sb_indices = sequential_bootstrap(sb_starts, sb_holds, sb_n)
            train_f = [train_f[i] for i in sb_indices]
            train_l = [train_l[i] for i in sb_indices]
            sw = sw[sb_indices]
            print(
                f"[{segment_id}] Fold {fold_idx}: sequential bootstrap "
                f"({sb_n} draws, {len(set(sb_indices))} unique)"
            )

        # Train models
        xgb = XGBoostModel(segment_id=segment_id, max_depth=get_xgboost_max_depth(segment_id))
        lgbm = LightGBMModel(segment_id=segment_id)
        cat = CatBoostModel(segment_id=segment_id, depth=get_catboost_depth(segment_id))

        xgb.fit(train_f, train_l, sample_weight=sw)
        lgbm.fit(train_f, train_l, sample_weight=sw)
        cat.fit(train_f, train_l, sample_weight=sw)

        models = [xgb, lgbm, cat]
        mean_uniq = float(uniq.mean()) if len(uniq) > 0 else 1.0

        # Compute avg hold bars for the test fold (for dynamic quality gates)
        if hold_bars is not None:
            test_hb = [hold_bars[i] for i in test_idx if i < len(hold_bars)]
            fold_avg_hold = float(_np.mean(test_hb)) if test_hb else 1.0
        else:
            fold_avg_hold = 1.0

        # Fit per-fold calibrator on calibration split for Brier evaluation
        fold_calibrator = None
        if cal_f:
            from finalayze.ml.calibration import EnsembleCalibrator as _EnsCalib  # noqa: PLC0415

            cal_raw_probas = []
            for feat in cal_f:
                cprobs = []
                for m in models:
                    trained = getattr(m, "_trained", None) or getattr(m, "_model", None)
                    if trained is None:
                        continue
                    try:
                        cprobs.append(m.predict_proba(feat))
                    except Exception:
                        continue
                cal_raw_probas.append(sum(cprobs) / len(cprobs) if cprobs else 0.5)

            _fc = _EnsCalib()
            _fc.fit(_np.array(cal_raw_probas), _np.array(cal_l))
            if _fc.is_fitted:
                fold_calibrator = _fc

        # Evaluate on test fold
        if test_f:
            fold_metrics = evaluate_fold_metrics(
                models,
                test_f,
                test_l,
                mean_uniq,
                avg_hold_bars=fold_avg_hold,
                calibrator=fold_calibrator,
            )
            _is_moex_seg = is_moex_segment(segment_id)
            gate_results = evaluate_fold(
                fold_metrics,
                min_sensitivity=0.30 if _is_moex_seg else 0.45,
                min_specificity=0.30 if _is_moex_seg else 0.45,
                min_class_ratio=0.20 if _is_moex_seg else 0.30,
            )
            all_fold_results.append(gate_results)

            passed_count = sum(1 for r in gate_results if r.passed)
            total_gates = len(gate_results)
            fold_n_eff = compute_n_eff(len(test_f), fold_avg_hold)
            print(
                f"[{segment_id}] Fold {fold_idx}: acc={fold_metrics.accuracy:.3f}, "
                f"brier={fold_metrics.brier_score:.3f}, "
                f"n_eff={fold_n_eff}, "
                f"gates={passed_count}/{total_gates}, "
                f"train={len(train_f)}, cal={len(cal_f)}, test={len(test_f)}"
            )

            # Always use the last fold (most temporally recent) -- no cherry-picking
            last_acc = fold_metrics.accuracy
            best_models = models
            best_selected_features = selected
            best_test_f = test_f
            best_test_l = test_l
            best_train_l = train_l

    if not all_fold_results:
        print(f"[{segment_id}] No folds produced results.")
        return None

    min_ratio = MOEX_MIN_PASSING_FOLDS_RATIO if is_moex_segment(segment_id) else 0.60
    overall_passed, gate_pass_rates = evaluate_walk_forward(
        all_fold_results, min_passing_folds_ratio=min_ratio
    )

    status_str = "PASS" if overall_passed else "FAIL"
    print(
        f"\n[{segment_id}] Walk-forward results (overall: {status_str}, min_ratio={min_ratio:.0%}):"
    )
    for gate_name, rate in sorted(gate_pass_rates.items()):
        status = "PASS" if rate >= min_ratio else "FAIL"
        print(f"  {gate_name:>20s}: {rate:.1%} [{status}]")

    # Always save gate results for diagnostics (even when models are not saved)
    if best_models:
        segment_dir = output_dir / segment_id
        segment_dir.mkdir(parents=True, exist_ok=True)

        # S2.3: mark force-saved artefacts so the loader can warn at runtime
        # and audit tools can spot bypassed gates without re-reading
        # overall_passed.
        will_force_save = not overall_passed and force_save

        gate_results_path = segment_dir / "wf_gate_results.json"
        gate_results_path.write_text(
            json.dumps(
                {
                    "overall_passed": overall_passed,
                    "gate_pass_rates": gate_pass_rates,
                    "n_folds": len(all_fold_results),
                    "best_accuracy": last_acc,
                    "force_saved": will_force_save,
                },
                indent=2,
            )
        )

    # Quality gate enforcement: skip saving models if gates failed
    if not overall_passed and not force_save:
        print(
            f"[{segment_id}] Quality gates FAILED -- models NOT saved. "
            f"Use --force-save to override."
        )
        return gate_pass_rates

    # Save best models from walk-forward
    if best_models:
        segment_dir = output_dir / segment_id
        segment_dir.mkdir(parents=True, exist_ok=True)

        if not overall_passed and force_save:
            print(f"[{segment_id}] Quality gates FAILED but --force-save is set, saving anyway.")

        best_models[0].save(segment_dir / "xgb.pkl")
        best_models[1].save(segment_dir / "lgbm.pkl")
        best_models[2].save(segment_dir / "catboost.pkl")  # type: ignore[union-attr]

        if best_selected_features:
            (segment_dir / "selected_features.json").write_text(json.dumps(best_selected_features))

        # Compute and save model weights from best fold's test evaluation
        if best_test_f and best_test_l:
            from sklearn.metrics import accuracy_score as _acc  # noqa: PLC0415

            model_accs: dict[str, float] = {}
            names = ["xgboost", "lightgbm", "catboost"]
            for m, name in zip(best_models, names, strict=True):
                probas = [m.predict_proba(f) for f in best_test_f]
                preds = [round(p) for p in probas]
                model_accs[name] = float(_acc(best_test_l, preds))
            # Compute squared-edge weights: max(0, acc - 0.50)^2
            model_weights: dict[str, float] = {}
            for name, acc_val in model_accs.items():
                model_weights[name] = max(0.0, acc_val - 0.50) ** 2
        else:
            model_weights = {"xgboost": 0.33, "lightgbm": 0.33, "catboost": 0.34}
        weights_path = segment_dir / "model_weights.json"
        weights_path.write_text(json.dumps(model_weights, indent=2))
        print(f"[{segment_id}] Saved model_weights.json: {model_weights}")

        # Compute base_rate from best fold's training labels only (no test data leakage)
        positive_count = sum(1 for y in best_train_l if y > 0)
        base_rate = positive_count / len(best_train_l) if len(best_train_l) > 0 else 0.50
        meta = {"base_rate": round(base_rate, 4)}
        meta_path = segment_dir / "segment_meta.json"
        meta_path.write_text(json.dumps(meta, indent=2))
        print(f"[{segment_id}] Saved segment_meta.json: base_rate={base_rate:.4f}")

        # Fit stacking meta-learner on OOF predictions from the best fold's test set
        fit_and_save_meta_learner(segment_id, segment_dir, best_models, best_test_f, best_test_l)

    return gate_pass_rates


def apply_bh_across_segments(
    segment_accuracies: dict[str, float],
    output_dir: Path,
) -> None:
    """Apply BH multiple testing correction across all segments (D3).

    Converts accuracies to p-values using binomial test, then applies BH correction.
    Disables models in segments that fail the correction.
    """
    from scipy.stats import binomtest  # noqa: PLC0415

    segment_ids = list(segment_accuracies.keys())
    p_values: list[float] = []

    for seg_id in segment_ids:
        acc = segment_accuracies[seg_id]
        # Load n_test from wf results
        results_path = output_dir / seg_id / "wf_gate_results.json"
        n_folds = 1
        if results_path.exists():
            wf_data = json.loads(results_path.read_text())
            n_folds = wf_data.get("n_folds", 1)

        # Approximate: accuracy > 0.5 is the null hypothesis test
        n_correct = int(acc * n_folds * 100)  # approximate
        n_total = n_folds * 100
        result = binomtest(n_correct, n_total, 0.5, alternative="greater")
        p_values.append(float(result.pvalue))

    passes = apply_bh_correction(p_values, fdr=BH_FDR)

    print("\n=== BH Multiple Testing Correction (D3) ===")
    for seg_id, p_val, passed in zip(segment_ids, p_values, passes, strict=True):
        status = "PASS" if passed else "FAIL (disabled)"
        print(f"  {seg_id:>20s}: p={p_val:.4f} [{status}]")
        if not passed:
            # Mark segment as failed in its results file
            results_path = output_dir / seg_id / "wf_gate_results.json"
            if results_path.exists():
                wf_data = json.loads(results_path.read_text())
                wf_data["bh_passed"] = False
                results_path.write_text(json.dumps(wf_data, indent=2))
