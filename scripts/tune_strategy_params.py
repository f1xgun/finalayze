"""Optuna-based strategy parameter tuning per segment.

Optimizes strategy-level parameters (confidence thresholds, ADX routing,
RSI thresholds, vol targets) using Bayesian optimization.  Each trial
applies candidate params to a temporary YAML config and runs the full
backtest pipeline, scoring results with a composite objective that
balances Sharpe, trade sufficiency, and drawdown.

Usage:
    uv run python scripts/tune_strategy_params.py --segment us_tech --n-trials 50
    uv run python scripts/tune_strategy_params.py --segment us_broad \
        --n-trials 100 --output-dir results/tuned_params/us_broad
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import re
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT))

import optuna
import yaml

optuna.logging.set_verbosity(optuna.logging.WARNING)

_DEFAULT_N_TRIALS = 50
_DEFAULT_OUTPUT_DIR = "results/tuned_params"

# Trade sufficiency target — score linearly scales up to this count
_TRADE_SUFFICIENCY_TARGET = 200

# Max drawdown threshold beyond which we start penalising
_DD_PENALTY_THRESHOLD = 0.02

# DD penalty multiplier
_DD_PENALTY_COEFF = 2.0

# ── Date ranges ──────────────────────────────────────────────────────────
_OPT_START = datetime(2023, 1, 1, tzinfo=UTC)
_OPT_END = datetime(2024, 6, 30, tzinfo=UTC)
_HOLDOUT_START = datetime(2024, 7, 1, tzinfo=UTC)
_HOLDOUT_END = datetime(2024, 12, 31, tzinfo=UTC)

# ── Overfitting guardrail thresholds ─────────────────────────────────────
_HOLDOUT_DEGRADATION_THRESHOLD = 0.50  # flag if holdout < 50% of opt
_PERTURBATION_PCT = 0.20  # ±20%
_PERTURBATION_DROP_THRESHOLD = 0.50  # flag if Sharpe drops >50%


def deflated_sharpe_ratio(sharpe: float, *, n_trials: int, n_trades: int) -> float:
    """Apply Harvey et al. (2016) multiple-testing haircut to Sharpe ratio.

    DSR = SR * max(0, 1 - ln(N) / (2T))

    This penalizes Sharpe ratios found after many optimization trials,
    accounting for the increased probability of finding spuriously high values.
    """
    if n_trades <= 0:
        return 0.0
    haircut = max(0.0, 1.0 - math.log(max(1, n_trials)) / (2 * n_trades))
    return sharpe * haircut


def composite_objective(wf_sharpe: float, trades: int, max_dd: float) -> float:
    """Composite objective: Sharpe scaled by trade sufficiency, penalized by DD.

    Args:
        wf_sharpe: Walk-forward Sharpe ratio (can be negative).
        trades: Total number of trades (higher is better up to target).
        max_dd: Maximum drawdown as a fraction (0.10 = 10%).

    Returns:
        A scalar score to maximize.  Higher is better.

    The formula is::

        score = wf_sharpe * min(1.0, trades / 200) - 2.0 * max(0, max_dd - 0.02)

    This rewards:
    - Higher Sharpe
    - At least 200 trades (linear ramp-up below that)
    - Drawdown below 2% (penalty above that)
    """
    trade_scale = min(1.0, trades / _TRADE_SUFFICIENCY_TARGET)
    dd_penalty = _DD_PENALTY_COEFF * max(0.0, max_dd - _DD_PENALTY_THRESHOLD)
    return wf_sharpe * trade_scale - dd_penalty


def create_search_space(trial: optuna.Trial) -> dict[str, Any]:
    """Define the Optuna search space for strategy parameters.

    Returns a dict of parameter names to suggested values.  These parameters
    map to YAML preset keys that control signal filtering and regime routing.
    """
    min_combined = trial.suggest_float("min_combined_confidence", 0.20, 0.40)
    min_exit = trial.suggest_float("min_exit_confidence", 0.15, 0.35)
    # Clamp: exit confidence must not exceed combined confidence
    min_exit = min(min_exit, min_combined)
    return {
        "min_combined_confidence": min_combined,
        "min_exit_confidence": min_exit,
        "vol_target": trial.suggest_float("vol_target", 0.15, 0.30),
        "trend_threshold": trial.suggest_int("trend_threshold", 28, 40),
        "mr_threshold": trial.suggest_int("mr_threshold", 10, 22),
        "rsi_buy_threshold": trial.suggest_float("rsi_buy_threshold", 5.0, 15.0),
        "rsi_sell_threshold": trial.suggest_float("rsi_sell_threshold", 85.0, 95.0),
    }


def apply_params_to_yaml(segment_id: str, params: dict[str, Any]) -> None:
    """Apply tuned params to the segment YAML preset using surgical text edits.

    Uses regex replacement instead of yaml.dump to preserve formatting and
    avoid silent value corruption.

    Args:
        segment_id: Segment identifier (e.g. ``us_tech``).
        params: Dict of parameter names to values from the best trial.
    """
    presets_dir = _PROJECT_ROOT / "src" / "finalayze" / "strategies" / "presets"
    yaml_path = presets_dir / f"{segment_id}.yaml"
    text = yaml_path.read_text()

    if "min_combined_confidence" in params:
        val = round(float(params["min_combined_confidence"]), 2)
        text = re.sub(
            r"(min_combined_confidence:\s*)\S+",
            rf"\g<1>{val}",
            text,
        )

    if "min_exit_confidence" in params:
        val = round(float(params["min_exit_confidence"]), 2)
        text = re.sub(
            r"(min_exit_confidence:\s*)\S+",
            rf"\g<1>{val}",
            text,
        )

    if "vol_target" in params:
        val = round(float(params["vol_target"]), 2)
        text = re.sub(
            r"(vol_target:\s*)\S+",
            rf"\g<1>{val}",
            text,
            flags=re.MULTILINE,
        )

    if "trend_threshold" in params:
        val = int(params["trend_threshold"])
        text = re.sub(
            r"(trend_threshold:\s*)\S+",
            rf"\g<1>{val}",
            text,
        )

    if "mr_threshold" in params:
        val = int(params["mr_threshold"])
        text = re.sub(
            r"(mr_threshold:\s*)\S+",
            rf"\g<1>{val}",
            text,
        )

    if "rsi_buy_threshold" in params:
        val = round(float(params["rsi_buy_threshold"]), 1)
        text = re.sub(
            r"(rsi_buy_threshold:\s*)\S+",
            rf"\g<1>{val}",
            text,
        )

    if "rsi_sell_threshold" in params:
        val = round(float(params["rsi_sell_threshold"]), 1)
        text = re.sub(
            r"(rsi_sell_threshold:\s*)\S+",
            rf"\g<1>{val}",
            text,
        )

    yaml_path.write_text(text)


def _apply_params_to_config(config: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    """Apply trial params to a config dict (non-mutating, returns copy).

    This is the internal helper used by ``run_backtest_for_trial`` to create
    a temporary config without touching the on-disk YAML file.
    """
    config = copy.deepcopy(config)

    if "min_combined_confidence" in params:
        config["min_combined_confidence"] = round(float(params["min_combined_confidence"]), 2)

    if "min_exit_confidence" in params:
        config["min_exit_confidence"] = round(float(params["min_exit_confidence"]), 2)

    if "vol_target" in params:
        for strat_cfg in config.get("strategies", {}).values():
            if isinstance(strat_cfg, dict) and strat_cfg.get("params", {}).get(
                "vol_target_enabled"
            ):
                strat_cfg["params"]["vol_target"] = round(float(params["vol_target"]), 2)

    routing = config.get("regime_routing", {})
    if isinstance(routing, dict):
        if "trend_threshold" in params:
            routing["trend_threshold"] = int(params["trend_threshold"])
        if "mr_threshold" in params:
            routing["mr_threshold"] = int(params["mr_threshold"])

    rsi2 = config.get("strategies", {}).get("rsi2_connors", {})
    if isinstance(rsi2, dict):
        p = rsi2.setdefault("params", {})
        if "rsi_buy_threshold" in params:
            p["rsi_buy_threshold"] = round(float(params["rsi_buy_threshold"]), 1)
        if "rsi_sell_threshold" in params:
            p["rsi_sell_threshold"] = round(float(params["rsi_sell_threshold"]), 1)

    return config


def run_backtest_for_trial(  # noqa: PLR0915
    segment_id: str,
    params: dict[str, Any],
    *,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
) -> dict[str, float]:
    """Run a backtest with given params and return metrics.

    Creates a temporary YAML config with the trial params applied, writes it
    to a temp directory as ``{segment_id}.yaml``, and overrides the combiner's
    preset directory so ``_load_config`` reads the trial config instead of the
    original preset.

    Args:
        segment_id: Segment identifier (e.g. ``us_tech``).
        params: Trial parameter values from ``create_search_space``.
        start_date: Backtest start (default ``_OPT_START``).
        end_date: Backtest end (default ``_OPT_END``).

    Returns:
        Dict with keys ``wf_sharpe``, ``trades``, ``max_dd``.
        On failure, returns ``{wf_sharpe: -1.0, trades: 0, max_dd: 1.0}``.
    """
    presets_dir = _PROJECT_ROOT / "src" / "finalayze" / "strategies" / "presets"
    yaml_path = presets_dir / f"{segment_id}.yaml"

    with yaml_path.open() as f:
        base_config = yaml.safe_load(f)

    config = _apply_params_to_config(base_config, params)

    # Write temp YAML to a temp directory using the segment filename so that
    # StrategyCombiner._load_config can find it via _presets_dir override.
    tmp_dir = tempfile.mkdtemp(prefix="optuna_preset_")
    tmp_preset_path = Path(tmp_dir) / f"{segment_id}.yaml"
    with tmp_preset_path.open("w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    try:
        # Import the iteration runner's internal machinery
        from scripts.run_iteration import (  # noqa: PLC0415
            UNIVERSE,
            BacktestConfig,
            CachingFetcher,
            DecisionJournal,
            JournalingStrategyCombiner,
            PerformanceAnalyzer,
            RollingKelly,
            YFinanceFetcher,
            _build_regime_provider,
            _build_strategies,
        )

        from finalayze.backtest.engine import BacktestEngine  # noqa: PLC0415

        symbols = UNIVERSE.get(segment_id, [])
        if not symbols:
            return {"wf_sharpe": -1.0, "trades": 0, "max_dd": 1.0}

        from decimal import Decimal  # noqa: PLC0415

        start = start_date or _OPT_START
        end = end_date or _OPT_END

        market_id = "moex" if segment_id.startswith("ru_") else "us"
        if segment_id.startswith("ru_"):
            import os  # noqa: PLC0415

            from finalayze.data.fetchers.tinkoff_data import TinkoffFetcher  # noqa: PLC0415
            from finalayze.markets.instruments import build_default_registry  # noqa: PLC0415

            token = os.environ.get("FINALAYZE_TINKOFF_TOKEN", "")
            if not token:
                return {"wf_sharpe": -1.0, "trades": 0, "max_dd": 1.0}
            registry = build_default_registry()
            fetcher = CachingFetcher(TinkoffFetcher(token=token, registry=registry, sandbox=False))
            cash = Decimal(1_000_000)
        else:
            fetcher = CachingFetcher(YFinanceFetcher(market_id=market_id))
            cash = Decimal(100_000)
        strategies = _build_strategies(segment_id, fetcher, start, end, None, symbols=symbols)

        regime_provider = _build_regime_provider("vix", segment_id, start, end)

        all_trades = []
        all_snapshots = []

        random.seed(42)
        sample_symbols = random.sample(symbols, min(5, len(symbols)))
        for symbol in sample_symbols:  # Random sample of 5 symbols for speed
            try:
                candles = fetcher.fetch_candles(symbol, start, end)
                if not candles:
                    continue
            except Exception:
                continue

            combiner = JournalingStrategyCombiner(
                strategies=strategies,
                allocation_mode="hrp",
            )
            # Override presets dir so combiner reads the trial config
            combiner._presets_dir = Path(tmp_dir)
            journal = DecisionJournal()

            engine = BacktestEngine(
                strategy=combiner,
                config=BacktestConfig(
                    initial_cash=cash,
                    decision_journal=journal,
                    rolling_kelly=RollingKelly(),
                    stop_loss_mode="chandelier",
                ),
                regime_provider=regime_provider,
            )
            trades, snapshots = engine.run(symbol, segment_id, candles)
            all_trades.extend(trades)
            all_snapshots.extend(snapshots)

        if not all_trades:
            return {"wf_sharpe": -1.0, "trades": 0, "max_dd": 1.0}

        result = PerformanceAnalyzer().analyze(all_trades, all_snapshots)
        return {
            "wf_sharpe": float(result.sharpe) if result else -1.0,
            "trades": len(all_trades),
            "max_dd": float(result.max_drawdown) if result else 1.0,
        }
    except Exception:
        return {"wf_sharpe": -1.0, "trades": 0, "max_dd": 1.0}
    finally:
        import shutil  # noqa: PLC0415

        shutil.rmtree(tmp_dir, ignore_errors=True)


def run_holdout_validation(
    segment_id: str, params: dict[str, Any], *, opt_sharpe: float
) -> dict[str, Any]:
    """Run best params on holdout period and check for degradation.

    If the holdout Sharpe is less than 50% of the optimization Sharpe,
    the check fails — indicating likely overfitting to the training period.
    """
    if opt_sharpe <= 0:
        return {
            "passed": True,
            "reason": "opt_sharpe <= 0, skip holdout",
            "opt_sharpe": opt_sharpe,
            "holdout_sharpe": None,
            "degradation_ratio": None,
            "threshold": _HOLDOUT_DEGRADATION_THRESHOLD,
        }

    metrics = run_backtest_for_trial(
        segment_id, params, start_date=_HOLDOUT_START, end_date=_HOLDOUT_END
    )
    holdout_sharpe = metrics.get("wf_sharpe", -1.0)
    ratio = holdout_sharpe / opt_sharpe
    passed = ratio >= _HOLDOUT_DEGRADATION_THRESHOLD

    return {
        "passed": passed,
        "opt_sharpe": opt_sharpe,
        "holdout_sharpe": holdout_sharpe,
        "degradation_ratio": round(ratio, 3),
        "threshold": _HOLDOUT_DEGRADATION_THRESHOLD,
    }


def run_perturbation_check(
    segment_id: str, params: dict[str, Any], *, opt_sharpe: float
) -> dict[str, Any]:
    """Perturb each param ±20% and check Sharpe stability.

    For each parameter, create +20% and -20% variants. If any variant's
    Sharpe drops more than 50% vs the optimization Sharpe, flag that
    parameter as "fragile".
    """
    if opt_sharpe <= 0:
        return {
            "passed": True,
            "reason": "opt_sharpe <= 0, skip perturbation",
            "fragile_params": [],
            "details": {},
            "threshold": _PERTURBATION_DROP_THRESHOLD,
        }

    fragile: list[str] = []
    details: dict[str, float] = {}
    for key, val in params.items():
        if not isinstance(val, (int, float)):
            continue
        for direction, factor in [("up", 1 + _PERTURBATION_PCT), ("down", 1 - _PERTURBATION_PCT)]:
            perturbed = {**params, key: type(val)(val * factor)}
            metrics = run_backtest_for_trial(segment_id, perturbed)
            perturbed_sharpe = metrics.get("wf_sharpe", -1.0)
            ratio = perturbed_sharpe / opt_sharpe
            details[f"{key}_{direction}"] = round(ratio, 3)
            if ratio < _PERTURBATION_DROP_THRESHOLD and key not in fragile:
                fragile.append(key)

    return {
        "passed": len(fragile) == 0,
        "fragile_params": fragile,
        "details": details,
        "threshold": _PERTURBATION_DROP_THRESHOLD,
    }


def _clamp_exit_confidence(params: dict[str, Any]) -> None:
    """Ensure min_exit_confidence <= min_combined_confidence in-place.

    ``study.best_params`` returns the raw values from ``trial.suggest_float``,
    bypassing the clamp applied inside ``create_search_space``.  This helper
    re-applies the constraint after retrieving the best trial.
    """
    if "min_exit_confidence" in params and "min_combined_confidence" in params:
        params["min_exit_confidence"] = min(
            params["min_exit_confidence"], params["min_combined_confidence"]
        )


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Optuna strategy parameter tuning")
    parser.add_argument("--segment", required=True, help="Segment ID (e.g., us_tech)")
    parser.add_argument("--n-trials", type=int, default=_DEFAULT_N_TRIALS, help="Number of trials")
    parser.add_argument("--output-dir", default=None, help="Output directory for results")
    parser.add_argument("--apply", action="store_true", help="Apply best params to YAML preset")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampler")
    parser.add_argument(
        "--skip-guardrails",
        action="store_true",
        help="Skip overfitting guardrails (DSR, holdout, perturbation)",
    )
    return parser.parse_args()


def main() -> None:
    """Run Optuna strategy parameter tuning."""
    args = _parse_args()

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else _PROJECT_ROOT / _DEFAULT_OUTPUT_DIR / args.segment
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Optuna strategy tuning: segment={args.segment}, trials={args.n_trials}")
    print(f"  Output: {output_dir}")

    def objective(trial: optuna.Trial) -> float:
        params = create_search_space(trial)
        metrics = run_backtest_for_trial(
            args.segment, params, start_date=_OPT_START, end_date=_OPT_END
        )
        score = composite_objective(
            wf_sharpe=metrics.get("wf_sharpe", -1.0),
            trades=int(metrics.get("trades", 0)),
            max_dd=metrics.get("max_dd", 1.0),
        )
        trial.set_user_attr("metrics", metrics)
        return score

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(
        direction="maximize",
        study_name=f"tune_{args.segment}",
        sampler=sampler,
    )
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)

    best = study.best_params
    _clamp_exit_confidence(best)
    best_score = study.best_value
    best_metrics = study.best_trial.user_attrs.get("metrics", {})

    print(f"\nBest params for {args.segment}:")
    print(json.dumps(best, indent=2))
    print(f"\nBest composite score: {best_score:.4f}")
    print(f"Best trial metrics: {json.dumps(best_metrics, indent=2)}")

    # ── Overfitting guardrails ────────────────────────────────────────────
    opt_sharpe = best_metrics.get("wf_sharpe", -1.0)
    guardrails: dict[str, Any] = {}

    # 1. Deflated Sharpe Ratio (always computed, cheap)
    dsr = deflated_sharpe_ratio(
        opt_sharpe,
        n_trials=args.n_trials,
        n_trades=int(best_metrics.get("trades", 0)),
    )
    guardrails["deflated_sharpe"] = {"value": round(dsr, 4), "raw_sharpe": opt_sharpe}

    if not args.skip_guardrails:
        # 2. Holdout validation
        print("\nRunning holdout validation...")
        holdout = run_holdout_validation(args.segment, best, opt_sharpe=opt_sharpe)
        guardrails["holdout"] = holdout

        # 3. Perturbation check
        print("Running perturbation check...")
        perturbation = run_perturbation_check(args.segment, best, opt_sharpe=opt_sharpe)
        guardrails["perturbation"] = perturbation

        # Print summary
        all_passed = holdout["passed"] and perturbation["passed"]
        print(f"\nGuardrails: {'PASS' if all_passed else 'WARN'}")
        print(f"  DSR: {dsr:.4f} (raw {opt_sharpe:.4f})")
        print(f"  Holdout: {'PASS' if holdout['passed'] else 'FAIL'}")
        print(f"  Perturbation: {'PASS' if perturbation['passed'] else 'FAIL'}")
        if perturbation.get("fragile_params"):
            print(f"  Fragile params: {perturbation['fragile_params']}")
    else:
        print("\nGuardrails: SKIPPED (--skip-guardrails)")
        print(f"  DSR: {dsr:.4f} (raw {opt_sharpe:.4f})")

    result_path = output_dir / "strategy_params.json"
    with result_path.open("w") as f:
        json.dump(
            {
                "segment": args.segment,
                "n_trials": args.n_trials,
                "params": best,
                "score": best_score,
                "metrics": best_metrics,
                "guardrails": guardrails,
            },
            f,
            indent=2,
        )
    print(f"\nSaved to {result_path}")

    if args.apply:
        apply_params_to_yaml(args.segment, best)
        print(f"Applied best params to {args.segment}.yaml")


if __name__ == "__main__":
    main()
