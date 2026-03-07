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
import re
import sys
import tempfile
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
_DD_PENALTY_THRESHOLD = 0.10

# DD penalty multiplier
_DD_PENALTY_COEFF = 2.0


def composite_objective(wf_sharpe: float, trades: int, max_dd: float) -> float:
    """Composite objective: Sharpe scaled by trade sufficiency, penalized by DD.

    Args:
        wf_sharpe: Walk-forward Sharpe ratio (can be negative).
        trades: Total number of trades (higher is better up to target).
        max_dd: Maximum drawdown as a fraction (0.10 = 10%).

    Returns:
        A scalar score to maximize.  Higher is better.

    The formula is::

        score = wf_sharpe * min(1.0, trades / 200) - 2.0 * max(0, max_dd - 0.10)

    This rewards:
    - Higher Sharpe
    - At least 200 trades (linear ramp-up below that)
    - Drawdown below 10% (penalty above that)
    """
    trade_scale = min(1.0, trades / _TRADE_SUFFICIENCY_TARGET)
    dd_penalty = _DD_PENALTY_COEFF * max(0.0, max_dd - _DD_PENALTY_THRESHOLD)
    return wf_sharpe * trade_scale - dd_penalty


def create_search_space(trial: optuna.Trial) -> dict[str, Any]:
    """Define the Optuna search space for strategy parameters.

    Returns a dict of parameter names to suggested values.  These parameters
    map to YAML preset keys that control signal filtering and regime routing.
    """
    return {
        "min_combined_confidence": trial.suggest_float("min_combined_confidence", 0.20, 0.40),
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


def run_backtest_for_trial(segment_id: str, params: dict[str, Any]) -> dict[str, float]:
    """Run a backtest with given params and return metrics.

    Creates a temporary YAML config with the trial params applied, writes it
    to a temp directory as ``{segment_id}.yaml``, and overrides the combiner's
    preset directory so ``_load_config`` reads the trial config instead of the
    original preset.

    Args:
        segment_id: Segment identifier (e.g. ``us_tech``).
        params: Trial parameter values from ``create_search_space``.

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

        from datetime import UTC, datetime  # noqa: PLC0415
        from decimal import Decimal  # noqa: PLC0415

        start = datetime(2023, 1, 1, tzinfo=UTC)
        end = datetime(2024, 12, 31, tzinfo=UTC)
        cash = Decimal(100_000)

        market_id = "moex" if segment_id.startswith("ru_") else "us"
        fetcher = CachingFetcher(YFinanceFetcher(market_id=market_id))
        strategies = _build_strategies(segment_id, fetcher, start, end, None, symbols=symbols)

        regime_provider = _build_regime_provider("vix", segment_id, start, end)

        all_trades = []
        all_snapshots = []

        for symbol in symbols[:5]:  # Limit to first 5 symbols for speed
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


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Optuna strategy parameter tuning")
    parser.add_argument("--segment", required=True, help="Segment ID (e.g., us_tech)")
    parser.add_argument("--n-trials", type=int, default=_DEFAULT_N_TRIALS, help="Number of trials")
    parser.add_argument("--output-dir", default=None, help="Output directory for results")
    parser.add_argument("--apply", action="store_true", help="Apply best params to YAML preset")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampler")
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
        metrics = run_backtest_for_trial(args.segment, params)
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
    best_score = study.best_value
    best_metrics = study.best_trial.user_attrs.get("metrics", {})

    print(f"\nBest params for {args.segment}:")
    print(json.dumps(best, indent=2))
    print(f"\nBest composite score: {best_score:.4f}")
    print(f"Best trial metrics: {json.dumps(best_metrics, indent=2)}")

    result_path = output_dir / "strategy_params.json"
    with result_path.open("w") as f:
        json.dump(
            {
                "segment": args.segment,
                "n_trials": args.n_trials,
                "params": best,
                "score": best_score,
                "metrics": best_metrics,
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
