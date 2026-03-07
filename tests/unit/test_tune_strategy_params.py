"""Unit tests for Optuna strategy parameter tuning script components."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

# Ensure scripts/ is importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT))

from tune_strategy_params import (  # noqa: E402
    _apply_params_to_config,
    apply_params_to_yaml,
    composite_objective,
    create_search_space,
    deflated_sharpe_ratio,
    run_holdout_validation,
    run_perturbation_check,
)

# ── Constants ─────────────────────────────────────────────────────────────
# Avoid magic numbers (ruff PLR2004)
ENOUGH_TRADES = 300
LOW_TRADES = 50
ZERO_TRADES = 0
TARGET_TRADES = 200
LOW_DD = 0.05
HIGH_DD = 0.20
SHARPE_POSITIVE = 1.0
SHARPE_NEGATIVE = -0.5

# Expected scores
LOW_TRADES_SCALE = LOW_TRADES / TARGET_TRADES  # 0.25
DD_PENALTY_FOR_HIGH = 2.0 * (HIGH_DD - 0.10)  # 0.20


class TestCompositeObjective:
    """Tests for the composite_objective scoring function."""

    def test_penalizes_low_trade_count(self) -> None:
        """With low trades, Sharpe is scaled down by trades/200."""
        score = composite_objective(wf_sharpe=SHARPE_POSITIVE, trades=LOW_TRADES, max_dd=LOW_DD)
        # trade_scale = 50/200 = 0.25, dd_penalty = 0 (0.05 < 0.10)
        expected = SHARPE_POSITIVE * LOW_TRADES_SCALE
        assert score == pytest.approx(expected)

    def test_full_score_with_enough_trades(self) -> None:
        """Enough trades + low DD gives full Sharpe score."""
        score = composite_objective(wf_sharpe=SHARPE_POSITIVE, trades=ENOUGH_TRADES, max_dd=LOW_DD)
        # trade_scale = min(1.0, 300/200) = 1.0, dd_penalty = 0
        assert score == pytest.approx(SHARPE_POSITIVE)

    def test_dd_penalty_applied(self) -> None:
        """High drawdown reduces the score."""
        score_low_dd = composite_objective(
            wf_sharpe=SHARPE_POSITIVE, trades=ENOUGH_TRADES, max_dd=LOW_DD
        )
        score_high_dd = composite_objective(
            wf_sharpe=SHARPE_POSITIVE, trades=ENOUGH_TRADES, max_dd=HIGH_DD
        )
        # dd_penalty = 2.0 * max(0, 0.20 - 0.10) = 0.20
        assert score_high_dd < score_low_dd
        assert score_high_dd == pytest.approx(SHARPE_POSITIVE - DD_PENALTY_FOR_HIGH)

    def test_zero_trades_produces_zero_score(self) -> None:
        """Zero trades makes trade_scale=0, so score is 0 (minus any DD penalty)."""
        score = composite_objective(wf_sharpe=SHARPE_POSITIVE, trades=ZERO_TRADES, max_dd=LOW_DD)
        assert score == pytest.approx(0.0)

    def test_negative_sharpe_passes_through(self) -> None:
        """Negative Sharpe is not floored -- it scales with trade count."""
        score = composite_objective(wf_sharpe=SHARPE_NEGATIVE, trades=TARGET_TRADES, max_dd=LOW_DD)
        # trade_scale = 1.0, dd_penalty = 0
        assert score == pytest.approx(SHARPE_NEGATIVE)

    def test_dd_at_threshold_no_penalty(self) -> None:
        """Drawdown exactly at 10% produces zero penalty."""
        threshold_dd = 0.10
        score = composite_objective(
            wf_sharpe=SHARPE_POSITIVE, trades=ENOUGH_TRADES, max_dd=threshold_dd
        )
        assert score == pytest.approx(SHARPE_POSITIVE)

    def test_trades_at_target_gives_scale_one(self) -> None:
        """Exactly 200 trades gives trade_scale=1.0."""
        score = composite_objective(wf_sharpe=SHARPE_POSITIVE, trades=TARGET_TRADES, max_dd=LOW_DD)
        assert score == pytest.approx(SHARPE_POSITIVE)


class TestCreateSearchSpace:
    """Tests for the Optuna search space definition."""

    def test_returns_expected_keys(self) -> None:
        """Search space contains all expected parameter names."""
        import optuna

        study = optuna.create_study()
        trial = study.ask()
        params = create_search_space(trial)

        expected_keys = {
            "min_combined_confidence",
            "vol_target",
            "trend_threshold",
            "mr_threshold",
            "rsi_buy_threshold",
            "rsi_sell_threshold",
        }
        assert set(params.keys()) == expected_keys

    def test_values_are_within_bounds(self) -> None:
        """All suggested values fall within their defined ranges."""
        import optuna

        study = optuna.create_study()
        trial = study.ask()
        params = create_search_space(trial)

        min_conf_low = 0.20
        min_conf_high = 0.40
        vol_low = 0.15
        vol_high = 0.30
        trend_low = 28
        trend_high = 40
        mr_low = 10
        mr_high = 22
        rsi_buy_low = 5.0
        rsi_buy_high = 15.0
        rsi_sell_low = 85.0
        rsi_sell_high = 95.0

        assert min_conf_low <= params["min_combined_confidence"] <= min_conf_high
        assert vol_low <= params["vol_target"] <= vol_high
        assert trend_low <= params["trend_threshold"] <= trend_high
        assert mr_low <= params["mr_threshold"] <= mr_high
        assert rsi_buy_low <= params["rsi_buy_threshold"] <= rsi_buy_high
        assert rsi_sell_low <= params["rsi_sell_threshold"] <= rsi_sell_high


class TestApplyParamsToConfig:
    """Tests for _apply_params_to_config (non-mutating config patching)."""

    @pytest.fixture
    def sample_config(self) -> dict[str, Any]:
        """A minimal YAML-like config dict for testing."""
        return {
            "segment_id": "us_tech",
            "min_combined_confidence": 0.30,
            "regime_routing": {
                "enabled": True,
                "trend_threshold": 35,
                "mr_threshold": 15,
            },
            "strategies": {
                "momentum": {
                    "enabled": True,
                    "weight": 0.20,
                    "params": {
                        "vol_target_enabled": True,
                        "vol_target": 0.20,
                    },
                },
                "rsi2_connors": {
                    "enabled": True,
                    "weight": 0.15,
                    "params": {
                        "rsi_buy_threshold": 10.0,
                        "rsi_sell_threshold": 90.0,
                    },
                },
                "mean_reversion": {
                    "enabled": True,
                    "weight": 0.30,
                    "params": {},
                },
            },
        }

    def test_min_combined_confidence_applied(self, sample_config: dict[str, Any]) -> None:
        """min_combined_confidence is updated in the config."""
        new_conf = 0.25
        result = _apply_params_to_config(sample_config, {"min_combined_confidence": new_conf})
        assert result["min_combined_confidence"] == new_conf

    def test_vol_target_applied_to_enabled_strategies(self, sample_config: dict[str, Any]) -> None:
        """vol_target is applied only to strategies with vol_target_enabled=True."""
        new_vol = 0.18
        result = _apply_params_to_config(sample_config, {"vol_target": new_vol})
        # momentum has vol_target_enabled=True
        assert result["strategies"]["momentum"]["params"]["vol_target"] == pytest.approx(new_vol)
        # mean_reversion does not have vol_target_enabled
        assert "vol_target" not in result["strategies"]["mean_reversion"]["params"]

    def test_regime_routing_thresholds_applied(self, sample_config: dict[str, Any]) -> None:
        """trend_threshold and mr_threshold update regime_routing."""
        new_trend = 32
        new_mr = 18
        result = _apply_params_to_config(
            sample_config, {"trend_threshold": new_trend, "mr_threshold": new_mr}
        )
        assert result["regime_routing"]["trend_threshold"] == new_trend
        assert result["regime_routing"]["mr_threshold"] == new_mr

    def test_rsi_thresholds_applied(self, sample_config: dict[str, Any]) -> None:
        """RSI buy/sell thresholds update rsi2_connors params."""
        new_buy = 8.0
        new_sell = 92.0
        result = _apply_params_to_config(
            sample_config, {"rsi_buy_threshold": new_buy, "rsi_sell_threshold": new_sell}
        )
        assert result["strategies"]["rsi2_connors"]["params"]["rsi_buy_threshold"] == pytest.approx(
            new_buy
        )
        assert result["strategies"]["rsi2_connors"]["params"][
            "rsi_sell_threshold"
        ] == pytest.approx(new_sell)

    def test_original_config_not_mutated(self, sample_config: dict[str, Any]) -> None:
        """_apply_params_to_config returns a deep copy, not mutating the original."""
        original_confidence = sample_config["min_combined_confidence"]
        _apply_params_to_config(sample_config, {"min_combined_confidence": 0.99})
        assert sample_config["min_combined_confidence"] == original_confidence

    def test_empty_params_returns_copy(self, sample_config: dict[str, Any]) -> None:
        """Empty params dict returns an unchanged copy."""
        result = _apply_params_to_config(sample_config, {})
        assert result == sample_config
        assert result is not sample_config  # it's a copy


class TestApplyParamsToYaml:
    """Tests for apply_params_to_yaml (writes to disk)."""

    def test_roundtrip_yaml_update(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """apply_params_to_yaml reads, patches, and writes back the YAML file."""
        # Set up a fake presets directory
        presets_dir = tmp_path / "src" / "finalayze" / "strategies" / "presets"
        presets_dir.mkdir(parents=True)

        config = {
            "segment_id": "test_seg",
            "min_combined_confidence": 0.30,
            "regime_routing": {
                "enabled": True,
                "trend_threshold": 35,
                "mr_threshold": 15,
            },
            "strategies": {
                "momentum": {
                    "enabled": True,
                    "params": {
                        "vol_target_enabled": True,
                        "vol_target": 0.20,
                    },
                },
                "rsi2_connors": {
                    "enabled": True,
                    "params": {
                        "rsi_buy_threshold": 10.0,
                        "rsi_sell_threshold": 90.0,
                    },
                },
            },
        }

        yaml_path = presets_dir / "test_seg.yaml"
        with yaml_path.open("w") as f:
            yaml.dump(config, f, default_flow_style=False)

        # Monkeypatch _PROJECT_ROOT so apply_params_to_yaml finds the temp dir
        import tune_strategy_params as module

        monkeypatch.setattr(module, "_PROJECT_ROOT", tmp_path)

        new_params = {
            "min_combined_confidence": 0.25,
            "vol_target": 0.18,
            "trend_threshold": 30,
            "mr_threshold": 20,
            "rsi_buy_threshold": 7.0,
            "rsi_sell_threshold": 93.0,
        }
        apply_params_to_yaml("test_seg", new_params)

        with yaml_path.open() as f:
            updated = yaml.safe_load(f)

        assert updated["min_combined_confidence"] == 0.25
        assert updated["strategies"]["momentum"]["params"]["vol_target"] == pytest.approx(0.18)
        assert updated["regime_routing"]["trend_threshold"] == 30
        assert updated["regime_routing"]["mr_threshold"] == 20
        rsi_params = updated["strategies"]["rsi2_connors"]["params"]
        assert rsi_params["rsi_buy_threshold"] == pytest.approx(7.0)
        assert rsi_params["rsi_sell_threshold"] == pytest.approx(93.0)


# ── Guardrail constants ──────────────────────────────────────────────────
_N_TRIALS_MANY = 50
_N_TRADES_MANY = 200
_N_TRADES_FEW = 30
_OPT_SHARPE = 0.10
_OPT_SHARPE_NEG = -0.10
_HOLDOUT_SHARPE_OK = 0.08  # 80% of opt → pass
_HOLDOUT_SHARPE_BAD = 0.02  # 20% of opt → fail

_BEST_PARAMS: dict[str, Any] = {
    "min_combined_confidence": 0.30,
    "vol_target": 0.20,
    "trend_threshold": 35,
    "mr_threshold": 15,
    "rsi_buy_threshold": 10.0,
    "rsi_sell_threshold": 90.0,
}


class TestDeflatedSharpeRatio:
    """Tests for deflated_sharpe_ratio (Harvey et al. haircut)."""

    def test_single_trial_no_haircut(self) -> None:
        """N=1 → ln(1)=0 → no haircut."""
        assert deflated_sharpe_ratio(1.0, n_trials=1, n_trades=_N_TRADES_MANY) == pytest.approx(1.0)

    def test_many_trials_haircut(self) -> None:
        """More trials → bigger haircut."""
        dsr = deflated_sharpe_ratio(1.0, n_trials=_N_TRIALS_MANY, n_trades=_N_TRADES_MANY)
        assert dsr < 1.0
        assert dsr > 0.0

    def test_few_trades_bigger_haircut(self) -> None:
        """Fewer trades → bigger haircut."""
        dsr_many = deflated_sharpe_ratio(1.0, n_trials=_N_TRIALS_MANY, n_trades=_N_TRADES_MANY)
        dsr_few = deflated_sharpe_ratio(1.0, n_trials=_N_TRIALS_MANY, n_trades=_N_TRADES_FEW)
        assert dsr_few < dsr_many

    def test_negative_sharpe_stays_negative(self) -> None:
        """Negative Sharpe gets proportionally reduced (stays negative)."""
        dsr = deflated_sharpe_ratio(-0.5, n_trials=_N_TRIALS_MANY, n_trades=_N_TRADES_MANY)
        assert dsr < 0.0

    def test_zero_trades_returns_zero(self) -> None:
        """Edge case: 0 trades → return 0."""
        assert deflated_sharpe_ratio(1.0, n_trials=_N_TRIALS_MANY, n_trades=0) == pytest.approx(0.0)


class TestRunHoldoutValidation:
    """Tests for run_holdout_validation."""

    def test_pass_when_holdout_sharpe_ok(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Holdout Sharpe >= 50% of opt Sharpe → pass."""
        import tune_strategy_params as module

        monkeypatch.setattr(
            module,
            "run_backtest_for_trial",
            lambda *_a, **_kw: {"wf_sharpe": _HOLDOUT_SHARPE_OK, "trades": 100, "max_dd": 0.05},
        )
        result = run_holdout_validation("us_tech", _BEST_PARAMS, opt_sharpe=_OPT_SHARPE)
        assert result["passed"] is True

    def test_fail_when_holdout_degrades(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Holdout Sharpe < 50% of opt Sharpe → fail."""
        import tune_strategy_params as module

        monkeypatch.setattr(
            module,
            "run_backtest_for_trial",
            lambda *_a, **_kw: {"wf_sharpe": _HOLDOUT_SHARPE_BAD, "trades": 100, "max_dd": 0.05},
        )
        result = run_holdout_validation("us_tech", _BEST_PARAMS, opt_sharpe=_OPT_SHARPE)
        assert result["passed"] is False

    def test_negative_opt_sharpe_auto_passes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """If optimization Sharpe <= 0, holdout check is meaningless → auto-pass."""
        import tune_strategy_params as module

        monkeypatch.setattr(
            module,
            "run_backtest_for_trial",
            lambda *_a, **_kw: {"wf_sharpe": -0.5, "trades": 100, "max_dd": 0.05},
        )
        result = run_holdout_validation("us_tech", _BEST_PARAMS, opt_sharpe=_OPT_SHARPE_NEG)
        assert result["passed"] is True


class TestRunPerturbationCheck:
    """Tests for run_perturbation_check."""

    def test_stable_params_pass(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """All perturbed runs have Sharpe within 50% → pass."""
        import tune_strategy_params as module

        # Return stable Sharpe for all perturbations
        monkeypatch.setattr(
            module,
            "run_backtest_for_trial",
            lambda *_a, **_kw: {"wf_sharpe": 0.09, "trades": 100, "max_dd": 0.05},
        )
        result = run_perturbation_check("us_tech", _BEST_PARAMS, opt_sharpe=_OPT_SHARPE)
        assert result["passed"] is True
        assert result["fragile_params"] == []

    def test_fragile_param_flagged(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """If one param's perturbation drops Sharpe >50% → flagged as fragile."""
        import tune_strategy_params as module

        def _mock_backtest(_seg: str, params: dict[str, Any], **_kw: Any) -> dict[str, float]:
            # vol_target perturbation causes big Sharpe drop
            vol = params.get("vol_target", 0.20)
            base_vol = 0.20
            if abs(vol - base_vol) > 0.01:
                return {"wf_sharpe": 0.01, "trades": 100, "max_dd": 0.05}
            return {"wf_sharpe": 0.09, "trades": 100, "max_dd": 0.05}

        monkeypatch.setattr(module, "run_backtest_for_trial", _mock_backtest)
        result = run_perturbation_check("us_tech", _BEST_PARAMS, opt_sharpe=_OPT_SHARPE)
        assert result["passed"] is False
        assert "vol_target" in result["fragile_params"]

    def test_negative_opt_sharpe_auto_passes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """If optimization Sharpe <= 0, perturbation check is meaningless → auto-pass."""
        import tune_strategy_params as module

        monkeypatch.setattr(
            module,
            "run_backtest_for_trial",
            lambda *_a, **_kw: {"wf_sharpe": -0.5, "trades": 100, "max_dd": 0.05},
        )
        result = run_perturbation_check("us_tech", _BEST_PARAMS, opt_sharpe=_OPT_SHARPE_NEG)
        assert result["passed"] is True
