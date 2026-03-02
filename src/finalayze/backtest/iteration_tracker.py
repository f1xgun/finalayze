"""Iteration tracker -- snapshot, measure, gate, persist, compare iterations.

Orchestrates the capture of git provenance, computation of iteration metrics,
evaluation of acceptance gates, and atomic persistence of results.

See docs/architecture/DEPENDENCY_LAYERS.md for layering rules.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import statistics
import subprocess
import tempfile
from typing import TYPE_CHECKING, Any

from finalayze.core.schemas import (
    GateResult,
    IterationComparison,
    IterationMetadata,
    IterationMetrics,
)

if TYPE_CHECKING:
    from pathlib import Path

    from finalayze.backtest.decision_journal import DecisionJournal
    from finalayze.backtest.monte_carlo import BootstrapResult
    from finalayze.backtest.walk_forward import WalkForwardResult
    from finalayze.core.schemas import PortfolioState, TradeResult

# ── Gate thresholds ─────────────────────────────────────────────────────────
_S1_SHARPE_FLOOR = 0.0
_S2_MAX_DD_CEILING = 15.0  # percent
_S3_MC_SHARPE_FLOOR = 0.0
_C1_SHARPE_REGRESSION_LIMIT = 0.05
_C2_DD_REGRESSION_LIMIT = 2.0  # percent
_C3_MIN_TRADES_PER_FOLD = 60
_C4_MAX_SEGMENT_SHARE = 0.35
_C6_MAX_PARAM_CV = 0.30
_CALIBRATION_REJECT_THRESHOLD = 2
_ANNUALIZATION_FACTOR = 252
_LARGE_RATIO_SENTINEL = 999.0
_MIN_SNAPSHOTS_FOR_SORTINO = 3


class IterationTracker:
    """Track, persist, and compare backtest iterations."""

    def __init__(self, results_root: Path) -> None:
        self._root = results_root

    # ── Git context ─────────────────────────────────────────────────────────

    def snapshot_context(self) -> dict[str, Any]:
        """Capture git SHA, describe, dirty flag via subprocess."""
        sha = self._git("rev-parse", "HEAD").strip()
        describe = self._git("describe", "--dirty", "--always").strip()
        # returncode != 0 means dirty
        dirty = self._git_returncode("diff", "--quiet") != 0
        return {
            "git_sha": sha,
            "git_describe": describe,
            "git_dirty": dirty,
        }

    @staticmethod
    def _git(*args: str) -> str:
        result = subprocess.run(  # noqa: S603
            ["git", *args],  # noqa: S607
            capture_output=True,
            text=True,
            check=False,
        )
        return result.stdout

    @staticmethod
    def _git_returncode(*args: str) -> int:
        result = subprocess.run(  # noqa: S603
            ["git", *args],  # noqa: S607
            capture_output=True,
            text=True,
            check=False,
        )
        return result.returncode

    # ── Config hash ─────────────────────────────────────────────────────────

    @staticmethod
    def compute_config_hash(
        backtest_config: dict[str, Any],
        strategy_configs: dict[str, Any],
    ) -> str:
        """SHA-256 of deterministically serialized config + strategy YAMLs."""
        payload = json.dumps(
            {"backtest": backtest_config, "strategies": strategy_configs},
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    # ── Compute metrics ─────────────────────────────────────────────────────

    def compute_metrics(
        self,
        wf_result: WalkForwardResult,
        trades: list[TradeResult],
        snapshots: list[PortfolioState],
        segment_trades: dict[str, list[TradeResult]],
        mc_result: BootstrapResult,
        journal: DecisionJournal | None = None,
    ) -> IterationMetrics:
        """Compute all iteration metrics from backtest outputs."""
        segment_pnl = self._compute_segment_pnl(trades, segment_trades)
        win_rate_by_seg = self._compute_win_rates(segment_trades)
        profit_factor = self._compute_profit_factor(trades)
        avg_hold = len(snapshots) / len(trades) if trades else 0.0
        calmar = self._compute_calmar(snapshots, wf_result.oos_max_drawdown_pct)
        sortino = self._compute_sortino(snapshots)
        turnover_adj = self._compute_turnover_adjusted(trades, snapshots)

        # Param stability CV
        param_cv = self._compute_param_stability_cv(wf_result.per_fold_sharpes)

        # Model disagreement and per-model proba mean from journal
        model_disagreement = 0.0
        per_model_proba_mean: dict[str, float] = {}
        if journal is not None:
            model_disagreement, per_model_proba_mean = self._compute_model_metrics(journal)

        return IterationMetrics(
            wf_sharpe=wf_result.oos_sharpe,
            wf_max_drawdown=wf_result.oos_max_drawdown_pct,
            profit_factor=round(profit_factor, 4),
            calmar_ratio=round(calmar, 4),
            trade_count=len(trades),
            avg_hold_bars=round(avg_hold, 2),
            segment_pnl_share=segment_pnl,
            sortino_ratio=round(sortino, 4),
            win_rate_by_segment=win_rate_by_seg,
            information_ratio=None,
            mc_5th_pct_sharpe=mc_result.sharpe_ratio.lower,
            model_disagreement=round(model_disagreement, 4),
            turnover_adjusted_return=round(turnover_adj, 4),
            gross_sharpe=round(wf_result.oos_sharpe, 4),
            net_sharpe=round(wf_result.oos_sharpe, 4),
            param_stability_cv=round(param_cv, 4),
            per_model_proba_mean=per_model_proba_mean,
        )

    @staticmethod
    def _compute_segment_pnl(
        trades: list[TradeResult],
        segment_trades: dict[str, list[TradeResult]],
    ) -> dict[str, float]:
        total_pnl = sum(float(t.pnl) for t in trades)
        result: dict[str, float] = {}
        for seg_id, seg_trades in segment_trades.items():
            seg_pnl = sum(float(t.pnl) for t in seg_trades)
            result[seg_id] = seg_pnl / total_pnl if total_pnl != 0 else 0.0
        return result

    @staticmethod
    def _compute_win_rates(
        segment_trades: dict[str, list[TradeResult]],
    ) -> dict[str, float]:
        result: dict[str, float] = {}
        for seg_id, seg_trades in segment_trades.items():
            if seg_trades:
                wins = sum(1 for t in seg_trades if t.pnl > 0)
                result[seg_id] = wins / len(seg_trades)
            else:
                result[seg_id] = 0.0
        return result

    @staticmethod
    def _compute_profit_factor(trades: list[TradeResult]) -> float:
        gross_profit = sum(float(t.pnl) for t in trades if t.pnl > 0)
        gross_loss = abs(sum(float(t.pnl) for t in trades if t.pnl < 0))
        return gross_profit / gross_loss if gross_loss > 0 else _LARGE_RATIO_SENTINEL

    @staticmethod
    def _compute_calmar(
        snapshots: list[PortfolioState],
        max_dd_pct: float,
    ) -> float:
        if len(snapshots) < 2:  # noqa: PLR2004
            return 0.0
        initial_eq = float(snapshots[0].equity)
        final_eq = float(snapshots[-1].equity)
        if initial_eq <= 0:
            return 0.0
        total_ret = (final_eq - initial_eq) / initial_eq
        n_days = len(snapshots) - 1
        if n_days <= 0 or total_ret <= 0:
            return 0.0
        ann_ret: float = (1 + total_ret) ** (_ANNUALIZATION_FACTOR / n_days) - 1
        if max_dd_pct <= 0:
            return 0.0
        return ann_ret / (max_dd_pct / 100)

    @staticmethod
    def _compute_sortino(snapshots: list[PortfolioState]) -> float:
        """Compute annualised Sortino ratio from equity snapshots."""
        if len(snapshots) < _MIN_SNAPSHOTS_FOR_SORTINO:
            return 0.0
        equities = [float(s.equity) for s in snapshots]
        returns = [
            (equities[i] - equities[i - 1]) / equities[i - 1]
            for i in range(1, len(equities))
            if equities[i - 1] > 0
        ]
        if not returns:
            return 0.0
        mean_ret = statistics.mean(returns)
        if mean_ret <= 0:
            return 0.0
        downside = [min(0.0, r) for r in returns]
        downside_var = statistics.mean([d**2 for d in downside])
        downside_std = math.sqrt(downside_var)
        if downside_std == 0:
            return _LARGE_RATIO_SENTINEL
        return (mean_ret / downside_std) * math.sqrt(_ANNUALIZATION_FACTOR)

    @staticmethod
    def _compute_turnover_adjusted(
        trades: list[TradeResult],
        snapshots: list[PortfolioState],
    ) -> float:
        if len(snapshots) < 2:  # noqa: PLR2004
            return 0.0
        initial_eq = float(snapshots[0].equity)
        final_eq = float(snapshots[-1].equity)
        total_ret = (final_eq - initial_eq) / initial_eq if initial_eq > 0 else 0.0
        total_notional = sum(float(t.quantity) * float(t.entry_price) for t in trades)
        avg_eq = statistics.mean([float(s.equity) for s in snapshots]) if snapshots else 1.0
        turnover = total_notional / avg_eq if avg_eq > 0 else 0.0
        return total_ret / turnover if turnover > 0 else total_ret

    @staticmethod
    def _compute_param_stability_cv(per_fold_sharpes: list[float]) -> float:
        """CV of per-fold Sharpes as proxy for param stability."""
        if len(per_fold_sharpes) < 2:  # noqa: PLR2004
            return 0.0
        mean_s = statistics.mean(per_fold_sharpes)
        if mean_s == 0:
            return 0.0
        return statistics.stdev(per_fold_sharpes) / abs(mean_s)

    @staticmethod
    def _compute_model_metrics(
        journal: DecisionJournal,
    ) -> tuple[float, dict[str, float]]:
        """Extract model disagreement and per-model proba means."""
        model_probas_list: dict[str, list[float]] = {}
        for record in journal.records:
            if record.model_probas:
                for model_name, proba in record.model_probas.items():
                    model_probas_list.setdefault(model_name, []).append(proba)

        per_model_mean: dict[str, float] = {
            name: statistics.mean(vals) for name, vals in model_probas_list.items()
        }

        if len(per_model_mean) >= 2:  # noqa: PLR2004
            disagreement = statistics.stdev(list(per_model_mean.values()))
        else:
            disagreement = 0.0

        return disagreement, per_model_mean

    # ── Acceptance gates ────────────────────────────────────────────────────

    def evaluate_gates(
        self,
        metrics: IterationMetrics,
        baseline: IterationMetrics | None,
    ) -> tuple[list[GateResult], str]:
        """Evaluate safety and calibration gates, return (results, verdict)."""
        gates: list[GateResult] = [
            self._gate_s1(metrics),
            self._gate_s2(metrics),
            self._gate_s3(metrics),
            self._gate_c1(metrics, baseline),
            self._gate_c2(metrics, baseline),
            self._gate_c3(metrics),
            self._gate_c4(metrics),
            self._gate_c5(metrics, baseline),
            self._gate_c6(metrics),
        ]

        safety_failed = any(not g.passed for g in gates if g.gate_type == "safety")
        calibration_failures = sum(
            1 for g in gates if g.gate_type == "calibration" and not g.passed
        )

        if safety_failed or calibration_failures >= _CALIBRATION_REJECT_THRESHOLD:
            verdict = "REJECT"
        elif calibration_failures == 1:
            verdict = "WARN"
        else:
            verdict = "PASS"

        return gates, verdict

    @staticmethod
    def _gate_s1(metrics: IterationMetrics) -> GateResult:
        passed = metrics.wf_sharpe >= _S1_SHARPE_FLOOR
        return GateResult(
            name="S1",
            gate_type="safety",
            passed=passed,
            value=metrics.wf_sharpe,
            threshold=_S1_SHARPE_FLOOR,
            message=("WF Sharpe >= 0.0" if passed else f"WF Sharpe {metrics.wf_sharpe:.4f} < 0.0"),
        )

    @staticmethod
    def _gate_s2(metrics: IterationMetrics) -> GateResult:
        passed = metrics.wf_max_drawdown < _S2_MAX_DD_CEILING
        return GateResult(
            name="S2",
            gate_type="safety",
            passed=passed,
            value=metrics.wf_max_drawdown,
            threshold=_S2_MAX_DD_CEILING,
            message=(
                "Max DD within ceiling"
                if passed
                else (f"Max DD {metrics.wf_max_drawdown:.1f}% >= {_S2_MAX_DD_CEILING}%")
            ),
        )

    @staticmethod
    def _gate_s3(metrics: IterationMetrics) -> GateResult:
        passed = metrics.mc_5th_pct_sharpe >= _S3_MC_SHARPE_FLOOR
        return GateResult(
            name="S3",
            gate_type="safety",
            passed=passed,
            value=metrics.mc_5th_pct_sharpe,
            threshold=_S3_MC_SHARPE_FLOOR,
            message=(
                "MC 5th-pct Sharpe >= 0.0"
                if passed
                else (f"MC 5th-pct Sharpe {metrics.mc_5th_pct_sharpe:.4f} < 0.0")
            ),
        )

    @staticmethod
    def _gate_c1(
        metrics: IterationMetrics,
        baseline: IterationMetrics | None,
    ) -> GateResult:
        if baseline is None:
            return GateResult(
                name="C1",
                gate_type="calibration",
                passed=True,
                value=0.0,
                threshold=_C1_SHARPE_REGRESSION_LIMIT,
                message="No baseline for comparison",
            )
        regression = baseline.wf_sharpe - metrics.wf_sharpe
        passed = regression <= _C1_SHARPE_REGRESSION_LIMIT
        return GateResult(
            name="C1",
            gate_type="calibration",
            passed=passed,
            value=round(regression, 4),
            threshold=_C1_SHARPE_REGRESSION_LIMIT,
            message=(
                "Sharpe regression within tolerance"
                if passed
                else (f"Sharpe regressed by {regression:.4f} > {_C1_SHARPE_REGRESSION_LIMIT}")
            ),
        )

    @staticmethod
    def _gate_c2(
        metrics: IterationMetrics,
        baseline: IterationMetrics | None,
    ) -> GateResult:
        if baseline is None:
            return GateResult(
                name="C2",
                gate_type="calibration",
                passed=True,
                value=0.0,
                threshold=_C2_DD_REGRESSION_LIMIT,
                message="No baseline for comparison",
            )
        increase = metrics.wf_max_drawdown - baseline.wf_max_drawdown
        passed = increase <= _C2_DD_REGRESSION_LIMIT
        return GateResult(
            name="C2",
            gate_type="calibration",
            passed=passed,
            value=round(increase, 4),
            threshold=_C2_DD_REGRESSION_LIMIT,
            message=(
                "DD regression within tolerance"
                if passed
                else (f"DD increased by {increase:.1f}% > {_C2_DD_REGRESSION_LIMIT}%")
            ),
        )

    @staticmethod
    def _gate_c3(metrics: IterationMetrics) -> GateResult:
        """Trade count >= 60 per fold."""
        passed = metrics.trade_count >= _C3_MIN_TRADES_PER_FOLD
        return GateResult(
            name="C3",
            gate_type="calibration",
            passed=passed,
            value=float(metrics.trade_count),
            threshold=float(_C3_MIN_TRADES_PER_FOLD),
            message=(
                "Sufficient trades"
                if passed
                else (f"Trade count {metrics.trade_count} < {_C3_MIN_TRADES_PER_FOLD}")
            ),
        )

    @staticmethod
    def _gate_c4(metrics: IterationMetrics) -> GateResult:
        max_share = max(metrics.segment_pnl_share.values()) if metrics.segment_pnl_share else 0.0
        passed = max_share <= _C4_MAX_SEGMENT_SHARE
        return GateResult(
            name="C4",
            gate_type="calibration",
            passed=passed,
            value=round(max_share, 4),
            threshold=_C4_MAX_SEGMENT_SHARE,
            message=(
                "Segment diversification OK"
                if passed
                else (f"Max segment share {max_share:.1%} > {_C4_MAX_SEGMENT_SHARE:.0%}")
            ),
        )

    @staticmethod
    def _gate_c5(
        metrics: IterationMetrics,
        baseline: IterationMetrics | None,
    ) -> GateResult:
        if baseline is None:
            return GateResult(
                name="C5",
                gate_type="calibration",
                passed=True,
                value=metrics.mc_5th_pct_sharpe,
                threshold=0.0,
                message="No baseline for comparison",
            )
        passed = metrics.mc_5th_pct_sharpe >= baseline.mc_5th_pct_sharpe
        return GateResult(
            name="C5",
            gate_type="calibration",
            passed=passed,
            value=metrics.mc_5th_pct_sharpe,
            threshold=baseline.mc_5th_pct_sharpe,
            message=(
                "MC robustness maintained"
                if passed
                else (
                    f"MC 5th-pct declined: "
                    f"{metrics.mc_5th_pct_sharpe:.4f} "
                    f"< {baseline.mc_5th_pct_sharpe:.4f}"
                )
            ),
        )

    @staticmethod
    def _gate_c6(metrics: IterationMetrics) -> GateResult:
        passed = metrics.param_stability_cv < _C6_MAX_PARAM_CV
        return GateResult(
            name="C6",
            gate_type="calibration",
            passed=passed,
            value=metrics.param_stability_cv,
            threshold=_C6_MAX_PARAM_CV,
            message=(
                "Param stability OK"
                if passed
                else (f"Param CV {metrics.param_stability_cv:.2f} >= {_C6_MAX_PARAM_CV}")
            ),
        )

    # ── Persistence ─────────────────────────────────────────────────────────

    def save(self, metadata: IterationMetadata) -> Path:
        """Atomic write of iteration data to results_root/<name>/."""
        iter_dir = self._root / metadata.name
        iter_dir.mkdir(parents=True, exist_ok=True)

        target = iter_dir / "metadata.json"
        self._atomic_write(target, metadata.model_dump_json(indent=2))

        history_entry = {
            "name": metadata.name,
            "created_at": metadata.created_at.isoformat(),
            "git_sha": metadata.git_sha,
            "verdict": metadata.verdict,
            "wf_sharpe": metadata.metrics.wf_sharpe,
            "wf_max_drawdown": metadata.metrics.wf_max_drawdown,
            "trade_count": metadata.metrics.trade_count,
        }
        history_path = self._root / "history.jsonl"
        with history_path.open("a") as f:
            f.write(json.dumps(history_entry) + "\n")

        return iter_dir

    def load(self, name: str) -> IterationMetadata:
        """Load iteration by name from results_root/<name>/metadata.json."""
        path = self._root / name / "metadata.json"
        data = json.loads(path.read_text())
        return IterationMetadata(**data)

    def load_latest(self) -> IterationMetadata | None:
        """Load the most recent iteration from history.jsonl."""
        history_path = self._root / "history.jsonl"
        if not history_path.exists():
            return None
        lines = history_path.read_text().strip().split("\n")
        if not lines or not lines[-1].strip():
            return None
        last_entry = json.loads(lines[-1])
        return self.load(last_entry["name"])

    def compare(
        self,
        current: str,
        baseline: str,
    ) -> IterationComparison:
        """Compute metric deltas between two iterations."""
        current_meta = self.load(current)
        baseline_meta = self.load(baseline)

        cm = current_meta.metrics
        bm = baseline_meta.metrics

        deltas: dict[str, float] = {
            "wf_sharpe": round(cm.wf_sharpe - bm.wf_sharpe, 4),
            "wf_max_drawdown": round(cm.wf_max_drawdown - bm.wf_max_drawdown, 4),
            "profit_factor": round(cm.profit_factor - bm.profit_factor, 4),
            "calmar_ratio": round(cm.calmar_ratio - bm.calmar_ratio, 4),
            "trade_count": float(cm.trade_count - bm.trade_count),
            "avg_hold_bars": round(cm.avg_hold_bars - bm.avg_hold_bars, 4),
            "sortino_ratio": round(cm.sortino_ratio - bm.sortino_ratio, 4),
            "mc_5th_pct_sharpe": round(cm.mc_5th_pct_sharpe - bm.mc_5th_pct_sharpe, 4),
            "model_disagreement": round(cm.model_disagreement - bm.model_disagreement, 4),
            "param_stability_cv": round(cm.param_stability_cv - bm.param_stability_cv, 4),
        }

        gate_results, verdict = self.evaluate_gates(current_meta.metrics, baseline_meta.metrics)

        return IterationComparison(
            current=current,
            baseline=baseline,
            metric_deltas=deltas,
            gate_results=gate_results,
            verdict=verdict,
        )

    def list_iterations(self) -> list[dict[str, Any]]:
        """Read history.jsonl and return list of iteration summaries."""
        history_path = self._root / "history.jsonl"
        if not history_path.exists():
            return []
        return [
            json.loads(line)
            for line in history_path.read_text().strip().split("\n")
            if line.strip()
        ]

    # ── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _atomic_write(target: Path, content: str) -> None:
        """Write content to target via temp file + rename for atomicity."""
        target.parent.mkdir(parents=True, exist_ok=True)
        fd = tempfile.NamedTemporaryFile(  # noqa: SIM115
            dir=target.parent,
            suffix=".tmp",
            delete=False,
            mode="w",
        )
        try:
            fd.write(content)
            fd.flush()
            os.fsync(fd.fileno())
            fd.close()
            os.rename(fd.name, target)
        except Exception:
            fd.close()
            with contextlib.suppress(OSError):
                os.unlink(fd.name)
            raise
