"""Tests for experiment-linked backtest runner integration.

Tests --hypothesis / --run-name CLI flags in run_iteration.py and
interaction test comparison table formatting.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.run_interaction_test import (  # noqa: E402, I001
    _format_comparison_table,
    _parse_args as _parse_interaction_args,
)
from scripts.run_iteration import _deep_merge, _parse_args  # noqa: E402


class TestDeepMerge:
    """Tests for _deep_merge helper."""

    def test_deep_merge_basic(self) -> None:
        result = _deep_merge({"a": 1, "b": 2}, {"b": 3})
        assert result == {"a": 1, "b": 3}

    def test_deep_merge_nested(self) -> None:
        result = _deep_merge({"a": {"x": 1, "y": 2}}, {"a": {"y": 3}})
        assert result == {"a": {"x": 1, "y": 3}}

    def test_deep_merge_new_key(self) -> None:
        result = _deep_merge({"a": 1}, {"b": 2})
        assert result == {"a": 1, "b": 2}

    def test_deep_merge_does_not_mutate_base(self) -> None:
        base = {"a": {"x": 1}}
        override = {"a": {"y": 2}}
        _deep_merge(base, override)
        assert base == {"a": {"x": 1}}

    def test_deep_merge_empty_override(self) -> None:
        result = _deep_merge({"a": 1}, {})
        assert result == {"a": 1}

    def test_deep_merge_empty_base(self) -> None:
        result = _deep_merge({}, {"b": 2})
        assert result == {"b": 2}


class TestHypothesisArgs:
    """Tests for --hypothesis and --run-name CLI arg parsing."""

    def test_hypothesis_args_parsed(self) -> None:
        test_argv = [
            "run_iteration.py",
            "--name",
            "test-iter",
            "--description",
            "test desc",
            "--hypothesis",
            "test-001",
            "--run-name",
            "A-only",
        ]
        with patch("sys.argv", test_argv):
            args = _parse_args()
        assert args.hypothesis == "test-001"
        assert args.run_name == "A-only"

    def test_hypothesis_not_set_backward_compat(self) -> None:
        test_argv = [
            "run_iteration.py",
            "--name",
            "test-iter",
            "--description",
            "test desc",
        ]
        with patch("sys.argv", test_argv):
            args = _parse_args()
        assert args.hypothesis is None
        assert args.run_name == "main"


class TestComparisonTable:
    """Tests for interaction test comparison table formatting."""

    def test_comparison_table_format(self) -> None:
        a = {
            "wf_sharpe": 0.1000,
            "profit_factor": 1.20,
            "wf_max_drawdown": 0.0500,
            "trade_count": 100,
        }
        b = {
            "wf_sharpe": 0.0800,
            "profit_factor": 1.10,
            "wf_max_drawdown": 0.0600,
            "trade_count": 80,
        }
        ab = {
            "wf_sharpe": 0.1500,
            "profit_factor": 1.30,
            "wf_max_drawdown": 0.0400,
            "trade_count": 120,
        }

        table = _format_comparison_table(a, b, ab)
        assert "| Metric |" in table
        assert "WF Sharpe" in table
        assert "Profit Factor" in table
        assert "Max Drawdown" in table
        assert "Trade Count" in table
        # Verify delta columns exist
        assert "Delta(A)" in table
        assert "Delta(B)" in table
        # Verify A+B values present
        assert "0.1500" in table
        assert "1.3000" in table
        assert "120" in table

    def test_comparison_table_with_zero_values(self) -> None:
        a = {"wf_sharpe": 0.0, "profit_factor": 0.0, "wf_max_drawdown": 0.0, "trade_count": 0}
        b = {"wf_sharpe": 0.0, "profit_factor": 0.0, "wf_max_drawdown": 0.0, "trade_count": 0}
        ab = {"wf_sharpe": 0.0, "profit_factor": 0.0, "wf_max_drawdown": 0.0, "trade_count": 0}
        table = _format_comparison_table(a, b, ab)
        assert "WF Sharpe" in table


class TestInteractionArgs:
    """Tests for interaction test CLI arg parsing."""

    def test_interaction_args_parsed(self) -> None:
        test_argv = [
            "run_interaction_test.py",
            "--experiment-a",
            "exp-001",
            "--experiment-b",
            "exp-002",
            "--segments",
            "us_tech",
        ]
        with patch("sys.argv", test_argv):
            args = _parse_interaction_args()
        assert args.experiment_a == "exp-001"
        assert args.experiment_b == "exp-002"
        assert args.segments == "us_tech"
