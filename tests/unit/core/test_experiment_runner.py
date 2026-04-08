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
            "--name", "test-iter",
            "--description", "test desc",
            "--hypothesis", "test-001",
            "--run-name", "A-only",
        ]
        with patch("sys.argv", test_argv):
            args = _parse_args()
        assert args.hypothesis == "test-001"
        assert args.run_name == "A-only"

    def test_hypothesis_not_set_backward_compat(self) -> None:
        test_argv = [
            "run_iteration.py",
            "--name", "test-iter",
            "--description", "test desc",
        ]
        with patch("sys.argv", test_argv):
            args = _parse_args()
        assert args.hypothesis is None
        assert args.run_name == "main"
