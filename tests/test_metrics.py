"""Tests for evaluation metrics and comparison-table assembly."""

from __future__ import annotations

import numpy as np

from src.config import Config
from src.evaluation.comparison import build_comparison_table
from src.evaluation.metrics import (
    EvaluationResult,
    classification_metrics,
    full_metrics,
)


def test_perfect_classification_metrics() -> None:
    y_true = np.array([0, 1, 0, 1])
    metrics = classification_metrics(y_true, y_true)
    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["false_pass_rate"] == 0.0
    assert metrics["false_fail_rate"] == 0.0


def test_false_pass_rate_definition() -> None:
    # Two true FAILs; one is wrongly predicted PASS -> false-pass rate 0.5.
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([1, 0, 0, 0])
    metrics = classification_metrics(y_true, y_pred)
    assert metrics["false_pass_rate"] == 0.5
    assert metrics["false_fail_rate"] == 0.0


def test_full_metrics_cost_reduction(small_config: Config) -> None:
    result = EvaluationResult(
        true_labels=np.array([0, 1]),
        predicted_labels=np.array([0, 1]),
        rewards=np.array([10.0, 10.0]),
        test_costs=np.array([1.0, 1.0]),
    )
    metrics = full_metrics(result, small_config)
    full_cost = small_config.env.n_stages * small_config.reward.test_cost
    expected = (full_cost - 1.0) / full_cost * 100.0
    assert abs(metrics["cost_reduction_pct"] - expected) < 1e-6


def test_build_comparison_table() -> None:
    results = {
        "A": {"accuracy": 0.9, "f1": 0.8, "cost_reduction_pct": 50.0},
        "B": {"accuracy": 0.7, "f1": 0.6, "cost_reduction_pct": 10.0},
    }
    table = build_comparison_table(results)
    assert list(table.index) == ["A", "B"]
    assert "Accuracy" in table.columns
    assert table.loc["A", "Accuracy"] == 0.9
