"""Classification and cost metrics for the chip-testing problem.

Label convention: ``1 = FAIL`` (defective) is treated as the positive class,
because detecting defects is the safety-critical objective.

* **False Pass** (FN): a defective chip classified PASS - it escapes to the
  customer. This is the most costly error.
* **False Fail** (FP): a good chip classified FAIL - it is needlessly scrapped.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.config import CONFIG, Config
from src.environment.chip_testing_env import LABEL_FAIL, LABEL_PASS


@dataclass
class EvaluationResult:
    """Per-chip outcomes collected during a policy rollout or prediction."""

    true_labels: np.ndarray
    predicted_labels: np.ndarray
    rewards: np.ndarray
    test_costs: np.ndarray
    action_counts: dict[int, int] = field(default_factory=dict)
    # Number of test stages run per chip (profile-independent test effort).
    tests_run: np.ndarray | None = None


def classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> dict[str, float]:
    """Compute classification metrics with FAIL as the positive class.

    Args:
        y_true: Ground-truth labels (0 = PASS, 1 = FAIL).
        y_pred: Predicted labels.

    Returns:
        Dict with accuracy, precision, recall, F1, false-pass and false-fail
        rates.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, pos_label=LABEL_FAIL, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=LABEL_FAIL, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, pos_label=LABEL_FAIL, zero_division=0)),
    }

    cm = confusion_matrix(y_true, y_pred, labels=[LABEL_PASS, LABEL_FAIL])
    tn, fp, fn, tp = cm.ravel()
    actual_fail = tp + fn
    actual_pass = tn + fp
    metrics["false_pass_rate"] = float(fn / actual_fail) if actual_fail else 0.0
    metrics["false_fail_rate"] = float(fp / actual_pass) if actual_pass else 0.0
    return metrics


def full_metrics(result: EvaluationResult, config: Config = CONFIG) -> dict[str, float]:
    """Combine classification and cost metrics into a single report.

    Args:
        result: The collected per-chip evaluation outcomes.
        config: Project configuration (for full-testing cost reference).

    Returns:
        Dict including classification metrics plus average reward, average test
        cost and cost-reduction percentage relative to full testing.
    """
    metrics = classification_metrics(result.true_labels, result.predicted_labels)
    metrics["avg_reward"] = float(np.mean(result.rewards))
    metrics["avg_test_cost"] = float(np.mean(result.test_costs))

    full_cost = config.env.n_stages * config.reward.test_cost
    metrics["cost_reduction_pct"] = (
        float((full_cost - metrics["avg_test_cost"]) / full_cost * 100.0)
        if full_cost
        else 0.0
    )
    # Average number of test stages run per chip. This is independent of the
    # reward profile's per-stage cost, so it is directly comparable across runs.
    if result.tests_run is not None:
        metrics["avg_tests_run"] = float(np.mean(result.tests_run))
    metrics["n_samples"] = int(len(result.true_labels))
    return metrics


def confusion_matrix_counts(
    y_true: np.ndarray, y_pred: np.ndarray
) -> np.ndarray:
    """Return the 2x2 confusion matrix with rows/cols ordered [PASS, FAIL]."""
    return confusion_matrix(
        np.asarray(y_true).astype(int),
        np.asarray(y_pred).astype(int),
        labels=[LABEL_PASS, LABEL_FAIL],
    )
