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
from src.environment.actions import LABEL_FAIL, LABEL_PASS


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
    # Multi-stage extras: the stage index at which each episode stopped
    # (0 = before Stage-2, 1 = after Stage-2, 2 = sent to Stage-3) and a
    # per-chip flag for chips that failed Stage-2.
    stages_stopped: np.ndarray | None = None
    is_stage2_fail: np.ndarray | None = None


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


def full_metrics(
    result: EvaluationResult,
    config: Config = CONFIG,
    *,
    full_testing_cost: float | None = None,
) -> dict[str, float]:
    """Combine classification and cost metrics into a single report.

    Args:
        result: The collected per-chip evaluation outcomes.
        config: Project configuration (for full-testing cost reference).
        full_testing_cost: Cost of fully testing one chip. Defaults to
            Stage-2 + Stage-3 costs from the reward profile; callers may
            override explicitly.

    Returns:
        Dict including classification metrics plus average reward, average test
        cost and cost-reduction percentage relative to full testing. For
        multi-stage rollouts additional stage-routing metrics are included.
    """
    metrics = classification_metrics(result.true_labels, result.predicted_labels)
    metrics["avg_reward"] = float(np.mean(result.rewards))
    metrics["avg_test_cost"] = float(np.mean(result.test_costs))

    full_cost = (
        full_testing_cost
        if full_testing_cost is not None
        else config.reward.stage_cost(1) + config.reward.stage_cost(2)
    )
    metrics["cost_reduction_pct"] = (
        float((full_cost - metrics["avg_test_cost"]) / full_cost * 100.0)
        if full_cost
        else 0.0
    )
    # Average number of test stages run per chip. This is independent of the
    # reward profile's per-stage cost, so it is directly comparable across runs.
    if result.tests_run is not None:
        metrics["avg_tests_run"] = float(np.mean(result.tests_run))
    metrics.update(multi_stage_metrics(result))
    metrics["n_samples"] = int(len(result.true_labels))
    return metrics


def multi_stage_metrics(result: EvaluationResult) -> dict[str, float]:
    """Compute stage-routing metrics for a multi-stage rollout.

    Returns an empty dict when the result has no stage information (single-stage
    rollouts or supervised baselines without stage routing).

    Metrics:
        * ``pct_stopped_before_stage2`` - stopped at State 0 (metadata only).
        * ``pct_stopped_after_stage2`` - stopped at State 1 (ran Stage-2 only).
        * ``pct_sent_to_stage3`` - reached State 2 (ran Stage-3).
        * ``pct_stage2_fail_correctly_stopped`` - of Stage-2-failed chips, the
          fraction classified FAIL.
        * ``pct_stage2_fail_incorrectly_passed`` - of Stage-2-failed chips, the
          fraction classified PASS.
    """
    if result.stages_stopped is None:
        return {}
    stages = np.asarray(result.stages_stopped)
    n = len(stages)
    metrics: dict[str, float] = {
        "pct_stopped_before_stage2": float(np.mean(stages == 0) * 100.0) if n else 0.0,
        "pct_stopped_after_stage2": float(np.mean(stages == 1) * 100.0) if n else 0.0,
        "pct_sent_to_stage3": float(np.mean(stages >= 2) * 100.0) if n else 0.0,
    }
    if result.is_stage2_fail is not None:
        s2_fail = np.asarray(result.is_stage2_fail).astype(bool)
        preds = np.asarray(result.predicted_labels).astype(int)
        n_fail = int(s2_fail.sum())
        if n_fail:
            caught = float(np.mean(preds[s2_fail] == LABEL_FAIL) * 100.0)
            metrics["pct_stage2_fail_correctly_stopped"] = caught
            metrics["pct_stage2_fail_incorrectly_passed"] = float(
                np.mean(preds[s2_fail] == LABEL_PASS) * 100.0
            )
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
