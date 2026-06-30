"""Matplotlib plotting helpers that save figures to ``results/figures``.

All functions use the non-interactive ``Agg`` backend and return the path of
the saved figure so callers (scripts and notebooks) can log/display it.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Safe for headless script/CI execution.

import matplotlib.pyplot as plt
import numpy as np

from src.environment.actions import Action
from src.utils.helpers import get_logger

logger = get_logger(__name__)


def _save(fig: plt.Figure, path: Path | str) -> Path:
    """Save and close a figure, creating parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved figure to %s", path)
    return path


def plot_reward_curve(
    rewards: list[float] | np.ndarray,
    path: Path | str,
    *,
    title: str = "Training Reward Curve",
    smoothing_window: int = 50,
) -> Path:
    """Plot a (smoothed) reward-vs-episode curve.

    Args:
        rewards: Per-episode rewards.
        path: Output figure path.
        title: Plot title.
        smoothing_window: Moving-average window for the smoothed overlay.

    Returns:
        The saved figure path.
    """
    rewards = np.asarray(rewards, dtype=float)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(rewards, alpha=0.3, label="Episode reward", color="tab:blue")
    if len(rewards) >= smoothing_window > 1:
        kernel = np.ones(smoothing_window) / smoothing_window
        smoothed = np.convolve(rewards, kernel, mode="valid")
        offset = smoothing_window - 1
        ax.plot(
            np.arange(offset, offset + len(smoothed)),
            smoothed,
            color="tab:red",
            label=f"Moving avg ({smoothing_window})",
        )
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    return _save(fig, path)


def plot_confusion_matrix(
    cm: np.ndarray,
    path: Path | str,
    *,
    title: str = "Confusion Matrix",
    labels: tuple[str, str] = ("PASS", "FAIL"),
) -> Path:
    """Plot a 2x2 confusion matrix heatmap.

    Args:
        cm: Confusion matrix ordered ``[PASS, FAIL]`` on both axes.
        path: Output figure path.
        title: Plot title.
        labels: Class labels for ticks.

    Returns:
        The saved figure path.
    """
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set_xticks([0, 1], labels=labels)
    ax.set_yticks([0, 1], labels=labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    threshold = cm.max() / 2.0 if cm.max() else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                int(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > threshold else "black",
            )
    return _save(fig, path)


def plot_cost_savings(
    cost_reduction_by_method: dict[str, float],
    path: Path | str,
    *,
    title: str = "Cost Reduction by Method",
) -> Path:
    """Bar chart of cost-reduction percentage per method.

    Args:
        cost_reduction_by_method: Mapping ``method -> cost_reduction_pct``.
        path: Output figure path.
        title: Plot title.

    Returns:
        The saved figure path.
    """
    methods = list(cost_reduction_by_method.keys())
    values = [cost_reduction_by_method[m] for m in methods]
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(methods, values, color="tab:green", alpha=0.8)
    ax.set_ylabel("Cost Reduction (%)")
    ax.set_title(title)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.grid(True, axis="y", alpha=0.3)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.1f}%",
            ha="center",
            va="bottom",
        )
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    return _save(fig, path)


def plot_action_distribution(
    action_counts: dict[int, int],
    path: Path | str,
    *,
    title: str = "Policy Action Distribution",
) -> Path:
    """Bar chart of how often each action was taken by a policy.

    Args:
        action_counts: Mapping ``action_index -> count``.
        path: Output figure path.
        title: Plot title.

    Returns:
        The saved figure path.
    """
    names = {a.value: a.name for a in Action}
    actions = sorted(names.keys())
    counts = [action_counts.get(a, 0) for a in actions]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar([names[a] for a in actions], counts, color="tab:purple", alpha=0.8)
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    return _save(fig, path)


def plot_precision_recall_comparison(
    metrics_by_method: dict[str, dict[str, float]],
    path: Path | str,
    *,
    title: str = "Precision / Recall / F1 Comparison",
) -> Path:
    """Grouped bar chart comparing precision, recall and F1 across methods.

    Args:
        metrics_by_method: Mapping ``method -> metrics_dict``.
        path: Output figure path.
        title: Plot title.

    Returns:
        The saved figure path.
    """
    methods = list(metrics_by_method.keys())
    metric_keys = ["precision", "recall", "f1"]
    x = np.arange(len(methods))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, key in enumerate(metric_keys):
        values = [metrics_by_method[m].get(key, 0.0) for m in methods]
        ax.bar(x + (i - 1) * width, values, width, label=key.capitalize())
    ax.set_xticks(x, labels=methods)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    return _save(fig, path)
