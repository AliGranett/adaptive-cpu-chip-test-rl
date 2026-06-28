"""Publication-quality figures for the reward-sensitivity sweep.

All figures are saved as both PNG and PDF under ``<output>/plots/`` with
consistent styling (readable fonts, grids, legends). Functions are defensive:
they degrade gracefully when a single seed is used (no error bars) or when an
optional metric column is missing.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

plt.rcParams.update(
    {
        "figure.dpi": 120,
        "savefig.dpi": 200,
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "axes.grid": True,
        "grid.alpha": 0.3,
    }
)

_SEED_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
_PENALTY_FMT = "{:.0f}"


def _save(fig: plt.Figure, plots_dir: Path, name: str) -> None:
    """Save a figure as PNG and PDF, then close it."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(plots_dir / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)


def _penalty_label(value: float) -> str:
    return _PENALTY_FMT.format(value)


# ---------------------------------------------------------------------------
# Figure 1 - Pareto frontier (aggregated)
# ---------------------------------------------------------------------------
def figure_pareto_aggregated(by_penalty: pd.DataFrame, plots_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    x = by_penalty["FalsePassRate_mean"].to_numpy()
    y = by_penalty["CostReduction_mean"].to_numpy()
    xerr = by_penalty.get("FalsePassRate_std", pd.Series(np.zeros(len(x)))).to_numpy()
    yerr = by_penalty.get("CostReduction_std", pd.Series(np.zeros(len(y)))).to_numpy()

    ax.errorbar(
        x, y, xerr=xerr, yerr=yerr, fmt="o", color="#4C72B0", ecolor="#aaaaaa",
        elinewidth=1, capsize=3, markersize=8, label="DQN policy (mean +/- std)",
    )
    for _, row in by_penalty.iterrows():
        ax.annotate(
            _penalty_label(row["Penalty"]),
            (row["FalsePassRate_mean"], row["CostReduction_mean"]),
            textcoords="offset points", xytext=(6, 6), fontsize=9,
        )

    if "Pareto_Optimal" in by_penalty.columns and by_penalty["Pareto_Optimal"].any():
        pareto = by_penalty[by_penalty["Pareto_Optimal"]].sort_values("FalsePassRate_mean")
        ax.plot(
            pareto["FalsePassRate_mean"], pareto["CostReduction_mean"],
            "-", color="#C44E52", linewidth=2, zorder=1,
        )
        ax.scatter(
            pareto["FalsePassRate_mean"], pareto["CostReduction_mean"],
            s=160, facecolors="none", edgecolors="#C44E52", linewidths=2,
            label="Pareto-optimal", zorder=3,
        )

    ax.set_xlabel("False Pass Rate (mean)")
    ax.set_ylabel("Cost Reduction % (mean)")
    ax.set_title("Pareto Frontier of DQN Test Policies")
    ax.legend()
    _save(fig, plots_dir, "figure_1_pareto_aggregated")


# ---------------------------------------------------------------------------
# Figure 2 - per-seed Pareto scatter
# ---------------------------------------------------------------------------
def figure_per_seed_scatter(all_runs: pd.DataFrame, plots_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    seeds = sorted(all_runs["Seed"].unique())
    cmap = plt.get_cmap("tab10")
    for i, seed in enumerate(seeds):
        sub = all_runs[all_runs["Seed"] == seed]
        ax.scatter(
            sub["FalsePassRate"], sub["CostReduction"],
            marker=_SEED_MARKERS[i % len(_SEED_MARKERS)], s=90,
            color=cmap(i), label=f"seed {seed}", edgecolors="black", linewidths=0.5,
        )
        for _, row in sub.iterrows():
            ax.annotate(
                _penalty_label(row["Penalty"]),
                (row["FalsePassRate"], row["CostReduction"]),
                textcoords="offset points", xytext=(5, 4), fontsize=8,
            )
    ax.set_xlabel("False Pass Rate")
    ax.set_ylabel("Cost Reduction %")
    ax.set_title("Per-Seed DQN Policies (stability across seeds)")
    ax.legend(title="Seed")
    _save(fig, plots_dir, "figure_2_per_seed_scatter")


# ---------------------------------------------------------------------------
# Figures 3 & 4 - metric vs penalty line plots
# ---------------------------------------------------------------------------
def _line_vs_penalty(
    all_runs: pd.DataFrame,
    by_penalty: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    plots_dir: Path,
    name: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    seeds = sorted(all_runs["Seed"].unique())
    cmap = plt.get_cmap("tab10")
    for i, seed in enumerate(seeds):
        sub = all_runs[all_runs["Seed"] == seed].sort_values("Penalty")
        ax.plot(
            sub["Penalty"], sub[metric], marker=_SEED_MARKERS[i % len(_SEED_MARKERS)],
            color=cmap(i), alpha=0.7, linewidth=1.3, label=f"seed {seed}",
        )
    mean_col = f"{metric}_mean"
    if mean_col in by_penalty.columns:
        agg = by_penalty.sort_values("Penalty")
        ax.plot(
            agg["Penalty"], agg[mean_col], color="black", linewidth=3,
            marker="o", markersize=8, label="across-seed mean", zorder=5,
        )
    ax.set_xlabel("False Pass Penalty")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    _save(fig, plots_dir, name)


def figure_fpr_vs_penalty(all_runs, by_penalty, plots_dir: Path) -> None:
    _line_vs_penalty(
        all_runs, by_penalty, "FalsePassRate", "False Pass Rate",
        "False Pass Rate vs False Pass Penalty", plots_dir, "figure_3_fpr_vs_penalty",
    )


def figure_cost_vs_penalty(all_runs, by_penalty, plots_dir: Path) -> None:
    _line_vs_penalty(
        all_runs, by_penalty, "CostReduction", "Cost Reduction %",
        "Cost Reduction vs False Pass Penalty", plots_dir, "figure_4_cost_vs_penalty",
    )


# ---------------------------------------------------------------------------
# Figure 5 - dual-axis (aggregated)
# ---------------------------------------------------------------------------
def figure_dual_axis(by_penalty: pd.DataFrame, plots_dir: Path) -> None:
    agg = by_penalty.sort_values("Penalty")
    x = agg["Penalty"].to_numpy()
    fig, ax1 = plt.subplots(figsize=(8, 6))

    color1 = "#C44E52"
    fpr = agg["FalsePassRate_mean"].to_numpy()
    fpr_std = agg.get("FalsePassRate_std", pd.Series(np.zeros(len(x)))).to_numpy()
    ax1.plot(x, fpr, color=color1, marker="o", linewidth=2, label="False Pass Rate")
    ax1.fill_between(x, fpr - fpr_std, fpr + fpr_std, color=color1, alpha=0.15)
    ax1.set_xlabel("False Pass Penalty")
    ax1.set_ylabel("False Pass Rate (mean)", color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)

    ax2 = ax1.twinx()
    ax2.grid(False)
    color2 = "#4C72B0"
    cost = agg["CostReduction_mean"].to_numpy()
    cost_std = agg.get("CostReduction_std", pd.Series(np.zeros(len(x)))).to_numpy()
    ax2.plot(x, cost, color=color2, marker="s", linewidth=2, label="Cost Reduction %")
    ax2.fill_between(x, cost - cost_std, cost + cost_std, color=color2, alpha=0.15)
    ax2.set_ylabel("Cost Reduction % (mean)", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)

    ax1.set_title("False Pass Rate and Cost Reduction vs Penalty")
    _save(fig, plots_dir, "figure_5_dual_axis")


# ---------------------------------------------------------------------------
# Figure 6 - average tests per chip (bar)
# ---------------------------------------------------------------------------
def figure_avg_tests_bar(by_penalty: pd.DataFrame, plots_dir: Path) -> None:
    agg = by_penalty.sort_values("Penalty")
    labels = [_penalty_label(p) for p in agg["Penalty"]]
    means = agg["AverageTests_mean"].to_numpy()
    stds = agg.get("AverageTests_std", pd.Series(np.zeros(len(means)))).to_numpy()
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(labels, means, yerr=stds, capsize=4, color="#55A868", edgecolor="black")
    ax.set_xlabel("False Pass Penalty")
    ax.set_ylabel("Average Tests Per Chip (mean)")
    ax.set_title("Average Tests Per Chip vs Penalty")
    _save(fig, plots_dir, "figure_6_avg_tests")


# ---------------------------------------------------------------------------
# Figure 7 - grouped classification metrics (bar)
# ---------------------------------------------------------------------------
def figure_grouped_metrics(by_penalty: pd.DataFrame, plots_dir: Path) -> None:
    agg = by_penalty.sort_values("Penalty")
    labels = [_penalty_label(p) for p in agg["Penalty"]]
    metrics = [("Accuracy_mean", "Accuracy"), ("F1_mean", "F1"),
               ("Precision_mean", "Precision"), ("Recall_mean", "Recall")]
    metrics = [(c, n) for c, n in metrics if c in agg.columns]
    x = np.arange(len(labels))
    width = 0.8 / max(len(metrics), 1)
    fig, ax = plt.subplots(figsize=(9, 6))
    cmap = plt.get_cmap("Set2")
    for i, (col, name) in enumerate(metrics):
        ax.bar(x + i * width - 0.4 + width / 2, agg[col].to_numpy(),
               width, label=name, color=cmap(i), edgecolor="black", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("False Pass Penalty")
    ax.set_ylabel("Score (mean)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Classification Metrics vs Penalty")
    ax.legend(ncol=len(metrics))
    _save(fig, plots_dir, "figure_7_grouped_metrics")


# ---------------------------------------------------------------------------
# Figure 8 - policy routing
# ---------------------------------------------------------------------------
def figure_policy_routing(by_penalty: pd.DataFrame, plots_dir: Path) -> None:
    agg = by_penalty.sort_values("Penalty")
    labels = [_penalty_label(p) for p in agg["Penalty"]]
    routing = [
        ("PctStopBeforeStage2_mean", "Stop Before Stage 2", "#4C72B0"),
        ("PctStopAfterStage2_mean", "Stop After Stage 2", "#DD8452"),
        ("PctContinueToStage3_mean", "Continue To Stage 3", "#55A868"),
    ]
    routing = [(c, n, col) for c, n, col in routing if c in agg.columns]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9, 6))
    bottom = np.zeros(len(labels))
    for col, name, color in routing:
        vals = agg[col].to_numpy()
        ax.bar(x, vals, bottom=bottom, label=name, color=color, edgecolor="black", linewidth=0.4)
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("False Pass Penalty")
    ax.set_ylabel("Percent of Chips (mean)")
    ax.set_title("Policy Routing vs Penalty")
    ax.legend()
    _save(fig, plots_dir, "figure_8_policy_routing")


# ---------------------------------------------------------------------------
# Figure 9 - training curves (aggregated grid by penalty)
# ---------------------------------------------------------------------------
def figure_training_curves(
    run_logs: dict[tuple[int, float], pd.DataFrame], plots_dir: Path
) -> None:
    if not run_logs:
        return
    penalties = sorted({p for _, p in run_logs})
    seeds = sorted({s for s, _ in run_logs})
    cmap = plt.get_cmap("tab10")

    for metric, ylabel, fname in (
        ("mean_reward_100", "Mean Episode Reward (rolling 100)", "figure_9_training_reward"),
        ("loss", "Training Loss", "figure_9_training_loss"),
    ):
        ncols = min(3, len(penalties))
        nrows = int(np.ceil(len(penalties) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
        for idx, penalty in enumerate(penalties):
            ax = axes[idx // ncols][idx % ncols]
            for si, seed in enumerate(seeds):
                df = run_logs.get((seed, penalty))
                if df is None or metric not in df.columns:
                    continue
                ax.plot(df["timestep"], df[metric], color=cmap(si),
                        linewidth=1.3, label=f"seed {seed}")
            ax.set_title(f"penalty {_penalty_label(penalty)}")
            ax.set_xlabel("Timestep")
            ax.set_ylabel(ylabel)
            ax.legend(fontsize=8)
        for j in range(len(penalties), nrows * ncols):
            axes[j // ncols][j % ncols].axis("off")
        fig.suptitle(f"DQN Training Curves grouped by penalty ({ylabel})", y=1.0)
        fig.tight_layout()
        _save(fig, plots_dir, fname)


def plot_run_training_curves(
    training_log: pd.DataFrame, run_dir: Path, seed: int, penalty: float
) -> None:
    """Save a per-run 2-panel training-curve figure (reward + loss)."""
    if training_log.empty:
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.plot(training_log["timestep"], training_log["mean_reward_100"], color="#4C72B0")
    ax1.set_xlabel("Timestep")
    ax1.set_ylabel("Mean Episode Reward (rolling 100)")
    ax1.set_title("Episode Reward")
    ax2.plot(training_log["timestep"], training_log["loss"], color="#C44E52")
    ax2.set_xlabel("Timestep")
    ax2.set_ylabel("Training Loss")
    ax2.set_title("Loss")
    fig.suptitle(f"DQN training (seed {seed}, penalty {_penalty_label(penalty)})")
    fig.tight_layout()
    for ext in ("png",):
        fig.savefig(run_dir / f"training_curves.{ext}", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def generate_all(
    all_runs: pd.DataFrame,
    by_penalty: pd.DataFrame,
    run_logs: dict[tuple[int, float], pd.DataFrame],
    plots_dir: Path,
    recall_constraint: float,
) -> None:
    """Generate all nine figures (PNG + PDF) under ``plots_dir``."""
    figure_pareto_aggregated(by_penalty, plots_dir)
    figure_per_seed_scatter(all_runs, plots_dir)
    figure_fpr_vs_penalty(all_runs, by_penalty, plots_dir)
    figure_cost_vs_penalty(all_runs, by_penalty, plots_dir)
    figure_dual_axis(by_penalty, plots_dir)
    figure_avg_tests_bar(by_penalty, plots_dir)
    figure_grouped_metrics(by_penalty, plots_dir)
    figure_policy_routing(by_penalty, plots_dir)
    figure_training_curves(run_logs, plots_dir)
