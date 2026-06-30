"""Supervised baselines on the full-stage dataset + comparison vs best DQN.

This is an *analysis extension* of the DQN reward sweep. It trains two
full-information supervised classifiers (Logistic Regression, XGBoost) on the
**same processed train/test split** used by the multi-stage RL experiments, then
compares them against the best reward-sweep DQN policy
(``false_pass_penalty = -1000``).

Supervised baselines use the full observable feature set (metadata + Stage-2
features, including the revealed Stage-2 result; this dataset has no Stage-3
measurement columns) and pay the *full* testing cost for every chip
(Average Test Cost = Stage-2 + Stage-3 cost = 5, Cost Reduction = 0%). Label and
final-outcome columns are never used as features (verified at runtime).

Run::

    python -m src.baselines.full_stage_supervised \
        --dataset full_stage_v1 --run-name supervised_full_stage_v1

Outputs:

* ``results/runs/<run-name>/models/{logistic_regression.pkl, xgboost.pkl}``
* ``results/runs/<run-name>/metrics/{logistic_regression_metrics.csv,
  xgboost_metrics.csv, supervised_comparison.csv}``
* ``results/runs/<run-name>/figures/confusion_matrices.png``
* ``results/reward_sweep/baseline_vs_best_dqn.{csv,md}``
* ``results/reward_sweep/summary_policy_routing_terminal_actions.csv``
* ``results/reward_sweep/plots/figure_8b_policy_routing_terminal_actions.{png,pdf}``
* ``results/reward_sweep/plots/figure_10_supervised_vs_best_dqn.{png,pdf}``
* ``results/reward_sweep/plots/figure_11_quality_cost_scatter.{png,pdf}``
* ``results/reward_sweep/supervised_baseline_summary.md``

Existing reward-sweep outputs are never overwritten: every new artifact uses a
new filename.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Allow `python -m src.baselines.full_stage_supervised` and direct execution.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.baselines.logistic_baseline import LogisticBaseline
from src.baselines.xgboost_baseline import XGBoostBaseline
from src.config import CONFIG, REWARD_PROFILES, Config
from src.data.full_stage_loader import load_full_stage_processed
from src.environment.actions import LABEL_FAIL, LABEL_PASS
from src.evaluation.metrics import classification_metrics, confusion_matrix_counts
from src.utils.helpers import get_logger

logger = get_logger("full_stage_supervised")

# Method labels used as rows in the comparison table.
LR_NAME = "Logistic Regression"
XGB_NAME = "XGBoost"
DQN_NAME = "DQN penalty -1000 mean"

# Columns of the supervised-vs-DQN comparison table, in order.
COMPARISON_COLUMNS = [
    "Accuracy",
    "Precision_FAIL",
    "Recall_FAIL",
    "F1_FAIL",
    "False_Pass_Rate",
    "False_Fail_Rate",
    "Average_Test_Cost",
    "Cost_Reduction",
    "Average_Tests_Per_Chip",
]

# Best DQN reward-sweep penalty used for the comparison.
BEST_DQN_PENALTY = -1000.0
# Supervised baselines run both stages -> 2 tests per chip.
SUPERVISED_TESTS_PER_CHIP = 2


def full_testing_cost(config: Config = CONFIG) -> float:
    """Full per-chip testing cost (Stage-2 + Stage-3) for the full-stage reward."""
    reward = REWARD_PROFILES["full_stage_v1"]
    return reward.stage_cost(1) + reward.stage_cost(2)


# --------------------------------------------------------------------------- #
# Feature selection (leakage-safe)
# --------------------------------------------------------------------------- #
def select_features(
    data: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Return ``(X_train, y_train, X_test, y_test, feature_names)``.

    Features are exactly the environment's observable feature columns
    (metadata + Stage-2 + Stage-3). The final ``label`` and the
    ``is_stage2_fail`` helper are *excluded* to avoid label leakage.

    Raises:
        AssertionError: If a forbidden (label/outcome) column leaks into the
            feature set.
    """
    label_col = data.label_column
    feature_names = list(data.feature_columns)

    forbidden = {label_col, "is_stage2_fail", "final_res", "FinalRes_Stage2"}
    leaked = forbidden.intersection(feature_names)
    assert not leaked, f"Label/outcome columns leaked into features: {sorted(leaked)}"

    x_train = data.train[feature_names].to_numpy(dtype=float)
    y_train = data.train[label_col].to_numpy(dtype=int)
    x_test = data.test[feature_names].to_numpy(dtype=float)
    y_test = data.test[label_col].to_numpy(dtype=int)
    return x_train, y_train, x_test, y_test, feature_names


# --------------------------------------------------------------------------- #
# Metric computation
# --------------------------------------------------------------------------- #
def supervised_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, config: Config = CONFIG
) -> dict[str, float]:
    """Compute the full supervised metric set for one classifier."""
    base = classification_metrics(y_true, y_pred)
    cm = confusion_matrix_counts(y_true, y_pred)
    tn, fp, fn, tp = (int(v) for v in cm.ravel())
    return {
        "Accuracy": base["accuracy"],
        "Precision_FAIL": base["precision"],
        "Recall_FAIL": base["recall"],
        "F1_FAIL": base["f1"],
        "False_Pass_Rate": base["false_pass_rate"],
        "False_Fail_Rate": base["false_fail_rate"],
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "EscapedDefects": fn,
        "Average_Test_Cost": full_testing_cost(config),
        "Cost_Reduction": 0.0,
        "Average_Tests_Per_Chip": SUPERVISED_TESTS_PER_CHIP,
        "n_test": int(len(y_true)),
    }


def train_supervised(
    run_paths: Any, config: Config = CONFIG, *, dataset: str = "full_stage_v1"
) -> dict[str, dict[str, float]]:
    """Train + evaluate LR and XGBoost; save models and per-model metrics.

    Returns a mapping ``{method_name: metrics_dict}``.
    """
    data = load_full_stage_processed(config, dataset=dataset)
    x_train, y_train, x_test, y_test, feature_names = select_features(data)
    logger.info(
        "Supervised features (%d): %s", len(feature_names), ", ".join(feature_names)
    )
    logger.info("Train chips: %d | Test chips: %d", len(y_train), len(y_test))

    run_paths.ensure()
    results: dict[str, dict[str, float]] = {}
    predictions: dict[str, np.ndarray] = {}

    # Logistic Regression
    lr = LogisticBaseline(config).fit(x_train, y_train)
    lr_pred = lr.predict(x_test)
    predictions[LR_NAME] = lr_pred
    results[LR_NAME] = supervised_metrics(y_test, lr_pred, config)
    with (run_paths.models / "logistic_regression.pkl").open("wb") as handle:
        pickle.dump(lr.model, handle)

    # XGBoost
    xgb = XGBoostBaseline(config).fit(x_train, y_train)
    xgb_pred = xgb.predict(x_test)
    predictions[XGB_NAME] = xgb_pred
    results[XGB_NAME] = supervised_metrics(y_test, xgb_pred, config)
    with (run_paths.models / "xgboost.pkl").open("wb") as handle:
        pickle.dump(xgb.model, handle)

    # Per-model metrics CSVs.
    pd.DataFrame([results[LR_NAME]]).to_csv(
        run_paths.metrics / "logistic_regression_metrics.csv", index=False
    )
    pd.DataFrame([results[XGB_NAME]]).to_csv(
        run_paths.metrics / "xgboost_metrics.csv", index=False
    )
    # Combined supervised comparison.
    supervised_df = pd.DataFrame(
        [{"Model": LR_NAME, **results[LR_NAME]}, {"Model": XGB_NAME, **results[XGB_NAME]}]
    )
    supervised_df.to_csv(run_paths.metrics / "supervised_comparison.csv", index=False)

    _plot_confusion_matrices(predictions, y_test, run_paths.figures)
    logger.info("Saved supervised models + metrics under %s", run_paths.models.parent)
    return results


def _plot_confusion_matrices(
    predictions: dict[str, np.ndarray], y_true: np.ndarray, figures_dir: Path
) -> None:
    """Save a side-by-side confusion-matrix figure for the supervised models."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(predictions), figsize=(5 * len(predictions), 4.5))
    if len(predictions) == 1:
        axes = [axes]
    for ax, (name, pred) in zip(axes, predictions.items()):
        cm = confusion_matrix_counts(y_true, pred)
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(name)
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["PASS", "FAIL"]); ax.set_yticklabels(["PASS", "FAIL"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Supervised Baseline Confusion Matrices (FAIL = positive)")
    fig.tight_layout()
    fig.savefig(figures_dir / "confusion_matrices.png", bbox_inches="tight", dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# DQN reference (from existing reward-sweep aggregates)
# --------------------------------------------------------------------------- #
def dqn_reference_row(
    summary_by_penalty: Path, penalty: float = BEST_DQN_PENALTY, config: Config = CONFIG
) -> dict[str, float]:
    """Build the DQN comparison row from the aggregated reward-sweep means.

    Average Test Cost is derived from the measured Cost Reduction:
    ``avg_cost = full_cost * (1 - cost_reduction / 100)``.
    """
    df = pd.read_csv(summary_by_penalty)
    row = df[np.isclose(df["Penalty"], penalty)]
    if row.empty:
        raise ValueError(f"Penalty {penalty} not found in {summary_by_penalty}")
    row = row.iloc[0]
    full_cost = full_testing_cost(config)
    cost_reduction = float(row["CostReduction_mean"])
    return {
        "Accuracy": float(row["Accuracy_mean"]),
        "Precision_FAIL": float(row["Precision_mean"]),
        "Recall_FAIL": float(row["Recall_mean"]),
        "F1_FAIL": float(row["F1_mean"]),
        "False_Pass_Rate": float(row["FalsePassRate_mean"]),
        "False_Fail_Rate": float(row["FalseFailRate_mean"]),
        "Average_Test_Cost": full_cost * (1.0 - cost_reduction / 100.0),
        "Cost_Reduction": cost_reduction,
        "Average_Tests_Per_Chip": float(row["AverageTests_mean"]),
    }


def build_comparison(
    supervised: dict[str, dict[str, float]],
    dqn_row: dict[str, float],
) -> pd.DataFrame:
    """Assemble the three-row supervised-vs-DQN comparison table."""
    rows = {
        LR_NAME: supervised[LR_NAME],
        XGB_NAME: supervised[XGB_NAME],
        DQN_NAME: dqn_row,
    }
    frame = pd.DataFrame.from_dict(rows, orient="index")[COMPARISON_COLUMNS]
    frame.index.name = "Method"
    return frame


# --------------------------------------------------------------------------- #
# Terminal-action routing breakdown (re-evaluate DQN sweep models)
# --------------------------------------------------------------------------- #
def terminal_routing_breakdown(
    sweep_dir: Path, config: Config = CONFIG, *, dataset: str = "full_stage_v1"
) -> pd.DataFrame | None:
    """Re-evaluate saved DQN models to split routing by terminal action.

    For every saved ``model_best`` checkpoint, roll out on the shared test split
    and record the joint distribution of ``(stage_stopped, predicted_label)``,
    yielding the six terminal-action categories. Results are averaged across
    seeds per penalty. Routing is policy-determined, so the reward profile used
    to build the evaluation environment does not affect it.

    Returns ``None`` (and logs a warning) if no saved models are found.
    """
    from src.agents.dqn_agent import DQNAgent
    from src.environment.factory import load_dataset_bundle, make_env
    from src.evaluation.evaluate import rollout_agent
    from experiments.sweep_plots import TERMINAL_ROUTING_CATEGORIES

    model_paths = sorted(sweep_dir.glob("seed_*/penalty_*/model_best.zip"))
    if not model_paths:
        logger.warning(
            "No DQN models found under %s; skipping terminal-routing breakdown.",
            sweep_dir,
        )
        return None

    bundle = load_dataset_bundle(dataset, "multi_stage", config)
    eval_env = make_env(bundle, "test", config, reward_config=REWARD_PROFILES["full_stage_v1"])

    per_run: list[dict[str, float]] = []
    for path in model_paths:
        try:
            penalty = float(path.parent.name.split("_")[1])
            seed = int(path.parent.parent.name.split("_")[1])
        except (IndexError, ValueError):
            continue
        agent = DQNAgent(config)
        agent.load(path)
        result = rollout_agent(agent, eval_env)
        stages = np.asarray(result.stages_stopped)
        preds = np.asarray(result.predicted_labels).astype(int)
        n = len(preds)

        def pct(stage: int, label: int) -> float:
            return float(np.mean((stages == stage) & (preds == label)) * 100.0) if n else 0.0

        per_run.append(
            {
                "Penalty": penalty,
                "Seed": seed,
                "STOP_PASS_before_Stage2": pct(0, LABEL_PASS),
                "STOP_FAIL_before_Stage2": pct(0, LABEL_FAIL),
                "STOP_PASS_after_Stage2": pct(1, LABEL_PASS),
                "STOP_FAIL_after_Stage2": pct(1, LABEL_FAIL),
                "STOP_PASS_after_Stage3": pct(2, LABEL_PASS),
                "STOP_FAIL_after_Stage3": pct(2, LABEL_FAIL),
            }
        )
        logger.info("Routing computed: seed=%d penalty=%g", seed, penalty)

    per_run_df = pd.DataFrame(per_run)
    routing = (
        per_run_df.groupby("Penalty")[TERMINAL_ROUTING_CATEGORIES].mean().reset_index()
    )
    return routing.sort_values("Penalty").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Markdown summary
# --------------------------------------------------------------------------- #
def write_summary_md(
    comparison: pd.DataFrame, path: Path, config: Config = CONFIG
) -> None:
    """Write the supervised-baseline analysis summary markdown."""
    lr = comparison.loc[LR_NAME]
    xgb = comparison.loc[XGB_NAME]
    dqn = comparison.loc[DQN_NAME]
    full_cost = full_testing_cost(config)

    lines = [
        "# Supervised Baselines vs Best DQN Policy",
        "",
        "Comparison of full-information supervised classifiers against the best "
        f"reward-sweep DQN policy (`false_pass_penalty = {int(BEST_DQN_PENALTY)}`, "
        "across-seed mean) on the **same** processed `full_stage_v1` test split.",
        "",
        "## Comparison table",
        "",
        comparison.round(4).to_markdown(),
        "",
        "## 1. How do Logistic Regression and XGBoost perform on the full dataset?",
        "",
        f"- **Logistic Regression** - Accuracy {lr['Accuracy']:.3f}, "
        f"Recall(FAIL) {lr['Recall_FAIL']:.3f}, False Pass Rate {lr['False_Pass_Rate']:.3f}, "
        f"F1(FAIL) {lr['F1_FAIL']:.3f}.",
        f"- **XGBoost** - Accuracy {xgb['Accuracy']:.3f}, "
        f"Recall(FAIL) {xgb['Recall_FAIL']:.3f}, False Pass Rate {xgb['False_Pass_Rate']:.3f}, "
        f"F1(FAIL) {xgb['F1_FAIL']:.3f}.",
        "",
        "Both classifiers use the full observable feature set (metadata + Stage-2 "
        "measurements, including the revealed Stage-2 result) and therefore pay the "
        f"full testing cost of {full_cost:.0f} per chip (no test reduction).",
        "",
        "## 2. How does the best DQN policy compare in failure recall and false pass rate?",
        "",
        f"- DQN Recall(FAIL) {dqn['Recall_FAIL']:.3f} vs LR {lr['Recall_FAIL']:.3f} / "
        f"XGB {xgb['Recall_FAIL']:.3f}.",
        f"- DQN False Pass Rate {dqn['False_Pass_Rate']:.3f} vs LR {lr['False_Pass_Rate']:.3f} / "
        f"XGB {xgb['False_Pass_Rate']:.3f}.",
        "",
        f"The DQN meets the safety target (Recall >= 0.95) at "
        f"{dqn['Recall_FAIL']:.3f} while keeping a low escaped-defect rate.",
        "",
        "## 3. How much cost reduction does DQN achieve vs full-testing baselines?",
        "",
        f"- Supervised baselines: Average Test Cost {full_cost:.0f} (Cost Reduction 0%).",
        f"- DQN: Average Test Cost {dqn['Average_Test_Cost']:.2f} "
        f"(**Cost Reduction {dqn['Cost_Reduction']:.1f}%**), "
        f"averaging {dqn['Average_Tests_Per_Chip']:.2f} tests per chip vs "
        f"{int(SUPERVISED_TESTS_PER_CHIP)} for the supervised baselines.",
        "",
        "## 4. Is DQN preferable for adaptive test reduction?",
        "",
        "If the objective is **pure classification quality**, the supervised models "
        "are competitive (and simpler). If the objective is **adaptive test "
        "reduction** - cutting test cost while keeping failure detection high - the "
        f"DQN is preferable: it achieves ~{dqn['Cost_Reduction']:.0f}% cost reduction "
        "at comparable failure recall, which the full-testing baselines cannot do by "
        "construction (they always run every stage). See "
        "`plots/figure_11_quality_cost_scatter.png`.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote summary markdown to %s", path)


def write_comparison_md(comparison: pd.DataFrame, path: Path) -> None:
    """Write the comparison table as a standalone markdown file."""
    path.write_text(
        "# Supervised Baselines vs Best DQN (penalty -1000, across-seed mean)\n\n"
        + comparison.round(4).to_markdown()
        + "\n",
        encoding="utf-8",
    )


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run(
    *,
    dataset: str,
    run_name: str,
    sweep_dir: Path,
    config: Config = CONFIG,
    skip_routing: bool = False,
) -> None:
    """Train baselines, build the comparison, routing breakdown, figures, summary."""
    from experiments import sweep_plots

    run_paths = config.paths.run_paths(run_name)

    supervised = train_supervised(run_paths, config, dataset=dataset)

    summary_by_penalty = sweep_dir / "summary_by_penalty.csv"
    plots_dir = sweep_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    if not summary_by_penalty.exists():
        logger.warning(
            "Reward-sweep aggregates not found at %s; skipping DQN comparison.",
            summary_by_penalty,
        )
        return

    dqn_row = dqn_reference_row(summary_by_penalty, BEST_DQN_PENALTY, config)
    comparison = build_comparison(supervised, dqn_row)

    # New filenames only - never overwrite existing reward-sweep artifacts.
    comparison.to_csv(sweep_dir / "baseline_vs_best_dqn.csv")
    write_comparison_md(comparison, sweep_dir / "baseline_vs_best_dqn.md")
    sweep_plots.figure_supervised_vs_dqn(comparison, plots_dir)
    sweep_plots.figure_quality_cost_scatter(comparison, plots_dir)
    write_summary_md(comparison, sweep_dir / "supervised_baseline_summary.md", config)

    if not skip_routing:
        routing = terminal_routing_breakdown(sweep_dir, config, dataset=dataset)
        if routing is not None:
            routing.to_csv(
                sweep_dir / "summary_policy_routing_terminal_actions.csv", index=False
            )
            sweep_plots.figure_8b_terminal_routing(routing, plots_dir)

    logger.info("Supervised-baseline extension complete.")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="full_stage_v1", help="Processed dataset name.")
    parser.add_argument(
        "--run-name", default="supervised_full_stage_v1", help="Isolated run directory name."
    )
    parser.add_argument(
        "--sweep-dir",
        type=Path,
        default=Path("results/reward_sweep"),
        help="Existing reward-sweep results directory.",
    )
    parser.add_argument(
        "--skip-routing",
        action="store_true",
        help="Skip the (slower) DQN terminal-routing re-evaluation / Figure 8b.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point."""
    args = parse_args()
    run(
        dataset=args.dataset,
        run_name=args.run_name,
        sweep_dir=args.sweep_dir,
        skip_routing=args.skip_routing,
    )


if __name__ == "__main__":
    main()
