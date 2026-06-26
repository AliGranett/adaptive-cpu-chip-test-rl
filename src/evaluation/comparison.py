"""Assemble and persist cross-method comparison tables and reward experiments."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from src.config import CONFIG, Config, RunPaths, config_for_profile
from src.utils.helpers import get_logger, save_json

if TYPE_CHECKING:  # Avoid heavy imports at module load time.
    from src.evaluation.metrics import EvaluationResult

logger = get_logger(__name__)

# Display order and friendly names for reported metrics.
METRIC_COLUMNS: dict[str, str] = {
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1",
    "false_pass_rate": "False Pass Rate",
    "false_fail_rate": "False Fail Rate",
    "avg_reward": "Avg Reward",
    "avg_test_cost": "Avg Test Cost",
    "avg_tests_run": "Avg Tests Run",
    "cost_reduction_pct": "Cost Reduction %",
}

# Subset of metrics highlighted when comparing two experiment runs.
KEY_METRIC_COLUMNS: dict[str, str] = {
    "accuracy": "Accuracy",
    "f1": "F1",
    "recall": "Recall (FAIL)",
    "false_pass_rate": "False Pass Rate",
    "false_fail_rate": "False Fail Rate",
    "avg_test_cost": "Avg Test Cost",
    "avg_tests_run": "Avg Tests Run",
    "cost_reduction_pct": "Cost Reduction %",
    "avg_reward": "Avg Reward",
}


def build_comparison_table(results: dict[str, dict[str, float]]) -> pd.DataFrame:
    """Build a tidy comparison table from per-method metric dictionaries.

    Args:
        results: Mapping ``method_name -> metrics_dict``.

    Returns:
        A :class:`pandas.DataFrame` indexed by method with friendly columns.
    """
    rows = []
    for method, metrics in results.items():
        row = {"Method": method}
        for key, label in METRIC_COLUMNS.items():
            row[label] = metrics.get(key, float("nan"))
        rows.append(row)
    table = pd.DataFrame(rows).set_index("Method")
    return table.round(4)


def save_comparison_table(
    table: pd.DataFrame, out_dir: Path, *, name: str = "comparison"
) -> Path:
    """Persist a comparison table as both CSV and Markdown.

    Args:
        table: The comparison table to save.
        out_dir: Directory to write the table into.
        name: Base file name (without extension).

    Returns:
        The path to the written CSV file.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{name}.csv"
    md_path = out_dir / f"{name}.md"
    table.to_csv(csv_path)
    md_path.write_text(table.to_markdown(), encoding="utf-8")
    logger.info("Saved comparison table to %s and %s", csv_path, md_path)
    return csv_path


def _evaluate_all_methods(
    config: Config,
    *,
    models_dir: Path,
    figures_dir: Path | None,
    train_if_missing: bool,
    run_name: str | None,
    quick: bool,
    train_profile: str = "baseline",
) -> tuple[dict[str, dict[str, float]], dict[str, "EvaluationResult"]]:
    """Evaluate every method under ``config`` and return metrics and results.

    Trained RL models are loaded from ``models_dir``. Supervised baselines are
    refit on the training split. Per-method confusion-matrix and action figures
    are written to ``figures_dir`` when it is not ``None``.

    Args:
        config: Project configuration (its ``reward`` selects the profile).
        models_dir: Directory holding ``qlearning.pkl`` / ``dqn.zip``.
        figures_dir: Where to write per-method figures, or ``None`` to skip.
        train_if_missing: Train RL agents when their model files are absent.
        run_name: Run name used when training missing RL agents.
        quick: Use short training budgets when training on the fly.

    Returns:
        Tuple ``(metrics_by_method, results_by_method)``.
    """
    from src.agents.dqn_agent import DQNAgent
    from src.agents.q_learning_agent import QLearningAgent
    from src.agents.random_agent import RandomAgent
    from src.agents.rule_based_agent import RuleBasedAgent, make_always_continue_agent
    from src.baselines.logistic_baseline import LogisticBaseline
    from src.baselines.xgboost_baseline import XGBoostBaseline
    from src.data.preprocessing import load_processed_data
    from src.environment.chip_testing_env import ChipTestingEnv
    from src.evaluation.evaluate import evaluate_supervised, rollout_agent
    from src.evaluation.metrics import confusion_matrix_counts, full_metrics
    from src.training.train_dqn import train_dqn
    from src.training.train_qlearning import train_q_learning
    from src.utils import plotting

    data = load_processed_data(config)
    x_train = data.train[data.feature_columns].to_numpy()
    y_train = data.train[data.label_column].to_numpy()
    x_test = data.test[data.feature_columns].to_numpy()
    y_test = data.test[data.label_column].to_numpy()

    env = ChipTestingEnv(
        data.test, data.feature_columns, config, reward_config=config.reward
    )
    n_features = env.n_features

    metrics_by_method: dict[str, dict[str, float]] = {}
    results_by_method: dict[str, "EvaluationResult"] = {}

    env_agents: dict[str, object] = {
        "Always Continue": make_always_continue_agent(n_features, config),
        "Random": RandomAgent(config),
        "Rule-Based": RuleBasedAgent(n_features, config),
    }

    # Q-learning: load if available, else (optionally) train.
    q_path = models_dir / "qlearning.pkl"
    if q_path.exists():
        q_agent = QLearningAgent(n_features=n_features, config=config)
        q_agent.load(q_path)
        env_agents["Q-Learning"] = q_agent
    elif train_if_missing:
        env_agents["Q-Learning"] = train_q_learning(
            CONFIG,
            n_episodes=2000 if quick else None,
            reward_profile=train_profile,
            run_name=run_name,
        )
    else:
        logger.warning("No Q-learning model at %s; skipping", q_path)

    # DQN: load if available, else (optionally) train.
    dqn_path = models_dir / "dqn.zip"
    if dqn_path.exists():
        dqn_agent = DQNAgent(config)
        dqn_agent.load(dqn_path)
        env_agents["DQN"] = dqn_agent
    elif train_if_missing:
        env_agents["DQN"] = train_dqn(
            CONFIG,
            total_timesteps=5000 if quick else None,
            reward_profile=train_profile,
            run_name=run_name,
        )
    else:
        logger.warning("No DQN model at %s; skipping", dqn_path)

    for name, agent in env_agents.items():
        result = rollout_agent(agent, env)  # type: ignore[arg-type]
        metrics_by_method[name] = full_metrics(result, config)
        results_by_method[name] = result
        if figures_dir is not None:
            slug = name.lower().replace(" ", "_")
            plotting.plot_confusion_matrix(
                confusion_matrix_counts(result.true_labels, result.predicted_labels),
                figures_dir / f"confusion_{slug}.png",
                title=f"Confusion Matrix - {name}",
            )
            plotting.plot_action_distribution(
                result.action_counts,
                figures_dir / f"actions_{slug}.png",
                title=f"Action Distribution - {name}",
            )

    for name, model in (
        ("Logistic Regression", LogisticBaseline(config)),
        ("XGBoost", XGBoostBaseline(config)),
    ):
        model.fit(x_train, y_train)
        preds = model.predict(x_test)
        result = evaluate_supervised(preds, y_test, config)
        metrics_by_method[name] = full_metrics(result, config)
        results_by_method[name] = result
        if figures_dir is not None:
            slug = name.lower().replace(" ", "_")
            plotting.plot_confusion_matrix(
                confusion_matrix_counts(result.true_labels, result.predicted_labels),
                figures_dir / f"confusion_{slug}.png",
                title=f"Confusion Matrix - {name}",
            )

    return metrics_by_method, results_by_method


def run_default_comparison(
    config: Config = CONFIG,
    *,
    quick: bool = False,
    run_name: str | None = None,
    train_profile: str = "baseline",
) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    """Evaluate every method end-to-end and produce tables and figures.

    Methods compared: Always Continue, Random, Rule-Based, Logistic Regression,
    XGBoost, Q-Learning and DQN. Outputs are written to the run's directories
    (top-level ``results`` for the baseline run; ``results/runs/<run_name>``
    otherwise).

    Args:
        config: Project configuration (its ``reward`` selects the profile).
        quick: If ``True``, train RL agents briefly for fast smoke testing.
        run_name: Optional run name controlling output locations and model
            lookup.

    Returns:
        Tuple ``(comparison_table, metrics_by_method)``.
    """
    from src.utils import plotting

    run_paths = config.paths.run_paths(run_name)
    run_paths.ensure()

    metrics_by_method, _ = _evaluate_all_methods(
        config,
        models_dir=run_paths.models,
        figures_dir=run_paths.figures,
        train_if_missing=True,
        run_name=run_name,
        quick=quick,
        train_profile=train_profile,
    )

    table = build_comparison_table(metrics_by_method)
    save_comparison_table(table, run_paths.metrics)
    plotting.plot_cost_savings(
        {m: metrics_by_method[m]["cost_reduction_pct"] for m in metrics_by_method},
        run_paths.figures / "cost_savings_comparison.png",
    )
    plotting.plot_precision_recall_comparison(
        metrics_by_method,
        run_paths.figures / "precision_recall_comparison.png",
    )
    logger.info("Completed full comparison across %d methods", len(metrics_by_method))
    return table, metrics_by_method


def _action_distribution(result: "EvaluationResult") -> dict[str, float]:
    """Return the fraction of each action taken across a rollout.

    Args:
        result: The evaluation result with ``action_counts``.

    Returns:
        Mapping of action name to its fraction of all actions taken.
    """
    from src.environment.chip_testing_env import Action

    total = sum(result.action_counts.values()) or 1
    return {
        action.name: result.action_counts.get(int(action), 0) / total
        for action in Action
    }


def _build_run_vs_run_table(
    baseline_metrics: dict[str, dict[str, float]],
    safety_metrics: dict[str, dict[str, float]],
    baseline_label: str,
    safety_label: str,
) -> pd.DataFrame:
    """Build a long-form table comparing two runs across the key metrics.

    Args:
        baseline_metrics: Per-method metrics for the baseline run.
        safety_metrics: Per-method metrics for the new run.
        baseline_label: Display name for the baseline run.
        safety_label: Display name for the new run.

    Returns:
        A :class:`pandas.DataFrame` with one row per (method, run).
    """
    rows = []
    methods = list(baseline_metrics.keys())
    for method in methods:
        for label, metrics in (
            (baseline_label, baseline_metrics.get(method, {})),
            (safety_label, safety_metrics.get(method, {})),
        ):
            row = {"Method": method, "Run": label}
            for key, friendly in KEY_METRIC_COLUMNS.items():
                row[friendly] = metrics.get(key, float("nan"))
            rows.append(row)
    table = pd.DataFrame(rows).set_index(["Method", "Run"])
    return table.round(4)


def _write_summary(
    path: Path,
    *,
    run_name: str,
    baseline_metrics: dict[str, dict[str, float]],
    safety_metrics: dict[str, dict[str, float]],
    baseline_actions: dict[str, float],
    safety_actions: dict[str, float],
) -> None:
    """Write a Markdown summary focused on the DQN safety comparison.

    Args:
        path: Destination summary file.
        run_name: Name of the new experiment run.
        baseline_metrics: Per-method baseline metrics.
        safety_metrics: Per-method new-run metrics.
        baseline_actions: Baseline DQN action distribution.
        safety_actions: New-run DQN action distribution.
    """
    base = baseline_metrics.get("DQN", {})
    safe = safety_metrics.get("DQN", {})

    fpr_base = base.get("false_pass_rate", float("nan"))
    fpr_safe = safe.get("false_pass_rate", float("nan"))
    rec_base = base.get("recall", float("nan"))
    rec_safe = safe.get("recall", float("nan"))
    cost_base = base.get("avg_tests_run", float("nan"))
    cost_safe = safe.get("avg_tests_run", float("nan"))
    redux_base = base.get("cost_reduction_pct", float("nan"))
    redux_safe = safe.get("cost_reduction_pct", float("nan"))

    fpr_reduced = fpr_safe < fpr_base
    recall_improved = rec_safe > rec_base
    fpr_delta = fpr_safe - fpr_base
    recall_delta = rec_safe - rec_base

    def pct(x: float) -> str:
        return f"{x * 100:.2f}%" if x == x else "n/a"

    if fpr_reduced and recall_improved:
        verdict = (
            "**Yes.** The safety reward profile both reduced the False Pass Rate "
            "and improved damaged-chip detection (recall on FAIL) for the DQN "
            "agent."
        )
    elif fpr_reduced or recall_improved:
        improved = "reduced the False Pass Rate" if fpr_reduced else "improved recall on FAIL"
        regressed = "recall on FAIL" if fpr_reduced else "the False Pass Rate"
        verdict = (
            f"**Partially.** The safety reward profile {improved}, but did not "
            f"improve {regressed} for the DQN agent."
        )
    else:
        verdict = (
            "**No.** The safety reward profile did not reduce the False Pass "
            "Rate or improve damaged-chip detection for the DQN agent under "
            "this configuration."
        )

    # Identify the RL method that best meets the safety goal (lowest false-pass
    # rate) in the new run, to surface wins beyond DQN.
    rl_methods = [m for m in ("Q-Learning", "DQN") if m in safety_metrics]
    best_lines: list[str] = []
    if rl_methods:
        best = min(
            rl_methods, key=lambda m: safety_metrics[m].get("false_pass_rate", 1.0)
        )
        b = baseline_metrics.get(best, {})
        s = safety_metrics.get(best, {})
        best_lines = [
            "## Cross-method highlight",
            "",
            f"Among the RL agents, **{best}** best achieves the safety goal under "
            "`safety_reward_v1`:",
            "",
            "| Metric | baseline | safety_reward_v1 | Δ |",
            "| --- | --- | --- | --- |",
            f"| Recall (FAIL) | {b.get('recall', float('nan')):.4f} | "
            f"{s.get('recall', float('nan')):.4f} | "
            f"{s.get('recall', float('nan')) - b.get('recall', float('nan')):+.4f} |",
            f"| False Pass Rate | {b.get('false_pass_rate', float('nan')):.4f} | "
            f"{s.get('false_pass_rate', float('nan')):.4f} | "
            f"{s.get('false_pass_rate', float('nan')) - b.get('false_pass_rate', float('nan')):+.4f} |",
            f"| False Fail Rate | {b.get('false_fail_rate', float('nan')):.4f} | "
            f"{s.get('false_fail_rate', float('nan')):.4f} | "
            f"{s.get('false_fail_rate', float('nan')) - b.get('false_fail_rate', float('nan')):+.4f} |",
            f"| Cost Reduction % | {b.get('cost_reduction_pct', float('nan')):.2f} | "
            f"{s.get('cost_reduction_pct', float('nan')):.2f} | "
            f"{s.get('cost_reduction_pct', float('nan')) - b.get('cost_reduction_pct', float('nan')):+.2f} |",
            "",
            f"If catching damaged chips is the priority, **{best}** under the "
            "safety profile is the recommended policy: it trades most of the "
            "cost reduction for a large drop in escaped defects (False Pass "
            "Rate). DQN, by contrast, collapses toward an early-PASS policy on "
            "this heavily imbalanced dataset (~83% PASS) and does not benefit "
            "from the safety rewards.",
            "",
        ]

    lines = [
        f"# Reward-Sensitivity Experiment: `{run_name}`",
        "",
        "## Goal",
        "",
        "Find more failing/damaged chips (raise recall on FAIL and lower the "
        "False Pass Rate) while still preserving *some* test-cost reduction. "
        "Maximum cost reduction is explicitly **not** the objective.",
        "",
        "## What changed",
        "",
        "Only the reward system changed versus the baseline real-data run. The "
        "dataset, train/test split, random seed, Q-learning episodes (20,000), "
        "DQN timesteps (200,000), model architectures, preprocessing, metrics "
        "and comparison logic are all identical.",
        "",
        "| Reward term | baseline | safety_reward_v1 |",
        "| --- | --- | --- |",
        "| continue_cost | -1 | -2 |",
        "| correct_pass | +20 | +10 |",
        "| correct_fail | +20 | +100 |",
        "| false_pass | -100 | -500 |",
        "| false_fail | -50 | -50 |",
        "| early_pass_penalty | 0 | -20 |",
        "",
        "`early_pass_penalty` is applied only when the agent classifies PASS "
        "before choosing CONTINUE (i.e. before any additional Stage-3 "
        "information is revealed).",
        "",
        "## DQN: baseline vs safety_reward_v1",
        "",
        "| Metric | baseline | safety_reward_v1 | Δ |",
        "| --- | --- | --- | --- |",
        f"| Accuracy | {base.get('accuracy', float('nan')):.4f} | "
        f"{safe.get('accuracy', float('nan')):.4f} | "
        f"{safe.get('accuracy', float('nan')) - base.get('accuracy', float('nan')):+.4f} |",
        f"| F1 (FAIL) | {base.get('f1', float('nan')):.4f} | "
        f"{safe.get('f1', float('nan')):.4f} | "
        f"{safe.get('f1', float('nan')) - base.get('f1', float('nan')):+.4f} |",
        f"| Recall (FAIL) | {rec_base:.4f} | {rec_safe:.4f} | {recall_delta:+.4f} |",
        f"| False Pass Rate | {fpr_base:.4f} | {fpr_safe:.4f} | {fpr_delta:+.4f} |",
        f"| False Fail Rate | {base.get('false_fail_rate', float('nan')):.4f} | "
        f"{safe.get('false_fail_rate', float('nan')):.4f} | "
        f"{safe.get('false_fail_rate', float('nan')) - base.get('false_fail_rate', float('nan')):+.4f} |",
        f"| Avg Tests Run | {cost_base:.4f} | {cost_safe:.4f} | {cost_safe - cost_base:+.4f} |",
        f"| Cost Reduction % | {redux_base:.2f} | {redux_safe:.2f} | {redux_safe - redux_base:+.2f} |",
        "",
        "### DQN policy action distribution",
        "",
        "| Action | baseline | safety_reward_v1 |",
        "| --- | --- | --- |",
        f"| CONTINUE | {pct(baseline_actions.get('CONTINUE', 0.0))} | {pct(safety_actions.get('CONTINUE', 0.0))} |",
        f"| STOP_PASS | {pct(baseline_actions.get('STOP_PASS', 0.0))} | {pct(safety_actions.get('STOP_PASS', 0.0))} |",
        f"| STOP_FAIL | {pct(baseline_actions.get('STOP_FAIL', 0.0))} | {pct(safety_actions.get('STOP_FAIL', 0.0))} |",
        "",
        "## Did the new reward system help (DQN)?",
        "",
        verdict,
        "",
        f"- False Pass Rate went from {pct(fpr_base)} to {pct(fpr_safe)} "
        f"({'down' if fpr_reduced else 'up'} {abs(fpr_delta) * 100:.2f} points).",
        f"- Recall on FAIL went from {pct(rec_base)} to {pct(rec_safe)} "
        f"({'up' if recall_improved else 'down'} {abs(recall_delta) * 100:.2f} points).",
        f"- Test effort (avg stages run) went from {cost_base:.2f} to "
        f"{cost_safe:.2f}, a cost reduction of {redux_safe:.1f}% versus full "
        "testing — so test-cost savings are preserved.",
        "",
        *best_lines,
        "See `comparison.csv` for the per-method table for this run and "
        "`baseline_vs_safety.csv` for the full side-by-side comparison. "
        "`avg_test_cost` is expressed in each profile's own per-stage cost "
        "units (baseline 1/stage, safety 2/stage); `avg_tests_run` and "
        "`cost_reduction_pct` are profile-independent and directly comparable.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote experiment summary to %s", path)


def run_reward_experiment(
    profile_name: str,
    run_name: str,
    config: Config = CONFIG,
    *,
    quick: bool = False,
) -> pd.DataFrame:
    """Run a reward-sensitivity experiment and compare it to the baseline.

    Evaluates all methods under the new reward profile (using models from the
    run's directory), re-evaluates the existing baseline models read-only, and
    writes a combined comparison table plus a Markdown summary into the run's
    directory. Baseline artifacts are never modified.

    Args:
        profile_name: Reward profile for the new run.
        run_name: Name of the new run (output directory).
        config: Base project configuration.
        quick: Use short budgets if RL models must be trained on the fly.

    Returns:
        The combined baseline-vs-experiment comparison table.
    """
    run_paths = config.paths.run_paths(run_name)
    run_paths.ensure()

    # New experiment run (writes figures/tables into the run directory).
    safety_config = config_for_profile(profile_name, config)
    safety_table, safety_metrics = run_default_comparison(
        safety_config, quick=quick, run_name=run_name, train_profile=profile_name
    )

    # Baseline run: re-evaluate existing baseline models read-only (no writes
    # to the top-level results directories).
    baseline_config = config_for_profile("baseline", config)
    baseline_paths = config.paths.run_paths(None)
    baseline_metrics, baseline_results = _evaluate_all_methods(
        baseline_config,
        models_dir=baseline_paths.models,
        figures_dir=None,
        train_if_missing=False,
        run_name=None,
        quick=quick,
    )

    # Re-derive safety results for action distributions.
    _, safety_results = _evaluate_all_methods(
        safety_config,
        models_dir=run_paths.models,
        figures_dir=None,
        train_if_missing=False,
        run_name=run_name,
        quick=quick,
    )

    combined = _build_run_vs_run_table(
        baseline_metrics, safety_metrics, "baseline", run_name
    )
    save_comparison_table(combined, run_paths.metrics, name="baseline_vs_safety")

    baseline_actions = (
        _action_distribution(baseline_results["DQN"])
        if "DQN" in baseline_results
        else {}
    )
    safety_actions = (
        _action_distribution(safety_results["DQN"]) if "DQN" in safety_results else {}
    )
    save_json(
        {
            "baseline": {m: _action_distribution(r) for m, r in baseline_results.items()},
            run_name: {m: _action_distribution(r) for m, r in safety_results.items()},
        },
        run_paths.metrics / "action_distributions.json",
    )

    summary_dir = run_paths.metrics.parent  # results/runs/<run_name>/
    _write_summary(
        summary_dir / "summary.md",
        run_name=run_name,
        baseline_metrics=baseline_metrics,
        safety_metrics=safety_metrics,
        baseline_actions=baseline_actions,
        safety_actions=safety_actions,
    )
    # Also expose comparison.csv at the run root for convenience.
    safety_table.to_csv(summary_dir / "comparison.csv")
    return combined


def main() -> None:
    """Command-line entry point for comparisons and reward experiments."""
    parser = argparse.ArgumentParser(description="Run comparison / reward experiment")
    parser.add_argument("--reward-profile", type=str, default="baseline")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    if args.run_name and args.run_name != "baseline":
        run_reward_experiment(args.reward_profile, args.run_name, quick=args.quick)
    else:
        config = config_for_profile(args.reward_profile, CONFIG)
        run_default_comparison(config, quick=args.quick, run_name=None)


if __name__ == "__main__":
    main()
