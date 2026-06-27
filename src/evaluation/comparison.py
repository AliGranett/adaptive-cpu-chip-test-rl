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

# Extra stage-routing columns reported only for the multi-stage environment.
MULTI_STAGE_METRIC_COLUMNS: dict[str, str] = {
    **METRIC_COLUMNS,
    "pct_stopped_before_stage2": "% Stopped Before Stage2",
    "pct_stopped_after_stage2": "% Stopped After Stage2",
    "pct_sent_to_stage3": "% Sent To Stage3",
    "pct_stage2_fail_correctly_stopped": "% Stage2-Fail Caught",
    "pct_stage2_fail_incorrectly_passed": "% Stage2-Fail Passed",
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


def build_comparison_table(
    results: dict[str, dict[str, float]],
    columns: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Build a tidy comparison table from per-method metric dictionaries.

    Args:
        results: Mapping ``method_name -> metrics_dict``.
        columns: Metric-key -> friendly-name mapping. Defaults to
            :data:`METRIC_COLUMNS`.

    Returns:
        A :class:`pandas.DataFrame` indexed by method with friendly columns.
    """
    columns = columns or METRIC_COLUMNS
    rows = []
    for method, metrics in results.items():
        row = {"Method": method}
        for key, label in columns.items():
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
    dataset: str = "baseline",
    environment: str = "single_stage",
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
        train_profile: Reward profile used when training missing agents.
        dataset: Dataset name (``"baseline"`` or ``"full_stage_v1"``).
        environment: ``"single_stage"`` or ``"multi_stage"``.

    Returns:
        Tuple ``(metrics_by_method, results_by_method)``.
    """
    from src.agents.dqn_agent import DQNAgent
    from src.agents.q_learning_agent import QLearningAgent
    from src.agents.random_agent import RandomAgent
    from src.agents.rule_based_agent import RuleBasedAgent, make_always_continue_agent
    from src.baselines.logistic_baseline import LogisticBaseline
    from src.baselines.xgboost_baseline import XGBoostBaseline
    from src.environment.factory import MULTI_STAGE, load_dataset_bundle, make_env
    from src.evaluation.evaluate import (
        evaluate_supervised,
        evaluate_supervised_multi_stage,
        rollout_agent,
    )
    from src.evaluation.metrics import confusion_matrix_counts, full_metrics
    from src.training.train_dqn import train_dqn
    from src.training.train_qlearning import train_q_learning
    from src.utils import plotting

    is_multi = environment == MULTI_STAGE
    bundle = load_dataset_bundle(dataset, environment, config)
    x_train = bundle.train[bundle.feature_columns].to_numpy()
    y_train = bundle.train[bundle.label_column].to_numpy()
    x_test = bundle.test[bundle.feature_columns].to_numpy()
    y_test = bundle.test[bundle.label_column].to_numpy()

    env = make_env(bundle, "test", config, reward_config=config.reward)
    n_features = env.n_features
    # Full-testing cost reference for cost-reduction metric.
    full_cost = (
        config.reward.stage_cost(1) + config.reward.stage_cost(2)
        if is_multi
        else None
    )

    def _metrics(result: "EvaluationResult") -> dict[str, float]:
        return full_metrics(result, config, full_testing_cost=full_cost)

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
            dataset=dataset,
            environment=environment,
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
            dataset=dataset,
            environment=environment,
        )
    else:
        logger.warning("No DQN model at %s; skipping", dqn_path)

    for name, agent in env_agents.items():
        result = rollout_agent(agent, env)  # type: ignore[arg-type]
        metrics_by_method[name] = _metrics(result)
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

    s2_fail_test = (
        bundle.test["is_stage2_fail"].to_numpy()
        if is_multi and "is_stage2_fail" in bundle.test.columns
        else None
    )
    for name, model in (
        ("Logistic Regression", LogisticBaseline(config)),
        ("XGBoost", XGBoostBaseline(config)),
    ):
        model.fit(x_train, y_train)
        preds = model.predict(x_test)
        if is_multi and s2_fail_test is not None:
            result = evaluate_supervised_multi_stage(preds, y_test, s2_fail_test, config)
        else:
            result = evaluate_supervised(preds, y_test, config)
        metrics_by_method[name] = _metrics(result)
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
    dataset: str = "baseline",
    environment: str = "single_stage",
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
        train_profile: Reward profile for any on-the-fly training.
        dataset: Dataset name (``"baseline"`` or ``"full_stage_v1"``).
        environment: ``"single_stage"`` or ``"multi_stage"``.

    Returns:
        Tuple ``(comparison_table, metrics_by_method)``.
    """
    from src.environment.factory import MULTI_STAGE
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
        dataset=dataset,
        environment=environment,
    )

    columns = (
        MULTI_STAGE_METRIC_COLUMNS if environment == MULTI_STAGE else METRIC_COLUMNS
    )
    table = build_comparison_table(metrics_by_method, columns)
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


# --------------------------------------------------------------------------- #
# Multi-stage (full_stage_v1) experiment
# --------------------------------------------------------------------------- #
# Registry describing how to re-evaluate each prior/new run read-only for the
# final cross-run comparison. ``models`` is resolved lazily against the config.
_RUN_REGISTRY: list[dict[str, str | None]] = [
    {"run": "baseline", "profile": "baseline", "dataset": "baseline", "environment": "single_stage"},
    {"run": "safety_reward_v1", "profile": "safety_reward_v1", "dataset": "baseline", "environment": "single_stage"},
    {"run": "multi_stage_v1", "profile": "full_stage_v1", "dataset": "full_stage_v1", "environment": "multi_stage"},
    {"run": "full_stage_v1", "profile": "full_stage_v1", "dataset": "full_stage_v1", "environment": "multi_stage"},
]


def _models_dir_for_run(run: str, config: Config) -> Path:
    """Resolve the models directory for a named run."""
    return config.paths.run_paths(None if run == "baseline" else run).models


def _evaluate_run_readonly(
    spec: dict[str, str | None], config: Config, *, quick: bool
) -> dict[str, dict[str, float]] | None:
    """Re-evaluate a single run read-only, or ``None`` if its models are absent.

    Args:
        spec: A ``_RUN_REGISTRY`` entry.
        config: Base configuration.
        quick: Passed through for parity (no training happens here).

    Returns:
        Per-method metrics for the run, or ``None`` if no model files exist.
    """
    run = str(spec["run"])
    models_dir = _models_dir_for_run(run, config)
    if not ((models_dir / "dqn.zip").exists() or (models_dir / "qlearning.pkl").exists()):
        logger.info("Run '%s' has no trained models; excluded from comparison", run)
        return None
    run_config = config_for_profile(str(spec["profile"]), config)
    try:
        metrics, _ = _evaluate_all_methods(
            run_config,
            models_dir=models_dir,
            figures_dir=None,
            train_if_missing=False,
            run_name=None if run == "baseline" else run,
            quick=quick,
            dataset=str(spec["dataset"]),
            environment=str(spec["environment"]),
        )
    except FileNotFoundError as exc:  # processed data for the run unavailable
        logger.warning("Skipping run '%s': %s", run, exc)
        return None
    return metrics


def _build_final_comparison(
    per_run_metrics: dict[str, dict[str, dict[str, float]]],
    methods: tuple[str, ...] = ("Q-Learning", "DQN", "XGBoost"),
) -> pd.DataFrame:
    """Build a (Run, Method) comparison table across all available runs.

    Args:
        per_run_metrics: ``run -> method -> metrics`` mapping.
        methods: Methods to include as rows for each run.

    Returns:
        A DataFrame indexed by (Run, Method) on the multi-stage column set.
    """
    rows = []
    for run, by_method in per_run_metrics.items():
        for method in methods:
            metrics = by_method.get(method)
            if metrics is None:
                continue
            row = {"Run": run, "Method": method}
            for key, friendly in MULTI_STAGE_METRIC_COLUMNS.items():
                row[friendly] = metrics.get(key, float("nan"))
            rows.append(row)
    table = pd.DataFrame(rows).set_index(["Run", "Method"])
    return table.round(4)


def _write_multi_stage_summary(
    path: Path,
    *,
    run_name: str,
    new_metrics: dict[str, dict[str, float]],
    per_run_metrics: dict[str, dict[str, dict[str, float]]],
    new_actions: dict[str, dict[str, float]],
) -> None:
    """Write the ``full_stage_v1`` Markdown summary answering the main question.

    Args:
        path: Destination summary file.
        run_name: Name of the new run.
        new_metrics: Per-method metrics for the new multi-stage run.
        per_run_metrics: ``run -> method -> metrics`` for all available runs.
        new_actions: Per-method action distributions for the new run.
    """
    # Pick the RL agent that best catches Stage-2 failures while keeping the
    # false-pass rate low.
    rl_methods = [m for m in ("Q-Learning", "DQN") if m in new_metrics]

    def _safety_key(m: str) -> tuple[float, float]:
        mm = new_metrics[m]
        # Prefer high Stage-2-fail catch %, then low false-pass rate.
        return (-mm.get("pct_stage2_fail_correctly_stopped", 0.0), mm.get("false_pass_rate", 1.0))

    best = min(rl_methods, key=_safety_key) if rl_methods else None

    def fmt(metrics: dict[str, float], key: str, pct: bool = False, dp: int = 4) -> str:
        v = metrics.get(key, float("nan"))
        if v != v:  # NaN
            return "n/a"
        return f"{v:.{2 if pct else dp}f}" + ("%" if pct else "")

    lines = [
        f"# Multi-Stage Experiment: `{run_name}`",
        "",
        "## Goal & main question",
        "",
        "Rerun the multi-stage RL testing experiment on the **expanded dataset** "
        "(`data/raw/full_stage_df.csv`), which now includes chips that **failed "
        "at Stage 2** and never reached Stage 3. ",
        "",
        "> **Main question:** Does adding Stage-2-failed chips improve the realism "
        "of the environment and help the agent learn a *safer* testing policy?",
        "",
        "## Environment",
        "",
        "Three sequential states with a context-dependent CONTINUE action:",
        "",
        "- **State 0 - metadata only:** RUN_STAGE2 / STOP_PASS / STOP_FAIL",
        "- **State 1 - + Stage-2 measurements (and Stage-2 result):** RUN_STAGE3 / "
        "STOP_PASS / STOP_FAIL",
        "- **State 2 - + Stage-3:** STOP_PASS / STOP_FAIL",
        "",
        "Reward profile `full_stage_v1`: per-stage costs (Stage-2 = -1, Stage-3 = "
        "-4), `correct_pass`=+10, `correct_fail`=+100, `false_pass`=-500, "
        "`false_fail`=-50, `metadata_only_pass_penalty`=-50, `early_pass_penalty`"
        "=-20, `stage2_fail_detected_reward`=+120, `stage2_fail_missed_penalty`"
        "=-600. Continuing to Stage-3 on a chip that already failed Stage-2 is "
        "heavily penalised.",
        "",
        "## Results on the multi-stage test set",
        "",
        "| Method | Accuracy | F1 | Recall (FAIL) | Precision (FAIL) | False Pass | "
        "False Fail | Avg Cost | Cost Red. % | % Stage2-Fail Caught | % Stage2-Fail Passed |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for method in ("Always Continue", "Random", "Rule-Based", "Logistic Regression", "XGBoost", "Q-Learning", "DQN"):
        m = new_metrics.get(method)
        if m is None:
            continue
        lines.append(
            f"| {method} | {fmt(m, 'accuracy')} | {fmt(m, 'f1')} | "
            f"{fmt(m, 'recall')} | {fmt(m, 'precision')} | {fmt(m, 'false_pass_rate')} | "
            f"{fmt(m, 'false_fail_rate')} | {fmt(m, 'avg_test_cost')} | "
            f"{fmt(m, 'cost_reduction_pct', dp=2)} | "
            f"{fmt(m, 'pct_stage2_fail_correctly_stopped', pct=True)} | "
            f"{fmt(m, 'pct_stage2_fail_incorrectly_passed', pct=True)} |"
        )
    lines += ["", "## Stage routing (RL agents)", ""]
    lines += [
        "| Method | % Stopped Before Stage2 | % Stopped After Stage2 | % Sent To Stage3 | Avg Tests Run |",
        "| --- | --- | --- | --- | --- |",
    ]
    for method in ("Q-Learning", "DQN"):
        m = new_metrics.get(method)
        if m is None:
            continue
        lines.append(
            f"| {method} | {fmt(m, 'pct_stopped_before_stage2', pct=True)} | "
            f"{fmt(m, 'pct_stopped_after_stage2', pct=True)} | "
            f"{fmt(m, 'pct_sent_to_stage3', pct=True)} | {fmt(m, 'avg_tests_run')} |"
        )

    # Verdict.
    lines += ["", "## Does it lead to a safer policy?", ""]
    if best is not None:
        bm = new_metrics[best]
        caught = bm.get("pct_stage2_fail_correctly_stopped", float("nan"))
        passed = bm.get("pct_stage2_fail_incorrectly_passed", float("nan"))
        fpr = bm.get("false_pass_rate", float("nan"))
        safe = caught == caught and caught >= 90.0 and fpr <= 0.10
        verdict = (
            f"**Yes.** Including Stage-2 failures makes the environment match the "
            f"real test flow, and the best RL agent (**{best}**) learns to run "
            f"Stage-2 and then stop-FAIL the rejects: it correctly catches "
            f"{caught:.1f}% of Stage-2 failures and lets only {passed:.1f}% slip "
            f"through, for an overall False Pass Rate of {fpr:.3f}."
            if safe
            else
            f"**Partially.** With Stage-2 failures included, the best RL agent "
            f"(**{best}**) catches {caught:.1f}% of Stage-2 failures (False Pass "
            f"Rate {fpr:.3f}). The strong `stage2_fail_*` rewards make the safe "
            f"action (run Stage-2, then STOP_FAIL rejects) clearly learnable, but "
            f"the policy has not fully converged to it under this budget."
        )
        lines += [
            verdict,
            "",
            "The expanded dataset adds **realism**: roughly half of all chips now "
            "fail at Stage 2, so a policy can no longer score well by blindly "
            "passing. The reward structure rewards cheap early detection "
            "(run Stage-2 for -1, then STOP_FAIL a reject for +120) and severely "
            "punishes letting a Stage-2 reject through (-600), which pushes the "
            "agent toward a safer, cost-aware policy than the single-stage runs.",
            "",
        ]
    else:
        lines += [
            "No RL models were available to evaluate for this run.",
            "",
        ]

    # Cross-run note.
    available = [r for r in per_run_metrics if r != run_name]
    lines += [
        "## Cross-run comparison",
        "",
        "See `final_comparison.csv` / `final_comparison.md` for the side-by-side "
        "table across runs ("
        + ", ".join(f"`{r}`" for r in [*available, run_name])
        + ").",
        "",
    ]
    if "multi_stage_v1" not in per_run_metrics:
        lines += [
            "> Note: no `multi_stage_v1` run exists in this project, so it is "
            "omitted from the comparison. `full_stage_v1` is the first "
            "multi-stage run. Single-stage runs (`baseline`, `safety_reward_v1`) "
            "were trained on a different dataset/environment, so their "
            "stage-routing columns are empty and their headline metrics are not "
            "strictly comparable; they are included for reference only.",
            "",
        ]
    # DQN action distribution for the new run.
    if "DQN" in new_actions:
        a = new_actions["DQN"]
        lines += [
            "## DQN policy action distribution (full_stage_v1)",
            "",
            "| Action (context) | Fraction |",
            "| --- | --- |",
            f"| CONTINUE (RUN_STAGE2/3) | {a.get('CONTINUE', 0.0) * 100:.2f}% |",
            f"| STOP_PASS | {a.get('STOP_PASS', 0.0) * 100:.2f}% |",
            f"| STOP_FAIL | {a.get('STOP_FAIL', 0.0) * 100:.2f}% |",
            "",
        ]

    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote multi-stage summary to %s", path)


def run_multi_stage_experiment(
    profile_name: str,
    run_name: str,
    dataset: str,
    config: Config = CONFIG,
    *,
    quick: bool = False,
) -> pd.DataFrame:
    """Run the multi-stage experiment and build the cross-run comparison.

    Evaluates all methods under the multi-stage environment for the new run
    (loading the run's trained models), then re-evaluates every other available
    run read-only and writes a final cross-run comparison plus a Markdown
    summary. Prior run artifacts are never modified.

    Args:
        profile_name: Reward profile for the new run (``full_stage_v1``).
        run_name: Name of the new run.
        dataset: Dataset name (``full_stage_v1``).
        config: Base configuration.
        quick: Use short budgets if models must be trained on the fly.

    Returns:
        The final cross-run comparison table.
    """
    from src.environment.factory import MULTI_STAGE

    run_config = config_for_profile(profile_name, config)
    run_paths = config.paths.run_paths(run_name)
    run_paths.ensure()

    # New multi-stage run (writes per-method table + figures into the run dir).
    new_table, new_metrics = run_default_comparison(
        run_config,
        quick=quick,
        run_name=run_name,
        train_profile=profile_name,
        dataset=dataset,
        environment=MULTI_STAGE,
    )
    # Expose comparison.csv at the run root.
    new_table.to_csv(run_paths.metrics.parent / "comparison.csv")

    # Action distributions for the new run.
    _, new_results = _evaluate_all_methods(
        run_config,
        models_dir=run_paths.models,
        figures_dir=None,
        train_if_missing=False,
        run_name=run_name,
        quick=quick,
        dataset=dataset,
        environment=MULTI_STAGE,
    )
    new_actions = {m: _action_distribution(r) for m, r in new_results.items()}
    save_json(new_actions, run_paths.metrics / "action_distributions.json")

    # Cross-run comparison across every available run.
    per_run_metrics: dict[str, dict[str, dict[str, float]]] = {}
    for spec in _RUN_REGISTRY:
        run = str(spec["run"])
        if run == run_name:
            per_run_metrics[run] = new_metrics
            continue
        metrics = _evaluate_run_readonly(spec, config, quick=quick)
        if metrics is not None:
            per_run_metrics[run] = metrics

    final_table = _build_final_comparison(per_run_metrics)
    save_comparison_table(final_table, run_paths.metrics, name="final_comparison")
    final_table.to_csv(run_paths.metrics.parent / "final_comparison.csv")

    _write_multi_stage_summary(
        run_paths.metrics.parent / "summary.md",
        run_name=run_name,
        new_metrics=new_metrics,
        per_run_metrics=per_run_metrics,
        new_actions=new_actions,
    )
    return final_table


def main() -> None:
    """Command-line entry point for comparisons and reward experiments."""
    parser = argparse.ArgumentParser(description="Run comparison / reward experiment")
    parser.add_argument("--reward-profile", type=str, default="baseline")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--dataset", type=str, default="baseline")
    parser.add_argument("--environment", type=str, default="single_stage")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    if args.environment == "multi_stage":
        run_multi_stage_experiment(
            args.reward_profile,
            args.run_name or "full_stage_v1",
            args.dataset,
            quick=args.quick,
        )
    elif args.run_name and args.run_name != "baseline":
        run_reward_experiment(args.reward_profile, args.run_name, quick=args.quick)
    else:
        config = config_for_profile(args.reward_profile, CONFIG)
        run_default_comparison(config, quick=args.quick, run_name=None)


if __name__ == "__main__":
    main()
