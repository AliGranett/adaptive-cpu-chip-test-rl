"""Assemble and persist cross-method comparison tables for full_stage_v1."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from src.config import CONFIG, Config, config_for_profile
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
    train_profile: str = "full_stage_v1",
    dataset: str = "full_stage_v1",
    environment: str = "multi_stage",
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
        dataset: Processed dataset name (``full_stage_v1``).
        environment: ``multi_stage``.

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
    from src.evaluation.evaluate import evaluate_supervised_multi_stage, rollout_agent
    from src.evaluation.metrics import confusion_matrix_counts, full_metrics
    from src.training.train_dqn import train_dqn
    from src.training.train_qlearning import train_q_learning
    from src.utils import plotting

    bundle = load_dataset_bundle(dataset, environment, config)
    x_train = bundle.train[bundle.feature_columns].to_numpy()
    y_train = bundle.train[bundle.label_column].to_numpy()
    x_test = bundle.test[bundle.feature_columns].to_numpy()
    y_test = bundle.test[bundle.label_column].to_numpy()

    env = make_env(bundle, "test", config, reward_config=config.reward)
    n_features = env.n_features
    full_cost = config.reward.stage_cost(1) + config.reward.stage_cost(2)

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

    s2_fail_test = bundle.test["is_stage2_fail"].to_numpy()
    for name, model in (
        ("Logistic Regression", LogisticBaseline(config)),
        ("XGBoost", XGBoostBaseline(config)),
    ):
        model.fit(x_train, y_train)
        preds = model.predict(x_test)
        result = evaluate_supervised_multi_stage(preds, y_test, s2_fail_test, config)
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
    train_profile: str = "full_stage_v1",
    dataset: str = "full_stage_v1",
    environment: str = "multi_stage",
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
        dataset: Processed dataset name (``full_stage_v1``).
        environment: ``multi_stage``.

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

    table = build_comparison_table(metrics_by_method, MULTI_STAGE_METRIC_COLUMNS)
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
    from src.environment.actions import Action

    total = sum(result.action_counts.values()) or 1
    return {
        action.name: result.action_counts.get(int(action), 0) / total
        for action in Action
    }


def _write_multi_stage_summary(
    path: Path,
    *,
    run_name: str,
    new_metrics: dict[str, dict[str, float]],
    new_actions: dict[str, dict[str, float]],
) -> None:
    """Write the ``full_stage_v1`` Markdown summary."""
    rl_methods = [m for m in ("Q-Learning", "DQN") if m in new_metrics]

    def _safety_key(m: str) -> tuple[float, float]:
        mm = new_metrics[m]
        return (-mm.get("pct_stage2_fail_correctly_stopped", 0.0), mm.get("false_pass_rate", 1.0))

    best = min(rl_methods, key=_safety_key) if rl_methods else None

    def fmt(metrics: dict[str, float], key: str, pct: bool = False, dp: int = 4) -> str:
        v = metrics.get(key, float("nan"))
        if v != v:
            return "n/a"
        return f"{v:.{2 if pct else dp}f}" + ("%" if pct else "")

    lines = [
        f"# Multi-Stage Experiment: `{run_name}`",
        "",
        "## Goal & main question",
        "",
        "Evaluate multi-stage RL testing on the expanded dataset "
        "(`data/raw/full_stage_df.csv`), including chips that failed at Stage 2.",
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
        "=-600.",
        "",
        "## Results on the multi-stage test set",
        "",
        "| Method | Accuracy | F1 | Recall (FAIL) | Precision (FAIL) | False Pass | "
        "False Fail | Avg Cost | Cost Red. % | % Stage2-Fail Caught | % Stage2-Fail Passed |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for method in (
        "Always Continue",
        "Random",
        "Rule-Based",
        "Logistic Regression",
        "XGBoost",
        "Q-Learning",
        "DQN",
    ):
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

    lines += ["", "## Does it lead to a safer policy?", ""]
    if best is not None:
        bm = new_metrics[best]
        caught = bm.get("pct_stage2_fail_correctly_stopped", float("nan"))
        passed = bm.get("pct_stage2_fail_incorrectly_passed", float("nan"))
        fpr = bm.get("false_pass_rate", float("nan"))
        safe = caught == caught and caught >= 90.0 and fpr <= 0.10
        verdict = (
            f"**Yes.** The best RL agent (**{best}**) correctly catches "
            f"{caught:.1f}% of Stage-2 failures with False Pass Rate {fpr:.3f}."
            if safe
            else
            f"**Partially.** The best RL agent (**{best}**) catches "
            f"{caught:.1f}% of Stage-2 failures (False Pass Rate {fpr:.3f}, "
            f"{passed:.1f}% incorrectly passed)."
        )
        lines += [verdict, ""]
    else:
        lines += ["No RL models were available to evaluate for this run.", ""]

    if "DQN" in new_actions:
        a = new_actions["DQN"]
        lines += [
            "## DQN policy action distribution",
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
    """Run the multi-stage comparison for ``full_stage_v1``."""
    from src.environment.factory import MULTI_STAGE

    run_config = config_for_profile(profile_name, config)
    run_paths = config.paths.run_paths(run_name)
    run_paths.ensure()

    new_table, new_metrics = run_default_comparison(
        run_config,
        quick=quick,
        run_name=run_name,
        train_profile=profile_name,
        dataset=dataset,
        environment=MULTI_STAGE,
    )
    new_table.to_csv(run_paths.metrics.parent / "comparison.csv")

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

    _write_multi_stage_summary(
        run_paths.metrics.parent / "summary.md",
        run_name=run_name,
        new_metrics=new_metrics,
        new_actions=new_actions,
    )
    return new_table


def main() -> None:
    """Command-line entry point for the full_stage_v1 comparison."""
    parser = argparse.ArgumentParser(description="Run full_stage_v1 comparison")
    parser.add_argument("--reward-profile", type=str, default="full_stage_v1")
    parser.add_argument("--run-name", type=str, default="full_stage_v1")
    parser.add_argument("--dataset", type=str, default="full_stage_v1")
    parser.add_argument("--environment", type=str, default="multi_stage")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    run_multi_stage_experiment(
        args.reward_profile,
        args.run_name,
        args.dataset,
        quick=args.quick,
    )


if __name__ == "__main__":
    main()
