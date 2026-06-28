"""Multi-objective reward-sensitivity sweep for the DQN agent.

Trains the *existing* DQN agent (unchanged algorithm / architecture /
hyperparameters) on the multi-stage environment under a range of
``false_pass_penalty`` values and random seeds, evaluates every model on the
exact same held-out test split, aggregates results across seeds, performs a
Pareto-frontier analysis (minimise False Pass Rate, maximise Cost Reduction,
subject to a FAIL-recall constraint) and produces publication-quality figures.

Run the full experiment with a single command::

    python experiments/run_reward_sweep.py --config configs/reward_sweep.yaml

Useful flags::

    --quick            # fast smoke test (1 seed, 2 penalties, tiny budget)
    --skip-training    # re-aggregate / re-plot from existing run folders
    --seeds 42 123     # override the seeds from the config
    --penalties -100 -500
    --timesteps 50000

Outputs are written under ``results/reward_sweep/`` (configurable via
``--output-dir``)::

    results/reward_sweep/
        seed_42/penalty_-100/{model_last.zip, model_best.zip,
                              config.yaml, metrics.csv, training_log.csv,
                              run_metadata.json, training_curves.png}
        ...
        summary_all_runs.csv
        summary_by_penalty.csv
        final_summary_table.csv
        sweep_metadata.json
        plots/figure_*.png|pdf
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import subprocess
import sys
import time
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

# Allow `python experiments/run_reward_sweep.py` from the project root by
# ensuring the project root (which contains the `src` and `experiments`
# packages) is importable regardless of the script's launch directory.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd
import yaml
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from src.agents.dqn_agent import DQNAgent
from src.config import (
    CONFIG,
    Config,
    RewardConfig,
    reward_config_from_mapping,
)
from src.environment.chip_testing_env import LABEL_FAIL, LABEL_PASS
from src.environment.factory import load_dataset_bundle, make_env
from src.evaluation.evaluate import rollout_agent
from src.evaluation.metrics import confusion_matrix_counts, full_metrics
from src.utils.helpers import get_logger, save_json, set_global_seed

from experiments import sweep_plots

logger = get_logger("reward_sweep")

# Packages whose versions are worth recording for reproducibility.
_TRACKED_PACKAGES = (
    "numpy",
    "pandas",
    "scikit-learn",
    "matplotlib",
    "gymnasium",
    "stable-baselines3",
    "torch",
    "xgboost",
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class SweepConfig:
    """Parsed sweep configuration (from YAML + CLI overrides)."""

    name: str
    dataset: str
    environment: str
    total_timesteps: int
    seeds: list[int]
    false_pass_penalty_sweep: list[float]
    recall_constraint: float
    reward_defaults: dict[str, float]


def load_sweep_config(path: Path) -> SweepConfig:
    """Load and validate the sweep YAML configuration."""
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    exp = raw["experiment"]
    return SweepConfig(
        name=str(exp.get("name", "reward_sweep")),
        dataset=str(exp.get("dataset", "full_stage_v1")),
        environment=str(exp.get("environment", "multi_stage")),
        total_timesteps=int(exp.get("total_timesteps", 200_000)),
        seeds=[int(s) for s in exp.get("seeds", [42, 123, 2024])],
        false_pass_penalty_sweep=[
            float(p) for p in exp.get("false_pass_penalty_sweep", [-100, -500])
        ],
        recall_constraint=float(exp.get("recall_constraint", 0.95)),
        reward_defaults={str(k): float(v) for k, v in raw["reward"].items()},
    )


# ---------------------------------------------------------------------------
# Training callback (logging + best-checkpoint selection)
# ---------------------------------------------------------------------------
class TrainingMonitorCallback(BaseCallback):
    """Records training curves and checkpoints the best model.

    The underlying DQN algorithm is *not* modified; this callback only observes
    episode statistics exposed by the ``Monitor`` wrapper and periodically
    snapshots a smoothed training log and the best-so-far model.

    ``model_best`` selection: lowest-cost full-test evaluation of every
    checkpoint across 3 seeds x 6 penalties would be prohibitively expensive, so
    (as permitted by the spec) we checkpoint the model with the best smoothed
    training episode reward. This is documented in each run's metadata.
    """

    def __init__(
        self,
        best_path: Path,
        *,
        snapshot_every: int = 1_000,
        window: int = 100,
        best_check_every: int = 2_000,
        best_min_episodes: int = 50,
    ) -> None:
        super().__init__()
        self.best_path = best_path
        self.snapshot_every = snapshot_every
        self.best_check_every = best_check_every
        self.best_min_episodes = best_min_episodes
        self.recent: collections.deque[float] = collections.deque(maxlen=window)
        self.records: list[dict[str, float]] = []
        self.episodes = 0
        self.best_mean = -np.inf
        self.best_saved = False
        self._next_snapshot = snapshot_every
        self._next_best_check = best_check_every

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            episode = info.get("episode")
            if episode is not None:
                self.episodes += 1
                self.recent.append(float(episode["r"]))

        t = int(self.num_timesteps)
        if t >= self._next_snapshot:
            self._next_snapshot += self.snapshot_every
            loss = self.model.logger.name_to_value.get("train/loss", float("nan"))
            mean_r = float(np.mean(self.recent)) if self.recent else float("nan")
            self.records.append(
                {
                    "timestep": float(t),
                    "episodes": float(self.episodes),
                    "mean_reward_100": mean_r,
                    "loss": float(loss),
                }
            )

        if t >= self._next_best_check:
            self._next_best_check += self.best_check_every
            if self.episodes >= self.best_min_episodes and self.recent:
                mean_r = float(np.mean(self.recent))
                if mean_r > self.best_mean:
                    self.best_mean = mean_r
                    self.model.save(str(self.best_path))
                    self.best_saved = True
        return True


# ---------------------------------------------------------------------------
# Reproducibility helpers
# ---------------------------------------------------------------------------
def _git_commit_hash() -> str | None:
    """Return the current git commit hash, or ``None`` if unavailable."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).resolve().parent,
        )
        return out.stdout.strip()
    except Exception:  # pragma: no cover - git may be absent
        return None


def _package_versions() -> dict[str, str]:
    """Capture versions of the key scientific / RL packages."""
    versions: dict[str, str] = {}
    for pkg in _TRACKED_PACKAGES:
        try:
            versions[pkg] = importlib_metadata.version(pkg)
        except importlib_metadata.PackageNotFoundError:  # pragma: no cover
            continue
    return versions


def _dqn_hyperparameters(config: Config) -> dict[str, Any]:
    """Return the DQN hyperparameters used for every run (unchanged)."""
    return dataclasses.asdict(config.dqn)


def _reward_to_yaml(reward: RewardConfig) -> dict[str, Any]:
    """Serialise a :class:`RewardConfig` to a plain mapping for YAML output."""
    return dataclasses.asdict(reward)


# ---------------------------------------------------------------------------
# Train + evaluate one run
# ---------------------------------------------------------------------------
def _full_testing_cost(reward: RewardConfig) -> float:
    """Cost of fully testing one chip (Stage-2 + Stage-3)."""
    return reward.stage_cost(1) + reward.stage_cost(2)


def evaluate_model(
    model_path: Path,
    bundle: Any,
    run_config: Config,
    reward: RewardConfig,
) -> dict[str, float]:
    """Evaluate a saved DQN model on the shared test split.

    Returns a flat metrics dict using the canonical column names used in the
    summary CSVs.
    """
    eval_env = make_env(bundle, "test", run_config, reward_config=reward)
    agent = DQNAgent(run_config)
    agent.load(model_path)

    t0 = time.perf_counter()
    result = rollout_agent(agent, eval_env)
    inference_time = time.perf_counter() - t0

    full_cost = _full_testing_cost(reward)
    metrics = full_metrics(result, run_config, full_testing_cost=full_cost)

    cm = confusion_matrix_counts(result.true_labels, result.predicted_labels)
    tn, fp, fn, tp = (int(x) for x in cm.ravel())

    preds = np.asarray(result.predicted_labels).astype(int)
    n_test = int(len(preds))

    return {
        "Accuracy": metrics["accuracy"],
        "Precision": metrics["precision"],
        "Recall": metrics["recall"],
        "F1": metrics["f1"],
        "FalsePassRate": metrics["false_pass_rate"],
        "FalseFailRate": metrics["false_fail_rate"],
        "EscapedDefects": fn,
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "CostReduction": metrics["cost_reduction_pct"],
        "AverageTests": metrics.get("avg_tests_run", float("nan")),
        "AverageReward": metrics["avg_reward"],
        "InferenceTime": inference_time,
        "PctStopBeforeStage2": metrics.get("pct_stopped_before_stage2", float("nan")),
        "PctStopAfterStage2": metrics.get("pct_stopped_after_stage2", float("nan")),
        "PctContinueToStage3": metrics.get("pct_sent_to_stage3", float("nan")),
        "PctStopPass": float(np.mean(preds == LABEL_PASS) * 100.0),
        "PctStopFail": float(np.mean(preds == LABEL_FAIL) * 100.0),
        "PctStage2FailCaught": metrics.get(
            "pct_stage2_fail_correctly_stopped", float("nan")
        ),
        "NTest": n_test,
    }


def run_single(
    run_dir: Path,
    seed: int,
    penalty: float,
    sweep: SweepConfig,
    bundle: Any,
    base_config: Config,
    *,
    git_hash: str | None,
    pkg_versions: dict[str, str],
) -> dict[str, float]:
    """Train + evaluate one (seed, penalty) combination; persist all artifacts."""
    run_dir.mkdir(parents=True, exist_ok=True)

    reward_mapping = dict(sweep.reward_defaults)
    reward_mapping["false_pass_penalty"] = penalty
    reward = reward_config_from_mapping(reward_mapping)
    run_config = dataclasses.replace(base_config, seed=seed, reward=reward)

    logger.info("=== seed=%d penalty=%g : training DQN ===", seed, penalty)
    set_global_seed(seed)
    train_env = Monitor(make_env(bundle, "train", run_config, reward_config=reward))
    agent = DQNAgent(run_config)
    agent.build(train_env)

    best_path = run_dir / "model_best"
    callback = TrainingMonitorCallback(best_path)
    t0 = time.perf_counter()
    agent.train(train_env, total_timesteps=sweep.total_timesteps, callback=callback)
    training_time = time.perf_counter() - t0

    agent.save(run_dir / "model_last")
    if not callback.best_saved:
        # Fallback: never accumulated enough episodes to checkpoint a "best".
        agent.save(best_path)

    training_log = pd.DataFrame(callback.records)
    training_log.to_csv(run_dir / "training_log.csv", index=False)
    sweep_plots.plot_run_training_curves(training_log, run_dir, seed, penalty)

    # Evaluate the *best* checkpoint on the shared test split.
    eval_metrics = evaluate_model(best_path, bundle, run_config, reward)
    eval_metrics_last = evaluate_model(run_dir / "model_last", bundle, run_config, reward)

    row: dict[str, float] = {"Seed": seed, "Penalty": penalty, **eval_metrics}
    row["TrainingTime"] = training_time

    # Per-run metrics.csv (best checkpoint is the reported model).
    pd.DataFrame([row]).to_csv(run_dir / "metrics.csv", index=False)

    # Exact reward + DQN config for this run.
    with (run_dir / "config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            {
                "seed": seed,
                "false_pass_penalty": penalty,
                "reward": _reward_to_yaml(reward),
                "dqn": _dqn_hyperparameters(run_config),
                "total_timesteps": sweep.total_timesteps,
                "dataset": sweep.dataset,
                "environment": sweep.environment,
            },
            handle,
            sort_keys=False,
        )

    save_json(
        {
            "seed": seed,
            "false_pass_penalty": penalty,
            "reward_config": _reward_to_yaml(reward),
            "dqn_hyperparameters": _dqn_hyperparameters(run_config),
            "dataset": sweep.dataset,
            "environment": sweep.environment,
            "total_timesteps": sweep.total_timesteps,
            "n_train": int(len(bundle.train)),
            "n_test": int(len(bundle.test)),
            "training_time_s": training_time,
            "inference_time_s": eval_metrics["InferenceTime"],
            "model_best_selection": (
                "best smoothed training episode reward (window=100); "
                "see TrainingMonitorCallback docstring"
            ),
            "model_best_metrics": eval_metrics,
            "model_last_metrics": eval_metrics_last,
            "git_commit": git_hash,
            "package_versions": pkg_versions,
        },
        run_dir / "run_metadata.json",
    )
    logger.info(
        "seed=%d penalty=%g -> Recall=%.3f FPR=%.4f CostRed=%.1f%% (train %.1fs)",
        seed,
        penalty,
        row["Recall"],
        row["FalsePassRate"],
        row["CostReduction"],
        training_time,
    )
    return row


# ---------------------------------------------------------------------------
# Aggregation + Pareto analysis
# ---------------------------------------------------------------------------
# Metrics aggregated (mean/std) across seeds in summary_by_penalty.csv.
_AGG_METRICS = [
    "Recall",
    "FalsePassRate",
    "FalseFailRate",
    "CostReduction",
    "AverageTests",
    "F1",
    "Accuracy",
    "Precision",
    "AverageReward",
    "EscapedDefects",
    "InferenceTime",
    "TrainingTime",
    "PctStopBeforeStage2",
    "PctStopAfterStage2",
    "PctContinueToStage3",
    "PctStopPass",
    "PctStopFail",
    "PctStage2FailCaught",
]


def aggregate_by_penalty(all_runs: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-run metrics to mean/std per ``false_pass_penalty``."""
    present = [m for m in _AGG_METRICS if m in all_runs.columns]
    grouped = all_runs.groupby("Penalty")[present].agg(["mean", "std"])
    grouped.columns = [f"{metric}_{stat}" for metric, stat in grouped.columns]
    grouped = grouped.reset_index()
    # std is NaN for a single seed; report 0.0 for readability.
    std_cols = [c for c in grouped.columns if c.endswith("_std")]
    grouped[std_cols] = grouped[std_cols].fillna(0.0)
    grouped["NumSeeds"] = (
        all_runs.groupby("Penalty")["Seed"].nunique().reset_index(drop=True)
    )
    return grouped


def pareto_flags(
    frame: pd.DataFrame,
    fpr_col: str,
    cost_col: str,
    recall_col: str,
    recall_min: float,
) -> tuple[pd.Series, pd.Series]:
    """Compute ``Meets_Recall_Constraint`` and ``Pareto_Optimal`` boolean series.

    Pareto objectives: minimise ``fpr_col``, maximise ``cost_col``. Only
    solutions meeting ``recall_col >= recall_min`` are considered for the
    frontier. A point is Pareto-optimal if no other valid point has both a
    lower-or-equal FPR and a higher-or-equal Cost Reduction, with at least one
    strict improvement.
    """
    meets = frame[recall_col] >= recall_min
    pareto = pd.Series(False, index=frame.index)
    valid_idx = list(frame.index[meets])
    for i in valid_idx:
        fpr_i = frame.at[i, fpr_col]
        cost_i = frame.at[i, cost_col]
        dominated = False
        for j in valid_idx:
            if j == i:
                continue
            fpr_j = frame.at[j, fpr_col]
            cost_j = frame.at[j, cost_col]
            if (
                fpr_j <= fpr_i
                and cost_j >= cost_i
                and (fpr_j < fpr_i or cost_j > cost_i)
            ):
                dominated = True
                break
        pareto[i] = not dominated
    return meets, pareto


def build_final_table(by_penalty: pd.DataFrame) -> pd.DataFrame:
    """Produce the ranked final summary table from aggregated results."""
    cols = [
        "Penalty",
        "Recall_mean",
        "Recall_std",
        "FalsePassRate_mean",
        "FalsePassRate_std",
        "FalseFailRate_mean",
        "FalseFailRate_std",
        "CostReduction_mean",
        "CostReduction_std",
        "AverageTests_mean",
        "AverageTests_std",
        "Accuracy_mean",
        "F1_mean",
        "Precision_mean",
        "EscapedDefects_mean",
        "Pareto_Optimal",
        "Meets_Recall_Constraint",
    ]
    available = [c for c in cols if c in by_penalty.columns]
    table = by_penalty[available].copy()
    table = table.sort_values(
        by=[
            "Meets_Recall_Constraint",
            "Pareto_Optimal",
            "FalsePassRate_mean",
            "CostReduction_mean",
        ],
        ascending=[False, False, True, False],
    ).reset_index(drop=True)
    return table


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def collect_existing_runs(output_dir: Path) -> pd.DataFrame:
    """Reload per-run metrics from existing ``seed_*/penalty_*`` folders."""
    rows = []
    for metrics_path in sorted(output_dir.glob("seed_*/penalty_*/metrics.csv")):
        rows.append(pd.read_csv(metrics_path))
    if not rows:
        raise FileNotFoundError(
            f"No existing run metrics found under {output_dir} (run without --skip-training first)"
        )
    return pd.concat(rows, ignore_index=True)


def run_experiment(sweep: SweepConfig, output_dir: Path, *, skip_training: bool) -> None:
    """Execute the full sweep: train, evaluate, aggregate, analyse, plot."""
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    base_config = CONFIG
    git_hash = _git_commit_hash()
    pkg_versions = _package_versions()

    if skip_training:
        logger.info("Skipping training; re-aggregating existing runs in %s", output_dir)
        all_runs = collect_existing_runs(output_dir)
    else:
        logger.info(
            "Loading dataset '%s' / environment '%s'", sweep.dataset, sweep.environment
        )
        bundle = load_dataset_bundle(sweep.dataset, sweep.environment, base_config)
        logger.info(
            "Train chips: %d | Test chips: %d", len(bundle.train), len(bundle.test)
        )

        rows: list[dict[str, float]] = []
        total = len(sweep.seeds) * len(sweep.false_pass_penalty_sweep)
        done = 0
        # Seed-outer loop: complete the full penalty sweep within each seed first.
        for seed in sweep.seeds:
            for penalty in sweep.false_pass_penalty_sweep:
                run_dir = output_dir / f"seed_{seed}" / f"penalty_{int(penalty)}"
                row = run_single(
                    run_dir,
                    seed,
                    penalty,
                    sweep,
                    bundle,
                    base_config,
                    git_hash=git_hash,
                    pkg_versions=pkg_versions,
                )
                rows.append(row)
                done += 1
                logger.info("Progress: %d/%d runs complete", done, total)
        all_runs = pd.DataFrame(rows)

    # --- summary_all_runs.csv + per-run Pareto ---
    meets_run, pareto_run = pareto_flags(
        all_runs, "FalsePassRate", "CostReduction", "Recall", sweep.recall_constraint
    )
    all_runs = all_runs.copy()
    all_runs["Meets_Recall_Constraint"] = meets_run.values
    all_runs["Pareto_Optimal"] = pareto_run.values
    all_runs = all_runs.sort_values(["Seed", "Penalty"]).reset_index(drop=True)
    all_runs.to_csv(output_dir / "summary_all_runs.csv", index=False)

    # --- summary_by_penalty.csv + aggregated Pareto ---
    by_penalty = aggregate_by_penalty(all_runs)
    meets_agg, pareto_agg = pareto_flags(
        by_penalty,
        "FalsePassRate_mean",
        "CostReduction_mean",
        "Recall_mean",
        sweep.recall_constraint,
    )
    by_penalty["Meets_Recall_Constraint"] = meets_agg.values
    by_penalty["Pareto_Optimal"] = pareto_agg.values
    by_penalty.to_csv(output_dir / "summary_by_penalty.csv", index=False)

    if not bool(meets_agg.any()):
        logger.warning(
            "No penalty value meets the Recall >= %.2f constraint; "
            "no valid Pareto frontier exists under the constraint. "
            "Tables are still saved.",
            sweep.recall_constraint,
        )

    # --- final ranked table ---
    final_table = build_final_table(by_penalty)
    final_table.to_csv(output_dir / "final_summary_table.csv", index=False)

    # --- sweep-level metadata ---
    save_json(
        {
            "name": sweep.name,
            "dataset": sweep.dataset,
            "environment": sweep.environment,
            "seeds": sweep.seeds,
            "false_pass_penalty_sweep": sweep.false_pass_penalty_sweep,
            "total_timesteps": sweep.total_timesteps,
            "recall_constraint": sweep.recall_constraint,
            "reward_defaults": sweep.reward_defaults,
            "dqn_hyperparameters": _dqn_hyperparameters(base_config),
            "git_commit": git_hash,
            "package_versions": pkg_versions,
            "n_valid_under_constraint": int(meets_agg.sum()),
        },
        output_dir / "sweep_metadata.json",
    )

    # --- plots ---
    logger.info("Generating figures under %s", plots_dir)
    run_logs = _load_run_logs(output_dir, sweep)
    sweep_plots.generate_all(all_runs, by_penalty, run_logs, plots_dir, sweep.recall_constraint)

    logger.info("Done. Summary tables and %d figures written to %s", 9, output_dir)
    _log_final_table(final_table)


def _load_run_logs(output_dir: Path, sweep: SweepConfig) -> dict[tuple[int, float], pd.DataFrame]:
    """Load training logs for every run for aggregated training-curve plots."""
    logs: dict[tuple[int, float], pd.DataFrame] = {}
    for log_path in sorted(output_dir.glob("seed_*/penalty_*/training_log.csv")):
        try:
            seed = int(log_path.parent.parent.name.split("_")[1])
            penalty = float(log_path.parent.name.split("_")[1])
        except (IndexError, ValueError):
            continue
        df = pd.read_csv(log_path)
        if not df.empty:
            logs[(seed, penalty)] = df
    return logs


def _log_final_table(table: pd.DataFrame) -> None:
    """Pretty-print the final ranked table to the log."""
    show_cols = [
        c
        for c in [
            "Penalty",
            "Recall_mean",
            "FalsePassRate_mean",
            "CostReduction_mean",
            "AverageTests_mean",
            "Pareto_Optimal",
            "Meets_Recall_Constraint",
        ]
        if c in table.columns
    ]
    logger.info("Final ranked summary:\n%s", table[show_cols].to_string(index=False))


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/reward_sweep.yaml"),
        help="Path to the reward-sweep YAML config.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/reward_sweep"),
        help="Directory for all sweep outputs.",
    )
    parser.add_argument("--seeds", type=int, nargs="+", help="Override seeds.")
    parser.add_argument(
        "--penalties", type=float, nargs="+", help="Override false_pass_penalty sweep."
    )
    parser.add_argument("--timesteps", type=int, help="Override total_timesteps per run.")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Fast smoke test (1 seed, 2 penalties, 2000 timesteps).",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Re-aggregate / re-plot from existing run folders only.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point."""
    args = parse_args()
    sweep = load_sweep_config(args.config)

    if args.quick:
        sweep.seeds = [42]
        sweep.false_pass_penalty_sweep = [-100.0, -500.0]
        sweep.total_timesteps = 2_000
    if args.seeds:
        sweep.seeds = args.seeds
    if args.penalties:
        sweep.false_pass_penalty_sweep = args.penalties
    if args.timesteps:
        sweep.total_timesteps = args.timesteps

    logger.info(
        "Reward sweep '%s': seeds=%s penalties=%s timesteps=%d",
        sweep.name,
        sweep.seeds,
        sweep.false_pass_penalty_sweep,
        sweep.total_timesteps,
    )
    run_experiment(sweep, args.output_dir, skip_training=args.skip_training)


if __name__ == "__main__":
    main()
