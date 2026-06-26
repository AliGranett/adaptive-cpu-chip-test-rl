"""Central configuration for the Adaptive CPU Chip Test Reduction project.

All tunable parameters live here: reward values, test costs, file paths,
train/test split, the global random seed and reinforcement-learning
hyperparameters. No path is hardcoded as an absolute string - every path is
derived from :data:`PROJECT_ROOT`, which is computed relative to this file.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
# ``config.py`` lives in ``project/src/`` so the project root is two parents up.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class RunPaths:
    """Output directories for a single named experiment run.

    The baseline run writes to the top-level ``results/{models,figures,metrics}``
    directories. Named runs are isolated under ``results/runs/<run_name>/`` so
    experiments never overwrite one another.
    """

    models: Path
    figures: Path
    metrics: Path

    def ensure(self) -> None:
        """Create the run's output directories if they do not yet exist."""
        for directory in (self.models, self.figures, self.metrics):
            directory.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class Paths:
    """Filesystem layout of the project.

    Every attribute is an absolute :class:`~pathlib.Path` derived from
    :data:`PROJECT_ROOT`, keeping the project relocatable.
    """

    root: Path = PROJECT_ROOT
    data: Path = PROJECT_ROOT / "data"
    raw_data: Path = PROJECT_ROOT / "data" / "raw"
    processed_data: Path = PROJECT_ROOT / "data" / "processed"
    results: Path = PROJECT_ROOT / "results"
    figures: Path = PROJECT_ROOT / "results" / "figures"
    metrics: Path = PROJECT_ROOT / "results" / "metrics"
    models: Path = PROJECT_ROOT / "results" / "models"
    runs: Path = PROJECT_ROOT / "results" / "runs"

    # Canonical data file names.
    base_dataset: Path = PROJECT_ROOT / "data" / "raw" / "base_data.csv"
    raw_dataset: Path = PROJECT_ROOT / "data" / "raw" / "chip_tests.csv"
    train_dataset: Path = PROJECT_ROOT / "data" / "processed" / "train.csv"
    test_dataset: Path = PROJECT_ROOT / "data" / "processed" / "test.csv"

    def ensure(self) -> None:
        """Create every directory in the layout if it does not already exist."""
        for directory in (
            self.data,
            self.raw_data,
            self.processed_data,
            self.results,
            self.figures,
            self.metrics,
            self.models,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    def run_paths(self, run_name: str | None = None) -> RunPaths:
        """Resolve the output directories for a given run.

        Args:
            run_name: Name of the experiment run. ``None`` or ``"baseline"``
                maps to the top-level ``results`` directories; any other name
                is isolated under ``results/runs/<run_name>/``.

        Returns:
            The :class:`RunPaths` for the requested run.
        """
        if run_name is None or run_name == "baseline":
            return RunPaths(self.models, self.figures, self.metrics)
        base = self.runs / run_name
        return RunPaths(base / "models", base / "figures", base / "metrics")


@dataclass(frozen=True)
class RewardConfig:
    """Reward structure for the chip-testing environment.

    Action semantics (see :class:`~src.environment.chip_testing_env.Action`):

    * ``CONTINUE`` -> ``-test_cost`` per revealed test stage.
    * ``STOP_PASS`` on a truly good chip  -> ``correct_pass``.
    * ``STOP_FAIL`` on a truly bad chip   -> ``correct_fail``.
    * ``STOP_FAIL`` on a good chip (false fail) -> ``false_fail``.
    * ``STOP_PASS`` on a bad chip  (false pass) -> ``false_pass``.

    ``early_pass_penalty`` is an *additional* penalty added to a ``STOP_PASS``
    reward when the agent classifies PASS *before* ever choosing ``CONTINUE``
    (i.e. before any additional Stage-3 information has been revealed). It is
    not applied once the agent has continued at least once. With the default of
    ``0.0`` it has no effect, preserving the original baseline behaviour.
    """

    correct_pass: float = 20.0
    correct_fail: float = 20.0
    false_fail: float = -50.0
    false_pass: float = -100.0
    # Cost charged each time the agent chooses to CONTINUE testing.
    test_cost: float = 1.0
    # Extra penalty for passing a chip without any additional testing.
    early_pass_penalty: float = 0.0


# --------------------------------------------------------------------------- #
# Named reward profiles
# --------------------------------------------------------------------------- #
# ``baseline`` reproduces the original real-data run exactly. ``safety_reward_v1``
# is a safety-oriented profile: a much harsher false-pass penalty and a large
# correct-fail reward to push the policy towards catching more defective chips,
# while keeping continue cheap and discouraging passing without any testing.
REWARD_PROFILES: dict[str, RewardConfig] = {
    "baseline": RewardConfig(),
    "safety_reward_v1": RewardConfig(
        correct_pass=10.0,
        correct_fail=100.0,
        false_fail=-50.0,
        false_pass=-500.0,
        # ``continue_cost`` of -2 is expressed as a positive per-step magnitude.
        test_cost=2.0,
        early_pass_penalty=-20.0,
    ),
}


def get_reward_profile(name: str) -> RewardConfig:
    """Return the :class:`RewardConfig` for a named reward profile.

    Args:
        name: One of the keys in :data:`REWARD_PROFILES`.

    Returns:
        The matching :class:`RewardConfig`.

    Raises:
        KeyError: If ``name`` is not a known reward profile.
    """
    if name not in REWARD_PROFILES:
        valid = ", ".join(sorted(REWARD_PROFILES))
        raise KeyError(f"Unknown reward profile '{name}'. Valid profiles: {valid}")
    return REWARD_PROFILES[name]


@dataclass(frozen=True)
class EnvConfig:
    """Configuration of the sequential test-reveal dynamics."""

    # Number of sequential test stages a chip can go through. Features are
    # partitioned into this many groups and revealed one group per CONTINUE.
    n_stages: int = 5
    # Hard cap on environment steps per episode (truncation guard).
    max_steps: int = 6
    # Value used for not-yet-revealed features in the observation vector.
    masked_value: float = 0.0
    # Whether to sample chips with replacement during training rollouts.
    shuffle: bool = True
    # Column in the dataset holding the binary ground-truth label
    # (0 = good/PASS, 1 = defective/FAIL).
    label_column: str = "label"


@dataclass(frozen=True)
class DataConfig:
    """Data loading / splitting / synthetic-generation configuration."""

    test_size: float = 0.2
    val_size: float = 0.0
    # Synthetic dataset parameters (used when no raw CSV is present).
    n_synthetic_samples: int = 4000
    n_synthetic_features: int = 20
    synthetic_fail_rate: float = 0.25
    # Fraction of values to randomly drop to emulate missing measurements.
    synthetic_missing_rate: float = 0.02


@dataclass(frozen=True)
class QLearningConfig:
    """Hyperparameters for the tabular Q-learning agent."""

    n_episodes: int = 20_000
    learning_rate: float = 0.1
    discount_factor: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.9995
    # Number of bins used to discretise each continuous observation dimension.
    n_bins: int = 6
    # Only the first ``n_discretised_features`` observation dims are binned to
    # keep the tabular state space tractable.
    n_discretised_features: int = 4


@dataclass(frozen=True)
class DQNConfig:
    """Hyperparameters for the Stable-Baselines3 DQN agent."""

    total_timesteps: int = 200_000
    learning_rate: float = 1e-3
    buffer_size: int = 50_000
    learning_starts: int = 1_000
    batch_size: int = 64
    gamma: float = 0.99
    train_freq: int = 4
    target_update_interval: int = 1_000
    exploration_fraction: float = 0.2
    exploration_final_eps: float = 0.05
    net_arch: tuple[int, ...] = (128, 128)


@dataclass(frozen=True)
class Config:
    """Top-level configuration aggregating every sub-configuration."""

    seed: int = 42
    paths: Paths = field(default_factory=Paths)
    reward: RewardConfig = field(default_factory=RewardConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    data: DataConfig = field(default_factory=DataConfig)
    qlearning: QLearningConfig = field(default_factory=QLearningConfig)
    dqn: DQNConfig = field(default_factory=DQNConfig)


# A ready-to-import default configuration instance.
CONFIG = Config()
"""Default project configuration, importable as ``from src.config import CONFIG``."""


def config_for_profile(profile_name: str, base: Config = CONFIG) -> Config:
    """Return a copy of ``base`` with its reward set to a named profile.

    Args:
        profile_name: One of the keys in :data:`REWARD_PROFILES`.
        base: The configuration to derive from (defaults to :data:`CONFIG`).

    Returns:
        A new :class:`Config` whose ``reward`` is the requested profile. Every
        other setting (dataset, split, seed, hyperparameters) is unchanged.
    """
    return dataclasses.replace(base, reward=get_reward_profile(profile_name))
