"""Central configuration for the Adaptive CPU Chip Test Reduction project.

All tunable parameters live here: reward values, test costs, file paths,
train/test split, the global random seed and reinforcement-learning
hyperparameters. No path is hardcoded as an absolute string - every path is
derived from :data:`PROJECT_ROOT`, which is computed relative to this file.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
# ``config.py`` lives in ``project/src/`` so the project root is two parents up.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = "full_stage_v1"


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

    # Raw and processed data for the full-stage multi-stage dataset.
    full_stage_dataset: Path = PROJECT_ROOT / "data" / "raw" / "full_stage_df.csv"
    processed_train: Path = (
        PROJECT_ROOT / "data" / "processed" / DEFAULT_DATASET / "train.csv"
    )
    processed_test: Path = (
        PROJECT_ROOT / "data" / "processed" / DEFAULT_DATASET / "test.csv"
    )

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

    def processed_split_paths(self, dataset: str | None = None) -> tuple[Path, Path]:
        """Return the ``(train, test)`` processed CSV paths for a dataset."""
        name = dataset or DEFAULT_DATASET
        base = self.processed_data / name
        return base / "train.csv", base / "test.csv"


@dataclass(frozen=True)
class RewardConfig:
    """Reward structure for the chip-testing environment.

    Action semantics (see :class:`~src.environment.actions.Action`):

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

    # Multi-stage extensions (used only by MultiStageChipTestingEnv).
    # Per-stage testing costs (positive magnitudes; reward = -cost). When
    # ``None`` the multi-stage env falls back to ``test_cost``.
    stage2_cost: float | None = None
    stage3_cost: float | None = None
    # Penalty for classifying PASS using metadata only (Stage 0, before
    # running Stage 2 at all).
    metadata_only_pass_penalty: float = 0.0
    # Reward for correctly stopping (STOP_FAIL) a chip that failed Stage 2,
    # once the Stage-2 result is known. ``None`` -> falls back to correct_fail.
    stage2_fail_detected_reward: float | None = None
    # Penalty for passing (STOP_PASS) a chip that is known to have failed
    # Stage 2. ``None`` -> falls back to false_pass.
    stage2_fail_missed_penalty: float | None = None

    # Resolved per-stage cost helpers -------------------------------------- #
    def stage_cost(self, stage_index: int) -> float:
        """Return the testing cost for entering a given stage.

        Args:
            stage_index: 1 for the Stage-2 measurements, 2 for Stage-3.

        Returns:
            The positive cost magnitude for that stage.
        """
        if stage_index == 1:
            return self.stage2_cost if self.stage2_cost is not None else self.test_cost
        if stage_index == 2:
            return self.stage3_cost if self.stage3_cost is not None else self.test_cost
        return self.test_cost

    @property
    def resolved_stage2_fail_detected_reward(self) -> float:
        """Reward for catching a Stage-2 failure (falls back to correct_fail)."""
        if self.stage2_fail_detected_reward is not None:
            return self.stage2_fail_detected_reward
        return self.correct_fail

    @property
    def resolved_stage2_fail_missed_penalty(self) -> float:
        """Penalty for passing a Stage-2 failure (falls back to false_pass)."""
        if self.stage2_fail_missed_penalty is not None:
            return self.stage2_fail_missed_penalty
        return self.false_pass


# --------------------------------------------------------------------------- #
# Named reward profiles
# --------------------------------------------------------------------------- #
# Reward profile for the full-stage multi-stage experiments.
REWARD_PROFILES: dict[str, RewardConfig] = {
    "full_stage_v1": RewardConfig(
        correct_pass=10.0,
        correct_fail=100.0,
        false_pass=-500.0,
        false_fail=-50.0,
        test_cost=1.0,
        stage2_cost=1.0,
        stage3_cost=4.0,
        metadata_only_pass_penalty=-50.0,
        early_pass_penalty=-20.0,
        stage2_fail_detected_reward=120.0,
        stage2_fail_missed_penalty=-600.0,
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

    # Hard cap on environment steps per episode (truncation guard).
    max_steps: int = 4
    # Highest stage index in the multi-stage environment (0=metadata, 2=Stage-3).
    max_stage_index: int = 2
    # Value used for not-yet-revealed features in the observation vector.
    masked_value: float = 0.0
    # Whether to sample chips with replacement during training rollouts.
    shuffle: bool = True
    # Column in the dataset holding the binary ground-truth label
    # (0 = good/PASS, 1 = defective/FAIL).
    label_column: str = "label"


@dataclass(frozen=True)
class DataConfig:
    """Train/test split configuration for the full-stage dataset."""

    test_size: float = 0.2
    val_size: float = 0.0


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
    reward: RewardConfig = field(default_factory=lambda: REWARD_PROFILES["full_stage_v1"])
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


# Mapping from the (signed) YAML reward keys to :class:`RewardConfig` fields.
# Costs/penalties are written as *signed* values in YAML (e.g. ``-1``, ``-500``)
# for readability, but :class:`RewardConfig` stores per-stage costs as positive
# magnitudes (reward = ``-cost``); the conversion below handles that.
def reward_config_from_mapping(mapping: Mapping[str, float]) -> RewardConfig:
    """Build a :class:`RewardConfig` from a YAML-style reward mapping.

    Expected keys (signed values; rewards positive, penalties/costs negative)::

        continue_penalty            -> generic per-stage test cost fallback
        stage2_cost / stage3_cost   -> optional per-stage costs (override)
        correct_pass_reward         -> RewardConfig.correct_pass
        correct_fail_reward         -> RewardConfig.correct_fail
        false_fail_penalty          -> RewardConfig.false_fail
        false_pass_penalty          -> RewardConfig.false_pass
        stage2_fail_caught_reward   -> RewardConfig.stage2_fail_detected_reward
        stage2_fail_missed_penalty  -> RewardConfig.stage2_fail_missed_penalty
        metadata_only_pass_penalty  -> RewardConfig.metadata_only_pass_penalty
        early_pass_penalty          -> RewardConfig.early_pass_penalty

    With the default ``full_stage_v1`` values this reproduces
    ``REWARD_PROFILES["full_stage_v1"]`` exactly.

    Args:
        mapping: A mapping of the reward keys above to numeric values.

    Returns:
        The corresponding :class:`RewardConfig`.

    Raises:
        KeyError: If a required reward key is missing.
    """
    required = (
        "correct_pass_reward",
        "correct_fail_reward",
        "false_fail_penalty",
        "false_pass_penalty",
    )
    missing = [k for k in required if k not in mapping]
    if missing:
        raise KeyError(f"Missing required reward keys: {missing}")

    def _cost(key: str, default: float | None) -> float | None:
        """Return a positive cost magnitude for a signed YAML cost value."""
        if key not in mapping:
            return default
        return abs(float(mapping[key]))

    continue_cost = _cost("continue_penalty", 1.0) or 1.0
    return RewardConfig(
        correct_pass=float(mapping["correct_pass_reward"]),
        correct_fail=float(mapping["correct_fail_reward"]),
        false_fail=float(mapping["false_fail_penalty"]),
        false_pass=float(mapping["false_pass_penalty"]),
        test_cost=continue_cost,
        stage2_cost=_cost("stage2_cost", None),
        stage3_cost=_cost("stage3_cost", None),
        metadata_only_pass_penalty=float(mapping.get("metadata_only_pass_penalty", 0.0)),
        early_pass_penalty=float(mapping.get("early_pass_penalty", 0.0)),
        stage2_fail_detected_reward=(
            float(mapping["stage2_fail_caught_reward"])
            if "stage2_fail_caught_reward" in mapping
            else None
        ),
        stage2_fail_missed_penalty=(
            float(mapping["stage2_fail_missed_penalty"])
            if "stage2_fail_missed_penalty" in mapping
            else None
        ),
    )
