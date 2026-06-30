"""Factory for loading the full-stage dataset and building the multi-stage env."""

from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import pandas as pd

from src.config import CONFIG, Config, RewardConfig
from src.environment.multi_stage_env import MultiStageChipTestingEnv
from src.utils.helpers import get_logger

logger = get_logger(__name__)

MULTI_STAGE = "multi_stage"
DEFAULT_DATASET = "full_stage_v1"


@dataclass
class DatasetBundle:
    """Loaded train/test splits plus environment-construction metadata."""

    train: pd.DataFrame
    test: pd.DataFrame
    feature_columns: list[str]
    label_column: str
    environment: str
    stage_groups: list[list[str]]
    stage2_fail_column: str = "is_stage2_fail"

    def split(self, which: str) -> pd.DataFrame:
        """Return the requested split ('train' or 'test')."""
        return self.train if which == "train" else self.test


def load_dataset_bundle(
    dataset: str = DEFAULT_DATASET,
    environment: str = MULTI_STAGE,
    config: Config = CONFIG,
) -> DatasetBundle:
    """Load processed multi-stage train/test splits.

    Args:
        dataset: Processed dataset name (default ``full_stage_v1``).
        environment: Kept for CLI compatibility; must be ``multi_stage``.
        config: Project configuration.

    Returns:
        A :class:`DatasetBundle` ready for environment construction.

    Raises:
        ValueError: If ``environment`` is not ``multi_stage``.
        FileNotFoundError: If processed splits are missing.
    """
    if environment != MULTI_STAGE:
        raise ValueError(
            f"Only the multi-stage environment is supported (got {environment!r}). "
            "Run `python -m src.data.prepare_full_stage_data` first."
        )
    from src.data.full_stage_loader import load_full_stage_processed

    data = load_full_stage_processed(config, dataset=dataset)
    return DatasetBundle(
        train=data.train,
        test=data.test,
        feature_columns=data.feature_columns,
        label_column=data.label_column,
        environment=MULTI_STAGE,
        stage_groups=data.columns.stage_groups,
    )


def make_env(
    bundle: DatasetBundle,
    split: str,
    config: Config = CONFIG,
    *,
    reward_config: RewardConfig | None = None,
    render_mode: str | None = None,
) -> gym.Env:
    """Construct the multi-stage environment for a bundle and split."""
    frame = bundle.split(split)
    return MultiStageChipTestingEnv(
        frame,
        bundle.stage_groups,
        config,
        reward_config=reward_config,
        label_column=bundle.label_column,
        stage2_fail_column=bundle.stage2_fail_column,
        render_mode=render_mode,
    )
