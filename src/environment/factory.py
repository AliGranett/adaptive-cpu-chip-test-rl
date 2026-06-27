"""Factory for selecting datasets and environments from CLI options.

Centralises the ``--dataset`` / ``--environment`` plumbing so the training and
evaluation scripts can build the correct data splits and Gymnasium environment
without duplicating logic.
"""

from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import pandas as pd

from src.config import CONFIG, Config, RewardConfig
from src.environment.chip_testing_env import ChipTestingEnv
from src.environment.multi_stage_env import MultiStageChipTestingEnv
from src.utils.helpers import get_logger

logger = get_logger(__name__)

SINGLE_STAGE = "single_stage"
MULTI_STAGE = "multi_stage"


@dataclass
class DatasetBundle:
    """Loaded train/test splits plus environment-construction metadata."""

    train: pd.DataFrame
    test: pd.DataFrame
    feature_columns: list[str]
    label_column: str
    environment: str
    stage_groups: list[list[str]] | None = None
    stage2_fail_column: str | None = None

    @property
    def is_multi_stage(self) -> bool:
        """Whether this bundle targets the multi-stage environment."""
        return self.environment == MULTI_STAGE

    def split(self, which: str) -> pd.DataFrame:
        """Return the requested split ('train' or 'test')."""
        return self.train if which == "train" else self.test


def load_dataset_bundle(
    dataset: str = "baseline",
    environment: str = SINGLE_STAGE,
    config: Config = CONFIG,
) -> DatasetBundle:
    """Load the processed data for a dataset/environment combination.

    Args:
        dataset: Dataset name (``"baseline"`` or e.g. ``"full_stage_v1"``).
        environment: ``"single_stage"`` or ``"multi_stage"``.
        config: Project configuration.

    Returns:
        A :class:`DatasetBundle` ready for environment construction.
    """
    if environment == MULTI_STAGE:
        from src.data.full_stage_loader import load_full_stage_processed

        data = load_full_stage_processed(config, dataset=dataset)
        return DatasetBundle(
            train=data.train,
            test=data.test,
            feature_columns=data.feature_columns,
            label_column=data.label_column,
            environment=MULTI_STAGE,
            stage_groups=data.columns.stage_groups,
            stage2_fail_column="is_stage2_fail",
        )

    from src.data.preprocessing import load_processed_data

    data = load_processed_data(config)
    return DatasetBundle(
        train=data.train,
        test=data.test,
        feature_columns=data.feature_columns,
        label_column=data.label_column,
        environment=SINGLE_STAGE,
    )


def make_env(
    bundle: DatasetBundle,
    split: str,
    config: Config = CONFIG,
    *,
    reward_config: RewardConfig | None = None,
    render_mode: str | None = None,
) -> gym.Env:
    """Construct the appropriate environment for a bundle and split.

    Args:
        bundle: The loaded dataset bundle.
        split: ``"train"`` or ``"test"``.
        config: Project configuration.
        reward_config: Reward profile to apply.
        render_mode: Optional Gymnasium render mode.

    Returns:
        A constructed Gymnasium environment.
    """
    frame = bundle.split(split)
    if bundle.is_multi_stage:
        assert bundle.stage_groups is not None
        return MultiStageChipTestingEnv(
            frame,
            bundle.stage_groups,
            config,
            reward_config=reward_config,
            label_column=bundle.label_column,
            stage2_fail_column=bundle.stage2_fail_column or "is_stage2_fail",
            render_mode=render_mode,
        )
    return ChipTestingEnv(
        frame,
        bundle.feature_columns,
        config,
        reward_config=reward_config,
        render_mode=render_mode,
    )
