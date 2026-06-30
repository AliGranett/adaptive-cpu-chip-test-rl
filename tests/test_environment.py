"""Tests for the MultiStageChipTestingEnv Gymnasium environment."""

from __future__ import annotations

import numpy as np

from src.config import Config
from src.data.full_stage_loader import MultiStageData
from src.environment.actions import LABEL_FAIL, LABEL_PASS, Action
from src.environment.multi_stage_env import MultiStageChipTestingEnv


def _make_env(data: MultiStageData, config: Config) -> MultiStageChipTestingEnv:
    return MultiStageChipTestingEnv(
        data.train,
        data.columns.stage_groups,
        config,
    )


def test_observation_and_action_spaces(
    multi_stage_data: MultiStageData, small_config: Config
) -> None:
    env = _make_env(multi_stage_data, small_config)
    obs, info = env.reset()
    assert env.observation_space.shape == (2 * env.n_features + 1,)
    assert env.action_space.n == 3
    assert obs.shape == env.observation_space.shape
    assert obs.dtype == np.float32
    assert "true_label" in info
    assert info["stage"] == 0


def test_continue_reveals_stage2(
    multi_stage_data: MultiStageData, small_config: Config
) -> None:
    env = _make_env(multi_stage_data, small_config)
    obs, _ = env.reset(options={"index": 0})
    mask_before = obs[env.n_features : 2 * env.n_features].sum()
    obs, reward, terminated, truncated, info = env.step(Action.CONTINUE)
    mask_after = obs[env.n_features : 2 * env.n_features].sum()
    assert reward == -small_config.reward.stage_cost(1)
    assert not terminated
    assert info["stage"] == 1
    assert mask_after > mask_before


def test_correct_pass_reward(
    multi_stage_data: MultiStageData, small_config: Config
) -> None:
    env = _make_env(multi_stage_data, small_config)
    pass_index = int(np.where(env._labels == LABEL_PASS)[0][0])
    env.reset(options={"index": pass_index})
    env.step(Action.CONTINUE)
    _, reward, terminated, _, info = env.step(Action.STOP_PASS)
    assert terminated
    expected = (
        small_config.reward.correct_pass + small_config.reward.early_pass_penalty
    )
    assert reward == expected
    assert info["predicted_label"] == LABEL_PASS


def test_false_pass_is_most_penalised(
    multi_stage_data: MultiStageData, small_config: Config
) -> None:
    env = _make_env(multi_stage_data, small_config)
    fail_index = int(np.where(env._labels == LABEL_FAIL)[0][0])
    env.reset(options={"index": fail_index})
    _, reward, terminated, _, _ = env.step(Action.STOP_PASS)
    assert terminated
    assert reward == small_config.reward.false_pass + small_config.reward.metadata_only_pass_penalty


def test_episode_truncates_when_always_continuing(
    multi_stage_data: MultiStageData, small_config: Config
) -> None:
    env = _make_env(multi_stage_data, small_config)
    env.reset(options={"index": 0})
    terminated = truncated = False
    steps = 0
    while not (terminated or truncated):
        _, _, terminated, truncated, _ = env.step(Action.CONTINUE)
        steps += 1
        assert steps <= small_config.env.max_steps + 1
    assert truncated and not terminated
