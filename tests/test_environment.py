"""Tests for the ChipTestingEnv Gymnasium environment."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import Config
from src.data.preprocessing import ProcessedData
from src.environment.chip_testing_env import (
    LABEL_FAIL,
    LABEL_PASS,
    Action,
    ChipTestingEnv,
)


def _make_env(processed: ProcessedData, config: Config) -> ChipTestingEnv:
    return ChipTestingEnv(processed.train, processed.feature_columns, config)


def test_observation_and_action_spaces(processed: ProcessedData, small_config: Config) -> None:
    env = _make_env(processed, small_config)
    obs, info = env.reset()
    assert env.observation_space.shape == (2 * env.n_features + 1,)
    assert env.action_space.n == 3
    assert obs.shape == env.observation_space.shape
    assert obs.dtype == np.float32
    assert "true_label" in info


def test_continue_reveals_more_features(processed: ProcessedData, small_config: Config) -> None:
    env = _make_env(processed, small_config)
    obs, _ = env.reset(options={"index": 0})
    mask_before = obs[env.n_features : 2 * env.n_features].sum()
    obs, reward, terminated, truncated, info = env.step(Action.CONTINUE)
    mask_after = obs[env.n_features : 2 * env.n_features].sum()
    assert reward == -small_config.reward.test_cost
    assert not terminated
    assert mask_after >= mask_before


def test_correct_pass_reward(processed: ProcessedData, small_config: Config) -> None:
    env = _make_env(processed, small_config)
    # Find a chip whose true label is PASS.
    pass_index = int(np.where(env._labels == LABEL_PASS)[0][0])
    env.reset(options={"index": pass_index})
    _, reward, terminated, _, info = env.step(Action.STOP_PASS)
    assert terminated
    assert reward == small_config.reward.correct_pass
    assert info["predicted_label"] == LABEL_PASS


def test_false_pass_is_most_penalised(processed: ProcessedData, small_config: Config) -> None:
    env = _make_env(processed, small_config)
    fail_index = int(np.where(env._labels == LABEL_FAIL)[0][0])
    env.reset(options={"index": fail_index})
    _, reward, terminated, _, _ = env.step(Action.STOP_PASS)
    assert terminated
    assert reward == small_config.reward.false_pass


def test_episode_truncates_when_always_continuing(processed: ProcessedData, small_config: Config) -> None:
    env = _make_env(processed, small_config)
    env.reset(options={"index": 0})
    terminated = truncated = False
    steps = 0
    while not (terminated or truncated):
        _, _, terminated, truncated, _ = env.step(Action.CONTINUE)
        steps += 1
        assert steps <= small_config.env.max_steps + 1
    assert truncated and not terminated
