"""Tests for reward profiles, run paths and multi-stage penalties."""

from __future__ import annotations

import numpy as np

from src.config import (
    REWARD_PROFILES,
    Config,
    config_for_profile,
    get_reward_profile,
)
from src.data.full_stage_loader import MultiStageData
from src.environment.actions import LABEL_PASS, Action
from src.environment.multi_stage_env import MultiStageChipTestingEnv


def test_reward_profiles_registered() -> None:
    assert "full_stage_v1" in REWARD_PROFILES


def test_full_stage_profile_values() -> None:
    profile = get_reward_profile("full_stage_v1")
    assert profile.correct_fail == 100.0
    assert profile.false_pass == -500.0
    assert profile.stage2_cost == 1.0
    assert profile.stage3_cost == 4.0
    assert profile.metadata_only_pass_penalty == -50.0
    assert profile.early_pass_penalty == -20.0


def test_config_for_profile_replaces_reward(small_config: Config) -> None:
    cfg = config_for_profile("full_stage_v1", small_config)
    assert cfg.reward == REWARD_PROFILES["full_stage_v1"]


def test_unknown_profile_raises() -> None:
    try:
        get_reward_profile("does_not_exist")
    except KeyError:
        return
    raise AssertionError("Expected KeyError for unknown profile")


def test_run_paths_isolated(small_config: Config) -> None:
    baseline = small_config.paths.run_paths(None)
    named = small_config.paths.run_paths("full_stage_v1")
    assert baseline.models == small_config.paths.models
    assert "runs/full_stage_v1" in str(named.models)
    assert named.models != baseline.models


def _env(data: MultiStageData, config: Config) -> MultiStageChipTestingEnv:
    return MultiStageChipTestingEnv(
        data.train,
        data.columns.stage_groups,
        config,
    )


def test_metadata_only_pass_penalty(
    multi_stage_data: MultiStageData, small_config: Config
) -> None:
    env = _env(multi_stage_data, small_config)
    pass_index = int(np.where(env._labels == LABEL_PASS)[0][0])
    env.reset(options={"index": pass_index})
    _, reward, terminated, _, info = env.step(Action.STOP_PASS)
    assert terminated
    assert info["early_pass_penalty_applied"] is True
    assert reward == 10.0 + (-50.0)


def test_early_pass_penalty_after_stage2(
    multi_stage_data: MultiStageData, small_config: Config
) -> None:
    env = _env(multi_stage_data, small_config)
    pass_index = int(np.where(env._labels == LABEL_PASS)[0][0])
    env.reset(options={"index": pass_index})
    env.step(Action.CONTINUE)
    _, reward, terminated, _, info = env.step(Action.STOP_PASS)
    assert terminated
    assert info["early_pass_penalty_applied"] is True
    assert reward == 10.0 + (-20.0)


def test_stage2_fail_detected_reward(
    multi_stage_data: MultiStageData, small_config: Config
) -> None:
    env = _env(multi_stage_data, small_config)
    s2_fail_index = int(np.where(env._stage2_fail == 1)[0][0])
    env.reset(options={"index": s2_fail_index})
    env.step(Action.CONTINUE)
    _, reward, terminated, _, info = env.step(Action.STOP_FAIL)
    assert terminated
    assert info["early_pass_penalty_applied"] is False
    assert reward == 120.0


def test_stop_fail_on_good_chip(
    multi_stage_data: MultiStageData, small_config: Config
) -> None:
    env = _env(multi_stage_data, small_config)
    pass_index = int(np.where(env._labels == LABEL_PASS)[0][0])
    env.reset(options={"index": pass_index})
    _, reward, terminated, _, info = env.step(Action.STOP_FAIL)
    assert terminated
    assert info["early_pass_penalty_applied"] is False
    assert reward == small_config.reward.false_fail
