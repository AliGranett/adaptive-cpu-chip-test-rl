"""Tests for reward profiles, run paths and the early-pass penalty."""

from __future__ import annotations

import numpy as np

from src.config import (
    REWARD_PROFILES,
    Config,
    config_for_profile,
    get_reward_profile,
)
from src.data.preprocessing import ProcessedData
from src.environment.chip_testing_env import LABEL_FAIL, LABEL_PASS, Action, ChipTestingEnv


def test_reward_profiles_registered() -> None:
    assert "baseline" in REWARD_PROFILES
    assert "safety_reward_v1" in REWARD_PROFILES


def test_safety_profile_values() -> None:
    safety = get_reward_profile("safety_reward_v1")
    assert safety.correct_fail == 100.0
    assert safety.false_pass == -500.0
    assert safety.false_fail == -50.0
    assert safety.correct_pass == 10.0
    assert safety.test_cost == 2.0  # continue_cost of -2 as a magnitude
    assert safety.early_pass_penalty == -20.0


def test_baseline_profile_matches_defaults(small_config: Config) -> None:
    baseline = get_reward_profile("baseline")
    assert baseline.correct_pass == 20.0
    assert baseline.false_pass == -100.0
    assert baseline.early_pass_penalty == 0.0
    # Replacing with the baseline profile must not change behaviour.
    same = config_for_profile("baseline", small_config)
    assert same.reward.correct_pass == small_config.reward.correct_pass


def test_unknown_profile_raises() -> None:
    try:
        get_reward_profile("does_not_exist")
    except KeyError:
        return
    raise AssertionError("Expected KeyError for unknown profile")


def test_run_paths_isolated(small_config: Config) -> None:
    baseline = small_config.paths.run_paths(None)
    named = small_config.paths.run_paths("safety_reward_v1")
    assert baseline.models == small_config.paths.models
    assert "runs/safety_reward_v1" in str(named.models)
    assert named.models != baseline.models


def _safety_env(processed: ProcessedData, small_config: Config) -> ChipTestingEnv:
    cfg = config_for_profile("safety_reward_v1", small_config)
    return ChipTestingEnv(
        processed.train, processed.feature_columns, cfg, reward_config=cfg.reward
    )


def test_early_pass_penalty_applied_when_no_continue(
    processed: ProcessedData, small_config: Config
) -> None:
    env = _safety_env(processed, small_config)
    pass_index = int(np.where(env._labels == LABEL_PASS)[0][0])
    env.reset(options={"index": pass_index})
    # STOP_PASS immediately (no CONTINUE): correct_pass (10) + early penalty (-20).
    _, reward, terminated, _, info = env.step(Action.STOP_PASS)
    assert terminated
    assert info["early_pass_penalty_applied"] is True
    assert reward == 10.0 + (-20.0)


def test_early_pass_penalty_skipped_after_continue(
    processed: ProcessedData, small_config: Config
) -> None:
    env = _safety_env(processed, small_config)
    pass_index = int(np.where(env._labels == LABEL_PASS)[0][0])
    env.reset(options={"index": pass_index})
    env.step(Action.CONTINUE)  # reveal additional information first
    _, reward, terminated, _, info = env.step(Action.STOP_PASS)
    assert terminated
    assert info["early_pass_penalty_applied"] is False
    assert reward == 10.0  # correct_pass only, no early penalty


def test_early_penalty_not_applied_to_stop_fail(
    processed: ProcessedData, small_config: Config
) -> None:
    env = _safety_env(processed, small_config)
    fail_index = int(np.where(env._labels == LABEL_FAIL)[0][0])
    env.reset(options={"index": fail_index})
    _, reward, terminated, _, info = env.step(Action.STOP_FAIL)
    assert terminated
    assert info["early_pass_penalty_applied"] is False
    assert reward == 100.0  # correct_fail, no early penalty for FAIL decisions


def test_baseline_env_has_no_early_penalty(
    processed: ProcessedData, small_config: Config
) -> None:
    env = ChipTestingEnv(processed.train, processed.feature_columns, small_config)
    pass_index = int(np.where(env._labels == LABEL_PASS)[0][0])
    env.reset(options={"index": pass_index})
    _, reward, _, _, info = env.step(Action.STOP_PASS)
    assert info["early_pass_penalty_applied"] is False
    assert reward == small_config.reward.correct_pass
