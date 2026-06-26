"""Tests for the agent implementations."""

from __future__ import annotations

import numpy as np

from src.agents.q_learning_agent import QLearningAgent
from src.agents.random_agent import RandomAgent
from src.agents.rule_based_agent import RuleBasedAgent, make_always_continue_agent
from src.config import Config
from src.data.preprocessing import ProcessedData
from src.environment.chip_testing_env import Action, ChipTestingEnv


def _env(processed: ProcessedData, config: Config) -> ChipTestingEnv:
    return ChipTestingEnv(processed.train, processed.feature_columns, config)


def test_random_agent_returns_valid_action(small_config: Config) -> None:
    agent = RandomAgent(small_config)
    obs = np.zeros(10, dtype=np.float32)
    for _ in range(20):
        assert agent.act(obs) in {0, 1, 2}


def test_always_continue_agent_keeps_continuing(processed: ProcessedData, small_config: Config) -> None:
    env = _env(processed, small_config)
    agent = make_always_continue_agent(env.n_features, small_config)
    obs, _ = env.reset(options={"index": 0})
    # Before all stages are revealed, it must choose CONTINUE.
    assert agent.act(obs) == int(Action.CONTINUE)


def test_rule_based_decides_at_full_reveal(processed: ProcessedData, small_config: Config) -> None:
    env = _env(processed, small_config)
    agent = RuleBasedAgent(env.n_features, small_config, confidence_margin=float("inf"))
    obs, _ = env.reset(options={"index": 0})
    terminated = truncated = False
    actions = []
    while not (terminated or truncated):
        action = agent.act(obs)
        actions.append(action)
        obs, _, terminated, truncated, _ = env.step(action)
    # Final action should be a stop decision (PASS or FAIL).
    assert actions[-1] in {int(Action.STOP_PASS), int(Action.STOP_FAIL)}


def test_q_learning_update_changes_q_table(processed: ProcessedData, small_config: Config) -> None:
    env = _env(processed, small_config)
    agent = QLearningAgent(n_features=env.n_features, config=small_config)
    obs, _ = env.reset(options={"index": 0})
    next_obs, reward, terminated, truncated, _ = env.step(Action.STOP_FAIL)
    state = agent.discretise(obs)
    before = agent.q_table[state][int(Action.STOP_FAIL)]
    agent.update(obs, int(Action.STOP_FAIL), reward, next_obs, True)
    after = agent.q_table[state][int(Action.STOP_FAIL)]
    assert before != after


def test_q_learning_save_load_roundtrip(processed: ProcessedData, small_config: Config, tmp_path) -> None:
    env = _env(processed, small_config)
    agent = QLearningAgent(n_features=env.n_features, config=small_config)
    obs, _ = env.reset(options={"index": 0})
    agent.update(obs, int(Action.STOP_PASS), 1.0, obs, True)
    path = tmp_path / "q.pkl"
    agent.save(path)
    restored = QLearningAgent(n_features=env.n_features, config=small_config)
    restored.load(path)
    assert restored.n_states == agent.n_states
