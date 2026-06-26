"""Rollout and prediction routines that produce :class:`EvaluationResult`s."""

from __future__ import annotations

from collections import Counter

import numpy as np

from src.agents import Agent
from src.config import CONFIG, Config
from src.environment.chip_testing_env import LABEL_FAIL, Action, ChipTestingEnv
from src.evaluation.metrics import EvaluationResult
from src.utils.helpers import get_logger

logger = get_logger(__name__)


def rollout_agent(
    agent: Agent,
    env: ChipTestingEnv,
    *,
    indices: list[int] | None = None,
) -> EvaluationResult:
    """Run an agent over a set of chips and collect per-chip outcomes.

    Each chip is evaluated as one deterministic episode. If an episode is
    truncated without the agent committing to a classification, the prediction
    defaults to FAIL (the conservative industrial choice: do not ship a chip of
    unknown quality).

    Args:
        agent: Any object implementing :class:`~src.agents.Agent`.
        env: The chip-testing environment to roll out in.
        indices: Optional explicit chip indices. Defaults to every chip.

    Returns:
        An :class:`EvaluationResult` with per-chip labels, rewards and costs.
    """
    if indices is None:
        indices = list(range(env.n_samples))

    true_labels: list[int] = []
    predicted_labels: list[int] = []
    rewards: list[float] = []
    test_costs: list[float] = []
    tests_run: list[int] = []
    action_counter: Counter[int] = Counter()

    for index in indices:
        obs, info = env.reset(options={"index": index})
        episode_reward = 0.0
        predicted = LABEL_FAIL  # Conservative default if no decision is made.
        terminated = truncated = False

        while not (terminated or truncated):
            action = agent.act(obs, explore=False)
            action_counter[int(action)] += 1
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            if terminated and info.get("predicted_label") is not None:
                predicted = int(info["predicted_label"])

        true_labels.append(int(info["true_label"]))
        predicted_labels.append(predicted)
        rewards.append(episode_reward)
        test_costs.append(float(info["test_cost_incurred"]))
        tests_run.append(int(info["tests_run"]))

    logger.info("Rolled out agent over %d chips", len(indices))
    return EvaluationResult(
        true_labels=np.array(true_labels),
        predicted_labels=np.array(predicted_labels),
        rewards=np.array(rewards),
        test_costs=np.array(test_costs),
        action_counts={int(a): int(c) for a, c in action_counter.items()},
        tests_run=np.array(tests_run),
    )


def evaluate_supervised(
    predictions: np.ndarray,
    true_labels: np.ndarray,
    config: Config = CONFIG,
) -> EvaluationResult:
    """Wrap supervised-classifier predictions in an :class:`EvaluationResult`.

    Supervised baselines use *all* features, so they are charged the full
    per-chip testing cost (every stage). Rewards are computed with the same
    reward structure used by the environment for a terminal classification.

    Args:
        predictions: Predicted labels (0 = PASS, 1 = FAIL).
        true_labels: Ground-truth labels.
        config: Project configuration (rewards and full-testing cost).

    Returns:
        An :class:`EvaluationResult` consistent with environment rollouts.
    """
    predictions = np.asarray(predictions).astype(int)
    true_labels = np.asarray(true_labels).astype(int)
    reward_cfg = config.reward
    full_cost = config.env.n_stages * reward_cfg.test_cost

    rewards = np.empty(len(predictions), dtype=float)
    for i, (pred, truth) in enumerate(zip(predictions, true_labels)):
        if pred == LABEL_FAIL:
            class_reward = reward_cfg.correct_fail if truth == LABEL_FAIL else reward_cfg.false_fail
        else:  # predicted PASS
            class_reward = reward_cfg.correct_pass if truth == 0 else reward_cfg.false_pass
        # Full testing cost is paid before the classification reward.
        rewards[i] = class_reward - full_cost

    action_counts = {
        int(Action.STOP_PASS): int(np.sum(predictions == 0)),
        int(Action.STOP_FAIL): int(np.sum(predictions == LABEL_FAIL)),
        int(Action.CONTINUE): int(config.env.n_stages * len(predictions)),
    }
    return EvaluationResult(
        true_labels=true_labels,
        predicted_labels=predictions,
        rewards=rewards,
        test_costs=np.full(len(predictions), full_cost),
        action_counts=action_counts,
        tests_run=np.full(len(predictions), config.env.n_stages),
    )
