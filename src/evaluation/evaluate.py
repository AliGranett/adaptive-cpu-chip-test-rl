"""Rollout and prediction routines that produce :class:`EvaluationResult`s."""

from __future__ import annotations

from collections import Counter

import gymnasium as gym
import numpy as np

from src.agents import Agent
from src.config import CONFIG, Config
from src.environment.actions import LABEL_FAIL, LABEL_PASS, Action
from src.evaluation.metrics import EvaluationResult
from src.utils.helpers import get_logger

logger = get_logger(__name__)


def rollout_agent(
    agent: Agent,
    env: gym.Env,
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
    stages_stopped: list[int] = []
    stage2_fail_flags: list[int] = []
    has_stage_info = False
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
        if "stage" in info:
            has_stage_info = True
            stages_stopped.append(int(info["stage"]))
            stage2_fail_flags.append(int(info.get("is_stage2_fail", 0)))

    logger.info("Rolled out agent over %d chips", len(indices))
    return EvaluationResult(
        true_labels=np.array(true_labels),
        predicted_labels=np.array(predicted_labels),
        rewards=np.array(rewards),
        test_costs=np.array(test_costs),
        action_counts={int(a): int(c) for a, c in action_counter.items()},
        tests_run=np.array(tests_run),
        stages_stopped=np.array(stages_stopped) if has_stage_info else None,
        is_stage2_fail=np.array(stage2_fail_flags) if has_stage_info else None,
    )


def evaluate_supervised_multi_stage(
    predictions: np.ndarray,
    true_labels: np.ndarray,
    is_stage2_fail: np.ndarray,
    config: Config = CONFIG,
) -> EvaluationResult:
    """Wrap supervised predictions for the multi-stage environment.

    Supervised baselines use *all* available features (metadata + Stage-2), so
    they are charged the full multi-stage testing cost (Stage-2 + Stage-3) and
    are treated as always reaching Stage-3. Rewards use the same multi-stage
    reward logic as the environment, including the strong Stage-2-failure
    detection reward / miss penalty.

    Args:
        predictions: Predicted labels (0 = PASS, 1 = FAIL).
        true_labels: Ground-truth labels.
        is_stage2_fail: Per-chip flag marking Stage-2 failures.
        config: Project configuration (reward profile).

    Returns:
        An :class:`EvaluationResult` with stage routing fixed to Stage-3.
    """
    predictions = np.asarray(predictions).astype(int)
    true_labels = np.asarray(true_labels).astype(int)
    is_stage2_fail = np.asarray(is_stage2_fail).astype(int)
    reward_cfg = config.reward
    full_cost = reward_cfg.stage_cost(1) + reward_cfg.stage_cost(2)

    rewards = np.empty(len(predictions), dtype=float)
    for i, (pred, truth, s2f) in enumerate(zip(predictions, true_labels, is_stage2_fail)):
        if pred == LABEL_FAIL:
            if s2f:
                class_reward = reward_cfg.resolved_stage2_fail_detected_reward
            else:
                class_reward = (
                    reward_cfg.correct_fail if truth == LABEL_FAIL else reward_cfg.false_fail
                )
        else:  # predicted PASS
            if s2f:
                class_reward = reward_cfg.resolved_stage2_fail_missed_penalty
            elif truth == LABEL_PASS:
                class_reward = reward_cfg.correct_pass
            else:
                class_reward = reward_cfg.false_pass
        rewards[i] = class_reward - full_cost

    action_counts = {
        int(Action.STOP_PASS): int(np.sum(predictions == LABEL_PASS)),
        int(Action.STOP_FAIL): int(np.sum(predictions == LABEL_FAIL)),
        int(Action.CONTINUE): int(2 * len(predictions)),
    }
    # Supervised baselines always run the full pipeline -> Stage-3 (index 2).
    return EvaluationResult(
        true_labels=true_labels,
        predicted_labels=predictions,
        rewards=rewards,
        test_costs=np.full(len(predictions), full_cost),
        action_counts=action_counts,
        tests_run=np.full(len(predictions), 2),
        stages_stopped=np.full(len(predictions), 2),
        is_stage2_fail=is_stage2_fail,
    )
