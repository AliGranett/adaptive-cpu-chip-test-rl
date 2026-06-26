"""Gymnasium environment modelling adaptive CPU chip testing.

One chip corresponds to one episode. Test information is revealed
*sequentially*: the agent starts seeing only the first stage of measurements
and may pay to reveal additional stages (``CONTINUE``) or stop early and commit
to a classification (``STOP_PASS`` / ``STOP_FAIL``). The goal is to learn a
policy that minimises testing cost while keeping classification quality high.
"""

from __future__ import annotations

import enum
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from src.config import CONFIG, Config, RewardConfig
from src.utils.helpers import get_logger

logger = get_logger(__name__)


class Action(enum.IntEnum):
    """Discrete actions available to the testing agent."""

    CONTINUE = 0
    """Pay ``test_cost`` to reveal the next stage of measurements."""
    STOP_PASS = 1
    """Stop testing and classify the chip as PASS (good)."""
    STOP_FAIL = 2
    """Stop testing and classify the chip as FAIL (defective)."""


# Ground-truth label encoding.
LABEL_PASS = 0
LABEL_FAIL = 1


class ChipTestingEnv(gym.Env):
    """A Gymnasium environment for sequential, cost-aware chip testing.

    Observation:
        A continuous vector of length ``2 * n_features + 1`` consisting of:

        * the (standardised) feature values, with not-yet-revealed features
          set to :attr:`EnvConfig.masked_value`;
        * a binary mask indicating which features have been revealed;
        * a scalar testing-progress signal in ``[0, 1]``.

    Action space:
        :class:`gymnasium.spaces.Discrete` with three actions, see
        :class:`Action`.

    Reward:
        ``-test_cost`` per ``CONTINUE``; classification rewards/penalties as
        configured in :class:`~src.config.RewardConfig` on stopping.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        data: pd.DataFrame,
        feature_columns: list[str],
        config: Config = CONFIG,
        *,
        reward_config: RewardConfig | None = None,
        render_mode: str | None = None,
    ) -> None:
        """Initialise the environment.

        Args:
            data: Dataset with one row per chip; must contain
                ``feature_columns`` and the configured label column.
            feature_columns: Ordered feature columns (reveal order).
            config: Project configuration (rewards, env dynamics, seed).
            reward_config: Optional reward profile overriding ``config.reward``.
                When ``None`` the environment uses ``config.reward`` so the
                default (baseline) behaviour is preserved.
            render_mode: Optional Gymnasium render mode (``"human"``).
        """
        super().__init__()
        label_column = config.env.label_column
        if label_column not in data.columns:
            raise ValueError(f"Label column '{label_column}' missing from data")
        missing = [c for c in feature_columns if c not in data.columns]
        if missing:
            raise ValueError(f"Missing feature columns: {missing}")

        self.config = config
        self.reward_cfg = reward_config if reward_config is not None else config.reward
        self.env_cfg = config.env
        self.render_mode = render_mode

        self._features = data[feature_columns].to_numpy(dtype=np.float32)
        self._labels = data[label_column].to_numpy(dtype=np.int64)
        self.feature_columns = list(feature_columns)
        self.n_samples = self._features.shape[0]
        self.n_features = self._features.shape[1]

        # Partition feature indices into sequential test stages.
        self._stage_slices = np.array_split(
            np.arange(self.n_features), self.env_cfg.n_stages
        )
        self.n_stages = len(self._stage_slices)

        self.action_space = spaces.Discrete(len(Action))
        obs_dim = 2 * self.n_features + 1
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Episode state.
        self._rng = np.random.default_rng(config.seed)
        self._pointer = 0
        self._current_index = 0
        self._revealed_stages = 0
        self._step_count = 0
        self._tests_run = 0
        # Whether the agent has chosen CONTINUE at least once this episode.
        self._has_continued = False

    # ------------------------------------------------------------------ #
    # Gymnasium API
    # ------------------------------------------------------------------ #
    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Start a new episode for a single chip.

        Args:
            seed: Optional seed to reseed the internal RNG.
            options: Optional dict. Supports ``{"index": int}`` to force a
                specific chip (used for deterministic evaluation).

        Returns:
            A tuple ``(observation, info)``.
        """
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        if options and "index" in options:
            self._current_index = int(options["index"]) % self.n_samples
        elif self.env_cfg.shuffle:
            self._current_index = int(self._rng.integers(0, self.n_samples))
        else:
            self._current_index = self._pointer % self.n_samples
            self._pointer += 1

        self._revealed_stages = 1
        self._step_count = 0
        self._tests_run = 1  # First stage is always revealed at reset.
        self._has_continued = False
        return self._build_observation(), self._build_info()

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Apply an action and advance the episode.

        Args:
            action: One of :class:`Action`.

        Returns:
            Tuple ``(observation, reward, terminated, truncated, info)``.
        """
        action = Action(int(action))
        self._step_count += 1
        terminated = False
        truncated = False
        predicted: int | None = None
        early_penalty_applied = False

        if action == Action.CONTINUE:
            reward = -self.reward_cfg.test_cost
            if self._revealed_stages < self.n_stages:
                self._revealed_stages += 1
                self._tests_run += 1
            self._has_continued = True
        else:
            terminated = True
            predicted = LABEL_PASS if action == Action.STOP_PASS else LABEL_FAIL
            reward = self._classification_reward(action)
            # Penalise classifying PASS before any additional testing was done.
            if action == Action.STOP_PASS and not self._has_continued:
                reward += self.reward_cfg.early_pass_penalty
                early_penalty_applied = self.reward_cfg.early_pass_penalty != 0.0

        if not terminated and self._step_count >= self.env_cfg.max_steps:
            truncated = True

        info = self._build_info()
        info["action"] = int(action)
        info["predicted_label"] = predicted
        info["reward"] = reward
        info["early_pass_penalty_applied"] = early_penalty_applied
        return self._build_observation(), float(reward), terminated, truncated, info

    def render(self) -> None:
        """Render the current episode state to the logger (``human`` mode)."""
        if self.render_mode != "human":
            return
        logger.info(
            "Chip %d | stages revealed %d/%d | tests run %d | true label %s",
            self._current_index,
            self._revealed_stages,
            self.n_stages,
            self._tests_run,
            self._labels[self._current_index],
        )

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _classification_reward(self, action: Action) -> float:
        """Compute the reward for a terminal classification action.

        Args:
            action: ``STOP_PASS`` or ``STOP_FAIL``.

        Returns:
            The configured reward/penalty.
        """
        true_label = int(self._labels[self._current_index])
        if action == Action.STOP_PASS:
            if true_label == LABEL_PASS:
                return self.reward_cfg.correct_pass
            return self.reward_cfg.false_pass
        # STOP_FAIL
        if true_label == LABEL_FAIL:
            return self.reward_cfg.correct_fail
        return self.reward_cfg.false_fail

    def _revealed_mask(self) -> np.ndarray:
        """Return a boolean mask of currently revealed feature indices."""
        mask = np.zeros(self.n_features, dtype=bool)
        for stage in range(self._revealed_stages):
            mask[self._stage_slices[stage]] = True
        return mask

    def _build_observation(self) -> np.ndarray:
        """Construct the masked observation vector for the current state."""
        mask = self._revealed_mask()
        values = self._features[self._current_index].copy()
        values[~mask] = self.env_cfg.masked_value
        progress = np.float32(self._revealed_stages / self.n_stages)
        return np.concatenate(
            [values, mask.astype(np.float32), np.array([progress], dtype=np.float32)]
        ).astype(np.float32)

    def _build_info(self) -> dict[str, Any]:
        """Build the per-step ``info`` dictionary."""
        return {
            "chip_index": int(self._current_index),
            "true_label": int(self._labels[self._current_index]),
            "revealed_stages": int(self._revealed_stages),
            "tests_run": int(self._tests_run),
            "test_cost_incurred": float(self._tests_run * self.reward_cfg.test_cost),
            "all_revealed": self._revealed_stages >= self.n_stages,
        }
