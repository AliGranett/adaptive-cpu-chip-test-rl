"""Multi-stage Gymnasium environment for the full chip-testing flow.

This environment models the *full* manufacturing test process, where a chip may
fail at Stage 2 (and never reach Stage 3) or fail at the final / Stage-3 test:

    State 0: metadata only
        actions: RUN_STAGE2 (continue) / STOP_PASS / STOP_FAIL
    State 1: metadata + Stage-2 measurements (and the Stage-2 result)
        actions: RUN_STAGE3 (continue) / STOP_PASS / STOP_FAIL
    State 2: metadata + Stage-2 + Stage-3 measurements
        actions: STOP_PASS / STOP_FAIL

The single :class:`~src.environment.chip_testing_env.Action.CONTINUE` action is
context-dependent: at State 0 it means *RUN_STAGE2*, at State 1 it means
*RUN_STAGE3*. This keeps the action space at ``Discrete(3)`` so the existing
tabular Q-learning and DQN agents work unchanged, while the ``info`` dict and
the evaluation metrics expose the stage semantics.

Stage-2 failure handling (driven by the reward profile):

* Correctly stopping (STOP_FAIL) a Stage-2-failed chip once Stage-2 has been
  run is strongly rewarded (``stage2_fail_detected_reward``).
* Passing (STOP_PASS) a chip known to have failed Stage-2 is very strongly
  penalised (``stage2_fail_missed_penalty``).
* Continuing to Stage 3 on a chip that already failed Stage 2 is heavily
  penalised (it is wasteful: the chip is already a reject).

Masking: not-yet-revealed features (including all Stage-3 features, which carry
no real measurements in the current dataset) are set to ``masked_value`` and a
separate binary reveal-mask channel is included in the observation, so the
agent can always distinguish a *masked* feature from a true zero measurement.
"""

from __future__ import annotations

import enum
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from src.config import CONFIG, Config, RewardConfig
from src.environment.chip_testing_env import LABEL_FAIL, LABEL_PASS, Action
from src.utils.helpers import get_logger

logger = get_logger(__name__)


class Stage(enum.IntEnum):
    """Testing stages (also the observation's reveal level)."""

    METADATA = 0
    STAGE2 = 1
    STAGE3 = 2


# Friendly names for the context-dependent CONTINUE action, by current stage.
_CONTINUE_NAME = {Stage.METADATA: "RUN_STAGE2", Stage.STAGE2: "RUN_STAGE3"}


class MultiStageChipTestingEnv(gym.Env):
    """A 3-state Gymnasium environment for sequential, cost-aware chip testing.

    Observation:
        Vector of length ``2 * n_features + 1``: masked feature values, a binary
        reveal mask, and a scalar stage-progress signal in ``[0, 1]``.

    Action space:
        :class:`gymnasium.spaces.Discrete(3)` - CONTINUE / STOP_PASS / STOP_FAIL,
        where CONTINUE advances to the next stage (RUN_STAGE2 then RUN_STAGE3).
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        data: pd.DataFrame,
        stage_groups: list[list[str]],
        config: Config = CONFIG,
        *,
        reward_config: RewardConfig | None = None,
        label_column: str = "label",
        stage2_fail_column: str = "is_stage2_fail",
        render_mode: str | None = None,
    ) -> None:
        """Initialise the multi-stage environment.

        Args:
            data: Dataset with one row per chip; must contain every column in
                ``stage_groups``, the label column and the Stage-2-fail column.
            stage_groups: Feature columns partitioned per stage, ordered
                ``[metadata, stage2, stage3]``.
            config: Project configuration.
            reward_config: Reward profile; defaults to ``config.reward``.
            label_column: Binary final-label column (0 = PASS, 1 = FAIL).
            stage2_fail_column: Binary column flagging Stage-2 failures.
            render_mode: Optional Gymnasium render mode.
        """
        super().__init__()
        self.config = config
        self.reward_cfg = reward_config if reward_config is not None else config.reward
        self.env_cfg = config.env
        self.render_mode = render_mode
        self.label_column = label_column
        self.stage2_fail_column = stage2_fail_column

        self._stage_groups = [list(group) for group in stage_groups]
        self.feature_columns = [c for group in self._stage_groups for c in group]
        if label_column not in data.columns:
            raise ValueError(f"Label column '{label_column}' missing from data")
        missing = [c for c in self.feature_columns if c not in data.columns]
        if missing:
            raise ValueError(f"Missing feature columns: {missing}")

        self._features = data[self.feature_columns].to_numpy(dtype=np.float32)
        self._labels = data[label_column].to_numpy(dtype=np.int64)
        if stage2_fail_column in data.columns:
            self._stage2_fail = data[stage2_fail_column].to_numpy(dtype=np.int64)
        else:  # Derive from the revealed flag feature if helper column absent.
            self._stage2_fail = np.zeros(len(data), dtype=np.int64)

        self.n_samples = self._features.shape[0]
        self.n_features = self._features.shape[1]
        self.max_stage = len(self._stage_groups) - 1  # 2 (States 0, 1, 2).

        # Map each stage to the feature indices it reveals.
        self._stage_feature_indices: list[np.ndarray] = []
        cursor = 0
        for group in self._stage_groups:
            idx = np.arange(cursor, cursor + len(group), dtype=int)
            self._stage_feature_indices.append(idx)
            cursor += len(group)

        self.action_space = spaces.Discrete(len(Action))
        obs_dim = 2 * self.n_features + 1
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Episode state.
        self._rng = np.random.default_rng(config.seed)
        self._pointer = 0
        self._current_index = 0
        self._stage = Stage.METADATA
        self._step_count = 0
        self._cost_incurred = 0.0

    # ------------------------------------------------------------------ #
    # Gymnasium API
    # ------------------------------------------------------------------ #
    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Start a new episode for a single chip (State 0, metadata only)."""
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

        self._stage = Stage.METADATA
        self._step_count = 0
        self._cost_incurred = 0.0
        return self._build_observation(), self._build_info()

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Apply an action and advance the episode.

        Args:
            action: One of :class:`~src.environment.chip_testing_env.Action`.

        Returns:
            Tuple ``(observation, reward, terminated, truncated, info)``.
        """
        action = Action(int(action))
        self._step_count += 1
        terminated = False
        truncated = False
        predicted: int | None = None
        early_penalty_applied = False
        action_name = action.name

        if action == Action.CONTINUE:
            reward, action_name = self._continue()
            if self._step_count >= self.env_cfg.max_steps:
                truncated = True
        else:
            terminated = True
            predicted = LABEL_PASS if action == Action.STOP_PASS else LABEL_FAIL
            reward, early_penalty_applied = self._stop_reward(action)

        info = self._build_info()
        info["action"] = int(action)
        info["action_name"] = action_name
        info["predicted_label"] = predicted
        info["reward"] = reward
        info["early_pass_penalty_applied"] = early_penalty_applied
        return self._build_observation(), float(reward), terminated, truncated, info

    def render(self) -> None:
        """Render the current episode state to the logger (``human`` mode)."""
        if self.render_mode != "human":
            return
        logger.info(
            "Chip %d | stage %s | stage2_fail=%d | true label %s",
            self._current_index,
            self._stage.name,
            int(self._stage2_fail[self._current_index]),
            self._labels[self._current_index],
        )

    # ------------------------------------------------------------------ #
    # Reward logic
    # ------------------------------------------------------------------ #
    def _continue(self) -> tuple[float, str]:
        """Handle a CONTINUE action (advance to the next stage).

        Returns:
            ``(reward, action_name)`` where ``action_name`` is RUN_STAGE2 /
            RUN_STAGE3 / CONTINUE_INVALID.
        """
        is_stage2_fail = bool(self._stage2_fail[self._current_index])

        if self._stage == Stage.METADATA:
            self._stage = Stage.STAGE2
            cost = self.reward_cfg.stage_cost(1)
            self._cost_incurred += cost
            return -cost, _CONTINUE_NAME[Stage.METADATA]

        if self._stage == Stage.STAGE2:
            cost = self.reward_cfg.stage_cost(2)
            self._cost_incurred += cost
            self._stage = Stage.STAGE3
            if is_stage2_fail:
                # Wasteful: the chip already failed Stage-2. Heavily penalise
                # running Stage-3 on top of paying its cost.
                heavy = abs(self.reward_cfg.resolved_stage2_fail_detected_reward)
                return -cost - heavy, "RUN_STAGE3_ON_STAGE2_FAIL"
            return -cost, _CONTINUE_NAME[Stage.STAGE2]

        # Already at the final stage: CONTINUE is invalid; charge Stage-3 cost.
        cost = self.reward_cfg.stage_cost(2)
        self._cost_incurred += cost
        return -cost, "CONTINUE_INVALID"

    def _stop_reward(self, action: Action) -> tuple[float, bool]:
        """Compute the reward for a terminal STOP_PASS / STOP_FAIL action.

        Returns:
            ``(reward, early_pass_penalty_applied)``.
        """
        true_label = int(self._labels[self._current_index])
        is_stage2_fail = bool(self._stage2_fail[self._current_index])
        stage2_seen = self._stage >= Stage.STAGE2
        early_penalty_applied = False

        if action == Action.STOP_FAIL:
            if is_stage2_fail and stage2_seen:
                return self.reward_cfg.resolved_stage2_fail_detected_reward, False
            if true_label == LABEL_FAIL:
                return self.reward_cfg.correct_fail, False
            return self.reward_cfg.false_fail, False

        # STOP_PASS
        if is_stage2_fail and stage2_seen:
            base = self.reward_cfg.resolved_stage2_fail_missed_penalty
        elif true_label == LABEL_PASS:
            base = self.reward_cfg.correct_pass
        else:
            base = self.reward_cfg.false_pass

        # Stage-dependent early-pass penalties.
        if self._stage == Stage.METADATA:
            base += self.reward_cfg.metadata_only_pass_penalty
            early_penalty_applied = self.reward_cfg.metadata_only_pass_penalty != 0.0
        elif self._stage == Stage.STAGE2:
            base += self.reward_cfg.early_pass_penalty
            early_penalty_applied = self.reward_cfg.early_pass_penalty != 0.0
        return base, early_penalty_applied

    # ------------------------------------------------------------------ #
    # Observation helpers
    # ------------------------------------------------------------------ #
    def _revealed_mask(self) -> np.ndarray:
        """Boolean mask of currently revealed feature indices."""
        mask = np.zeros(self.n_features, dtype=bool)
        for stage in range(int(self._stage) + 1):
            mask[self._stage_feature_indices[stage]] = True
        return mask

    def _build_observation(self) -> np.ndarray:
        """Construct the masked observation vector for the current state."""
        mask = self._revealed_mask()
        values = self._features[self._current_index].copy()
        values[~mask] = self.env_cfg.masked_value
        progress = np.float32(int(self._stage) / self.max_stage) if self.max_stage else 0.0
        return np.concatenate(
            [values, mask.astype(np.float32), np.array([progress], dtype=np.float32)]
        ).astype(np.float32)

    def _build_info(self) -> dict[str, Any]:
        """Build the per-step ``info`` dictionary."""
        # tests_run = number of measurement stages executed (Stage index).
        return {
            "chip_index": int(self._current_index),
            "true_label": int(self._labels[self._current_index]),
            "is_stage2_fail": int(self._stage2_fail[self._current_index]),
            "stage": int(self._stage),
            "stage_name": self._stage.name,
            "tests_run": int(self._stage),
            "test_cost_incurred": float(self._cost_incurred),
            "all_revealed": self._stage >= self.max_stage,
        }
