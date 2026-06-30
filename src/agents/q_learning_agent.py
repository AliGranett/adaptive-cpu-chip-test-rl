"""Tabular Q-learning agent with observation discretisation.

The continuous observation is mapped to a discrete state key by binning the
first few feature dimensions and appending the (already discrete) reveal-stage
index. Q-values are stored in a dictionary keyed by that state.
"""

from __future__ import annotations

import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.config import CONFIG, Config
from src.environment.actions import Action


class QLearningAgent:
    """An epsilon-greedy tabular Q-learning agent."""

    def __init__(self, n_features: int, config: Config = CONFIG) -> None:
        """Initialise the agent.

        Args:
            n_features: Number of raw features (to slice the observation).
            config: Project configuration with Q-learning hyperparameters.
        """
        cfg = config.qlearning
        self.n_features = n_features
        self.n_actions = len(Action)
        self.learning_rate = cfg.learning_rate
        self.discount_factor = cfg.discount_factor
        self.epsilon = cfg.epsilon_start
        self.epsilon_end = cfg.epsilon_end
        self.epsilon_decay = cfg.epsilon_decay
        self.n_bins = cfg.n_bins
        self.n_disc = min(cfg.n_discretised_features, n_features)
        self.n_stages = config.env.max_stage_index

        # Bin edges for standardised features (roughly within +/- 3 sigma).
        self._bin_edges = np.linspace(-3.0, 3.0, self.n_bins - 1)
        self._rng = np.random.default_rng(config.seed)
        self.q_table: dict[tuple[int, ...], np.ndarray] = defaultdict(
            lambda: np.zeros(self.n_actions, dtype=np.float64)
        )

    def discretise(self, observation: np.ndarray) -> tuple[int, ...]:
        """Map a continuous observation to a discrete state key.

        Args:
            observation: The environment observation vector.

        Returns:
            A hashable tuple of integer bin indices plus the reveal stage.
        """
        values = observation[: self.n_disc]
        bins = tuple(int(np.digitize(v, self._bin_edges)) for v in values)
        stage = int(round(float(observation[-1]) * self.n_stages))
        return bins + (stage,)

    def act(self, observation: np.ndarray, *, explore: bool = False) -> int:
        """Choose an action via an epsilon-greedy policy.

        Args:
            observation: The environment observation vector.
            explore: If ``True``, explore with probability ``epsilon``.

        Returns:
            The selected action index.
        """
        if explore and self._rng.random() < self.epsilon:
            return int(self._rng.integers(0, self.n_actions))
        state = self.discretise(observation)
        q_values = self.q_table[state]
        return int(np.argmax(q_values))

    def update(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        done: bool,
    ) -> None:
        """Apply the Q-learning temporal-difference update.

        Args:
            observation: State before the action.
            action: Action taken.
            reward: Reward received.
            next_observation: State after the action.
            done: Whether the episode terminated/truncated.
        """
        state = self.discretise(observation)
        next_state = self.discretise(next_observation)
        best_next = 0.0 if done else float(np.max(self.q_table[next_state]))
        target = reward + self.discount_factor * best_next
        td_error = target - self.q_table[state][action]
        self.q_table[state][action] += self.learning_rate * td_error

    def decay_epsilon(self) -> None:
        """Decay the exploration rate towards its configured minimum."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    @property
    def n_states(self) -> int:
        """Number of distinct states currently stored in the Q-table."""
        return len(self.q_table)

    def save(self, path: Path | str) -> None:
        """Persist the Q-table and hyperparameters to a pickle file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "q_table": {k: v for k, v in self.q_table.items()},
            "n_features": self.n_features,
            "n_bins": self.n_bins,
            "n_disc": self.n_disc,
            "n_stages": self.n_stages,
            "epsilon": self.epsilon,
        }
        with path.open("wb") as handle:
            pickle.dump(payload, handle)

    def load(self, path: Path | str) -> None:
        """Load a Q-table and hyperparameters from a pickle file."""
        with Path(path).open("rb") as handle:
            payload = pickle.load(handle)
        self.q_table = defaultdict(
            lambda: np.zeros(self.n_actions, dtype=np.float64), payload["q_table"]
        )
        self.n_features = payload["n_features"]
        self.n_bins = payload["n_bins"]
        self.n_disc = payload["n_disc"]
        self.n_stages = payload["n_stages"]
        self.epsilon = payload.get("epsilon", self.epsilon_end)
