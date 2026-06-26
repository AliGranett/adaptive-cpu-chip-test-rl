"""Deep Q-Network agent built on top of Stable-Baselines3."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback

import gymnasium as gym

from src.config import CONFIG, Config
from src.utils.helpers import get_logger

logger = get_logger(__name__)


class DQNAgent:
    """A thin wrapper around :class:`stable_baselines3.DQN`.

    Exposes a uniform :meth:`act` interface consistent with the other agents,
    plus convenience training/persistence helpers.
    """

    def __init__(self, config: Config = CONFIG) -> None:
        """Initialise the agent (the underlying model is built lazily).

        Args:
            config: Project configuration with DQN hyperparameters.
        """
        self.config = config
        self.model: BaseAlgorithm | None = None

    def build(self, env: gym.Env) -> DQN:
        """Construct the underlying DQN model for the given environment.

        Args:
            env: A Gymnasium environment (or vectorised wrapper).

        Returns:
            The constructed :class:`stable_baselines3.DQN` model.
        """
        cfg = self.config.dqn
        self.model = DQN(
            policy="MlpPolicy",
            env=env,
            learning_rate=cfg.learning_rate,
            buffer_size=cfg.buffer_size,
            learning_starts=cfg.learning_starts,
            batch_size=cfg.batch_size,
            gamma=cfg.gamma,
            train_freq=cfg.train_freq,
            target_update_interval=cfg.target_update_interval,
            exploration_fraction=cfg.exploration_fraction,
            exploration_final_eps=cfg.exploration_final_eps,
            policy_kwargs={"net_arch": list(cfg.net_arch)},
            seed=self.config.seed,
            verbose=0,
        )
        return self.model

    def train(
        self,
        env: gym.Env,
        *,
        total_timesteps: int | None = None,
        callback: BaseCallback | None = None,
    ) -> None:
        """Train the agent for a number of timesteps.

        Args:
            env: The training environment.
            total_timesteps: Override for the configured timestep budget.
            callback: Optional Stable-Baselines3 callback (e.g. logging).
        """
        if self.model is None:
            self.build(env)
        assert self.model is not None
        steps = total_timesteps or self.config.dqn.total_timesteps
        logger.info("Training DQN for %d timesteps", steps)
        self.model.learn(total_timesteps=steps, callback=callback, progress_bar=False)

    def act(self, observation: np.ndarray, *, explore: bool = False) -> int:
        """Return the greedy (or exploratory) action for an observation.

        Args:
            observation: The environment observation vector.
            explore: If ``True``, sample stochastically rather than greedily.

        Returns:
            The selected action index.

        Raises:
            RuntimeError: If called before the model is built/trained/loaded.
        """
        if self.model is None:
            raise RuntimeError("DQN model is not initialised; call build/train/load")
        action, _ = self.model.predict(observation, deterministic=not explore)
        return int(action)

    def save(self, path: Path | str) -> None:
        """Persist the trained model to disk."""
        if self.model is None:
            raise RuntimeError("Nothing to save; model is not initialised")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(path))

    def load(self, path: Path | str, env: gym.Env | None = None) -> None:
        """Load a trained model from disk.

        Args:
            path: Path to the saved model (``.zip``).
            env: Optional environment to bind to the loaded model.
        """
        self.model = DQN.load(str(path), env=env)
