"""A random-policy agent used as a sanity-check baseline."""

from __future__ import annotations

import numpy as np

from src.config import CONFIG, Config
from src.environment.chip_testing_env import Action


class RandomAgent:
    """Selects actions uniformly at random (optionally weighted).

    Serves as the weakest baseline: it ignores the observation entirely.
    """

    def __init__(
        self,
        config: Config = CONFIG,
        *,
        action_probabilities: tuple[float, float, float] | None = None,
    ) -> None:
        """Initialise the random agent.

        Args:
            config: Project configuration (for the random seed).
            action_probabilities: Optional probabilities for
                ``(CONTINUE, STOP_PASS, STOP_FAIL)``. Defaults to uniform.
        """
        self._rng = np.random.default_rng(config.seed)
        self._n_actions = len(Action)
        if action_probabilities is not None:
            probs = np.asarray(action_probabilities, dtype=float)
            probs = probs / probs.sum()
            self._probs = probs
        else:
            self._probs = np.full(self._n_actions, 1.0 / self._n_actions)

    def act(self, observation: np.ndarray, *, explore: bool = False) -> int:
        """Return a random action (the observation is ignored)."""
        return int(self._rng.choice(self._n_actions, p=self._probs))
