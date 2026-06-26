"""Agents: baselines (random, rule-based) and learners (Q-learning, DQN)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Agent(Protocol):
    """Common interface implemented by every agent in this project."""

    def act(self, observation: np.ndarray, *, explore: bool = False) -> int:
        """Return an action in ``{0, 1, 2}`` for the given observation.

        Args:
            observation: The environment observation vector.
            explore: Whether to use exploratory (stochastic) behaviour.

        Returns:
            The chosen action index.
        """
        ...


__all__ = ["Agent"]
