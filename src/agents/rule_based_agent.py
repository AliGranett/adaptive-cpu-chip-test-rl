"""A deterministic, hand-crafted rule-based agent and the full-test baseline.

The rule-based agent inspects the *revealed* portion of the observation and
either commits to a classification (when confident or out of test budget) or
continues testing. By tuning ``confidence_margin`` it also expresses the
"Always Continue" full-testing baseline (never stop early).
"""

from __future__ import annotations

import numpy as np

from src.config import CONFIG, Config
from src.environment.actions import Action


class RuleBasedAgent:
    """Threshold heuristic over revealed standardised measurements.

    Healthy chips tend to have higher standardised measurements, so the mean of
    the revealed features acts as a health score: high -> PASS, low -> FAIL.
    """

    def __init__(
        self,
        n_features: int,
        config: Config = CONFIG,
        *,
        decision_threshold: float = 0.0,
        confidence_margin: float = 0.75,
        patience_progress: float = 1.0,
    ) -> None:
        """Initialise the rule-based agent.

        Args:
            n_features: Number of raw features (to slice the observation).
            config: Project configuration (unused but kept for symmetry).
            decision_threshold: Health-score boundary between FAIL and PASS.
            confidence_margin: Minimum ``|score - threshold|`` required to stop
                early. Set to ``float('inf')`` for an always-continue policy.
            patience_progress: Reveal progress in ``[0, 1]`` at which a decision
                is forced regardless of confidence.
        """
        self.n_features = n_features
        self.decision_threshold = decision_threshold
        self.confidence_margin = confidence_margin
        self.patience_progress = patience_progress

    def _health_score(self, observation: np.ndarray) -> float:
        """Return the mean of revealed feature values (the health score)."""
        values = observation[: self.n_features]
        mask = observation[self.n_features : 2 * self.n_features] > 0.5
        revealed = values[mask]
        if revealed.size == 0:
            return self.decision_threshold
        return float(np.mean(revealed))

    def act(self, observation: np.ndarray, *, explore: bool = False) -> int:
        """Return CONTINUE, STOP_PASS or STOP_FAIL per the heuristic."""
        progress = float(observation[-1])
        score = self._health_score(observation)
        confident = abs(score - self.decision_threshold) >= self.confidence_margin

        if progress >= self.patience_progress or confident:
            if score >= self.decision_threshold:
                return int(Action.STOP_PASS)
            return int(Action.STOP_FAIL)
        return int(Action.CONTINUE)


def make_always_continue_agent(
    n_features: int, config: Config = CONFIG
) -> RuleBasedAgent:
    """Create the "Always Continue" full-testing baseline.

    This agent never stops early; it reveals every test stage before
    classifying, representing the conventional exhaustive testing flow.

    Args:
        n_features: Number of raw features.
        config: Project configuration.

    Returns:
        A :class:`RuleBasedAgent` configured to never stop early.
    """
    return RuleBasedAgent(
        n_features,
        config,
        confidence_margin=float("inf"),
        patience_progress=1.0,
    )
