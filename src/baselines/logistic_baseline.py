"""Logistic-regression baseline classifier (full-information, no test reduction).

This mirrors the supervised approach of the original Data Science project: it
uses *all* Stage-2 features to predict failure, implicitly paying the full
testing cost for every chip.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from src.config import CONFIG, Config


class LogisticBaseline:
    """A logistic-regression PASS/FAIL classifier."""

    def __init__(self, config: Config = CONFIG) -> None:
        """Initialise the baseline.

        Args:
            config: Project configuration (for the random seed).
        """
        self.config = config
        self.model = LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=config.seed
        )

    def fit(self, x: np.ndarray, y: np.ndarray) -> "LogisticBaseline":
        """Fit the classifier.

        Args:
            x: Feature matrix of shape ``(n_samples, n_features)``.
            y: Binary labels (0 = PASS, 1 = FAIL).

        Returns:
            ``self`` for chaining.
        """
        self.model.fit(x, y)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict binary PASS/FAIL labels for ``x``."""
        return self.model.predict(x).astype(int)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Return the probability of the FAIL class for ``x``."""
        return self.model.predict_proba(x)[:, 1]

    def save(self, path: Path | str) -> None:
        """Persist the fitted model via pickle."""
        import pickle

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(self.model, handle)
