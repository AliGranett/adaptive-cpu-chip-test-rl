"""XGBoost baseline classifier (full-information, no test reduction).

Mirrors the gradient-boosting model from the original Data Science project,
using all Stage-2 features and thus the full per-chip testing cost.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from xgboost import XGBClassifier

from src.config import CONFIG, Config


class XGBoostBaseline:
    """An XGBoost PASS/FAIL classifier."""

    def __init__(self, config: Config = CONFIG) -> None:
        """Initialise the baseline.

        Args:
            config: Project configuration (for the random seed).
        """
        self.config = config
        self.model = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=config.seed,
            n_jobs=-1,
        )

    def fit(self, x: np.ndarray, y: np.ndarray) -> "XGBoostBaseline":
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
        """Persist the fitted model in XGBoost's native JSON format."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(path))
