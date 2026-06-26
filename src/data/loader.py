"""Dataset loading and synthetic-data generation.

The project is modelled on a Stage-2 CPU manufacturing test dataset. Because
the original proprietary data cannot be shipped, this module transparently
generates a realistic synthetic dataset when no raw CSV is found, so the full
pipeline is runnable out of the box.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import CONFIG, Config
from src.utils.helpers import get_logger

logger = get_logger(__name__)


def generate_synthetic_dataset(config: Config = CONFIG) -> pd.DataFrame:
    """Generate a synthetic CPU chip-testing dataset.

    The dataset emulates a Stage-2 manufacturing flow: several continuous test
    measurements, wafer/location metadata, and a binary failure label. Later
    measurements are deliberately more informative than earlier ones so that an
    adaptive policy can benefit from continuing to test only uncertain chips.

    Args:
        config: Project configuration controlling sample/feature counts.

    Returns:
        A :class:`pandas.DataFrame` with ``stage2_*`` measurement columns,
        wafer metadata columns and a ``label`` column (0 = PASS, 1 = FAIL).
    """
    rng = np.random.default_rng(config.seed)
    n = config.data.n_synthetic_samples
    n_features = config.data.n_synthetic_features

    logger.info("Generating synthetic dataset: %d samples, %d features", n, n_features)

    # Latent "true health" of each chip; defective chips have lower health.
    labels = (rng.random(n) < config.data.synthetic_fail_rate).astype(int)
    health = np.where(labels == 1, rng.normal(-1.2, 1.0, n), rng.normal(1.0, 1.0, n))

    # Each measurement reflects health with increasing signal-to-noise so that
    # later test stages are more discriminative.
    features = np.empty((n, n_features), dtype=np.float64)
    for j in range(n_features):
        signal_strength = 0.2 + 1.3 * (j / max(n_features - 1, 1))
        noise = rng.normal(0.0, 1.0, n)
        features[:, j] = signal_strength * health + noise

    columns = {f"stage2_m{j:02d}": features[:, j] for j in range(n_features)}
    frame = pd.DataFrame(columns)

    # Wafer / location metadata.
    frame["wafer_id"] = rng.integers(0, 25, size=n)
    frame["die_x"] = rng.integers(0, 40, size=n)
    frame["die_y"] = rng.integers(0, 40, size=n)
    # Radial distance from wafer centre correlates weakly with edge defects.
    frame["radial_dist"] = np.sqrt(
        (frame["die_x"] - 20) ** 2 + (frame["die_y"] - 20) ** 2
    )

    frame[config.env.label_column] = labels

    # Inject a small amount of missingness into the measurement columns only.
    if config.data.synthetic_missing_rate > 0:
        measure_cols = [c for c in frame.columns if c.startswith("stage2_m")]
        mask = rng.random((n, len(measure_cols))) < config.data.synthetic_missing_rate
        block = frame[measure_cols].to_numpy(copy=True)
        block[mask] = np.nan
        frame[measure_cols] = block

    return frame


def load_raw_dataset(
    config: Config = CONFIG, *, generate_if_missing: bool = True
) -> pd.DataFrame:
    """Load the raw dataset from disk, generating synthetic data if absent.

    Args:
        config: Project configuration with the raw dataset path.
        generate_if_missing: If ``True`` and the raw CSV does not exist, a
            synthetic dataset is generated, written to disk and returned.

    Returns:
        The raw dataset as a :class:`pandas.DataFrame`.

    Raises:
        FileNotFoundError: If the file is missing and ``generate_if_missing``
            is ``False``.
    """
    path: Path = config.paths.raw_dataset
    if path.exists():
        logger.info("Loading raw dataset from %s", path)
        return pd.read_csv(path)

    if not generate_if_missing:
        raise FileNotFoundError(f"Raw dataset not found at {path}")

    config.paths.ensure()
    frame = generate_synthetic_dataset(config)
    frame.to_csv(path, index=False)
    logger.info("Synthetic raw dataset written to %s", path)
    return frame


def get_feature_columns(frame: pd.DataFrame, config: Config = CONFIG) -> list[str]:
    """Return the ordered list of feature columns (everything but the label).

    Args:
        frame: The dataset.
        config: Project configuration (for the label column name).

    Returns:
        Ordered list of feature column names.
    """
    return [c for c in frame.columns if c != config.env.label_column]
