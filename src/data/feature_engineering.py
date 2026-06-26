"""Feature engineering for the chip-testing dataset.

Adds engineered features derived from the raw Stage-2 measurements and wafer
metadata. The transformation is deterministic and stateless so it can be
applied identically to train and test splits.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import CONFIG, Config
from src.utils.helpers import get_logger

logger = get_logger(__name__)


def engineer_features(frame: pd.DataFrame, config: Config = CONFIG) -> pd.DataFrame:
    """Append engineered features to a copy of the input frame.

    Engineered features:

    * ``feat_measure_mean`` - row-wise mean of Stage-2 measurements.
    * ``feat_measure_std``  - row-wise standard deviation of measurements.
    * ``feat_measure_min`` / ``feat_measure_max`` - measurement extremes.
    * ``feat_neg_count``    - count of negative (out-of-spec) measurements.
    * ``feat_edge_flag``    - whether the die sits near the wafer edge.

    Args:
        frame: Raw dataset (measurements may contain NaNs).
        config: Project configuration (for the label column name).

    Returns:
        A new frame with engineered ``feat_*`` columns appended.
    """
    out = frame.copy()
    measure_cols = [c for c in out.columns if c.startswith("stage2_m")]
    if not measure_cols:
        logger.warning("No 'stage2_m*' measurement columns found for engineering")
        return out

    measures = out[measure_cols]
    out["feat_measure_mean"] = measures.mean(axis=1, skipna=True)
    out["feat_measure_std"] = measures.std(axis=1, skipna=True).fillna(0.0)
    out["feat_measure_min"] = measures.min(axis=1, skipna=True)
    out["feat_measure_max"] = measures.max(axis=1, skipna=True)
    # For synthetic data (standardised negatives); for real power/speed values
    # count measurements below the row median instead.
    neg_count = (measures < 0).sum(axis=1)
    if neg_count.max() == 0:
        row_medians = measures.median(axis=1, skipna=True)
        out["feat_below_median_count"] = measures.lt(row_medians, axis=0).sum(axis=1)
    else:
        out["feat_neg_count"] = neg_count

    if "radial_dist" in out.columns:
        threshold = out["radial_dist"].quantile(0.8)
        out["feat_edge_flag"] = (out["radial_dist"] >= threshold).astype(int)

    logger.info("Engineered %d new features", out.shape[1] - frame.shape[1])
    return out
