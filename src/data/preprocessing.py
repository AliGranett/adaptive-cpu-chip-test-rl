"""Preprocessing: imputation, scaling, feature ordering and train/test split.

The output of this module is a pair of processed CSV files (train/test) with a
deterministic column ordering. Column ordering matters because the environment
reveals features stage-by-stage in column order, so raw Stage-2 measurements
are placed first and aggregate/engineered features last (to avoid leaking
information from yet-to-be-revealed measurements).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import CONFIG, Config
from src.data.feature_engineering import engineer_features
from src.data.loader import load_raw_dataset
from src.utils.helpers import get_logger

logger = get_logger(__name__)


@dataclass
class ProcessedData:
    """Container holding the processed train/test splits and column metadata."""

    train: pd.DataFrame
    test: pd.DataFrame
    feature_columns: list[str]
    label_column: str


def _order_feature_columns(frame: pd.DataFrame, label_column: str) -> list[str]:
    """Return feature columns ordered for sequential reveal.

    Ordering: raw Stage-2 measurements first (revealed early), then wafer
    metadata, then aggregate engineered features (revealed last).

    Args:
        frame: The dataset.
        label_column: Name of the label column to exclude.

    Returns:
        Ordered list of feature column names.
    """
    measure = sorted(c for c in frame.columns if c.startswith("stage2_m"))
    engineered = sorted(c for c in frame.columns if c.startswith("feat_"))
    metadata = [
        c
        for c in frame.columns
        if c not in measure and c not in engineered and c != label_column
    ]
    return measure + metadata + engineered


def preprocess(
    frame: pd.DataFrame | None = None, config: Config = CONFIG
) -> ProcessedData:
    """Run the full preprocessing pipeline and return processed splits.

    Steps: load (if needed) -> engineer features -> train/test split ->
    fit imputer+scaler on train -> transform both splits.

    Args:
        frame: Optional raw frame. If ``None`` the raw dataset is loaded
            (and synthetically generated if missing).
        config: Project configuration.

    Returns:
        A :class:`ProcessedData` with scaled train/test frames.
    """
    if frame is None:
        frame = load_raw_dataset(config)

    label_column = config.env.label_column
    frame = engineer_features(frame, config)
    feature_columns = _order_feature_columns(frame, label_column)

    x = frame[feature_columns]
    y = frame[label_column].astype(int)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=config.data.test_size,
        random_state=config.seed,
        stratify=y,
    )

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    x_train_arr = scaler.fit_transform(imputer.fit_transform(x_train))
    x_test_arr = scaler.transform(imputer.transform(x_test))

    train = pd.DataFrame(x_train_arr, columns=feature_columns)
    train[label_column] = y_train.to_numpy()
    test = pd.DataFrame(x_test_arr, columns=feature_columns)
    test[label_column] = y_test.to_numpy()

    logger.info(
        "Preprocessed data: %d train / %d test rows, %d features",
        len(train),
        len(test),
        len(feature_columns),
    )
    return ProcessedData(train, test, feature_columns, label_column)


def preprocess_and_save(config: Config = CONFIG) -> ProcessedData:
    """Preprocess and persist the train/test splits to ``data/processed``.

    Args:
        config: Project configuration with output paths.

    Returns:
        The :class:`ProcessedData` that was written to disk.
    """
    config.paths.ensure()
    processed = preprocess(config=config)
    processed.train.to_csv(config.paths.train_dataset, index=False)
    processed.test.to_csv(config.paths.test_dataset, index=False)
    logger.info(
        "Saved processed splits to %s and %s",
        config.paths.train_dataset,
        config.paths.test_dataset,
    )
    return processed


def load_processed_data(
    config: Config = CONFIG, *, create_if_missing: bool = True
) -> ProcessedData:
    """Load processed train/test splits, creating them if necessary.

    Args:
        config: Project configuration with processed paths.
        create_if_missing: Whether to run preprocessing if files are absent.

    Returns:
        The loaded :class:`ProcessedData`.

    Raises:
        FileNotFoundError: If files are missing and ``create_if_missing`` is
            ``False``.
    """
    train_path = config.paths.train_dataset
    test_path = config.paths.test_dataset
    if not (train_path.exists() and test_path.exists()):
        if not create_if_missing:
            raise FileNotFoundError("Processed splits not found")
        return preprocess_and_save(config)

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    label_column = config.env.label_column
    feature_columns = [c for c in train.columns if c != label_column]
    logger.info("Loaded processed splits from %s", config.paths.processed_data)
    return ProcessedData(train, test, feature_columns, label_column)
