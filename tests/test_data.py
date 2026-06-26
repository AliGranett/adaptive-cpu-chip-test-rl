"""Tests for data loading, feature engineering and preprocessing."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import Config
from src.data.feature_engineering import engineer_features
from src.data.preprocessing import ProcessedData, preprocess


def test_synthetic_dataset_shape(synthetic_frame: pd.DataFrame, small_config: Config) -> None:
    assert len(synthetic_frame) == small_config.data.n_synthetic_samples
    assert small_config.env.label_column in synthetic_frame.columns
    assert set(synthetic_frame[small_config.env.label_column].unique()) <= {0, 1}


def test_label_is_imbalanced_but_present(synthetic_frame: pd.DataFrame, small_config: Config) -> None:
    fail_rate = synthetic_frame[small_config.env.label_column].mean()
    assert 0.05 < fail_rate < 0.6


def test_engineer_features_adds_columns(synthetic_frame: pd.DataFrame, small_config: Config) -> None:
    engineered = engineer_features(synthetic_frame, small_config)
    assert "feat_measure_mean" in engineered.columns
    assert engineered.shape[1] > synthetic_frame.shape[1]


def test_preprocess_outputs_scaled_split(processed: ProcessedData, small_config: Config) -> None:
    assert isinstance(processed, ProcessedData)
    assert len(processed.train) > len(processed.test)
    # No NaNs should remain after imputation.
    assert not processed.train[processed.feature_columns].isna().any().any()
    # Standardised training features have approximately zero mean.
    means = processed.train[processed.feature_columns].mean().abs()
    assert (means < 0.5).all()


def test_no_label_leakage_in_features(processed: ProcessedData) -> None:
    assert processed.label_column not in processed.feature_columns


def test_make_dataset_writes_files(small_config: Config) -> None:
    from src.data.make_dataset import make_dataset

    cfg = make_dataset(
        n_samples=200,
        n_features=10,
        write_splits=True,
        base_config=small_config,
    )
    assert cfg.paths.raw_dataset.exists()
    assert cfg.paths.train_dataset.exists()
    assert cfg.paths.test_dataset.exists()
    reloaded = pd.read_csv(cfg.paths.raw_dataset)
    assert len(reloaded) == 200
