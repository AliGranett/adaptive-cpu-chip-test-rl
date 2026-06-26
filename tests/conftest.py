"""Shared pytest fixtures for the test-suite."""

from __future__ import annotations

import dataclasses

import pandas as pd
import pytest

from src.config import CONFIG, Config
from src.data.loader import generate_synthetic_dataset
from src.data.preprocessing import ProcessedData, preprocess


@pytest.fixture(scope="session")
def small_config(tmp_path_factory: pytest.TempPathFactory) -> Config:
    """A lightweight configuration with a temp filesystem and tiny dataset."""
    base = tmp_path_factory.mktemp("project")
    paths = dataclasses.replace(
        CONFIG.paths,
        root=base,
        data=base / "data",
        raw_data=base / "data" / "raw",
        processed_data=base / "data" / "processed",
        results=base / "results",
        figures=base / "results" / "figures",
        metrics=base / "results" / "metrics",
        models=base / "results" / "models",
        raw_dataset=base / "data" / "raw" / "chip_tests.csv",
        train_dataset=base / "data" / "processed" / "train.csv",
        test_dataset=base / "data" / "processed" / "test.csv",
    )
    data_cfg = dataclasses.replace(
        CONFIG.data, n_synthetic_samples=400, n_synthetic_features=12
    )
    config = dataclasses.replace(CONFIG, paths=paths, data=data_cfg)
    config.paths.ensure()
    return config


@pytest.fixture(scope="session")
def synthetic_frame(small_config: Config) -> pd.DataFrame:
    """A small synthetic raw dataset."""
    return generate_synthetic_dataset(small_config)


@pytest.fixture(scope="session")
def processed(small_config: Config) -> ProcessedData:
    """Processed train/test splits for the small config."""
    return preprocess(config=small_config)
