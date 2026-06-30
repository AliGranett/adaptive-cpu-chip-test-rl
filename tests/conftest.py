"""Shared pytest fixtures for the test-suite."""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from src.config import CONFIG, Config
from src.data.full_stage_loader import (
    SRC_DURATION,
    SRC_FINAL,
    SRC_LOT,
    SRC_POWER2,
    SRC_RADIAL,
    SRC_SPEEDH2,
    SRC_SPEEDL2,
    SRC_STAGE2_RESULT,
    SRC_WAFER,
    SRC_X,
    SRC_Y,
    MultiStageData,
    convert_full_stage_dataset,
    preprocess_full_stage,
)


def _make_raw_full_stage_frame(n: int = 400) -> pd.DataFrame:
    """Build a tiny synthetic ``full_stage_df`` for tests."""
    rng = np.random.default_rng(42)
    rows = []
    for i in range(n):
        s2_fail = i % 5 == 0
        s2 = "fail" if s2_fail else "pass"
        final = pd.NA if s2_fail else ("fail" if i % 7 == 0 else "pass")
        rows.append(
            {
                SRC_LOT: i // 10,
                SRC_WAFER: i % 10,
                SRC_X: rng.normal(),
                SRC_Y: rng.normal(),
                SRC_RADIAL: abs(rng.normal()),
                SRC_STAGE2_RESULT: s2,
                SRC_DURATION: int(rng.integers(100, 500)),
                SRC_POWER2: rng.normal(),
                SRC_SPEEDH2: rng.normal(),
                SRC_SPEEDL2: rng.normal(),
                SRC_FINAL: final,
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture(scope="session")
def small_config(tmp_path_factory: pytest.TempPathFactory) -> Config:
    """A lightweight configuration with a temp filesystem."""
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
        runs=base / "results" / "runs",
        full_stage_dataset=base / "data" / "raw" / "full_stage_df.csv",
        processed_train=base / "data" / "processed" / "full_stage_v1" / "train.csv",
        processed_test=base / "data" / "processed" / "full_stage_v1" / "test.csv",
    )
    config = dataclasses.replace(CONFIG, paths=paths)
    config.paths.ensure()
    return config


@pytest.fixture(scope="session")
def multi_stage_data(small_config: Config) -> MultiStageData:
    """Processed multi-stage train/test splits for the small config."""
    raw = _make_raw_full_stage_frame()
    frame, cols = convert_full_stage_dataset(raw, small_config)
    return preprocess_full_stage(frame, cols, small_config)
