"""Tests for real dataset loading and conversion."""

from __future__ import annotations

import pandas as pd

from src.config import CONFIG
from src.data.real_data_loader import convert_real_dataset, load_real_dataset


def test_convert_real_dataset_schema() -> None:
    raw = pd.read_csv(CONFIG.paths.base_dataset, nrows=100)
    frame = convert_real_dataset(raw)
    assert "label" in frame.columns
    assert frame["label"].isin([0, 1]).all()
    assert "stage2_m00" in frame.columns
    assert "stage2_m06" in frame.columns
    assert "wafer_id" in frame.columns
    assert len(frame) <= 100


def test_load_real_dataset_from_disk() -> None:
    frame = load_real_dataset()
    assert len(frame) > 100_000
    fail_rate = frame["label"].mean()
    assert 0.05 < fail_rate < 0.35
