"""Tests for the full-stage data loader and preprocessing."""

from __future__ import annotations

from pathlib import Path

from src.config import Config
from src.data.full_stage_loader import (
    META_COLS,
    STAGE2_COLS,
    convert_full_stage_dataset,
    load_full_stage_processed,
    preprocess_full_stage,
)
from tests.conftest import _make_raw_full_stage_frame


def test_convert_full_stage_schema(small_config: Config) -> None:
    raw = _make_raw_full_stage_frame(100)
    frame, cols = convert_full_stage_dataset(raw, small_config)
    assert set(META_COLS + STAGE2_COLS).issubset(frame.columns)
    assert "label" in frame.columns
    assert "is_stage2_fail" in frame.columns
    assert cols.feature_columns == META_COLS + STAGE2_COLS


def test_preprocess_writes_splits(
    small_config: Config, tmp_path: Path
) -> None:
    raw = _make_raw_full_stage_frame(200)
    frame, cols = convert_full_stage_dataset(raw, small_config)
    data = preprocess_full_stage(frame, cols, small_config)
    assert len(data.train) + len(data.test) == len(frame)
    assert data.label_column == "label"
    assert len(data.feature_columns) == len(META_COLS) + len(STAGE2_COLS)


def test_load_processed_raises_when_missing(small_config: Config) -> None:
    try:
        load_full_stage_processed(small_config)
    except FileNotFoundError:
        return
    raise AssertionError("Expected FileNotFoundError for missing processed splits")
