"""Tests for the central configuration module."""

from __future__ import annotations

from pathlib import Path

from src.config import CONFIG, Config, DEFAULT_DATASET


def test_paths_are_absolute_and_relative_to_root() -> None:
    config = Config()
    assert config.paths.root.is_absolute()
    assert config.paths.full_stage_dataset.parent == config.paths.raw_data
    train, test = config.paths.processed_split_paths(DEFAULT_DATASET)
    assert train.parent == config.paths.processed_data / DEFAULT_DATASET
    assert test.parent == train.parent


def test_reward_structure_defaults() -> None:
    rewards = CONFIG.reward
    assert rewards.false_pass < rewards.false_fail < 0
    assert rewards.correct_pass > 0 and rewards.correct_fail > 0
    assert rewards.stage2_cost is not None
    assert rewards.stage3_cost is not None


def test_ensure_creates_directories(tmp_path: Path) -> None:
    import dataclasses

    paths = dataclasses.replace(
        CONFIG.paths,
        figures=tmp_path / "figs",
        metrics=tmp_path / "metrics",
        models=tmp_path / "models",
        data=tmp_path / "data",
        raw_data=tmp_path / "data" / "raw",
        processed_data=tmp_path / "data" / "processed",
        results=tmp_path / "results",
    )
    paths.ensure()
    assert paths.figures.exists()
    assert paths.processed_data.exists()
