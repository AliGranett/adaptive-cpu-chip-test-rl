"""Tests for the central configuration module."""

from __future__ import annotations

from pathlib import Path

from src.config import CONFIG, Config


def test_paths_are_absolute_and_relative_to_root() -> None:
    config = Config()
    assert config.paths.root.is_absolute()
    assert config.paths.raw_dataset.parent == config.paths.raw_data
    assert config.paths.train_dataset.parent == config.paths.processed_data


def test_reward_structure_defaults() -> None:
    rewards = CONFIG.reward
    # False pass (shipping a defect) must be the most severe penalty.
    assert rewards.false_pass < rewards.false_fail < 0
    assert rewards.correct_pass > 0 and rewards.correct_fail > 0


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
