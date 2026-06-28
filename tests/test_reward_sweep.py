"""Unit tests for the reward-sweep experiment plumbing.

These cover the pure logic (reward-config parsing, aggregation, Pareto
analysis, final-table ranking) without training any models.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import REWARD_PROFILES, reward_config_from_mapping
from experiments.run_reward_sweep import (
    aggregate_by_penalty,
    build_final_table,
    pareto_flags,
)

FULL_STAGE_DEFAULTS = {
    "continue_penalty": -1,
    "stage2_cost": -1,
    "stage3_cost": -4,
    "correct_pass_reward": 10,
    "correct_fail_reward": 100,
    "false_fail_penalty": -50,
    "false_pass_penalty": -500,
    "stage2_fail_caught_reward": 120,
    "stage2_fail_missed_penalty": -600,
    "metadata_only_pass_penalty": -50,
    "early_pass_penalty": -20,
}


def test_reward_mapping_matches_full_stage_v1() -> None:
    """Default YAML reward values reproduce the full_stage_v1 profile exactly."""
    built = reward_config_from_mapping(FULL_STAGE_DEFAULTS)
    assert built == REWARD_PROFILES["full_stage_v1"]


def test_reward_mapping_overrides_false_pass_penalty() -> None:
    """Changing only false_pass_penalty leaves all other rewards intact."""
    mapping = dict(FULL_STAGE_DEFAULTS)
    mapping["false_pass_penalty"] = -1000
    cfg = reward_config_from_mapping(mapping)
    assert cfg.false_pass == -1000.0
    assert cfg.correct_fail == 100.0
    assert cfg.stage_cost(1) == 1.0
    assert cfg.stage_cost(2) == 4.0


def test_pareto_flags_basic() -> None:
    """Pareto frontier respects the recall constraint and dominance rule."""
    frame = pd.DataFrame(
        {
            "FalsePassRate": [0.10, 0.05, 0.05, 0.20],
            "CostReduction": [80.0, 70.0, 60.0, 90.0],
            "Recall": [0.96, 0.97, 0.98, 0.90],  # last fails the constraint
        }
    )
    meets, pareto = pareto_flags(frame, "FalsePassRate", "CostReduction", "Recall", 0.95)
    assert list(meets) == [True, True, True, False]
    # Row 3 (lowest FPR but worse cost than row 1) is dominated by row 1? No:
    # row 0: FPR .10 cost 80 ; row 1: FPR .05 cost 70 ; row 2: FPR .05 cost 60.
    # row 2 is dominated by row 1 (same FPR, higher cost). rows 0 and 1 non-dominated.
    assert list(pareto) == [True, True, False, False]


def test_pareto_no_valid_solution() -> None:
    """When nothing meets the recall constraint, no point is Pareto-optimal."""
    frame = pd.DataFrame(
        {"FalsePassRate": [0.1, 0.2], "CostReduction": [80.0, 90.0], "Recall": [0.5, 0.6]}
    )
    meets, pareto = pareto_flags(frame, "FalsePassRate", "CostReduction", "Recall", 0.95)
    assert not meets.any()
    assert not pareto.any()


def test_aggregate_by_penalty_mean_std() -> None:
    """Aggregation produces per-penalty mean/std across seeds."""
    all_runs = pd.DataFrame(
        {
            "Seed": [42, 123, 42, 123],
            "Penalty": [-100, -100, -500, -500],
            "Recall": [0.90, 0.94, 0.96, 0.98],
            "FalsePassRate": [0.10, 0.06, 0.04, 0.02],
            "CostReduction": [80.0, 82.0, 70.0, 72.0],
            "AverageTests": [1.0, 1.0, 1.2, 1.2],
            "F1": [0.5, 0.5, 0.6, 0.6],
        }
    )
    agg = aggregate_by_penalty(all_runs)
    assert set(agg["Penalty"]) == {-100, -500}
    row = agg[agg["Penalty"] == -100].iloc[0]
    assert np.isclose(row["Recall_mean"], 0.92)
    assert row["Recall_std"] > 0
    assert "NumSeeds" in agg.columns


def test_build_final_table_sorting() -> None:
    """Final table sorts constraint-meeting Pareto points first."""
    by_penalty = pd.DataFrame(
        {
            "Penalty": [-100, -500, -1000],
            "Recall_mean": [0.90, 0.96, 0.97],
            "Recall_std": [0.0, 0.0, 0.0],
            "FalsePassRate_mean": [0.10, 0.04, 0.03],
            "FalsePassRate_std": [0.0, 0.0, 0.0],
            "FalseFailRate_mean": [0.01, 0.02, 0.03],
            "FalseFailRate_std": [0.0, 0.0, 0.0],
            "CostReduction_mean": [80.0, 70.0, 65.0],
            "CostReduction_std": [0.0, 0.0, 0.0],
            "AverageTests_mean": [1.0, 1.2, 1.3],
            "AverageTests_std": [0.0, 0.0, 0.0],
            "Accuracy_mean": [0.9, 0.9, 0.9],
            "F1_mean": [0.5, 0.6, 0.6],
            "Precision_mean": [0.5, 0.6, 0.6],
            "EscapedDefects_mean": [10, 4, 3],
        }
    )
    meets, pareto = pareto_flags(
        by_penalty, "FalsePassRate_mean", "CostReduction_mean", "Recall_mean", 0.95
    )
    by_penalty["Meets_Recall_Constraint"] = meets.values
    by_penalty["Pareto_Optimal"] = pareto.values
    table = build_final_table(by_penalty)
    # The -100 penalty fails the recall constraint -> ranked last.
    assert table.iloc[-1]["Penalty"] == -100
    assert bool(table.iloc[0]["Meets_Recall_Constraint"]) is True
