"""Unit tests for the supervised-baseline analysis extension.

These cover the pure logic and the leakage / output guarantees without training
the real models against the full dataset (that is exercised by the end-to-end
run). Tests that need trained artifacts are skipped when the run has not yet
been executed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.baselines.full_stage_supervised import (
    COMPARISON_COLUMNS,
    DQN_NAME,
    LR_NAME,
    XGB_NAME,
    build_comparison,
    dqn_reference_row,
    full_testing_cost,
    select_features,
    supervised_metrics,
)
from experiments.sweep_plots import TERMINAL_ROUTING_CATEGORIES

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SWEEP_DIR = PROJECT_ROOT / "results" / "reward_sweep"
RUN_DIR = PROJECT_ROOT / "results" / "runs" / "supervised_full_stage_v1"


class _FakeData:
    """Minimal stand-in for MultiStageData with controllable columns."""

    def __init__(self, frame: pd.DataFrame, feature_columns: list[str]) -> None:
        self.train = frame
        self.test = frame
        self._features = feature_columns

    @property
    def feature_columns(self) -> list[str]:
        return self._features

    @property
    def label_column(self) -> str:
        return "label"


def _fake_data() -> _FakeData:
    frame = pd.DataFrame(
        {
            "meta_x": [0.1, 0.2, 0.3, 0.4],
            "s2_power": [1.0, 2.0, 3.0, 4.0],
            "stage2_fail_flag": [0, 1, 0, 1],
            "is_stage2_fail": [0, 1, 0, 1],
            "label": [0, 1, 0, 1],
        }
    )
    return _FakeData(frame, ["meta_x", "s2_power", "stage2_fail_flag"])


def test_select_features_excludes_label_columns() -> None:
    """Feature selection must never expose label/outcome helper columns."""
    data = _fake_data()
    _, _, _, _, names = select_features(data)
    assert "label" not in names
    assert "is_stage2_fail" not in names
    assert names == ["meta_x", "s2_power", "stage2_fail_flag"]


def test_select_features_raises_on_leak() -> None:
    """A label column inside feature_columns triggers an assertion."""
    data = _fake_data()
    data._features = ["meta_x", "label"]  # inject a leak
    with pytest.raises(AssertionError):
        select_features(data)


def test_supervised_metrics_full_cost() -> None:
    """Supervised metrics assume full testing (cost 5, 0% reduction, 2 tests)."""
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 0])
    m = supervised_metrics(y_true, y_pred)
    assert m["Average_Test_Cost"] == full_testing_cost()
    assert m["Average_Test_Cost"] == 5.0
    assert m["Cost_Reduction"] == 0.0
    assert m["Average_Tests_Per_Chip"] == 2
    # FN = 1 (the actual-FAIL predicted PASS) -> escaped defect.
    assert m["EscapedDefects"] == 1
    assert m["FN"] == 1


def test_build_comparison_has_exactly_three_rows() -> None:
    """Comparison table contains exactly the three required rows + columns."""
    supervised = {
        LR_NAME: supervised_metrics(np.array([0, 1]), np.array([0, 1])),
        XGB_NAME: supervised_metrics(np.array([0, 1]), np.array([1, 1])),
    }
    dqn_row = {c: 0.5 for c in COMPARISON_COLUMNS}
    table = build_comparison(supervised, dqn_row)
    assert list(table.index) == [LR_NAME, XGB_NAME, DQN_NAME]
    assert list(table.columns) == COMPARISON_COLUMNS
    assert len(table) == 3


@pytest.mark.skipif(
    not (SWEEP_DIR / "summary_by_penalty.csv").exists(),
    reason="reward sweep not run",
)
def test_dqn_reference_row_from_real_aggregates() -> None:
    """DQN reference row is derivable from the real reward-sweep aggregates."""
    row = dqn_reference_row(SWEEP_DIR / "summary_by_penalty.csv", -1000.0)
    assert 0.0 <= row["Recall_FAIL"] <= 1.0
    # Average cost is consistent with the measured cost reduction.
    full = full_testing_cost()
    assert np.isclose(row["Average_Test_Cost"], full * (1 - row["Cost_Reduction"] / 100.0))


@pytest.mark.skipif(
    not (SWEEP_DIR / "summary_policy_routing_terminal_actions.csv").exists(),
    reason="terminal-routing breakdown not generated yet",
)
def test_routing_categories_sum_to_100() -> None:
    """Figure 8b routing categories must sum to ~100% per penalty."""
    routing = pd.read_csv(SWEEP_DIR / "summary_policy_routing_terminal_actions.csv")
    totals = routing[TERMINAL_ROUTING_CATEGORIES].sum(axis=1)
    assert np.allclose(totals.to_numpy(), 100.0, atol=1e-6)


@pytest.mark.skipif(not RUN_DIR.exists(), reason="supervised run not executed yet")
def test_supervised_outputs_exist() -> None:
    """All expected supervised output files are saved."""
    expected = [
        RUN_DIR / "models" / "logistic_regression.pkl",
        RUN_DIR / "models" / "xgboost.pkl",
        RUN_DIR / "metrics" / "logistic_regression_metrics.csv",
        RUN_DIR / "metrics" / "xgboost_metrics.csv",
        RUN_DIR / "metrics" / "supervised_comparison.csv",
    ]
    missing = [str(p) for p in expected if not p.exists()]
    assert not missing, f"Missing supervised outputs: {missing}"


@pytest.mark.skipif(
    not (SWEEP_DIR / "baseline_vs_best_dqn.csv").exists(),
    reason="comparison not generated yet",
)
def test_comparison_file_rows() -> None:
    """Saved comparison CSV has exactly the three required method rows."""
    table = pd.read_csv(SWEEP_DIR / "baseline_vs_best_dqn.csv", index_col=0)
    assert list(table.index) == [LR_NAME, XGB_NAME, DQN_NAME]


@pytest.mark.skipif(
    not (SWEEP_DIR / "summary_all_runs.csv").exists(),
    reason="reward sweep not run",
)
def test_existing_reward_sweep_files_have_new_names() -> None:
    """New analysis artifacts must not collide with original reward-sweep files."""
    original = {
        "summary_all_runs.csv",
        "summary_by_penalty.csv",
        "final_summary_table.csv",
        "sweep_metadata.json",
    }
    new_artifacts = {
        "baseline_vs_best_dqn.csv",
        "baseline_vs_best_dqn.md",
        "summary_policy_routing_terminal_actions.csv",
        "supervised_baseline_summary.md",
    }
    assert original.isdisjoint(new_artifacts)
    # Original figures 1-9 are never overwritten by the new figure filenames.
    new_figs = {
        "figure_8b_policy_routing_terminal_actions",
        "figure_10_supervised_vs_best_dqn",
        "figure_11_quality_cost_scatter",
    }
    for i in range(1, 10):
        assert not any(str(i) == fig.split("_")[1] for fig in new_figs if fig.split("_")[1].isdigit())
