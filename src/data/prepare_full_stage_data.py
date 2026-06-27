"""Prepare the expanded multi-stage dataset for the ``full_stage_v1`` run.

Loads ``data/raw/full_stage_df.csv``, derives labels (Stage-2 failures count
as FAIL), splits/scales the features and writes processed train/test CSVs under
``data/processed/full_stage_v1/``. It also prints and saves a data summary to
``results/runs/full_stage_v1/data_summary.md``.

Run as a module::

    python -m src.data.prepare_full_stage_data
"""

from __future__ import annotations

import argparse
import dataclasses
import json

import pandas as pd

from src.config import CONFIG, Config
from src.data.full_stage_loader import (
    compute_data_stats,
    convert_full_stage_dataset,
    load_full_stage_dataset,
    preprocess_full_stage,
)
from src.data.full_stage_loader import FullStageConfig
from src.utils.helpers import get_logger, set_global_seed

logger = get_logger(__name__)

DATASET_NAME = "full_stage_v1"
RUN_NAME = "full_stage_v1"


def _render_summary(stats: dict[str, object]) -> str:
    """Render the data-summary statistics as Markdown."""
    lines = [
        f"# Data Summary: `{DATASET_NAME}`",
        "",
        "Expanded multi-stage dataset that **includes chips that failed during "
        "Stage-2 testing** (previously excluded). Source file: "
        "`data/raw/full_stage_df.csv`.",
        "",
        "## Dataset composition",
        "",
        "| Quantity | Value |",
        "| --- | --- |",
        f"| Number of chips | {stats['n_chips']:,} |",
        f"| PASS (label 0) | {stats['n_pass']:,} |",
        f"| FAIL (label 1) | {stats['n_fail']:,} |",
        f"| Fail rate | {float(stats['fail_rate']) * 100:.2f}% |",
        f"| Stage-2 failures | {stats['n_stage2_failures']:,} |",
        f"| Stage-3 / final failures | {stats['n_stage3_failures']:,} |",
        f"| Chips with missing Stage-3 data | {stats['n_missing_stage3']:,} |",
        f"| Stage-2-pass chips with no final result (ambiguous -> PASS) | "
        f"{stats['n_ambiguous_pass_no_final']:,} |",
        "",
        "## Label logic",
        "",
        "- `FinalRes_Stage2 == fail` -> **FAIL** (even with no Stage-3 data).",
        "- `FinalRes_Stage2 == pass` and `final_res == fail` -> **FAIL**.",
        "- `FinalRes_Stage2 == pass` and `final_res == pass` -> **PASS**.",
        "- `FinalRes_Stage2 == pass` and `final_res` missing -> **PASS** "
        "(passed the only completed stage; no failure recorded).",
        "",
        "## Feature columns used at each state",
        "",
        f"- **State 0 (metadata only):** {', '.join(stats['state0_metadata_columns'])}",
        f"- **State 1 (after Stage-2):** {', '.join(stats['state1_stage2_columns'])}",
        f"- **State 2 (after Stage-3):** {', '.join(stats['state2_stage3_columns'])}",
        "",
        "## Leakage handling",
        "",
        f"Excluded outcome-encoding columns (they perfectly encode the Stage-2 "
        f"result and would leak the label at State 0): "
        f"{', '.join(stats['excluded_leaky_columns'])}.",
        "",
        "`Test_Duration` is treated as a **Stage-2 feature** (a by-product of "
        "running Stage-2), available only at State 1. The Stage-2 result itself "
        "is exposed at State 1 via the `stage2_fail_flag` feature.",
        "",
        "> **Note:** this dataset contains **no Stage-3 measurement columns**. "
        "The multi-stage environment still supports the Stage-3 step, but it "
        "reveals no real measurements here; Stage-3 features are always masked.",
        "",
    ]
    return "\n".join(lines)


def prepare(config: Config = CONFIG) -> dict[str, object]:
    """Run the full preparation pipeline and persist outputs.

    Args:
        config: Project configuration.

    Returns:
        The computed data-summary statistics dictionary.
    """
    set_global_seed(config.seed)
    config.paths.ensure()

    raw_frame = pd.read_csv(config.paths.full_stage_dataset)
    frame, cols = convert_full_stage_dataset(raw_frame, config, FullStageConfig())
    stats = compute_data_stats(frame, raw_frame, cols)

    # Print the important checks.
    logger.info("=== full_stage_v1 data checks ===")
    for key in (
        "n_chips",
        "n_pass",
        "n_fail",
        "fail_rate",
        "n_stage2_failures",
        "n_stage3_failures",
        "n_missing_stage3",
        "n_ambiguous_pass_no_final",
    ):
        logger.info("  %s = %s", key, stats[key])
    logger.info("  state0 metadata = %s", stats["state0_metadata_columns"])
    logger.info("  state1 stage2   = %s", stats["state1_stage2_columns"])
    logger.info("  state2 stage3   = %s", stats["state2_stage3_columns"])

    # Processed splits.
    processed = preprocess_full_stage(frame, cols, config)
    train_path, test_path = config.paths.processed_split_paths(DATASET_NAME)
    train_path.parent.mkdir(parents=True, exist_ok=True)
    processed.train.to_csv(train_path, index=False)
    processed.test.to_csv(test_path, index=False)
    logger.info("Saved processed splits to %s and %s", train_path, test_path)

    # Data summary (Markdown + JSON) under the run directory.
    run_dir = config.paths.runs / RUN_NAME
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "data_summary.md").write_text(_render_summary(stats), encoding="utf-8")
    (run_dir / "data_summary.json").write_text(
        json.dumps(stats, indent=2, default=str), encoding="utf-8"
    )
    logger.info("Wrote data summary to %s", run_dir / "data_summary.md")
    return stats


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Prepare the full_stage_v1 dataset")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    config = CONFIG
    if args.seed is not None:
        config = dataclasses.replace(CONFIG, seed=args.seed)
    prepare(config)


if __name__ == "__main__":
    main()
