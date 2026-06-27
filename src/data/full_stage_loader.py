"""Loader for the expanded multi-stage dataset (``full_stage_df.csv``).

This dataset extends the earlier real-data export by **including chips that
failed during Stage-2 testing** (which never proceed to Stage-3 / final test).
It therefore better represents the full manufacturing test flow:

    metadata  ->  Stage-2 test  ->  (Stage-3 / final test)  ->  final label

Column semantics (discovered by inspecting the raw file)::

    LOT_Ser_ID, WAFER_Ser_Num, X_cor, Y_cor, distance_from_center  -> metadata
    CHIP_Ser_ID                                                     -> identifier (dropped)
    FinalRes_Stage2                                                 -> Stage-2 result
    Category_code_, Test_code                                       -> failure-category codes
    Test_Duration(milliseconds)                                     -> Stage-2 test duration
    Power_Stage2, SpeedH_Stage2, SpeedL_Stage2                      -> Stage-2 measurements
    final_res                                                       -> final result

Label logic (PASS = 0, FAIL = 1):

* ``FinalRes_Stage2 == "fail"`` -> FAIL (Stage-2 failure; no Stage-3 data).
* ``FinalRes_Stage2 == "pass"`` and ``final_res == "fail"`` -> FAIL (final/Stage-3 failure).
* ``FinalRes_Stage2 == "pass"`` and ``final_res == "pass"`` -> PASS.
* ``FinalRes_Stage2 == "pass"`` and ``final_res`` missing -> ambiguous; labelled by
  :attr:`FullStageConfig.ambiguous_label` (default PASS), since the chip passed
  the only completed stage and no failure was recorded.

Leakage handling:

* ``Category_code_`` and ``Test_code`` *perfectly encode* the Stage-2 result
  (e.g. ``Category_code_ == 1`` iff Stage-2 passed). They are failure-category
  codes assigned **as a result** of testing, so using them as Stage-0 metadata
  would leak the outcome. They are therefore **excluded** from the observable
  features. The legitimate Stage-2 result is still exposed - but only at
  State 1, after Stage-2 has actually been run - via the ``stage2_fail_flag``
  feature.
* ``Test_Duration`` is a by-product of running Stage-2 (it strongly separates
  pass/fail) so it is treated as a **Stage-2 feature**, available only at
  State 1, never as Stage-0 metadata.

This dataset has **no Stage-3 measurement columns**; Stage-3 is structurally
supported by the environment but reveals no real measurements here (see
:mod:`src.environment.multi_stage_env`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import CONFIG, Config
from src.utils.helpers import get_logger

logger = get_logger(__name__)

# Source column names in the raw CSV.
SRC_LOT = "LOT_Ser_ID"
SRC_WAFER = "WAFER_Ser_Num"
SRC_X = "X_cor"
SRC_Y = "Y_cor"
SRC_RADIAL = "distance_from_center"
SRC_STAGE2_RESULT = "FinalRes_Stage2"
SRC_DURATION = "Test_Duration(milliseconds)"
SRC_POWER2 = "Power_Stage2"
SRC_SPEEDH2 = "SpeedH_Stage2"
SRC_SPEEDL2 = "SpeedL_Stage2"
SRC_FINAL = "final_res"
# Outcome-encoding codes that leak the Stage-2 result; intentionally excluded.
SRC_LEAKY_CODES = ("Category_code_", "Test_code")


@dataclass(frozen=True)
class FullStageConfig:
    """Options controlling conversion of the raw multi-stage dataset."""

    # Label assigned to chips that passed Stage-2 but have no final result.
    ambiguous_label: int = 0  # 0 = PASS


@dataclass
class StageColumns:
    """Names of the feature columns grouped by reveal stage.

    Attributes:
        metadata: Columns visible at State 0 (before any testing).
        stage2: Columns revealed by RUN_STAGE2 (State 1).
        stage3: Columns revealed by RUN_STAGE3 (State 2); empty for this
            dataset, which has no Stage-3 measurements.
        stage2_flag_col: Name of the binary Stage-2-failure flag (part of the
            Stage-2 group, used by the environment for reward/metric logic).
        label_col: Name of the final binary label column.
    """

    metadata: list[str]
    stage2: list[str]
    stage3: list[str] = field(default_factory=list)
    stage2_flag_col: str = "stage2_fail_flag"
    label_col: str = "label"

    @property
    def feature_columns(self) -> list[str]:
        """All feature columns in reveal order (metadata, Stage-2, Stage-3)."""
        return [*self.metadata, *self.stage2, *self.stage3]

    @property
    def stage_groups(self) -> list[list[str]]:
        """Feature columns partitioned per stage (index 0/1/2)."""
        return [list(self.metadata), list(self.stage2), list(self.stage3)]


# Canonical (post-conversion) column names.
META_COLS = ["meta_lot", "meta_wafer", "meta_x", "meta_y", "meta_radial"]
STAGE2_COLS = ["s2_power", "s2_speedh", "s2_speedl", "s2_duration", "stage2_fail_flag"]
STAGE3_COLS: list[str] = []  # No Stage-3 measurements in this dataset.


def stage_columns() -> StageColumns:
    """Return the canonical :class:`StageColumns` for the multi-stage dataset."""
    return StageColumns(
        metadata=list(META_COLS),
        stage2=list(STAGE2_COLS),
        stage3=list(STAGE3_COLS),
        stage2_flag_col="stage2_fail_flag",
        label_col="label",
    )


def _derive_label(
    stage2_result: pd.Series, final_result: pd.Series, ambiguous_label: int
) -> pd.Series:
    """Compute the binary label from Stage-2 and final results.

    Args:
        stage2_result: Lower-cased Stage-2 result series.
        final_result: Lower-cased final result series (may contain NaN).
        ambiguous_label: Label for Stage-2-pass chips with no final result.

    Returns:
        Integer series with 0 = PASS, 1 = FAIL.
    """
    label = pd.Series(ambiguous_label, index=stage2_result.index, dtype="int64")
    label[stage2_result == "fail"] = 1  # Stage-2 failure -> FAIL.
    passed_stage2 = stage2_result == "pass"
    label[passed_stage2 & (final_result == "fail")] = 1
    label[passed_stage2 & (final_result == "pass")] = 0
    return label


def convert_full_stage_dataset(
    frame: pd.DataFrame,
    config: Config = CONFIG,
    full_stage_config: FullStageConfig = FullStageConfig(),
) -> tuple[pd.DataFrame, StageColumns]:
    """Convert a raw ``full_stage_df`` frame to the canonical multi-stage schema.

    Args:
        frame: Raw dataframe read from ``full_stage_df.csv``.
        config: Project configuration.
        full_stage_config: Conversion options (ambiguous-label policy).

    Returns:
        ``(canonical_frame, stage_columns)``. The frame has metadata,
        Stage-2 features (including ``stage2_fail_flag``), the ``label`` column
        and a helper ``is_stage2_fail`` column used by the environment.

    Raises:
        ValueError: If required source columns are missing.
    """
    required = [
        SRC_LOT,
        SRC_WAFER,
        SRC_X,
        SRC_Y,
        SRC_RADIAL,
        SRC_STAGE2_RESULT,
        SRC_DURATION,
        SRC_POWER2,
        SRC_SPEEDH2,
        SRC_SPEEDL2,
        SRC_FINAL,
    ]
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns in full_stage dataset: {missing}")

    stage2_result = frame[SRC_STAGE2_RESULT].astype(str).str.strip().str.lower()
    final_result = frame[SRC_FINAL].astype(str).str.strip().str.lower()
    # Normalise textual NaNs ("nan", "none", "") to a real NaN sentinel.
    final_result = final_result.where(~final_result.isin({"nan", "none", ""}), other=np.nan)

    out = pd.DataFrame(index=frame.index)
    # Metadata (known before any testing).
    out["meta_lot"] = pd.to_numeric(frame[SRC_LOT], errors="coerce")
    out["meta_wafer"] = pd.to_numeric(frame[SRC_WAFER], errors="coerce")
    out["meta_x"] = pd.to_numeric(frame[SRC_X], errors="coerce")
    out["meta_y"] = pd.to_numeric(frame[SRC_Y], errors="coerce")
    out["meta_radial"] = pd.to_numeric(frame[SRC_RADIAL], errors="coerce")

    # Stage-2 measurements + duration (known only after running Stage-2).
    out["s2_power"] = pd.to_numeric(frame[SRC_POWER2], errors="coerce")
    out["s2_speedh"] = pd.to_numeric(frame[SRC_SPEEDH2], errors="coerce")
    out["s2_speedl"] = pd.to_numeric(frame[SRC_SPEEDL2], errors="coerce")
    out["s2_duration"] = pd.to_numeric(frame[SRC_DURATION], errors="coerce")

    # Stage-2 result, exposed as a feature revealed at State 1.
    is_stage2_fail = (stage2_result == "fail").astype(int)
    out["stage2_fail_flag"] = is_stage2_fail

    # Final label and a helper used by the environment for reward/metrics.
    out["label"] = _derive_label(
        stage2_result, final_result, full_stage_config.ambiguous_label
    ).to_numpy()
    out["is_stage2_fail"] = is_stage2_fail.to_numpy()

    cols = stage_columns()
    logger.info(
        "Converted full_stage dataset: %d rows, %.1f%% FAIL, %d Stage-2 failures",
        len(out),
        100.0 * out["label"].mean(),
        int(out["is_stage2_fail"].sum()),
    )
    return out, cols


def load_full_stage_dataset(
    path: Path | None = None,
    config: Config = CONFIG,
    full_stage_config: FullStageConfig = FullStageConfig(),
) -> tuple[pd.DataFrame, StageColumns]:
    """Read ``full_stage_df.csv`` and return the canonical frame + stage columns.

    Args:
        path: Optional override for the raw CSV path.
        config: Project configuration.
        full_stage_config: Conversion options.

    Returns:
        ``(canonical_frame, stage_columns)``.

    Raises:
        FileNotFoundError: If the raw dataset does not exist.
    """
    source = path or config.paths.full_stage_dataset
    if not source.exists():
        raise FileNotFoundError(f"Full-stage dataset not found at {source}")
    logger.info("Loading full-stage dataset from %s", source)
    raw = pd.read_csv(source)
    return convert_full_stage_dataset(raw, config, full_stage_config)


def compute_data_stats(
    frame: pd.DataFrame, raw: pd.DataFrame, cols: StageColumns
) -> dict[str, object]:
    """Compute summary statistics for the converted dataset.

    Args:
        frame: Converted canonical frame.
        raw: Raw dataframe (for Stage-2/Stage-3 result breakdowns).
        cols: Stage column groupings.

    Returns:
        A dictionary of human-readable dataset statistics.
    """
    stage2_result = raw[SRC_STAGE2_RESULT].astype(str).str.strip().str.lower()
    final_result = raw[SRC_FINAL].astype(str).str.strip().str.lower()
    final_result = final_result.where(~final_result.isin({"nan", "none", ""}), other=np.nan)

    n = len(frame)
    n_fail = int(frame["label"].sum())
    n_stage2_fail = int((stage2_result == "fail").sum())
    n_stage3_fail = int(((stage2_result == "pass") & (final_result == "fail")).sum())
    n_missing_stage3 = int(stage2_result.eq("fail").sum() + (
        (stage2_result == "pass") & (final_result.isna())
    ).sum())
    n_ambiguous = int(((stage2_result == "pass") & (final_result.isna())).sum())

    return {
        "n_chips": n,
        "n_pass": n - n_fail,
        "n_fail": n_fail,
        "fail_rate": n_fail / n if n else 0.0,
        "n_stage2_failures": n_stage2_fail,
        "n_stage3_failures": n_stage3_fail,
        "n_missing_stage3": n_missing_stage3,
        "n_ambiguous_pass_no_final": n_ambiguous,
        "excluded_leaky_columns": list(SRC_LEAKY_CODES),
        "state0_metadata_columns": cols.metadata,
        "state1_stage2_columns": cols.stage2,
        "state2_stage3_columns": cols.stage3 or ["(none - dataset has no Stage-3 measurements)"],
        "label_column": cols.label_col,
    }


# Helper (non-feature) columns carried alongside the features for the env.
HELPER_COLUMNS = ["is_stage2_fail"]
# Continuous columns to impute + standardise (binary flags are left as 0/1).
_CONTINUOUS_COLS = [*META_COLS, "s2_power", "s2_speedh", "s2_speedl", "s2_duration"]


@dataclass
class MultiStageData:
    """Processed multi-stage train/test splits plus stage metadata."""

    train: pd.DataFrame
    test: pd.DataFrame
    columns: StageColumns

    @property
    def feature_columns(self) -> list[str]:
        """Ordered feature columns (metadata, Stage-2, Stage-3)."""
        return self.columns.feature_columns

    @property
    def label_column(self) -> str:
        """Name of the label column."""
        return self.columns.label_col


def preprocess_full_stage(
    frame: pd.DataFrame, cols: StageColumns, config: Config = CONFIG
) -> MultiStageData:
    """Split, impute and standardise the converted multi-stage frame.

    Continuous metadata and Stage-2 measurement columns are median-imputed and
    standardised (fit on train only). Binary flags, the label and the helper
    ``is_stage2_fail`` column are passed through unchanged.

    Args:
        frame: Converted canonical frame from :func:`convert_full_stage_dataset`.
        cols: Stage column groupings.
        config: Project configuration (split/seed).

    Returns:
        A :class:`MultiStageData` with scaled train/test splits.
    """
    label_col = cols.label_col
    feature_cols = cols.feature_columns
    passthrough = [c for c in HELPER_COLUMNS if c in frame.columns]

    x = frame[feature_cols + passthrough]
    y = frame[label_col].astype(int)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=config.data.test_size,
        random_state=config.seed,
        stratify=y,
    )

    continuous = [c for c in _CONTINUOUS_COLS if c in feature_cols]
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    def _transform(block: pd.DataFrame, *, fit: bool) -> pd.DataFrame:
        out = block.copy().reset_index(drop=True)
        if fit:
            scaled = scaler.fit_transform(imputer.fit_transform(out[continuous]))
        else:
            scaled = scaler.transform(imputer.transform(out[continuous]))
        out[continuous] = scaled
        return out

    train = _transform(x_train, fit=True)
    train[label_col] = y_train.to_numpy()
    test = _transform(x_test, fit=False)
    test[label_col] = y_test.to_numpy()

    logger.info(
        "Preprocessed full-stage data: %d train / %d test rows, %d features",
        len(train),
        len(test),
        len(feature_cols),
    )
    return MultiStageData(train, test, cols)


def load_full_stage_processed(
    config: Config = CONFIG, *, dataset: str = "full_stage_v1"
) -> MultiStageData:
    """Load processed multi-stage train/test splits from disk.

    Args:
        config: Project configuration.
        dataset: Dataset name controlling the processed directory.

    Returns:
        The loaded :class:`MultiStageData`.

    Raises:
        FileNotFoundError: If the processed splits do not exist.
    """
    train_path, test_path = config.paths.processed_split_paths(dataset)
    if not (train_path.exists() and test_path.exists()):
        raise FileNotFoundError(
            f"Processed multi-stage splits not found for dataset '{dataset}'. "
            "Run `python -m src.data.prepare_full_stage_data` first."
        )
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    logger.info("Loaded processed multi-stage splits for dataset '%s'", dataset)
    return MultiStageData(train, test, stage_columns())
