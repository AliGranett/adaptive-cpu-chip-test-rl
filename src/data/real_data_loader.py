"""Load and convert the real CPU chip manufacturing dataset.

Maps ``data/raw/base_data.csv`` (from the original Data Science project) into
the canonical schema expected by preprocessing and :class:`ChipTestingEnv`:

* Stage-2 measurements  -> ``stage2_m00``–``stage2_m02``  (revealed first)
* Stage-3 measurements  -> ``stage2_m03``–``stage2_m06``  (revealed on CONTINUE)
* Wafer / location metadata
* Binary ``label`` (0 = PASS, 1 = FAIL) from ``FinalRes``

``FinalSpeed`` and other post-test outcome columns are excluded to avoid
label leakage.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import CONFIG, Config
from src.utils.helpers import get_logger

logger = get_logger(__name__)

# Source columns in ``base_data.csv``.
STAGE2_SOURCE = ("Power_Stage2", "SpeedH_Stage2", "SpeedL_Stage2")
STAGE3_SOURCE = (
    "Power_Stage3",
    "SpeedH_Stage3",
    "SpeedL_Stage3",
    "SpeedReference_Stage3",
)
LABEL_SOURCE = "FinalRes"


def convert_real_dataset(
    frame: pd.DataFrame, config: Config = CONFIG
) -> pd.DataFrame:
    """Convert a raw ``base_data`` frame to the canonical chip-testing schema.

    Args:
        frame: Raw manufacturing dataframe as read from ``base_data.csv``.
        config: Project configuration (for the label column name).

    Returns:
        Canonical dataframe with ``stage2_m*`` features, metadata and ``label``.

    Raises:
        ValueError: If required source columns are missing.
    """
    required = list(STAGE2_SOURCE) + list(STAGE3_SOURCE) + [LABEL_SOURCE]
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns in real dataset: {missing}")

    out = pd.DataFrame(index=frame.index)

    for i, col in enumerate(STAGE2_SOURCE):
        out[f"stage2_m{i:02d}"] = pd.to_numeric(frame[col], errors="coerce")
    for j, col in enumerate(STAGE3_SOURCE, start=len(STAGE2_SOURCE)):
        out[f"stage2_m{j:02d}"] = pd.to_numeric(frame[col], errors="coerce")

    out["wafer_id"] = pd.to_numeric(frame["WAFER_Ser_Num"], errors="coerce")
    out["die_x"] = pd.to_numeric(frame["X_cor"], errors="coerce")
    out["die_y"] = pd.to_numeric(frame["Y_cor"], errors="coerce")
    out["radial_dist"] = pd.to_numeric(frame["distance_from_center"], errors="coerce")

    if "Test_Duration(milliseconds)" in frame.columns:
        out["test_duration_ms"] = pd.to_numeric(
            frame["Test_Duration(milliseconds)"], errors="coerce"
        )
    if "Category_code" in frame.columns:
        out["category_code"] = pd.to_numeric(frame["Category_code"], errors="coerce")
    if "Test_code" in frame.columns:
        out["test_code"] = pd.to_numeric(frame["Test_code"], errors="coerce")

    labels = frame[LABEL_SOURCE].astype(str).str.strip().str.lower()
    unknown = ~labels.isin({"pass", "fail"})
    if unknown.any():
        n_bad = int(unknown.sum())
        logger.warning("Dropping %d rows with unknown FinalRes values", n_bad)
        out = out.loc[~unknown].copy()
        labels = labels.loc[~unknown]

    out[config.env.label_column] = (labels == "fail").astype(int)

    logger.info(
        "Converted real dataset: %d rows, %d features, %.1f%% FAIL",
        len(out),
        len([c for c in out.columns if c != config.env.label_column]),
        100.0 * out[config.env.label_column].mean(),
    )
    return out.reset_index(drop=True)


def load_real_dataset(
    path: Path | None = None, config: Config = CONFIG
) -> pd.DataFrame:
    """Read ``base_data.csv`` and return the canonical dataframe.

    Args:
        path: Optional override for the real-data CSV path.
        config: Project configuration.

    Returns:
        Canonical chip-testing dataframe.

    Raises:
        FileNotFoundError: If the real dataset file does not exist.
    """
    source = path or config.paths.base_dataset
    if not source.exists():
        raise FileNotFoundError(f"Real dataset not found at {source}")
    logger.info("Loading real dataset from %s", source)
    raw = pd.read_csv(source)
    return convert_real_dataset(raw, config)


def prepare_real_dataset(
    config: Config = CONFIG,
    *,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """Convert real data and write the canonical CSV used by the pipeline.

    Args:
        config: Project configuration.
        output_path: Destination for the canonical CSV. Defaults to
            ``config.paths.raw_dataset``.

    Returns:
        The converted canonical dataframe.
    """
    config.paths.ensure()
    frame = load_real_dataset(config=config)
    dest = output_path or config.paths.raw_dataset
    frame.to_csv(dest, index=False)
    logger.info("Wrote canonical real dataset to %s", dest)
    return frame
