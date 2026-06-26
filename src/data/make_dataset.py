"""Generate a (small) synthetic chip-testing dataset and processed splits.

This script materialises a synthetic Stage-2 dataset so the full training and
evaluation pipeline can be exercised end-to-end *before* the real proprietary
chip dataset is integrated. To switch to real data later, simply place a CSV
with the same schema (feature columns + a binary ``label`` column) at
``data/raw/chip_tests.csv`` and skip this script.

Run as a module::

    # Small dataset for a fast end-to-end smoke run.
    python -m src.data.make_dataset --samples 1500 --features 20

    # Generate raw data only (no train/test split).
    python -m src.data.make_dataset --samples 1500 --no-split
"""

from __future__ import annotations

import argparse
import dataclasses

from src.config import CONFIG, Config
from src.data.loader import generate_synthetic_dataset
from src.data.preprocessing import preprocess_and_save
from src.utils.helpers import get_logger, set_global_seed

logger = get_logger(__name__)


def make_dataset(
    *,
    n_samples: int = 1500,
    n_features: int = 20,
    fail_rate: float | None = None,
    seed: int | None = None,
    write_splits: bool = True,
    base_config: Config = CONFIG,
) -> Config:
    """Generate a synthetic dataset (and optionally processed splits) on disk.

    Args:
        n_samples: Number of chips (rows) to generate.
        n_features: Number of Stage-2 measurement features.
        fail_rate: Optional override for the defective-chip rate.
        seed: Optional override for the global/random seed.
        write_splits: If ``True``, also write processed train/test CSVs.
        base_config: Configuration to derive the run config from.

    Returns:
        The effective :class:`~src.config.Config` used for generation.
    """
    data_cfg = dataclasses.replace(
        base_config.data,
        n_synthetic_samples=n_samples,
        n_synthetic_features=n_features,
        synthetic_fail_rate=(
            fail_rate if fail_rate is not None else base_config.data.synthetic_fail_rate
        ),
    )
    config = dataclasses.replace(
        base_config,
        data=data_cfg,
        seed=seed if seed is not None else base_config.seed,
    )

    set_global_seed(config.seed)
    config.paths.ensure()

    frame = generate_synthetic_dataset(config)
    frame.to_csv(config.paths.raw_dataset, index=False)
    fail_pct = 100.0 * frame[config.env.label_column].mean()
    logger.info(
        "Wrote raw dataset (%d rows, %d cols, %.1f%% FAIL) to %s",
        frame.shape[0],
        frame.shape[1],
        fail_pct,
        config.paths.raw_dataset,
    )

    if write_splits:
        preprocess_and_save(config)

    return config


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Generate a synthetic dataset")
    parser.add_argument("--samples", type=int, default=1500, help="Number of chips")
    parser.add_argument("--features", type=int, default=20, help="Measurement features")
    parser.add_argument("--fail-rate", type=float, default=None, help="Defect rate")
    parser.add_argument("--seed", type=int, default=None, help="Random seed override")
    parser.add_argument(
        "--no-split",
        action="store_true",
        help="Only write the raw dataset (skip processed train/test splits)",
    )
    args = parser.parse_args()
    make_dataset(
        n_samples=args.samples,
        n_features=args.features,
        fail_rate=args.fail_rate,
        seed=args.seed,
        write_splits=not args.no_split,
    )


if __name__ == "__main__":
    main()
