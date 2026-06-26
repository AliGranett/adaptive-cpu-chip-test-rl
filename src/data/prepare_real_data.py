"""Prepare the real chip dataset for the RL pipeline.

Converts ``data/raw/base_data.csv`` into the canonical schema, writes
``data/raw/chip_tests.csv``, and builds processed train/test splits.

Run as a module::

    python -m src.data.prepare_real_data
"""

from __future__ import annotations

import argparse

from src.config import CONFIG, Config
from src.data.preprocessing import preprocess_and_save
from src.data.real_data_loader import prepare_real_dataset
from src.utils.helpers import get_logger, set_global_seed

logger = get_logger(__name__)


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Prepare real chip dataset")
    parser.add_argument("--seed", type=int, default=None, help="Random seed override")
    args = parser.parse_args()

    config = CONFIG
    if args.seed is not None:
        import dataclasses

        config = dataclasses.replace(CONFIG, seed=args.seed)

    set_global_seed(config.seed)
    config.paths.ensure()

    prepare_real_dataset(config)
    processed = preprocess_and_save(config)
    logger.info(
        "Real data ready: %d train / %d test / %d features",
        len(processed.train),
        len(processed.test),
        len(processed.feature_columns),
    )


if __name__ == "__main__":
    main()
