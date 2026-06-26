"""General-purpose helpers: logging, reproducibility and JSON serialisation."""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any

import numpy as np

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def configure_logging(level: int = logging.INFO) -> None:
    """Configure root logging once for the whole project.

    Args:
        level: Logging verbosity, e.g. :data:`logging.INFO`.
    """
    logging.basicConfig(level=level, format=_LOG_FORMAT, datefmt=_DATE_FORMAT)


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger, configuring logging on first use.

    Args:
        name: Usually ``__name__`` of the calling module.

    Returns:
        A configured :class:`logging.Logger`.
    """
    if not logging.getLogger().handlers:
        configure_logging()
    return logging.getLogger(name)


def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy (and, if present, PyTorch) for reproducibility.

    Args:
        seed: The integer seed to apply everywhere.
    """
    random.seed(seed)
    np.random.seed(seed)
    try:  # PyTorch is an optional (transitive) dependency via stable-baselines3.
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:  # pragma: no cover - torch always present with SB3.
        pass


def save_json(data: dict[str, Any], path: Path | str) -> None:
    """Serialise a dictionary to JSON, creating parent directories as needed.

    Args:
        data: JSON-serialisable mapping.
        path: Destination file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, default=_json_default)


def load_json(path: Path | str) -> dict[str, Any]:
    """Load a JSON file into a dictionary.

    Args:
        path: Source file path.

    Returns:
        The decoded JSON object.
    """
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _json_default(value: Any) -> Any:
    """Fallback encoder for NumPy scalar/array types in :func:`save_json`."""
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serialisable")
