"""Adaptive CPU Chip Test Reduction Using Reinforcement Learning.

A clean, modular reinforcement-learning project that learns an adaptive
chip-testing policy (continue / stop-PASS / stop-FAIL) to minimise testing
cost while preserving classification quality.
"""

from __future__ import annotations

import os

# macOS ships PyTorch (via stable-baselines3) and XGBoost with separate OpenMP
# runtimes. When both are loaded in the same process, multi-threaded OpenMP can
# corrupt memory and segfault. Pinning OpenMP to a single thread before either
# native library is imported avoids the conflict (and aids reproducibility).
# These are set with ``setdefault`` so an explicit user value always wins.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

__version__ = "1.0.0"
