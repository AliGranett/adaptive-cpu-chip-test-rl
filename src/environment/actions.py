"""Shared action and label definitions for the chip-testing environments."""

from __future__ import annotations

import enum


class Action(enum.IntEnum):
    """Discrete actions available to the testing agent."""

    CONTINUE = 0
    """Pay the stage cost and reveal the next test stage."""
    STOP_PASS = 1
    """Stop testing and classify the chip as PASS (good)."""
    STOP_FAIL = 2
    """Stop testing and classify the chip as FAIL (defective)."""


# Ground-truth label encoding.
LABEL_PASS = 0
LABEL_FAIL = 1
