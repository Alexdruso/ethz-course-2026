"""Shared pytest configuration for the hw1 test suite.

The exercises live in `src/` as plain (jupytext) scripts rather than an
installed package, so we put that directory on `sys.path` here and import the
exercise modules directly, exactly like the autograder does.
"""

import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture(autouse=True)
def _deterministic_rng():
    """Seed the global RNG before every test.

    Importing `ex1` runs its demo cells, one of which calls
    `torch.manual_seed(123)`. Re-seeding here keeps tests independent of import
    order and of each other.
    """
    import torch

    torch.manual_seed(0)
