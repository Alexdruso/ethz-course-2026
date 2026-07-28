"""Shared pytest configuration for the hw1 test suite.

The exercises live in `src/` as plain (jupytext) scripts rather than an
installed package, so we put that directory on `sys.path` here and import the
exercise modules directly, exactly like the autograder does.
"""

import importlib
import sys
from pathlib import Path
from unittest import mock

import pytest
import torch
from torchvision import datasets

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


class _FakeMNIST(torch.utils.data.Dataset):
    """Offline stand-in for `torchvision.datasets.MNIST`.

    `ex3` and `ex4` build the real MNIST datasets at module scope with
    `download=True`, so importing them in a test run would hit the network and
    write a `data/` directory. The samples here have the same shape, dtype and
    value range as `MNIST` + `ToTensor()`; only the pixels are meaningless.
    """

    n_train = 64
    n_test = 32

    def __init__(self, root=None, train=True, download=False, transform=None, **kwargs):
        self.train = train
        self.n = self.n_train if train else self.n_test
        g = torch.Generator().manual_seed(0 if train else 1)
        self.data = torch.rand(self.n, 1, 28, 28, generator=g)
        self.targets = torch.randint(0, 10, (self.n,), generator=g)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        return self.data[idx], int(self.targets[idx])


def _import_exercises_offline(*names: str) -> None:
    """Import exercise modules with MNIST stubbed out.

    conftest is imported before any test module, so patching here is what makes
    the plain `from ex3 import ...` at the top of the test files safe.

    A module that cannot be imported at all - e.g. a half-typed line in `ex4`
    while you are still working on it - must not take the rest of the suite
    down with it, so failures are swallowed here. The test files that import
    that module then fail on their own during collection, and the tests for the
    other exercises still run.
    """
    with mock.patch.object(datasets, "MNIST", _FakeMNIST):
        for name in names:
            try:
                importlib.import_module(name)
            except Exception:  # noqa: BLE001 - reported by the importing test file
                sys.modules.pop(name, None)


_import_exercises_offline("ex3", "ex4")


@pytest.fixture(autouse=True)
def _deterministic_rng():
    """Seed the global RNG before every test.

    Importing `ex1` runs its demo cells, one of which calls
    `torch.manual_seed(123)`. Re-seeding here keeps tests independent of import
    order and of each other.
    """
    import torch

    torch.manual_seed(0)
