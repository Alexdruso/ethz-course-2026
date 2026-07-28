"""Tests for the classification part of exercise 3.

Covers `cross_entropy_from_logits`, `ClassificationHead` and
`train_classifier`. All are still `TODO` in `src/ex3.py`; these tests are the
spec.

Assumed contract:
- `cross_entropy_from_logits(logits, targets)` returns the **mean** loss as a
  0-dim tensor that carries gradients, computed via log-softmax. It must not
  call `nn.CrossEntropyLoss` / `F.cross_entropy` - two tests enforce that by
  making those raise.
- `ClassificationHead(d_in, num_classes)` is a plain affine projection to raw
  logits (no softmax).
- `train_classifier(...)` returns one training loss **per update step**, i.e.
  `epochs * len(train_data_loader)` floats, and runs the model in `train()`
  mode while training and `eval()` mode while evaluating.

`accuracy(loader)` is deliberately **not** tested. Its signature takes no
model, so any test would have to invent a way for it to reach one (a
module-level global, say) and would then be checking that invention rather than
the exercise. The exercise calls it a helper you "can use", not a graded
interface, so how you wire it up is your call.
"""

import math

import pytest
import torch
from nn_helpers import ModeSpy, assert_is_affine
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from ex3 import (
    ClassificationHead,
    cross_entropy_from_logits,
    train_classifier,
)


def _ban_builtin_cross_entropy(monkeypatch):
    """Make PyTorch's own cross entropy explode, as the autograder does."""

    def boom(*args, **kwargs):
        raise AssertionError("use log-softmax, not PyTorch's cross entropy")

    monkeypatch.setattr(nn, "CrossEntropyLoss", boom)
    monkeypatch.setattr(nn.functional, "cross_entropy", boom)
    monkeypatch.setattr(torch.nn.functional, "nll_loss", boom)


def _image_loader(n: int, batch_size: int, seed: int = 0) -> DataLoader:
    """MNIST-shaped batches: `(B, 1, 28, 28)` floats in [0, 1] and int64 labels."""
    g = torch.Generator().manual_seed(seed)
    labels = torch.randint(0, 10, (n,), generator=g)
    # Give each class a distinguishable image so the toy problem is learnable.
    images = torch.rand(n, 1, 28, 28, generator=g) * 0.1
    images[torch.arange(n), 0, labels, :] += 1.0
    return DataLoader(
        TensorDataset(images, labels), batch_size=batch_size, shuffle=False
    )


class _FlatClassifier(nn.Sequential):
    def __init__(self, num_classes: int = 10):
        super().__init__(nn.Flatten(), nn.Linear(28 * 28, num_classes))


class TestCrossEntropyFromLogits:
    def test_matches_torch_cross_entropy(self):
        logits = torch.randn(7, 4)
        targets = torch.randint(0, 4, (7,))

        loss = cross_entropy_from_logits(logits, targets)

        assert loss.shape == ()
        assert abs(loss.item() - F.cross_entropy(logits, targets).item()) < 1e-6

    def test_averages_over_the_batch_instead_of_summing(self):
        logits = torch.randn(5, 3)
        targets = torch.randint(0, 3, (5,))

        single = cross_entropy_from_logits(logits, targets)
        doubled = cross_entropy_from_logits(logits.repeat(2, 1), targets.repeat(2))

        assert abs(single.item() - doubled.item()) < 1e-6
        assert (
            abs(
                single.item() - F.cross_entropy(logits, targets, reduction="sum").item()
            )
            > 1e-3
        )

    def test_uniform_logits_give_log_of_the_number_of_classes(self):
        loss = cross_entropy_from_logits(torch.zeros(4, 10), torch.randint(0, 10, (4,)))

        assert abs(loss.item() - math.log(10)) < 1e-6

    def test_stays_finite_for_huge_logits(self):
        logits = torch.tensor([[1e4, -1e4, 0.0], [-1e4, 1e4, 5e3]])
        targets = torch.tensor([0, 1])

        loss = cross_entropy_from_logits(logits, targets)

        assert torch.isfinite(loss).all()
        assert loss.item() < 1e-3  # confident and correct -> near-zero loss

    def test_is_invariant_to_a_constant_shift_of_the_logits(self):
        logits = torch.randn(6, 5)
        targets = torch.randint(0, 5, (6,))
        shift = torch.randn(6, 1) * 100

        a = cross_entropy_from_logits(logits, targets)
        b = cross_entropy_from_logits(logits + shift, targets)

        assert abs(a.item() - b.item()) < 1e-4

    def test_a_confidently_wrong_prediction_costs_far_more_than_a_right_one(self):
        logits = torch.tensor([[10.0, 0.0]])

        right = cross_entropy_from_logits(logits, torch.tensor([0]))
        wrong = cross_entropy_from_logits(logits, torch.tensor([1]))

        assert right.item() < 1e-3
        assert wrong.item() > 9.0

    def test_gradient_is_softmax_minus_one_hot_over_the_batch_size(self):
        logits = torch.randn(4, 3, requires_grad=True)
        targets = torch.tensor([0, 2, 1, 1])

        cross_entropy_from_logits(logits, targets).backward()

        expected = logits.softmax(dim=-1).detach()
        expected[torch.arange(4), targets] -= 1.0
        assert torch.allclose(logits.grad, expected / 4, atol=1e-6)

    def test_does_not_call_pytorchs_cross_entropy(self, monkeypatch):
        logits = torch.randn(5, 4)
        targets = torch.randint(0, 4, (5,))
        expected = F.cross_entropy(logits, targets).item()
        _ban_builtin_cross_entropy(monkeypatch)

        loss = cross_entropy_from_logits(logits, targets)

        assert abs(loss.item() - expected) < 1e-6


class TestClassificationHead:
    def test_projects_the_feature_dimension_to_the_class_dimension(self):
        head = ClassificationHead(d_in=6, num_classes=10)

        out = head(torch.randn(4, 6))

        assert out.shape == (4, 10)

    def test_supports_leading_batch_dimensions(self):
        head = ClassificationHead(d_in=6, num_classes=3).eval()
        x = torch.randn(2, 5, 6)

        out = head(x)

        assert out.shape == (2, 5, 3)
        assert torch.allclose(out[1, 3], head(x[1, 3]), atol=1e-5)

    def test_returns_raw_logits_rather_than_probabilities(self):
        head = ClassificationHead(d_in=6, num_classes=4).eval()

        out = head(torch.randn(20, 6) * 5)

        assert not torch.allclose(out.sum(dim=-1), torch.ones(20), atol=1e-3)
        assert_is_affine(head, torch.randn(8, 6))

    def test_gradients_reach_its_parameters(self):
        head = ClassificationHead(d_in=6, num_classes=4)

        head(torch.randn(4, 6)).pow(2).sum().backward()

        assert list(head.parameters())
        assert all(p.grad is not None for p in head.parameters())


class TestTrainClassifier:
    def test_returns_one_float_per_update_step(self):
        train_loader = _image_loader(n=24, batch_size=8)
        test_loader = _image_loader(n=8, batch_size=8, seed=1)
        model = _FlatClassifier()

        losses = train_classifier(model, train_loader, test_loader, lr=1e-2, epochs=2)

        assert isinstance(losses, list)
        assert len(losses) == 2 * len(train_loader) == 6
        assert all(isinstance(v, float) for v in losses)
        assert all(math.isfinite(v) for v in losses)

    def test_updates_the_model_parameters(self):
        train_loader = _image_loader(n=16, batch_size=8)
        test_loader = _image_loader(n=8, batch_size=8, seed=1)
        model = _FlatClassifier()
        before = [p.detach().clone() for p in model.parameters()]

        train_classifier(model, train_loader, test_loader, lr=1e-2, epochs=1)

        assert any(
            not torch.equal(b, p.detach()) for b, p in zip(before, model.parameters())
        )

    def test_the_loss_goes_down_on_a_learnable_problem(self):
        train_loader = _image_loader(n=64, batch_size=16)
        test_loader = _image_loader(n=16, batch_size=16, seed=1)
        model = _FlatClassifier()

        losses = train_classifier(model, train_loader, test_loader, lr=0.1, epochs=6)

        assert sum(losses[-4:]) / 4 < sum(losses[:4]) / 4

    def test_trains_in_train_mode_and_evaluates_in_eval_mode(self):
        train_loader = _image_loader(n=16, batch_size=8)
        test_loader = _image_loader(n=8, batch_size=8, seed=1)
        spy = ModeSpy(_FlatClassifier())

        train_classifier(spy, train_loader, test_loader, lr=1e-2, epochs=1)

        assert any(spy.modes), "forward passes during training must run in train() mode"
        assert not all(spy.modes), "the per-epoch evaluation must run in eval() mode"

    def test_does_not_use_pytorchs_cross_entropy(self, monkeypatch):
        _ban_builtin_cross_entropy(monkeypatch)
        train_loader = _image_loader(n=16, batch_size=8)
        test_loader = _image_loader(n=8, batch_size=8, seed=1)

        losses = train_classifier(
            _FlatClassifier(), train_loader, test_loader, lr=1e-2, epochs=1
        )

        assert len(losses) == 2

    def test_the_seed_makes_a_run_reproducible(self):
        # The model starts from the same weights each time, but the global RNG
        # is left in a *different* state right before the call. Shuffling and
        # dropout both draw from it, so the two runs can only agree if
        # `train_classifier` re-seeds with `seed` itself.
        def run(rng_nonce: int) -> list[float]:
            torch.manual_seed(1234)
            model = nn.Sequential(nn.Flatten(), nn.Dropout(0.5), nn.Linear(28 * 28, 10))
            train_loader = DataLoader(
                _image_loader(n=32, batch_size=8).dataset, batch_size=8, shuffle=True
            )
            torch.manual_seed(rng_nonce)
            return train_classifier(
                model,
                train_loader,
                _image_loader(n=8, batch_size=8, seed=1),
                lr=1e-2,
                epochs=2,
                seed=7,
            )

        assert run(1) == pytest.approx(run(2))
