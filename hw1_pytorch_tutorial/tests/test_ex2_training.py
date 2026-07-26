"""Tests for the single training step of exercise 2."""

import copy

import torch
from torch import nn

from ex2 import train_step_mse


def _model(seed: int = 0) -> nn.Linear:
    torch.manual_seed(seed)
    return nn.Linear(3, 2)


class TestTrainStepMse:
    def test_returns_the_loss_measured_before_the_update(self):
        model = _model()
        x = torch.randn(8, 3)
        y = torch.randn(8, 2)
        expected = nn.functional.mse_loss(model(x), y).item()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

        loss = train_step_mse(model, (x, y), optimizer)

        assert isinstance(loss, float)
        assert abs(loss - expected) < 1e-6

    def test_actually_updates_the_parameters(self):
        model = _model()
        before = [p.detach().clone() for p in model.parameters()]
        x = torch.randn(16, 3)
        y = torch.randn(16, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        train_step_mse(model, (x, y), optimizer)

        after = list(model.parameters())
        assert any(not torch.equal(b, a.detach()) for b, a in zip(before, after))

    def test_zeroes_stale_gradients_before_the_backward_pass(self):
        # Leftover garbage in .grad must not influence the SGD update.
        model = _model()
        for p in model.parameters():
            p.grad = torch.full_like(p, 1000.0)
        x = torch.randn(8, 3)
        y = torch.randn(8, 2)
        lr = 0.1
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)

        clean = copy.deepcopy(model)
        for p in clean.parameters():
            p.grad = None
        nn.functional.mse_loss(clean(x), y).backward()
        expected = [(p.detach() - lr * p.grad).clone() for p in clean.parameters()]

        train_step_mse(model, (x, y), optimizer)

        for p, e in zip(model.parameters(), expected):
            assert torch.allclose(p.detach(), e, atol=1e-6)

    def test_puts_the_model_into_train_mode(self):
        model = nn.Sequential(nn.Linear(3, 4), nn.Dropout(0.5), nn.Linear(4, 2))
        model.eval()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        train_step_mse(model, (torch.randn(4, 3), torch.randn(4, 2)), optimizer)

        assert model.training

    def test_repeated_steps_reduce_the_loss(self):
        model = _model()
        x = torch.randn(32, 3)
        y = x @ torch.randn(3, 2) + 0.1
        optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

        losses = [train_step_mse(model, (x, y), optimizer) for _ in range(25)]

        assert losses[-1] < losses[0]
