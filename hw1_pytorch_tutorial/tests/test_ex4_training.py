"""Tests for `train_one_run` in exercise 4.

The loop body is provided; the two `TODO`s are the criterion (`loss = ...`) and
the returned metrics dict. These tests are the spec for both.

Assumed contract:
- the criterion is the standard **mean** cross-entropy over the 10 classes;
- the returned dict contains at least one list of `epochs * len(train_loader)`
  training losses (one per optimizer step, as the provided body appends) and one
  list of `epochs` test accuracies. The tests look those up by length rather
  than by key name, so you are free to name them whatever you like.
"""

import math

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ex4 import TrainConfig, train_one_run

N_TRAIN, N_TEST, BATCH = 8, 4, 4
EPOCHS = 2
STEPS = EPOCHS * (N_TRAIN // BATCH)


def _loader(n: int, batch_size: int = BATCH, seed: int = 0) -> DataLoader:
    g = torch.Generator().manual_seed(seed)
    labels = torch.randint(0, 10, (n,), generator=g)
    images = torch.rand(n, 1, 28, 28, generator=g) * 0.1
    images[torch.arange(n), 0, labels, :] += 1.0  # class-dependent signal
    return DataLoader(
        TensorDataset(images, labels), batch_size=batch_size, shuffle=False
    )


def _model(zeroed: bool = False) -> nn.Module:
    model = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 10))
    if zeroed:
        with torch.no_grad():
            for p in model.parameters():
                p.zero_()
    return model


def _cfg(**overrides) -> TrainConfig:
    base = {
        "seed": 0,
        "batch_size": BATCH,
        "epochs": EPOCHS,
        "lr": 1e-2,
        "weight_decay": 0.01,
        "device": "cpu",
    }
    return TrainConfig(**{**base, **overrides})


def _float_series(result: dict, length: int) -> list[list[float]]:
    """Every list of `length` numbers in the result, regardless of its key."""
    return [
        list(v)
        for v in result.values()
        if isinstance(v, (list, tuple))
        and len(v) == length
        and all(isinstance(e, (int, float)) and not isinstance(e, bool) for e in v)
    ]


def _run(model: nn.Module | None = None, cfg: TrainConfig | None = None) -> dict:
    return train_one_run(
        "ffn",
        model if model is not None else _model(),
        _loader(N_TRAIN),
        _loader(N_TEST, seed=1),
        cfg if cfg is not None else _cfg(),
    )


class TestTrainOneRun:
    def test_returns_a_non_empty_metrics_dict(self):
        result = _run()

        assert isinstance(result, dict)
        assert result, "the returned dict must carry the metrics for the ablation"

    def test_reports_one_training_loss_per_optimizer_step(self):
        result = _run()

        series = _float_series(result, STEPS)

        assert series, (
            f"expected a list of {STEPS} per-step training losses, got {result}"
        )
        assert all(math.isfinite(v) for v in series[0])

    def test_reports_one_test_accuracy_per_epoch(self):
        result = _run()

        series = _float_series(result, EPOCHS)

        assert series, (
            f"expected a list of {EPOCHS} per-epoch test accuracies, got {result}"
        )
        assert any(all(0.0 <= v <= 1.0 for v in s) for s in series)

    def test_the_metric_lengths_follow_cfg_epochs(self):
        result = _run(cfg=_cfg(epochs=3))

        assert _float_series(result, 3 * (N_TRAIN // BATCH))
        assert _float_series(result, 3)

    def test_the_criterion_is_mean_cross_entropy_over_ten_classes(self):
        # A model whose weights are all zero emits uniform logits, so the very
        # first loss must be exactly ln(10) for mean cross entropy.
        result = _run(model=_model(zeroed=True))

        first = _float_series(result, STEPS)[0][0]

        assert abs(first - math.log(10)) < 1e-4, (
            f"first loss {first:.4f}, expected ln(10) = {math.log(10):.4f}"
        )

    def test_updates_the_model_parameters(self):
        model = _model()
        before = [p.detach().clone() for p in model.parameters()]

        _run(model=model)

        assert any(
            not torch.equal(b, p.detach()) for b, p in zip(before, model.parameters())
        )

    def test_the_training_loss_goes_down_on_a_learnable_problem(self):
        torch.manual_seed(0)
        model = _model()

        result = train_one_run(
            "ffn",
            model,
            _loader(64, batch_size=16),
            _loader(16, batch_size=16, seed=1),
            _cfg(epochs=8, lr=0.05),
        )

        losses = _float_series(result, 8 * 4)[0]
        assert sum(losses[-4:]) / 4 < sum(losses[:4]) / 4

    def test_leaves_the_model_in_eval_mode_after_the_final_evaluation(self):
        model = _model()

        _run(model=model)

        assert not model.training
