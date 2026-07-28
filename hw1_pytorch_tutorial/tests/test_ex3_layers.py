"""Tests for the basic layers of exercise 3: `Linear`, `Embedding`, `Dropout`.

These are still `TODO` in `src/ex3.py`, so the tests fail until you fill them
in - they are the executable spec. Work through one class at a time, e.g.

    make py-test ARGS="-k TestLinear"

What the exercise actually fixes, and therefore all these tests check:
- `Linear` is `y = x @ Wᵀ + b`, supports leading batch dimensions, registers
  its parameters as `nn.Parameter`, and stores its matrix as
  `(out_features, in_features)`. With `bias=False` there is no shift at all.
- `Embedding` maps ids of shape `(...)` to vectors of shape
  `(..., embedding_dim)` by indexing into a learnable
  `(num_embeddings, embedding_dim)` table.
- `Dropout` is a no-op in eval mode and scales kept values by `1/(1-p)` in
  training mode.
- None of the three may be implemented by delegating to the matching
  `torch.nn` module.

Attribute *names* are deliberately not part of that contract, so the tests
locate parameters by rank and shape - there is only ever one candidate - rather
than by reading `.weight` or `.bias`.
"""

import pytest
import torch
from nn_helpers import projection_shapes, vector_shapes
from torch import nn

from ex3 import Dropout, Embedding, Linear


def _matrix(module: nn.Module) -> nn.Parameter:
    """The module's single 2-D parameter, whatever it is called."""
    params = [p for p in module.parameters() if p.dim() == 2]
    assert len(params) == 1, f"expected exactly one matrix, found {len(params)}"
    return params[0]


def _vector(module: nn.Module) -> nn.Parameter:
    """The module's single 1-D parameter, whatever it is called."""
    params = [p for p in module.parameters() if p.dim() == 1]
    assert len(params) == 1, f"expected exactly one vector, found {len(params)}"
    return params[0]


class TestLinear:
    def test_registers_parameters_with_the_nn_linear_shapes(self):
        layer = Linear(4, 3)

        assert projection_shapes(layer) == [(3, 4)]  # (out_features, in_features)
        assert vector_shapes(layer) == [(3,)]
        params = list(layer.parameters())
        assert all(isinstance(p, nn.Parameter) and p.requires_grad for p in params)
        # "Usually functions shouldn't change the dtype of a tensor" - stay in
        # the default float32, otherwise every later comparison drifts.
        assert all(p.dtype == torch.float32 for p in params)

    def test_forward_is_x_matmul_weight_t_plus_bias(self):
        layer = Linear(4, 3)
        weight, bias = _matrix(layer), _vector(layer)
        with torch.no_grad():
            weight.copy_(torch.arange(12, dtype=torch.float32).reshape(3, 4))
            bias.copy_(torch.tensor([1.0, -2.0, 0.5]))
        x = torch.randn(5, 4)

        out = layer(x)

        assert out.shape == (5, 3)
        assert torch.allclose(out, x @ weight.T + bias, atol=1e-6)

    def test_supports_leading_batch_dimensions(self):
        layer = Linear(6, 2)
        x = torch.randn(3, 7, 6)

        out = layer(x)

        assert out.shape == (3, 7, 2)
        # Every position is transformed independently.
        assert torch.allclose(out[1, 4], layer(x[1, 4]), atol=1e-6)

    def test_bias_false_drops_the_shift_entirely(self):
        layer = Linear(4, 3, bias=False)

        # No second parameter to hold a shift...
        assert projection_shapes(layer) == [(3, 4)]
        assert vector_shapes(layer) == []
        # ...and behaviourally, a map with no shift sends zero to zero.
        assert torch.equal(layer(torch.zeros(2, 4)), torch.zeros(2, 3))
        x = torch.randn(2, 4)
        assert torch.allclose(layer(x), x @ _matrix(layer).T, atol=1e-6)

    def test_gradients_reach_both_parameters(self):
        layer = Linear(3, 2)

        layer(torch.randn(4, 3)).sum().backward()

        assert all(p.grad is not None for p in layer.parameters())
        assert _matrix(layer).grad.abs().sum() > 0
        # d(sum of outputs)/d(bias) is one per row of the batch.
        assert torch.allclose(_vector(layer).grad, torch.full((2,), 4.0), atol=1e-6)

    def test_does_not_wrap_nn_linear(self):
        layer = Linear(4, 3)

        assert layer(torch.randn(2, 4)).shape == (2, 3)
        assert not any(isinstance(m, nn.Linear) for m in layer.modules())

    def test_two_instances_do_not_share_parameters(self):
        a, b = Linear(4, 3), Linear(4, 3)

        assert _matrix(a) is not _matrix(b)
        # Randomly initialised, so two layers start out different.
        assert not torch.equal(_matrix(a), _matrix(b))


class TestEmbedding:
    def test_the_table_is_a_learnable_parameter_of_the_documented_shape(self):
        emb = Embedding(num_embeddings=5, embedding_dim=3)

        assert projection_shapes(emb) == [(5, 3)]
        assert all(isinstance(p, nn.Parameter) for p in emb.parameters())
        assert _matrix(emb).requires_grad

    def test_maps_a_batch_of_ids_to_vectors(self):
        emb = Embedding(num_embeddings=5, embedding_dim=3)
        idx = torch.tensor([0, 4, 2])

        out = emb(idx)

        assert out.shape == (3, 3)
        # Each row is the lookup of the corresponding id on its own.
        for position, token in enumerate(idx):
            assert torch.equal(out[position], emb(token))

    def test_a_lookup_is_deterministic_and_ids_map_to_distinct_vectors(self):
        emb = Embedding(4, 5)

        assert torch.equal(emb(torch.tensor(2)), emb(torch.tensor(2)))
        assert not torch.equal(emb(torch.tensor(2)), emb(torch.tensor(3)))

    def test_preserves_the_shape_of_multi_dimensional_indices(self):
        emb = Embedding(7, 4)
        idx = torch.randint(0, 7, (2, 3), dtype=torch.int64)

        out = emb(idx)

        assert out.shape == (2, 3, 4)
        for i in range(2):
            for j in range(3):
                assert torch.equal(out[i, j], emb(idx[i, j]))

    def test_a_scalar_index_returns_a_single_vector(self):
        emb = Embedding(6, 5)

        out = emb(torch.tensor(3))

        assert out.shape == (5,)

    def test_repeated_ids_return_the_same_row_of_the_table(self):
        emb = Embedding(4, 2)

        out = emb(torch.tensor([1, 1, 3]))

        assert torch.equal(out[0], out[1])
        assert not torch.equal(out[0], out[2])

    def test_gradients_accumulate_for_repeated_ids(self):
        emb = Embedding(4, 2)

        emb(torch.tensor([1, 1, 3])).sum().backward()

        # Row 1 was looked up twice, row 3 once, rows 0 and 2 never.
        expected = torch.tensor([[0.0, 0.0], [2.0, 2.0], [0.0, 0.0], [1.0, 1.0]])
        assert torch.allclose(_matrix(emb).grad, expected, atol=1e-6)

    def test_does_not_wrap_nn_embedding(self):
        emb = Embedding(5, 3)

        assert emb(torch.tensor([0, 1])).shape == (2, 3)
        assert not any(isinstance(m, nn.Embedding) for m in emb.modules())


class TestDropout:
    def test_eval_mode_returns_the_input_untouched(self):
        drop = Dropout(0.5).eval()
        x = torch.randn(200, 10)

        assert torch.equal(drop(x), x)
        assert torch.equal(drop(x), x)  # and it stays deterministic

    def test_train_mode_keeps_values_scaled_by_one_over_one_minus_p(self):
        p = 0.4
        drop = Dropout(p).train()
        x = torch.full((500, 20), 3.0)

        out = drop(x)

        kept = out != 0
        assert kept.any() and (~kept).any(), "expected a mix of kept and dropped units"
        assert torch.allclose(out[kept], torch.full_like(out[kept], 3.0 / (1 - p)))

    def test_drops_roughly_p_of_the_units(self):
        p = 0.3
        drop = Dropout(p).train()
        x = torch.ones(400, 100)

        dropped_fraction = (drop(x) == 0).float().mean().item()

        assert abs(dropped_fraction - p) < 0.02

    def test_preserves_the_expected_value(self):
        drop = Dropout(0.5).train()
        x = torch.full((500, 200), 2.0)

        # A missing 1/(1-p) rescale would halve the mean, so the bound is loose
        # on purpose - it only has to separate 2.0 from 1.0.
        assert abs(drop(x).mean().item() - 2.0) < 0.05

    def test_p_zero_is_the_identity_in_train_mode(self):
        drop = Dropout(0.0).train()
        x = torch.randn(50, 8)

        assert torch.allclose(drop(x), x, atol=1e-6)

    def test_masks_are_resampled_on_every_call(self):
        drop = Dropout(0.5).train()
        x = torch.ones(100, 100)

        assert not torch.equal(drop(x), drop(x))

    def test_preserves_shape_and_dtype_and_lets_gradients_through(self):
        drop = Dropout(0.25).train()
        x = torch.randn(4, 6, 8, dtype=torch.float32, requires_grad=True)

        out = drop(x)

        assert out.shape == x.shape and out.dtype == x.dtype
        out.sum().backward()
        assert x.grad is not None and x.grad.abs().sum() > 0

    def test_does_not_wrap_nn_dropout(self):
        drop = Dropout(0.5).train()

        assert drop(torch.ones(10, 10)) is not None
        assert not any(isinstance(m, nn.Dropout) for m in drop.modules())

    @pytest.mark.parametrize("p", [0.1, 0.5, 0.9])
    def test_switching_back_to_eval_restores_the_identity(self, p):
        drop = Dropout(p)
        x = torch.randn(100, 10)

        drop.train()
        drop(x)
        drop.eval()

        assert torch.equal(drop(x), x)
