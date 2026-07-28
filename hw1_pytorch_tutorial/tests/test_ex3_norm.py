"""Tests for the normalization layers of exercise 3: `LayerNorm` and `RMSNorm`.

Both are still `TODO` in `src/ex3.py`; these tests are the spec.

What the exercise fixes:
- `LayerNorm` normalizes over the **last** dimension using the **biased**
  variance (divide by `D`, not `D - 1`), then applies a learnable per-feature
  scale and shift, both of shape `(D,)` and initialised to ones/zeros. That is
  exactly `F.layer_norm(x, (D,), weight, bias, eps)`.
- `RMSNorm` is `x * rsqrt(mean(x^2) + eps) * weight`, with `eps` **inside** the
  square root, a scale of shape `(D,)` initialised to ones, and **no** shift
  and **no** mean subtraction.

Nothing here reads `.weight` or `.bias`. Where a test needs the learned scale
and shift it recovers them from the layer's own outputs (see
`nn_helpers.recover_affine`): `norm(0)` is the shift, and dividing the
remainder by a normalized probe gives the scale. That makes the tests care
about what the layer *computes*, not what it calls things.
"""

import torch
from nn_helpers import (
    projection_shapes,
    randomize_parameters,
    recover_affine,
    vector_shapes,
)
from torch import nn
from torch.nn import functional as F

from ex3 import LayerNorm, RMSNorm

D = 6


def _only_parameter(module: nn.Module) -> nn.Parameter:
    params = list(module.parameters())
    assert len(params) == 1, f"expected exactly one parameter, found {len(params)}"
    return params[0]


class TestLayerNorm:
    def test_has_a_learnable_scale_and_shift_of_the_feature_width(self):
        norm = LayerNorm(D)

        assert vector_shapes(norm) == [(D,), (D,)]
        assert projection_shapes(norm) == []  # normalization owns no matrices
        assert all(
            isinstance(p, nn.Parameter) and p.requires_grad for p in norm.parameters()
        )

    def test_matches_torch_layer_norm_at_initialisation(self):
        # Defaults are scale one and shift zero, so a fresh layer must agree
        # with the reference exactly - this pins both the formula and the init.
        norm = LayerNorm(D)
        x = torch.randn(4, D) * 3 + 2

        out = norm(x)

        assert torch.allclose(out, F.layer_norm(x, (D,), eps=1e-5), atol=1e-6)

    def test_matches_torch_layer_norm_with_learned_scale_and_shift(self):
        norm = LayerNorm(D)
        randomize_parameters(norm)
        scale, shift = recover_affine(norm, D, eps=1e-5)

        # Both parameters moved off their defaults, so a layer that ignored one
        # of them would show up here as a scale of ones or a shift of zeros.
        assert not torch.allclose(scale, torch.ones(D), atol=1e-3)
        assert not torch.allclose(shift, torch.zeros(D), atol=1e-3)

        x = torch.randn(3, 5, D)
        out = norm(x)

        assert out.shape == x.shape
        assert torch.allclose(
            out, F.layer_norm(x, (D,), scale, shift, eps=1e-5), atol=1e-5
        )

    def test_uses_the_biased_variance(self):
        # Recovering the affine parameters cannot catch this: an unbiased
        # variance just rescales them by a constant. Only a hand-computed value
        # pins the convention down.
        # For [0, 1, 2, 3]: biased var = 1.25, unbiased (Bessel) var = 1.6667.
        # The first output element is -1.5/sqrt(var + eps), which differs
        # clearly between the two conventions (-1.3416 vs -1.1619).
        norm = LayerNorm(4)
        x = torch.tensor([[0.0, 1.0, 2.0, 3.0]])

        out = norm(x)

        biased = -1.5 / (1.25 + 1e-5) ** 0.5
        assert abs(out[0, 0].item() - biased) < 1e-4

    def test_normalizes_each_row_independently(self):
        # The two halves of the batch live on very different scales; each row
        # must be centred using only its own statistics.
        norm = LayerNorm(D)
        x = torch.randn(2, 3, D) * torch.tensor([[[10.0]], [[0.1]]])

        out = norm(x)

        assert torch.allclose(out.mean(dim=-1), torch.zeros(2, 3), atol=1e-5)
        for i in range(2):
            for j in range(3):
                assert torch.allclose(out[i, j], norm(x[i, j]), atol=1e-6)
        # eps is negligible next to the large-scale rows, so those reach unit variance.
        assert torch.allclose(
            out[0].var(dim=-1, unbiased=False), torch.ones(3), atol=1e-4
        )

    def test_eps_is_used_and_keeps_constant_inputs_finite(self):
        norm = LayerNorm(D, eps=1e-2)
        x = torch.randn(4, D)

        assert torch.allclose(norm(x), F.layer_norm(x, (D,), eps=1e-2), atol=1e-6)
        constant = torch.full((2, D), 5.0)
        assert torch.isfinite(norm(constant)).all()

    def test_gradients_reach_both_parameters(self):
        norm = LayerNorm(D)

        norm(torch.randn(4, D)).pow(2).sum().backward()

        assert all(p.grad is not None for p in norm.parameters())
        assert all(p.grad.abs().sum() > 0 for p in norm.parameters())

    def test_does_not_wrap_nn_layernorm(self):
        norm = LayerNorm(D)

        assert norm(torch.randn(2, D)).shape == (2, D)
        assert not any(isinstance(m, nn.LayerNorm) for m in norm.modules())


class TestRMSNorm:
    def test_has_a_single_learnable_scale_and_no_shift(self):
        norm = RMSNorm(D)

        assert vector_shapes(norm) == [(D,)]
        assert projection_shapes(norm) == []
        assert isinstance(_only_parameter(norm), nn.Parameter)
        # A shift would survive an all-zero input; RMSNorm has none.
        assert torch.equal(norm(torch.zeros(2, D)), torch.zeros(2, D))

    def test_matches_the_closed_form_rms_formula(self):
        # The scale defaults to ones, so a fresh layer is the bare formula.
        norm = RMSNorm(D)
        x = torch.randn(4, 3, D) * 2 + 1

        out = norm(x)
        expected = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-8)

        assert out.shape == x.shape
        assert torch.allclose(out, expected, atol=1e-6)

    def test_applies_the_learnable_scale(self):
        norm = RMSNorm(D)
        scale = _only_parameter(norm)
        with torch.no_grad():
            scale.copy_(torch.arange(1, D + 1, dtype=torch.float32))
        x = torch.randn(5, D)

        out = norm(x)
        expected = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-8) * scale

        assert torch.allclose(out, expected, atol=1e-6)

    def test_matches_torch_rmsnorm(self):
        norm = RMSNorm(D)
        randomize_parameters(norm)
        scale, shift = recover_affine(norm, D, eps=1e-8, subtract_mean=False)

        assert not torch.allclose(scale, torch.ones(D), atol=1e-3)
        assert torch.allclose(shift, torch.zeros(D), atol=1e-6)

        reference = nn.RMSNorm(D, eps=1e-8)
        with torch.no_grad():
            reference.weight.copy_(scale)
        x = torch.randn(3, D) * 4

        assert torch.allclose(norm(x), reference(x), atol=1e-5)

    def test_does_not_subtract_the_mean(self):
        # A constant row has zero variance, so LayerNorm would map it to 0.
        # RMSNorm divides by its RMS instead and leaves the sign and unit scale.
        norm = RMSNorm(D)
        x = torch.full((1, D), 7.0)

        out = norm(x)

        assert torch.allclose(out, torch.ones(1, D), atol=1e-4)

    def test_output_has_unit_root_mean_square(self):
        norm = RMSNorm(D)
        x = torch.randn(4, D) * 100

        out = norm(x)

        assert torch.allclose(out.pow(2).mean(dim=-1), torch.ones(4), atol=1e-4)

    def test_custom_eps_is_respected(self):
        norm = RMSNorm(D, eps=0.5)
        x = torch.randn(3, D)

        expected = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + 0.5)

        assert torch.allclose(norm(x), expected, atol=1e-6)

    def test_gradients_reach_the_scale(self):
        norm = RMSNorm(D)

        norm(torch.randn(4, D)).pow(2).sum().backward()

        assert all(
            p.grad is not None and p.grad.abs().sum() > 0 for p in norm.parameters()
        )

    def test_does_not_wrap_nn_rmsnorm(self):
        norm = RMSNorm(D)

        assert norm(torch.randn(2, D)).shape == (2, D)
        assert not any(isinstance(m, nn.RMSNorm) for m in norm.modules())
