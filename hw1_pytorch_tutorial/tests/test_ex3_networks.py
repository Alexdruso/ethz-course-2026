"""Tests for the composed networks of exercise 3: `MLP`, `FeedForward`, `Residual`.

All three are still `TODO` in `src/ex3.py`; these tests are the spec.

Assumed contract (from the notebook text):
- `depth` is the **total number of Linear layers**, so `depth=1` is a single
  `in_dim -> out_dim` map that ignores `hidden_dim`, and `depth=n` is
  `in -> hidden -> ... -> hidden -> out` with `n - 1` activations in between.
- The activation is **GELU** and there is **no** activation after the last
  Linear (the network must be able to emit unbounded logits).
- `use_layernorm=True` inserts normalization between layers; `False` inserts none.
- `FeedForward` is `Linear(d_model -> d_ff) -> GELU -> Linear(d_ff -> d_model)`
  with `d_ff` defaulting to `4 * d_model`.
- `Residual(fn)` makes `fn`'s parameters part of the wrapper and returns
  `x + fn(x, ...)`.

How the layers are named, nested or ordered is up to you, so these tests
never look for a submodule by name. Layer *counts* come from counting matrices
(`p.dim() == 2` over `parameters()`), and the wiring is pinned by forcing every
projection to the identity, which collapses the network to a closed form.
"""

import pytest
import torch
from nn_helpers import (
    assert_is_affine,
    count_parameters,
    count_projections,
    make_all_projections_identity,
    projection_shapes,
)
from torch import nn
from torch.nn import functional as F

from ex3 import MLP, FeedForward, Linear, Residual


class TestMLP:
    @pytest.mark.parametrize("depth", [1, 2, 3, 4])
    def test_maps_in_dim_to_out_dim_for_every_depth(self, depth):
        model = MLP(in_dim=5, hidden_dim=8, out_dim=3, depth=depth)

        out = model(torch.randn(4, 5))

        assert out.shape == (4, 3)

    def test_supports_leading_batch_dimensions(self):
        model = MLP(in_dim=5, hidden_dim=8, out_dim=3, depth=3).eval()
        x = torch.randn(2, 7, 5)

        out = model(x)

        assert out.shape == (2, 7, 3)
        assert torch.allclose(out[1, 4], model(x[1, 4]), atol=1e-5)

    @pytest.mark.parametrize("depth", [1, 2, 3, 4])
    def test_depth_is_the_number_of_linear_layers(self, depth):
        model = MLP(in_dim=5, hidden_dim=8, out_dim=3, depth=depth)

        assert count_projections(model) == depth

    def test_depth_one_is_a_single_affine_map_that_ignores_hidden_dim(self):
        narrow = MLP(in_dim=4, hidden_dim=8, out_dim=3, depth=1)
        wide = MLP(in_dim=4, hidden_dim=512, out_dim=3, depth=1)

        assert projection_shapes(narrow) == [(3, 4)]
        assert projection_shapes(wide) == projection_shapes(narrow)
        assert_is_affine(narrow, torch.randn(4, 4))

    def test_hidden_widths_chain_in_to_hidden_to_out(self):
        model = MLP(in_dim=5, hidden_dim=8, out_dim=3, depth=3)

        assert projection_shapes(model) == sorted([(8, 5), (8, 8), (3, 8)])

    def test_uses_gelu_between_the_layers(self):
        # With every projection forced to the identity, a depth-2 MLP without
        # normalization collapses to exactly the activation function.
        model = MLP(in_dim=4, hidden_dim=4, out_dim=4, depth=2).eval()
        make_all_projections_identity(model)
        x = torch.randn(6, 4) * 2

        out = model(x)

        assert torch.allclose(out, F.gelu(x), atol=1e-5)
        assert not torch.allclose(out, F.relu(x), atol=1e-3)

    def test_has_no_activation_after_the_last_layer(self):
        # `depth` Linear layers means exactly `depth - 1` activations. Forcing
        # the projections to the identity makes a depth-3 MLP collapse to
        # gelu(gelu(x)); an activation after the final layer would add a third.
        model = MLP(in_dim=4, hidden_dim=4, out_dim=4, depth=3).eval()
        make_all_projections_identity(model)
        x = torch.linspace(-3, 3, 16).reshape(4, 4)

        out = model(x)

        assert torch.allclose(out, F.gelu(F.gelu(x)), atol=1e-5)
        assert not torch.allclose(out, F.gelu(F.gelu(F.gelu(x))), atol=1e-3)

    def test_use_layernorm_adds_parameters_without_adding_layers(self):
        plain = MLP(in_dim=5, hidden_dim=8, out_dim=3, depth=3, use_layernorm=False)
        normed = MLP(in_dim=5, hidden_dim=8, out_dim=3, depth=3, use_layernorm=True)

        # Normalization brings its own scale and shift...
        assert count_parameters(normed) > count_parameters(plain)
        # ...but no extra projections, and the width chain is untouched.
        assert count_projections(normed) == count_projections(plain) == 3
        assert projection_shapes(normed) == projection_shapes(plain)

    def test_use_layernorm_stops_the_input_scale_from_propagating(self):
        # This is what normalization is *for*, and unlike an isinstance check it
        # does not care where you put the norms or what type they are. Without
        # them the output grows with the input; with them it stays bounded.
        def growth(model: nn.Module) -> float:
            x = torch.randn(4, 5)
            return model(1000 * x).abs().max().item() / max(
                model(x).abs().max().item(), 1e-6
            )

        plain = MLP(5, 8, 3, depth=3, use_layernorm=False).eval()
        normed = MLP(5, 8, 3, depth=3, use_layernorm=True).eval()

        assert growth(plain) > 100
        assert growth(normed) < 10

    def test_layernorm_variant_still_produces_the_right_shape_and_differs(self):
        torch.manual_seed(0)
        plain = MLP(5, 8, 3, depth=3, use_layernorm=False).eval()
        torch.manual_seed(0)
        normed = MLP(5, 8, 3, depth=3, use_layernorm=True).eval()
        x = torch.randn(4, 5) * 10

        out = normed(x)

        assert out.shape == (4, 3)
        assert not torch.allclose(out, plain(x), atol=1e-4)

    def test_gradients_reach_every_parameter(self):
        model = MLP(in_dim=5, hidden_dim=8, out_dim=3, depth=3)

        model(torch.randn(4, 5)).pow(2).sum().backward()

        assert all(p.grad is not None for p in model.parameters())
        assert any(p.grad.abs().sum() > 0 for p in model.parameters())


class TestFeedForward:
    def test_default_inner_width_is_four_times_d_model(self):
        ff = FeedForward(d_model=8)

        assert projection_shapes(ff) == sorted([(32, 8), (8, 32)])

    def test_explicit_d_ff_overrides_the_default(self):
        ff = FeedForward(d_model=8, d_ff=5)

        assert projection_shapes(ff) == sorted([(5, 8), (8, 5)])

    def test_preserves_the_feature_dimension(self):
        ff = FeedForward(d_model=6).eval()
        x = torch.randn(3, 7, 6)

        out = ff(x)

        assert out.shape == (3, 7, 6)
        assert torch.allclose(out[2, 5], ff(x[2, 5]), atol=1e-5)

    def test_uses_gelu_between_the_two_projections(self):
        ff = FeedForward(d_model=4, d_ff=4).eval()
        make_all_projections_identity(ff)
        x = torch.randn(8, 4) * 2

        out = ff(x)

        assert torch.allclose(out, F.gelu(x), atol=1e-5)

    def test_uses_exactly_two_dense_layers(self):
        ff = FeedForward(d_model=8)

        assert count_projections(ff) == 2

    def test_gradients_reach_every_parameter(self):
        ff = FeedForward(d_model=6, d_ff=12)

        ff(torch.randn(4, 6)).pow(2).sum().backward()

        assert all(p.grad is not None for p in ff.parameters())


class TestResidual:
    def test_adds_the_wrapped_output_to_the_input(self):
        inner = Linear(4, 4)
        block = Residual(inner)
        x = torch.randn(3, 4)

        out = block(x)

        assert torch.allclose(out, x + inner(x), atol=1e-6)

    def test_registers_the_wrapped_module_so_its_parameters_are_visible(self):
        inner = nn.Linear(4, 4)
        block = Residual(inner)

        params = list(block.parameters())

        assert len(params) == 2
        assert any(p is inner.weight for p in params)
        assert any(m is inner for m in block.modules())

    def test_forwards_extra_positional_and_keyword_arguments(self):
        class Adder(nn.Module):
            def forward(self, x, bump, *, scale=1.0):
                return x * 0 + bump * scale

        block = Residual(Adder())
        x = torch.ones(2, 3)

        out = block(x, torch.full((2, 3), 2.0), scale=3.0)

        assert torch.allclose(out, torch.full((2, 3), 7.0))

    def test_a_zero_valued_branch_leaves_the_input_unchanged(self):
        class Zero(nn.Module):
            def forward(self, x):
                return torch.zeros_like(x)

        x = torch.randn(5, 4)

        assert torch.equal(Residual(Zero())(x), x)

    def test_gradients_flow_through_both_the_skip_and_the_branch(self):
        inner = nn.Linear(4, 4)
        block = Residual(inner)
        x = torch.randn(3, 4, requires_grad=True)

        block(x).sum().backward()

        assert inner.weight.grad is not None and inner.weight.grad.abs().sum() > 0
        assert x.grad is not None and x.grad.abs().sum() > 0
