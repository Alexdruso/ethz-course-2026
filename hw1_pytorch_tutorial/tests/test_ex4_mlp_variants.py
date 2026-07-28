"""Tests for the MLP variants of exercise 4: `FeedForward` and `GLUFeedForward`.

Both are still `TODO` in `src/ex4.py`; these tests are the spec.

Assumed contract (from the notebook text and Shazeer 2020):
- `FeedForward(d_model, d_ff, dropout)` is
  `Linear(d_model->d_ff) -> GELU -> Dropout -> Linear(d_ff->d_model) -> Dropout`.
- `GLUFeedForward(d_model, d_ff_gated, dropout, variant)` computes two
  projections to width `d_ff_gated`, applies the variant's activation to one of
  them, multiplies element-wise, and projects back:
  `W2( act(x W) * (x V) )`. The variants are the lowercase strings
  `"geglu"` (GELU gate) and `"swiglu"` (SiLU/Swish gate) - both named in the
  notebook - and an unknown variant raises.
- The caller applies Shazeer's 2/3 width rule, so `GLUFeedForward` uses
  `d_ff_gated` verbatim (see `TestTinyViT` for the parameter-parity check).
"""

import pytest
import torch
from nn_helpers import (
    count_parameters,
    count_projections,
    make_all_projections_identity,
    projection_shapes,
)
from torch.nn import functional as F

from ex4 import FeedForward, GLUFeedForward

VARIANTS = ["geglu", "swiglu"]
GATE = {"geglu": F.gelu, "swiglu": F.silu}


class TestFeedForward:
    def test_preserves_the_model_dimension(self):
        ff = FeedForward(d_model=8, d_ff=32, dropout=0.0).eval()

        out = ff(torch.randn(3, 5, 8))

        assert out.shape == (3, 5, 8)

    def test_uses_two_projections_of_the_requested_inner_width(self):
        ff = FeedForward(d_model=8, d_ff=32, dropout=0.0)

        assert projection_shapes(ff) == sorted([(32, 8), (8, 32)])
        assert count_projections(ff) == 2

    def test_the_activation_between_the_projections_is_gelu(self):
        ff = FeedForward(d_model=6, d_ff=6, dropout=0.0).eval()
        make_all_projections_identity(ff)
        x = torch.randn(4, 3, 6) * 2

        out = ff(x)

        assert torch.allclose(out, F.gelu(x), atol=1e-5)
        assert not torch.allclose(out, F.relu(x), atol=1e-3)

    def test_is_applied_independently_to_each_token(self):
        ff = FeedForward(d_model=6, d_ff=12, dropout=0.0).eval()
        x = torch.randn(2, 7, 6)

        out = ff(x)

        assert torch.allclose(out[1, 4], ff(x[1, 4]), atol=1e-5)

    def test_dropout_is_active_in_train_mode_only(self):
        ff = FeedForward(d_model=8, d_ff=32, dropout=0.5)
        x = torch.randn(16, 8)

        ff.train()
        assert not torch.allclose(ff(x), ff(x), atol=1e-6)

        ff.eval()
        assert torch.allclose(ff(x), ff(x), atol=1e-6)

    def test_zero_dropout_makes_train_and_eval_agree(self):
        ff = FeedForward(d_model=8, d_ff=32, dropout=0.0)
        x = torch.randn(4, 8)

        ff.eval()
        reference = ff(x)
        ff.train()

        assert torch.allclose(ff(x), reference, atol=1e-6)

    def test_gradients_reach_every_parameter(self):
        ff = FeedForward(d_model=6, d_ff=12, dropout=0.0)

        ff(torch.randn(4, 6)).pow(2).sum().backward()

        assert all(p.grad is not None for p in ff.parameters())


class TestGLUFeedForward:
    @pytest.mark.parametrize("variant", VARIANTS)
    def test_preserves_the_model_dimension(self, variant):
        mlp = GLUFeedForward(
            d_model=8, d_ff_gated=20, dropout=0.0, variant=variant
        ).eval()

        out = mlp(torch.randn(3, 5, 8))

        assert out.shape == (3, 5, 8)

    @pytest.mark.parametrize("variant", VARIANTS)
    def test_uses_three_projections_two_up_and_one_down(self, variant):
        mlp = GLUFeedForward(d_model=8, d_ff_gated=20, dropout=0.0, variant=variant)

        assert projection_shapes(mlp) == sorted([(20, 8), (20, 8), (8, 20)])

    @pytest.mark.parametrize("variant", VARIANTS)
    def test_gates_multiplicatively_with_the_variants_activation(self, variant):
        # With every projection forced to the identity the block collapses to
        # `act(x) * x`, which pins down both the gating and the activation.
        mlp = GLUFeedForward(
            d_model=6, d_ff_gated=6, dropout=0.0, variant=variant
        ).eval()
        make_all_projections_identity(mlp)
        x = torch.randn(4, 3, 6) * 2

        out = mlp(x)

        assert torch.allclose(out, GATE[variant](x) * x, atol=1e-5)

    def test_the_two_variants_do_not_compute_the_same_thing(self):
        torch.manual_seed(0)
        geglu = GLUFeedForward(8, 20, 0.0, "geglu").eval()
        torch.manual_seed(0)
        swiglu = GLUFeedForward(8, 20, 0.0, "swiglu").eval()
        x = torch.randn(4, 8) * 3

        assert not torch.allclose(geglu(x), swiglu(x), atol=1e-4)

    def test_an_unknown_variant_is_rejected(self):
        GLUFeedForward(d_model=8, d_ff_gated=20, dropout=0.0, variant="geglu")

        with pytest.raises(Exception) as excinfo:
            GLUFeedForward(
                d_model=8, d_ff_gated=20, dropout=0.0, variant="not-a-variant"
            )

        assert not isinstance(excinfo.value, NotImplementedError), (
            "reject the unknown variant explicitly, e.g. with a ValueError"
        )

    @pytest.mark.parametrize("variant", VARIANTS)
    def test_is_applied_independently_to_each_token(self, variant):
        mlp = GLUFeedForward(6, 16, 0.0, variant).eval()
        x = torch.randn(2, 7, 6)

        out = mlp(x)

        assert torch.allclose(out[1, 4], mlp(x[1, 4]), atol=1e-5)

    @pytest.mark.parametrize("variant", VARIANTS)
    def test_dropout_is_active_in_train_mode_only(self, variant):
        mlp = GLUFeedForward(8, 20, 0.5, variant)
        x = torch.randn(16, 8)

        mlp.train()
        assert not torch.allclose(mlp(x), mlp(x), atol=1e-6)

        mlp.eval()
        assert torch.allclose(mlp(x), mlp(x), atol=1e-6)

    @pytest.mark.parametrize("variant", VARIANTS)
    def test_gradients_reach_every_parameter(self, variant):
        mlp = GLUFeedForward(6, 12, 0.0, variant)

        mlp(torch.randn(4, 6)).pow(2).sum().backward()

        assert all(p.grad is not None for p in mlp.parameters())

    def test_two_thirds_width_matches_the_baseline_parameter_count(self):
        # Shazeer's rule: a gated MLP at 2/3 the inner width costs the same as
        # the baseline FFN (3 * 2/3 == 2 matrices' worth).
        d_model, d_ff = 64, 256
        baseline = FeedForward(d_model, d_ff, dropout=0.0)
        gated = GLUFeedForward(d_model, int(2 * d_ff / 3), dropout=0.0, variant="geglu")

        ratio = count_parameters(gated) / count_parameters(baseline)

        assert 0.9 < ratio < 1.1
