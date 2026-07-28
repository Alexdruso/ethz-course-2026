"""Tests for the encoder block and the ViT of exercise 4.

Covers `TransformerEncoderBlock` and `TinyViT`. Both are still `TODO` in
`src/ex4.py`; these tests are the spec.

Assumed contract (from the notebook text):
- `TransformerEncoderBlock` is **pre-LN**:
  `x = x + Dropout(SelfAttn(LN(x)))` then `x = x + Dropout(MLP(LN(x)))`,
  with the attention being an `nn.MultiheadAttention`. It consumes and returns
  `(B, T, D)`, which means `batch_first=True` - forgetting that flag silently
  swaps the batch and time axes instead of raising.
- `TinyViT` maps `(B, 1, 28, 28)` to `(B, 10)` logits via
  patchify -> patch embed -> positional embed -> blocks -> mean pool -> head.
- `mlp_kind` selects the MLP: `"ffn"` for the GELU baseline, `"geglu"` and
  `"swiglu"` for the gated variants, with the gated ones built at 2/3 of `d_ff`
  so all three cost about the same.

Submodule *names* are yours to choose, so the tests find things by type via
`modules()` - the `nn.MultiheadAttention` the exercise mandates, and the
`TransformerEncoderBlock` it defines - never by attribute name.
"""

import pytest
import torch
from nn_helpers import ZeroModule, count_parameters
from torch import nn

from ex4 import TinyViT, TransformerEncoderBlock

D_MODEL, N_HEADS = 8, 2
B, T = 3, 5  # deliberately different, so a transposed batch/time axis shows up
MLP_KINDS = ["ffn", "geglu", "swiglu"]


def _mlp(d_model: int = D_MODEL) -> nn.Module:
    return nn.Sequential(nn.Linear(d_model, 16), nn.GELU(), nn.Linear(16, d_model))


def _block(dropout: float = 0.0, mlp: nn.Module | None = None) -> nn.Module:
    return TransformerEncoderBlock(
        d_model=D_MODEL,
        n_heads=N_HEADS,
        mlp=mlp if mlp is not None else _mlp(),
        dropout=dropout,
    )


def _attention(block: nn.Module) -> nn.MultiheadAttention:
    attns = [m for m in block.modules() if isinstance(m, nn.MultiheadAttention)]
    assert len(attns) == 1, "expected exactly one nn.MultiheadAttention per block"
    return attns[0]


class TestTransformerEncoderBlock:
    def test_preserves_the_batch_time_feature_shape(self):
        block = _block().eval()

        out = block(torch.randn(B, T, D_MODEL))

        assert out.shape == (B, T, D_MODEL)

    def test_treats_the_first_axis_as_the_batch(self):
        # With batch_first left at its default the block would mix across the
        # batch instead of across tokens, so element 0 would react to element 1.
        block = _block().eval()
        x = torch.randn(B, T, D_MODEL)
        perturbed = x.clone()
        perturbed[1] = torch.randn(T, D_MODEL) * 5

        out, out_perturbed = block(x), block(perturbed)

        assert torch.allclose(out[0], out_perturbed[0], atol=1e-5), (
            "batch elements must not interact - pass batch_first=True"
        )
        assert not torch.allclose(out[1], out_perturbed[1], atol=1e-3)

    def test_attention_mixes_information_across_tokens(self):
        # Note: the perturbation must change the *shape* of the token, not just
        # shift it - a constant offset is removed again by the pre-LN.
        block = _block().eval()
        x = torch.randn(1, T, D_MODEL)
        perturbed = x.clone()
        perturbed[0, 0] = torch.randn(D_MODEL) * 5

        out, out_perturbed = block(x), block(perturbed)

        assert not torch.allclose(out[0, 1], out_perturbed[0, 1], atol=1e-4), (
            "changing one token must affect the others via self-attention"
        )

    def test_is_pre_ln_with_residuals_around_both_sublayers(self):
        # Zero out both sublayer outputs. A pre-LN block then returns x exactly;
        # a post-LN block would return a normalized version of x instead.
        block = _block(mlp=ZeroModule(D_MODEL)).eval()
        attn = _attention(block)
        with torch.no_grad():
            attn.out_proj.weight.zero_()
            if attn.out_proj.bias is not None:
                attn.out_proj.bias.zero_()
        x = torch.randn(B, T, D_MODEL) * 5 + 3

        out = block(x)

        assert torch.allclose(out, x, atol=1e-5), (
            "both sublayers must sit behind residual connections, with the "
            "LayerNorm inside the branch (pre-LN)"
        )

    def test_registers_the_mlp_it_was_given(self):
        mlp = _mlp()
        block = _block(mlp=mlp)

        assert any(m is mlp for m in block.modules())
        params = {id(p) for p in block.parameters()}
        assert all(id(p) in params for p in mlp.parameters())

    def test_dropout_is_active_in_train_mode_only(self):
        block = _block(dropout=0.5)
        x = torch.randn(B, T, D_MODEL)

        block.train()
        assert not torch.allclose(block(x), block(x), atol=1e-6)

        block.eval()
        assert torch.allclose(block(x), block(x), atol=1e-6)

    def test_gradients_reach_the_attention_and_the_mlp(self):
        mlp = _mlp()
        block = _block(mlp=mlp)

        block(torch.randn(B, T, D_MODEL)).pow(2).sum().backward()

        assert all(p.grad is not None for p in mlp.parameters())
        assert _attention(block).out_proj.weight.grad is not None


class TestTinyViT:
    @staticmethod
    def _model(
        mlp_kind: str = "ffn", patch_size: int = 4, n_layers: int = 2
    ) -> nn.Module:
        return TinyViT(
            patch_size=patch_size,
            d_model=16,
            n_heads=4,
            n_layers=n_layers,
            d_ff=32,
            dropout=0.0,
            mlp_kind=mlp_kind,
        )

    @pytest.mark.parametrize("mlp_kind", MLP_KINDS)
    def test_maps_mnist_images_to_ten_logits(self, mlp_kind):
        model = self._model(mlp_kind).eval()

        logits = model(torch.rand(4, 1, 28, 28))

        assert logits is not None, "TinyViT.forward still returns None"
        assert logits.shape == (4, 10)
        assert torch.isfinite(logits).all()

    @pytest.mark.parametrize("patch_size", [4, 7, 14, 28])
    def test_works_for_every_patch_size_that_divides_28(self, patch_size):
        model = self._model(patch_size=patch_size).eval()

        assert model(torch.rand(2, 1, 28, 28)).shape == (2, 10)

    def test_rejects_a_patch_size_that_does_not_divide_28(self):
        self._model(patch_size=4)  # a valid grid still builds

        with pytest.raises(AssertionError):
            self._model(patch_size=5)

    def test_builds_one_encoder_block_per_layer(self):
        for n_layers in (1, 3):
            model = self._model(n_layers=n_layers)
            blocks = [
                m for m in model.modules() if isinstance(m, TransformerEncoderBlock)
            ]
            assert len(blocks) == n_layers

    def test_batch_elements_are_classified_independently(self):
        model = self._model().eval()
        x = torch.rand(4, 1, 28, 28)
        perturbed = x.clone()
        perturbed[2] = torch.rand(1, 28, 28)

        logits, perturbed_logits = model(x), model(perturbed)

        assert torch.allclose(logits[0], perturbed_logits[0], atol=1e-5)
        assert not torch.allclose(logits[2], perturbed_logits[2], atol=1e-4)

    def test_positional_information_makes_patch_order_matter(self):
        # Swap two 4x4 patch blocks of the image. Without positional embeddings
        # (and with mean pooling) the logits would be unchanged.
        model = self._model().eval()
        x = torch.rand(2, 1, 28, 28)
        swapped = x.clone()
        swapped[:, :, 0:4, 0:4] = x[:, :, 8:12, 20:24]
        swapped[:, :, 8:12, 20:24] = x[:, :, 0:4, 0:4]

        assert not torch.allclose(model(x), model(swapped), atol=1e-4)

    @pytest.mark.parametrize("mlp_kind", MLP_KINDS)
    def test_gradients_reach_every_parameter(self, mlp_kind):
        model = self._model(mlp_kind)

        model(torch.rand(2, 1, 28, 28)).pow(2).sum().backward()

        missing = [n for n, p in model.named_parameters() if p.grad is None]
        assert not missing, f"no gradient for {missing}"

    def test_the_variants_cost_roughly_the_same_via_the_two_thirds_rule(self):
        counts = {kind: count_parameters(self._model(kind)) for kind in MLP_KINDS}

        baseline = counts["ffn"]
        for kind, n in counts.items():
            assert 0.9 < n / baseline < 1.1, f"{kind} has {n} params vs {baseline}"

    def test_the_variants_are_actually_different_models(self):
        torch.manual_seed(0)
        ffn = self._model("ffn").eval()
        torch.manual_seed(0)
        geglu = self._model("geglu").eval()
        x = torch.rand(2, 1, 28, 28)

        assert not torch.allclose(ffn(x), geglu(x), atol=1e-4)

    def test_dropout_is_wired_through_to_the_blocks(self):
        model = TinyViT(
            patch_size=4,
            d_model=16,
            n_heads=4,
            n_layers=2,
            d_ff=32,
            dropout=0.5,
            mlp_kind="ffn",
        )
        x = torch.rand(4, 1, 28, 28)

        model.train()
        assert not torch.allclose(model(x), model(x), atol=1e-6)

        model.eval()
        assert torch.allclose(model(x), model(x), atol=1e-6)
