"""Tests for the tokenization part of exercise 4.

Covers `patchify`, `PatchEmbed` and `PositionalEmbedding`. All are still `TODO`
in `src/ex4.py`; these tests are the spec.

Assumed contract (from the notebook text and the ViT paper):
- `patchify(x, patch_size)` turns `(B, C, H, W)` into `(B, N, C*P*P)` with
  `N = (H//P) * (W//P)`. Tokens are ordered **row-major over the patch grid**
  (left to right, then top to bottom) and the pixels inside a patch are
  flattened **row-major** as well. All tests use `C = 1` (MNIST), so the
  channel-vs-pixel ordering question never arises.
- `PatchEmbed(patch_dim, d_model)` is a single affine projection applied
  independently to each token: `(B, N, patch_dim) -> (B, N, d_model)`.
- `PositionalEmbedding(num_tokens, d_model)` holds one *learned* parameter of
  `num_tokens * d_model` values (ViT-style) and returns `x + pos`, identical
  for every element of the batch.
"""

import pytest
import torch
from nn_helpers import assert_is_affine
from torch import nn

from ex4 import PatchEmbed, PositionalEmbedding, patchify


def _ramp_image(b: int = 1, size: int = 28) -> torch.Tensor:
    """`x[.., i, j] = 100*i + j`, so every pixel says where it came from."""
    rows = torch.arange(size).view(size, 1) * 100
    cols = torch.arange(size).view(1, size)
    img = (rows + cols).float()
    return img.expand(b, 1, size, size).clone()


class TestPatchify:
    @pytest.mark.parametrize(
        "patch_size, num_tokens, patch_dim",
        [(4, 49, 16), (7, 16, 49), (14, 4, 196), (28, 1, 784)],
    )
    def test_token_count_and_width_follow_the_patch_grid(
        self, patch_size, num_tokens, patch_dim
    ):
        x = torch.randn(3, 1, 28, 28)

        out = patchify(x, patch_size)

        assert out.shape == (3, num_tokens, patch_dim)

    def test_the_first_token_is_the_top_left_patch_flattened_row_major(self):
        x = _ramp_image()

        out = patchify(x, patch_size=4)

        expected = torch.tensor(
            [0, 1, 2, 3, 100, 101, 102, 103, 200, 201, 202, 203, 300, 301, 302, 303],
            dtype=torch.float32,
        )
        assert torch.equal(out[0, 0], expected)

    def test_tokens_walk_the_patch_grid_row_major(self):
        x = _ramp_image()
        p, grid = 4, 7

        out = patchify(x, p)

        for token in range(grid * grid):
            gi, gj = divmod(token, grid)  # row-major over the patch grid
            patch = x[0, 0, gi * p : (gi + 1) * p, gj * p : (gj + 1) * p]
            assert torch.equal(out[0, token], patch.reshape(-1)), f"token {token}"

    def test_a_single_full_size_patch_is_the_flattened_image(self):
        x = torch.randn(2, 1, 28, 28)

        out = patchify(x, patch_size=28)

        assert out.shape == (2, 1, 784)
        assert torch.equal(out, x.reshape(2, 1, 784))

    def test_batch_elements_are_patchified_independently(self):
        x = torch.randn(4, 1, 28, 28)

        out = patchify(x, patch_size=7)

        for b in range(4):
            assert torch.equal(out[b], patchify(x[b : b + 1], 7)[0])

    def test_every_pixel_appears_exactly_once(self):
        x = _ramp_image()

        out = patchify(x, patch_size=4)

        assert torch.equal(out.flatten().sort().values, x.flatten().sort().values)

    def test_preserves_dtype_and_keeps_gradients_flowing(self):
        x = torch.randn(2, 1, 28, 28, dtype=torch.float32, requires_grad=True)

        out = patchify(x, 4)

        assert out.dtype == torch.float32
        out.sum().backward()
        assert x.grad is not None and torch.allclose(x.grad, torch.ones_like(x))


class TestPatchEmbed:
    def test_projects_patch_vectors_to_d_model(self):
        embed = PatchEmbed(patch_dim=16, d_model=8)

        out = embed(torch.randn(3, 49, 16))

        assert out.shape == (3, 49, 8)

    def test_applies_the_same_projection_to_every_token(self):
        embed = PatchEmbed(patch_dim=16, d_model=8).eval()
        x = torch.randn(2, 5, 16)

        out = embed(x)
        permuted = embed(x[:, [4, 3, 2, 1, 0]])

        assert torch.allclose(out[:, [4, 3, 2, 1, 0]], permuted, atol=1e-5)

    def test_is_an_affine_projection(self):
        embed = PatchEmbed(patch_dim=9, d_model=4).eval()

        assert_is_affine(embed, torch.randn(2, 6, 9))

    def test_gradients_reach_its_parameters(self):
        embed = PatchEmbed(patch_dim=16, d_model=8)

        embed(torch.randn(2, 4, 16)).pow(2).sum().backward()

        assert list(embed.parameters())
        assert all(p.grad is not None for p in embed.parameters())


class TestPositionalEmbedding:
    NUM_TOKENS, D_MODEL = 49, 8

    def _pos_parameter(self, module: nn.Module) -> nn.Parameter:
        params = [
            p
            for p in module.parameters()
            if p.numel() == self.NUM_TOKENS * self.D_MODEL
        ]
        assert len(params) == 1, "expected exactly one learned positional table"
        return params[0]

    def test_preserves_the_token_shape(self):
        pos = PositionalEmbedding(self.NUM_TOKENS, self.D_MODEL)

        out = pos(torch.randn(3, self.NUM_TOKENS, self.D_MODEL))

        assert out.shape == (3, self.NUM_TOKENS, self.D_MODEL)

    def test_holds_one_learnable_table_of_num_tokens_by_d_model(self):
        pos = PositionalEmbedding(self.NUM_TOKENS, self.D_MODEL)

        table = self._pos_parameter(pos)

        assert table.requires_grad
        assert table.shape in {
            (self.NUM_TOKENS, self.D_MODEL),
            (1, self.NUM_TOKENS, self.D_MODEL),
        }

    def test_adds_the_table_to_the_input(self):
        pos = PositionalEmbedding(self.NUM_TOKENS, self.D_MODEL)
        table = self._pos_parameter(pos)
        with torch.no_grad():
            table.copy_(
                torch.arange(table.numel(), dtype=torch.float32).reshape(table.shape)
            )
        x = torch.randn(2, self.NUM_TOKENS, self.D_MODEL)

        out = pos(x)

        expected = x + table.detach().reshape(self.NUM_TOKENS, self.D_MODEL)
        assert torch.allclose(out, expected, atol=1e-5)

    def test_each_position_gets_its_own_offset(self):
        pos = PositionalEmbedding(self.NUM_TOKENS, self.D_MODEL)
        table = self._pos_parameter(pos)
        with torch.no_grad():
            table.copy_(
                torch.arange(table.numel(), dtype=torch.float32).reshape(table.shape)
            )
        x = torch.zeros(1, self.NUM_TOKENS, self.D_MODEL)

        offsets = pos(x)[0]

        assert not torch.allclose(offsets[0], offsets[1]), "positions must differ"

    def test_the_same_offset_is_applied_to_every_batch_element(self):
        pos = PositionalEmbedding(self.NUM_TOKENS, self.D_MODEL)
        x = torch.randn(4, self.NUM_TOKENS, self.D_MODEL)

        delta = pos(x) - x

        for b in range(1, 4):
            assert torch.allclose(delta[0], delta[b], atol=1e-6)

    def test_gradients_reach_the_positional_table(self):
        pos = PositionalEmbedding(self.NUM_TOKENS, self.D_MODEL)

        pos(torch.randn(2, self.NUM_TOKENS, self.D_MODEL)).pow(2).sum().backward()

        assert self._pos_parameter(pos).grad is not None
        assert self._pos_parameter(pos).grad.abs().sum() > 0
