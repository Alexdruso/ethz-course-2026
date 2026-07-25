"""Tests for the einsum helpers of exercise 1.

All shapes use distinct axis sizes (B=2, T=4, D=3, H=5, Dh=7) so that an axis
mix-up cannot pass by coincidence.
"""

import torch

from shapes import B, D, Dh, H, T

from ex1 import (
    einsum_apply_attention,
    einsum_linear_btd_dh_to_bth,
    einsum_pairwise_dot,
    einsum_qk_scores,
)


class TestEinsumLinear:
    def test_matches_matmul_reference(self):
        x = torch.randn(B, T, D)
        W = torch.randn(D, H)

        y = einsum_linear_btd_dh_to_bth(x, W)

        assert y.shape == (B, T, H)
        assert torch.allclose(y, x @ W, atol=1e-5)

    def test_is_linear_in_the_input(self):
        x1 = torch.randn(B, T, D)
        x2 = torch.randn(B, T, D)
        W = torch.randn(D, H)

        combined = einsum_linear_btd_dh_to_bth(x1 + 2.0 * x2, W)
        expected = einsum_linear_btd_dh_to_bth(x1, W) + 2.0 * einsum_linear_btd_dh_to_bth(x2, W)

        assert torch.allclose(combined, expected, atol=1e-5)


class TestEinsumPairwiseDot:
    def test_matches_elementwise_sum_reference(self):
        x = torch.randn(B, T, D)
        y = torch.randn(B, T, D)

        dots = einsum_pairwise_dot(x, y)

        assert dots.shape == (B, T)
        assert torch.allclose(dots, (x * y).sum(dim=-1), atol=1e-5)

    def test_dot_with_itself_is_the_squared_norm(self):
        x = torch.randn(B, T, D)

        dots = einsum_pairwise_dot(x, x)

        assert torch.allclose(dots, x.pow(2).sum(dim=-1), atol=1e-5)


class TestEinsumQkScores:
    def test_returns_a_t_by_t_score_matrix(self):
        q = torch.randn(B, H, T, Dh)
        k = torch.randn(B, H, T, Dh)

        scores = einsum_qk_scores(q, k)

        assert scores.shape == (B, H, T, T)

    def test_entries_are_query_key_dot_products(self):
        q = torch.randn(B, H, T, Dh)
        k = torch.randn(B, H, T, Dh)

        scores = einsum_qk_scores(q, k)

        assert torch.allclose(scores, q @ k.transpose(-1, -2), atol=1e-5)


class TestEinsumApplyAttention:
    def test_matches_matmul_reference(self):
        weights = torch.randn(B, H, T, T)
        v = torch.randn(B, H, T, Dh)

        out = einsum_apply_attention(weights, v)

        assert out.shape == (B, H, T, Dh)
        assert torch.allclose(out, weights @ v, atol=1e-5)

    def test_one_hot_weights_select_a_single_value_row(self):
        weights = torch.zeros(B, H, T, T)
        weights[..., 0] = 1.0  # every query attends only to position 0
        v = torch.randn(B, H, T, Dh)

        out = einsum_apply_attention(weights, v)

        expected = v[:, :, 0:1, :].expand(B, H, T, Dh)
        assert torch.allclose(out, expected, atol=1e-6)
