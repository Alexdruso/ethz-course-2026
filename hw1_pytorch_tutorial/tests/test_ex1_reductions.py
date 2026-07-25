"""Tests for the broadcasting / reduction helpers of exercise 1."""

import torch

from ex1 import (
    argmax_over_dim,
    broadcast_add_vector,
    max_over_dim,
    mean_over_dim,
    sum_over_dim,
)


class TestSumOverDim:
    def test_reduces_the_requested_dim(self):
        x = torch.ones(2, 3)

        y = sum_over_dim(x, dim=1)

        assert y.shape == (2,)
        assert torch.equal(y, torch.tensor([3.0, 3.0]))

    def test_keepdim_keeps_a_size_one_dim(self):
        x = torch.ones(2, 3, 4)

        y = sum_over_dim(x, dim=1, keepdim=True)

        assert y.shape == (2, 1, 4)
        assert (x - y / 3).abs().max().item() == 0.0  # broadcasts back onto x


class TestMeanOverDim:
    def test_averages_over_the_requested_dim(self):
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

        y = mean_over_dim(x, dim=0)

        assert y.shape == (2,)
        assert torch.allclose(y, torch.tensor([2.0, 3.0]))

    def test_keepdim_allows_centering_by_broadcasting(self):
        x = torch.randn(2, 4, 3)

        m = mean_over_dim(x, dim=1, keepdim=True)
        centered = x - m

        assert m.shape == (2, 1, 3)
        assert torch.allclose(
            centered.mean(dim=1), torch.zeros(2, 3), atol=1e-6
        )


class TestMaxOverDim:
    def test_returns_values_and_indices(self):
        x = torch.tensor([[1.0, 5.0], [3.0, 2.0]])

        values, idx = max_over_dim(x, dim=1)

        assert torch.equal(values, torch.tensor([5.0, 3.0]))
        assert torch.equal(idx, torch.tensor([1, 0]))

    def test_reduces_the_correct_dim_on_a_non_square_input(self):
        x = torch.randn(2, 4, 3)

        values, idx = max_over_dim(x, dim=1)

        assert values.shape == (2, 3)
        assert idx.shape == (2, 3)
        assert torch.equal(values, x.gather(1, idx.unsqueeze(1)).squeeze(1))


class TestArgmaxOverDim:
    def test_returns_index_of_the_largest_entry(self):
        x = torch.tensor([[1.0, 5.0], [3.0, 2.0]])

        idx = argmax_over_dim(x, dim=1)

        assert idx.dtype == torch.int64
        assert torch.equal(idx, torch.tensor([1, 0]))

    def test_agrees_with_max_over_dim_on_a_non_square_input(self):
        x = torch.randn(2, 4, 3)

        idx = argmax_over_dim(x, dim=2)
        _, max_idx = max_over_dim(x, dim=2)

        assert idx.shape == (2, 4)
        assert torch.equal(idx, max_idx)


class TestBroadcastAddVector:
    def test_adds_the_vector_to_every_row(self):
        x = torch.zeros(3, 2)
        v = torch.tensor([10.0, 20.0])

        y = broadcast_add_vector(x, v)

        assert y.shape == (3, 2)
        assert torch.equal(y, torch.tensor([[10.0, 20.0]]).expand(3, 2))

    def test_broadcasts_along_the_trailing_dim_only(self):
        x = torch.arange(6, dtype=torch.float32).reshape(3, 2)
        v = torch.tensor([1.0, -1.0])

        y = broadcast_add_vector(x, v)

        assert torch.equal(y, torch.tensor([[1.0, 0.0], [3.0, 2.0], [5.0, 4.0]]))
