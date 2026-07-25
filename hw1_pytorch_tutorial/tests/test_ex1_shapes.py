"""Tests for the shape-manipulation helpers of exercise 1."""

import torch

from ex1 import (
    add_singleton_dim,
    flatten_from_dim,
    make_contiguous,
    permute_bhwc_to_bchw,
    remove_singleton_dims,
    reshape_tensor,
    transpose_last_two,
    view_tensor,
)


class TestReshapeTensor:
    def test_reshapes_in_row_major_order(self):
        y = reshape_tensor(torch.arange(6), (2, 3))

        assert y.shape == (2, 3)
        assert torch.equal(y, torch.tensor([[0, 1, 2], [3, 4, 5]]))

    def test_works_on_non_contiguous_input_by_copying(self):
        x = torch.arange(12).reshape(3, 4)[:, ::2]  # non-contiguous view
        assert not x.is_contiguous()

        y = reshape_tensor(x, (6,))

        assert torch.equal(y, torch.tensor([0, 2, 4, 6, 8, 10]))


class TestViewTensor:
    def test_shares_storage_when_input_is_contiguous(self):
        x = torch.arange(6)

        y = view_tensor(x, (2, 3))

        assert y.shape == (2, 3)
        assert y.data_ptr() == x.data_ptr()

    def test_handles_non_contiguous_input(self):
        x = torch.arange(12).reshape(3, 4)[:, ::2]
        assert not x.is_contiguous()

        y = view_tensor(x, (3, 2))

        assert torch.equal(y, torch.tensor([[0, 2], [4, 6], [8, 10]]))


class TestFlattenFromDim:
    def test_flattens_trailing_dims_only(self):
        x = torch.randn(2, 3, 4)

        flat = flatten_from_dim(x, start_dim=1)

        assert flat.shape == (2, 12)
        assert torch.equal(flat, x.reshape(2, 12))

    def test_start_dim_zero_flattens_everything(self):
        x = torch.randn(2, 3, 4)

        flat = flatten_from_dim(x)

        assert flat.shape == (24,)


class TestAddSingletonDim:
    def test_inserts_size_one_dim_at_position(self):
        x = torch.randn(5, 7)

        y = add_singleton_dim(x, dim=1)

        assert y.shape == (5, 1, 7)
        assert torch.equal(y.squeeze(1), x)

    def test_supports_negative_dim(self):
        x = torch.randn(5, 7)

        y = add_singleton_dim(x, dim=-1)

        assert y.shape == (5, 7, 1)


class TestRemoveSingletonDims:
    def test_removes_all_singleton_dims_when_dim_is_none(self):
        x = torch.randn(2, 1, 3, 1)

        y = remove_singleton_dims(x)

        assert y.shape == (2, 3)

    def test_removes_only_the_requested_dim(self):
        x = torch.randn(1, 1, 3)

        y = remove_singleton_dims(x, dim=1)

        assert y.shape == (1, 3)

    def test_dim_zero_removes_only_the_first_dim(self):
        """`dim=0` must behave like any other dim and leave dim 1 alone."""
        x = torch.randn(1, 1, 3)

        y = remove_singleton_dims(x, dim=0)

        assert y.shape == (1, 3)


class TestTransposeLastTwo:
    def test_swaps_last_two_dims_and_keeps_leading_ones(self):
        x = torch.randn(2, 3, 4)

        xt = transpose_last_two(x)

        assert xt.shape == (2, 4, 3)
        assert torch.equal(xt[0, 1, 2], x[0, 2, 1])

    def test_is_an_involution(self):
        x = torch.randn(2, 3, 4)

        assert torch.equal(transpose_last_two(transpose_last_two(x)), x)


class TestPermuteBhwcToBchw:
    def test_moves_channels_first(self):
        x = torch.randn(8, 32, 16, 3)  # B, H, W, C with distinct H and W

        y = permute_bhwc_to_bchw(x)

        assert y.shape == (8, 3, 32, 16)

    def test_preserves_element_values(self):
        x = torch.randn(2, 4, 5, 3)

        y = permute_bhwc_to_bchw(x)

        assert torch.equal(y[1, 2, 3, 4], x[1, 3, 4, 2])


class TestMakeContiguous:
    def test_returns_the_same_object_when_already_contiguous(self):
        x = torch.randn(4, 6)

        assert make_contiguous(x) is x

    def test_copies_non_contiguous_input_preserving_values(self):
        x = torch.randn(4, 6)[:, ::2]
        assert not x.is_contiguous()

        xc = make_contiguous(x)

        assert xc.is_contiguous()
        assert torch.equal(xc, x)
        assert xc.data_ptr() != x.data_ptr()
