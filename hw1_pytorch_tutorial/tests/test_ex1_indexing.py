"""Tests for the indexing helpers of exercise 1."""

import torch

from ex1 import gather_rows, get_diagonal, select_columns, set_subtensor, slice_rows


class TestSliceRows:
    def test_returns_the_requested_row_range(self):
        x = torch.arange(12).reshape(4, 3)

        rows = slice_rows(x, 1, 3)

        assert rows.shape == (2, 3)
        assert torch.equal(rows, torch.tensor([[3, 4, 5], [6, 7, 8]]))

    def test_basic_slicing_returns_a_view(self):
        x = torch.arange(12).reshape(4, 3)

        rows = slice_rows(x, 1, 3)
        rows[0, 0] = -1

        assert x[1, 0].item() == -1


class TestSelectColumns:
    def test_selects_columns_in_the_requested_order(self):
        x = torch.arange(12).reshape(4, 3)

        cols = select_columns(x, [2, 0])

        assert cols.shape == (4, 2)
        assert torch.equal(cols, torch.tensor([[2, 0], [5, 3], [8, 6], [11, 9]]))

    def test_fancy_indexing_returns_a_copy(self):
        x = torch.arange(12).reshape(4, 3)

        cols = select_columns(x, [0, 2])
        cols[0, 0] = -1

        assert x[0, 0].item() == 0


class TestGetDiagonal:
    def test_extracts_the_main_diagonal(self):
        d = get_diagonal(torch.tensor([[1, 2], [3, 4]]))

        assert torch.equal(d, torch.tensor([1, 4]))

    def test_works_on_a_non_square_matrix(self):
        x = torch.arange(12).reshape(3, 4)

        d = get_diagonal(x)

        assert torch.equal(d, torch.tensor([0, 5, 10]))


class TestSetSubtensor:
    def test_sets_the_requested_entry(self):
        out = set_subtensor(torch.zeros(2, 2), 0, 1, 5.0)

        assert torch.equal(out, torch.tensor([[0.0, 5.0], [0.0, 0.0]]))

    def test_leaves_the_input_unchanged(self):
        base = torch.zeros(2, 2)

        out = set_subtensor(base, 0, 1, 5.0)

        assert torch.equal(base, torch.zeros(2, 2))
        assert out.data_ptr() != base.data_ptr()


class TestGatherRows:
    def test_gathers_rows_in_index_order(self):
        x = torch.tensor([[10, 11, 12], [20, 21, 22], [30, 31, 32], [40, 41, 42]])
        idx = torch.tensor([2, 0])

        gathered = gather_rows(x, idx)

        assert gathered.shape == (2, 3)
        assert torch.equal(gathered, torch.tensor([[30, 31, 32], [10, 11, 12]]))

    def test_supports_repeated_indices(self):
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        idx = torch.tensor([1, 1, 0])

        gathered = gather_rows(x, idx)

        assert gathered.shape == (3, 2)
        assert torch.equal(
            gathered, torch.tensor([[3.0, 4.0], [3.0, 4.0], [1.0, 2.0]])
        )
