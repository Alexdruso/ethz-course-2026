"""Tests for the vectorization helpers of exercise 1."""

import torch

from ex1 import (
    batched_token_histogram,
    concat_tensors,
    cumsum_over_dim,
    expand_tensor,
    masked_mean,
    one_hot,
    repeat_tensor,
    scatter_add_1d,
    stack_tensors,
    where_select,
)


class TestConcatTensors:
    def test_concatenates_along_an_existing_dim(self):
        a = torch.zeros(2, 3)
        b = torch.ones(4, 3)

        out = concat_tensors([a, b], dim=0)

        assert out.shape == (6, 3)
        assert torch.equal(out[:2], a)
        assert torch.equal(out[2:], b)

    def test_concatenating_along_dim_one_keeps_the_batch_dim(self):
        a = torch.zeros(2, 3)
        b = torch.ones(2, 5)

        out = concat_tensors([a, b], dim=1)

        assert out.shape == (2, 8)
        assert out.data_ptr() != a.data_ptr()  # cat always allocates


class TestStackTensors:
    def test_creates_a_new_leading_dim(self):
        tensors = [torch.zeros(2, 3), torch.ones(2, 3), torch.full((2, 3), 2.0)]

        out = stack_tensors(tensors, dim=0)

        assert out.shape == (3, 2, 3)
        assert torch.equal(out[1], torch.ones(2, 3))

    def test_creates_a_new_dim_at_the_requested_position(self):
        tensors = [torch.zeros(2, 3), torch.ones(2, 3)]

        out = stack_tensors(tensors, dim=1)

        assert out.shape == (2, 2, 3)
        assert torch.equal(out[:, 1], torch.ones(2, 3))


class TestRepeatTensor:
    def test_tiles_the_tensor_along_each_dim(self):
        x = torch.tensor([[1, 2]])

        out = repeat_tensor(x, (3, 2))

        assert out.shape == (3, 4)
        assert torch.equal(out[0], torch.tensor([1, 2, 1, 2]))

    def test_result_owns_independent_memory(self):
        x = torch.tensor([[1.0, 2.0]])

        out = repeat_tensor(x, (2, 1))
        out[0, 0] = 99.0

        assert x[0, 0].item() == 1.0
        assert out[1, 0].item() == 1.0  # the other copy is untouched


class TestExpandTensor:
    def test_broadcasts_a_singleton_dim_without_copying(self):
        x = torch.tensor([[1.0, 2.0, 3.0]])  # (1, 3)

        out = expand_tensor(x, 4, -1)

        assert out.shape == (4, 3)
        assert out.data_ptr() == x.data_ptr()

    def test_expanded_rows_share_storage(self):
        x = torch.zeros(1, 3)

        out = expand_tensor(x, 2, 3)
        x[0, 0] = 7.0

        assert out[0, 0].item() == 7.0
        assert out[1, 0].item() == 7.0


class TestCumsumOverDim:
    def test_accumulates_along_dim_zero(self):
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

        out = cumsum_over_dim(x, dim=0)

        assert torch.equal(
            out, torch.tensor([[1.0, 2.0], [4.0, 6.0], [9.0, 12.0]])
        )

    def test_last_entry_equals_the_total_sum(self):
        x = torch.randn(2, 4, 3)

        out = cumsum_over_dim(x, dim=1)

        assert out.shape == x.shape
        assert torch.allclose(out[:, -1], x.sum(dim=1), atol=1e-6)


class TestWhereSelect:
    def test_selects_elementwise(self):
        mask = torch.tensor([[True, False], [False, True]])
        a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        b = torch.tensor([[-1.0, -2.0], [-3.0, -4.0]])

        out = where_select(mask, a, b)

        assert torch.equal(out, torch.tensor([[1.0, -2.0], [-3.0, 4.0]]))

    def test_broadcasts_the_mask(self):
        mask = torch.tensor([[True], [False]])  # (2, 1)
        a = torch.ones(2, 3)
        b = torch.zeros(2, 3)

        out = where_select(mask, a, b)

        assert out.shape == (2, 3)
        assert torch.equal(out, torch.tensor([[1.0] * 3, [0.0] * 3]))


class TestOneHot:
    def test_encodes_a_1d_index_tensor(self):
        out = one_hot(torch.tensor([0, 2, 1]), num_classes=3)

        assert out.shape == (3, 3)
        assert torch.equal(
            out,
            torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]]),
        )

    def test_works_for_arbitrary_leading_shape_and_dtype(self):
        indices = torch.randint(0, 5, (2, 4, 3))

        out = one_hot(indices, num_classes=5, dtype=torch.int64)

        assert out.shape == (2, 4, 3, 5)
        assert out.dtype == torch.int64
        assert torch.equal(out.sum(dim=-1), torch.ones(2, 4, 3, dtype=torch.int64))
        assert torch.equal(out.argmax(dim=-1), indices)


class TestScatterAdd1d:
    def test_sums_values_into_the_indexed_positions(self):
        values = torch.tensor([1.0, 2.0, 3.0])
        indices = torch.tensor([0, 2, 0])

        out = scatter_add_1d(values, indices, size=4)

        assert out.shape == (4,)
        assert torch.equal(out, torch.tensor([4.0, 0.0, 2.0, 0.0]))

    def test_preserves_dtype_and_handles_empty_input(self):
        values = torch.tensor([], dtype=torch.float64)
        indices = torch.tensor([], dtype=torch.int64)

        out = scatter_add_1d(values, indices, size=3)

        assert out.dtype == torch.float64
        assert torch.equal(out, torch.zeros(3, dtype=torch.float64))


class TestBatchedTokenHistogram:
    def test_counts_tokens_per_batch_item(self):
        tokens = torch.tensor([[0, 1, 1, 3], [2, 2, 2, 0]])

        counts = batched_token_histogram(tokens, vocab_size=4)

        assert counts.shape == (2, 4)
        assert torch.allclose(
            counts.float(), torch.tensor([[1.0, 2.0, 0.0, 1.0], [1.0, 0.0, 3.0, 0.0]])
        )

    def test_each_row_sums_to_the_sequence_length(self):
        B, T, vocab = 2, 4, 5
        tokens = torch.randint(0, vocab, (B, T))

        counts = batched_token_histogram(tokens, vocab_size=vocab)

        assert counts.shape == (B, vocab)
        assert torch.allclose(counts.sum(dim=1).float(), torch.full((B,), float(T)))


class TestMaskedMean:
    def test_averages_only_the_kept_entries(self):
        x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        mask = torch.tensor([[True, False, True], [False, True, True]])

        out = masked_mean(x, mask, dim=1)

        assert out.shape == (2,)
        assert torch.allclose(out, torch.tensor([2.0, 5.5]))

    def test_returns_zero_when_everything_is_masked_out(self):
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        mask = torch.tensor([[False, False], [True, True]])

        out = masked_mean(x, mask, dim=1)

        assert torch.allclose(out, torch.tensor([0.0, 3.5]))
