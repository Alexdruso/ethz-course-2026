"""Tests for the dataloading part of exercise 2.

Mask convention under test: `padding_mask[b, t] is True` means *padding*, and
`attention_mask` is its logical negation. Several tests deliberately use a
`pad_value` that also occurs as a **real** token, because a mask built as
`tokens == pad_value` looks correct until that happens.
"""

import torch
from torch.utils.data import DataLoader, Dataset

from ex2 import (
    NextTokenDataset,
    PaddedBatch,
    RandomCropSequenceDataset,
    TensorPairDataset,
    collate_next_token_batch,
    make_dataloader,
    pad_1d_sequences,
)


class _RangeDataset(Dataset):
    """Self-contained dataset so the DataLoader tests do not depend on ex2."""

    def __init__(self, n: int):
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.tensor([idx], dtype=torch.long)


class TestTensorPairDataset:
    def test_length_and_items_match_the_wrapped_tensors(self):
        x = torch.randn(5, 3)
        y = torch.randn(5)

        ds = TensorPairDataset(x, y)

        assert len(ds) == 5
        for i in range(5):
            xi, yi = ds[i]
            assert torch.equal(xi, x[i])
            assert torch.equal(yi, y[i])

    def test_supports_multi_dim_targets(self):
        x = torch.arange(4 * 2 * 3).reshape(4, 2, 3).float()
        y = torch.arange(4 * 2).reshape(4, 2).float()

        ds = TensorPairDataset(x, y)

        assert len(ds) == 4
        xi, yi = ds[2]
        assert xi.shape == (2, 3)
        assert yi.shape == (2,)
        assert torch.equal(xi, x[2])
        assert torch.equal(yi, y[2])

    def test_batches_with_the_default_dataloader_collation(self):
        x = torch.randn(6, 3)
        y = torch.randn(6, 1)

        ds = TensorPairDataset(x, y)
        batches = list(DataLoader(ds, batch_size=2, shuffle=False))

        assert len(batches) == 3
        bx, by = batches[0]
        assert bx.shape == (2, 3)
        assert by.shape == (2, 1)
        assert torch.equal(bx, x[:2])
        assert torch.equal(by, y[:2])


class TestNextTokenDataset:
    def test_shifts_the_sequence_by_one(self):
        tokens = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])

        ds = NextTokenDataset(tokens)

        assert len(ds) == 2
        inp, tgt = ds[0]
        assert torch.equal(inp, torch.tensor([1, 2, 3]))
        assert torch.equal(tgt, torch.tensor([2, 3, 4]))
        inp, tgt = ds[1]
        assert torch.equal(inp, torch.tensor([5, 6, 7]))
        assert torch.equal(tgt, torch.tensor([6, 7, 8]))

    def test_items_are_1d_of_length_t_minus_one_and_stay_integer(self):
        N, T = 3, 6
        tokens = torch.randint(0, 10, (N, T), dtype=torch.int64)

        ds = NextTokenDataset(tokens)

        assert len(ds) == N
        for i in range(N):
            inp, tgt = ds[i]
            assert inp.shape == (T - 1,)
            assert tgt.shape == (T - 1,)
            assert not torch.is_floating_point(inp)
            assert not torch.is_floating_point(tgt)
            assert torch.equal(inp, tokens[i, :-1])
            assert torch.equal(tgt, tokens[i, 1:])

    def test_targets_are_the_inputs_shifted_left(self):
        tokens = torch.arange(20).reshape(4, 5)

        ds = NextTokenDataset(tokens)

        for i in range(len(ds)):
            inp, tgt = ds[i]
            assert torch.equal(tgt[:-1], inp[1:])


class TestRandomCropSequenceDataset:
    @staticmethod
    def _rows(n: int, t_total: int) -> torch.Tensor:
        """Rows of strictly increasing, row-unique values.

        Row `i` is `[100*i, 100*i + 1, ...]`, so any crop identifies both the
        row it came from and its start offset.
        """
        return torch.arange(t_total).unsqueeze(0) + 100 * torch.arange(n).unsqueeze(1)

    def test_returns_contiguous_crops_of_the_requested_row(self):
        tokens = self._rows(n=3, t_total=10)
        crop_len = 4

        ds = RandomCropSequenceDataset(tokens, crop_len, seed=0)

        assert len(ds) == 3
        for i in range(3):
            crop = ds[i]
            assert crop.shape == (crop_len,)
            start = int(crop[0]) - 100 * i
            assert 0 <= start <= 10 - crop_len  # crop stays inside the row
            assert torch.equal(crop, tokens[i, start : start + crop_len])

    def test_same_seed_gives_the_same_crops(self):
        tokens = self._rows(n=4, t_total=12)

        a = RandomCropSequenceDataset(tokens, crop_len=5, seed=1234)
        b = RandomCropSequenceDataset(tokens, crop_len=5, seed=1234)

        crops_a = [a[i] for i in range(len(a))]
        crops_b = [b[i] for i in range(len(b))]
        for ca, cb in zip(crops_a, crops_b):
            assert torch.equal(ca, cb)

    def test_crop_len_equal_to_the_sequence_returns_the_whole_row(self):
        tokens = self._rows(n=2, t_total=6)

        ds = RandomCropSequenceDataset(tokens, crop_len=6, seed=0)

        for i in range(2):
            assert torch.equal(ds[i], tokens[i])

    def test_works_without_a_seed(self):
        tokens = self._rows(n=3, t_total=8)
        crop_len = 3

        ds = RandomCropSequenceDataset(tokens, crop_len)

        crop = ds[1]
        assert crop.shape == (crop_len,)
        start = int(crop[0]) - 100
        assert 0 <= start <= 8 - crop_len
        assert torch.equal(crop, tokens[1, start : start + crop_len])


class TestPad1dSequences:
    def test_pads_to_the_longest_sequence(self):
        seqs = [
            torch.tensor([1, 2, 3]),
            torch.tensor([4]),
            torch.tensor([5, 6]),
        ]

        out = pad_1d_sequences(seqs)

        assert isinstance(out, PaddedBatch)
        assert torch.equal(out.tokens, torch.tensor([[1, 2, 3], [4, 0, 0], [5, 6, 0]]))
        assert torch.equal(out.lengths, torch.tensor([3, 1, 2]))
        assert torch.equal(
            out.padding_mask,
            torch.tensor(
                [
                    [False, False, False],
                    [False, True, True],
                    [False, False, True],
                ]
            ),
        )

    def test_mask_follows_lengths_even_when_pad_value_is_a_real_token(self):
        # 7 appears both as a genuine token and as the padding filler.
        seqs = [torch.tensor([7, 1, 7, 2]), torch.tensor([7])]

        out = pad_1d_sequences(seqs, pad_value=7)

        assert torch.equal(out.tokens, torch.tensor([[7, 1, 7, 2], [7, 7, 7, 7]]))
        assert torch.equal(out.lengths, torch.tensor([4, 1]))
        assert torch.equal(
            out.padding_mask,
            torch.tensor([[False] * 4, [False, True, True, True]]),
        )

    def test_casts_tokens_and_lengths_to_long(self):
        seqs = [
            torch.tensor([1, 2], dtype=torch.int32),
            torch.tensor([3, 4, 5], dtype=torch.int32),
        ]

        out = pad_1d_sequences(seqs, pad_value=-1)

        assert out.tokens.dtype == torch.long
        assert out.lengths.dtype == torch.long
        assert out.padding_mask.dtype == torch.bool
        assert torch.equal(out.tokens, torch.tensor([[1, 2, -1], [3, 4, 5]]))

    def test_equal_length_sequences_need_no_padding(self):
        seqs = [torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6])]

        out = pad_1d_sequences(seqs)

        assert out.tokens.shape == (2, 3)
        assert torch.equal(out.lengths, torch.tensor([3, 3]))
        assert not out.padding_mask.any()


class TestCollateNextTokenBatch:
    def test_pads_inputs_and_targets_consistently(self):
        batch = [
            (torch.tensor([1, 2, 3]), torch.tensor([2, 3, 4])),
            (torch.tensor([5]), torch.tensor([6])),
        ]

        out = collate_next_token_batch(batch)

        assert set(out) >= {
            "input_ids",
            "target_ids",
            "attention_mask",
            "padding_mask",
        }
        assert torch.equal(out["input_ids"], torch.tensor([[1, 2, 3], [5, 0, 0]]))
        assert torch.equal(out["target_ids"], torch.tensor([[2, 3, 4], [6, 0, 0]]))
        assert torch.equal(
            out["padding_mask"],
            torch.tensor([[False, False, False], [False, True, True]]),
        )
        assert torch.equal(out["attention_mask"], ~out["padding_mask"])

    def test_masks_come_from_lengths_not_from_the_pad_value(self):
        # 9 is both a real token and the padding filler.
        batch = [
            (torch.tensor([9, 1]), torch.tensor([1, 9])),
            (torch.tensor([2, 3, 9]), torch.tensor([3, 9, 4])),
        ]

        out = collate_next_token_batch(batch, pad_value=9)

        assert out["input_ids"].shape == (2, 3)
        assert out["target_ids"].shape == (2, 3)
        assert torch.equal(out["input_ids"], torch.tensor([[9, 1, 9], [2, 3, 9]]))
        assert torch.equal(out["target_ids"], torch.tensor([[1, 9, 9], [3, 9, 4]]))
        assert torch.equal(
            out["padding_mask"],
            torch.tensor([[False, False, True], [False, False, False]]),
        )
        assert torch.equal(out["attention_mask"], ~out["padding_mask"])

    def test_attention_mask_is_boolean_and_counts_the_real_tokens(self):
        batch = [
            (torch.tensor([1, 2, 3, 4]), torch.tensor([2, 3, 4, 5])),
            (torch.tensor([7, 8]), torch.tensor([8, 9])),
            (torch.tensor([1]), torch.tensor([2])),
        ]

        out = collate_next_token_batch(batch)

        assert out["attention_mask"].dtype == torch.bool
        assert out["padding_mask"].dtype == torch.bool
        assert torch.equal(out["attention_mask"].sum(dim=1), torch.tensor([4, 2, 1]))


class TestMakeDataloader:
    def test_batches_in_order_when_shuffle_is_off(self):
        dl = make_dataloader(_RangeDataset(6), batch_size=2, shuffle=False)

        assert isinstance(dl, DataLoader)
        assert dl.batch_size == 2
        batches = list(dl)
        assert len(batches) == 3
        assert torch.equal(batches[0], torch.tensor([[0], [1]]))
        assert torch.equal(batches[1], torch.tensor([[2], [3]]))
        assert torch.equal(batches[2], torch.tensor([[4], [5]]))

    def test_drop_last_controls_the_trailing_short_batch(self):
        ds = _RangeDataset(7)

        kept = list(make_dataloader(ds, batch_size=3, shuffle=False, drop_last=False))
        dropped = list(make_dataloader(ds, batch_size=3, shuffle=False, drop_last=True))

        assert [b.shape[0] for b in kept] == [3, 3, 1]
        assert [b.shape[0] for b in dropped] == [3, 3]

    def test_uses_the_provided_collate_fn(self):
        def marker_collate(items):
            return {"size": len(items), "ids": torch.cat(items)}

        dl = make_dataloader(
            _RangeDataset(4), batch_size=2, shuffle=False, collate_fn=marker_collate
        )

        batches = list(dl)
        assert len(batches) == 2
        assert batches[0]["size"] == 2
        assert torch.equal(batches[0]["ids"], torch.tensor([0, 1]))
        assert torch.equal(batches[1]["ids"], torch.tensor([2, 3]))

    def test_shuffling_covers_every_sample_exactly_once(self):
        dl = make_dataloader(_RangeDataset(8), batch_size=4, shuffle=True)

        seen = torch.cat([b.flatten() for b in dl]).sort().values
        assert torch.equal(seen, torch.arange(8))
