"""Tests for the attention fundamentals of exercise 1.

These five functions are still `TODO: implement` in `src/ex1.py`, so these tests
fail until you fill them in. They are the executable spec for what each function
has to do - work through them one at a time, e.g.

    make py-test ARGS="-k TestStableSoftmax"
"""

import torch
from shapes import B, H, T

from ex1 import (
    apply_causal_mask,
    make_causal_mask,
    masked_fill_tensor,
    masked_softmax,
    stable_softmax,
)

class TestStableSoftmax:
    def test_output_is_a_distribution_along_dim(self):
        x = torch.randn(B, T, H)

        out = stable_softmax(x, dim=-1)

        assert out.shape == x.shape
        assert (out >= 0).all()
        assert torch.allclose(out.sum(dim=-1), torch.ones(B, T), atol=1e-6)
        assert torch.allclose(out, torch.softmax(x, dim=-1), atol=1e-6)

    def test_does_not_overflow_on_large_logits(self):
        x = torch.tensor([[1000.0, 1000.0, 1001.0]])

        out = stable_softmax(x, dim=-1)

        assert torch.isfinite(out).all()
        assert torch.allclose(out, torch.softmax(x, dim=-1), atol=1e-6)

    def test_reduces_over_a_non_default_dim(self):
        x = torch.randn(B, T, H)

        out = stable_softmax(x, dim=1)

        assert torch.allclose(out.sum(dim=1), torch.ones(B, H), atol=1e-6)


class TestMaskedFillTensor:
    def test_replaces_masked_positions(self):
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        mask = torch.tensor([[True, False], [False, True]])

        out = masked_fill_tensor(x, mask, -1.0)

        assert torch.equal(out, torch.tensor([[-1.0, 2.0], [3.0, -1.0]]))

    def test_leaves_the_input_unchanged(self):
        x = torch.zeros(2, 3)
        mask = torch.ones(2, 3, dtype=torch.bool)

        out = masked_fill_tensor(x, mask, 5.0)

        assert torch.equal(x, torch.zeros(2, 3))
        assert torch.equal(out, torch.full((2, 3), 5.0))

    def test_broadcasts_the_mask(self):
        x = torch.ones(2, 3)
        mask = torch.tensor([[True], [False]])  # (2, 1)

        out = masked_fill_tensor(x, mask, 0.0)

        assert torch.equal(out, torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]))


class TestMaskedSoftmax:
    def test_masked_positions_get_exactly_zero_probability(self):
        x = torch.tensor([[1.0, 2.0, 3.0]])
        mask = torch.tensor([[False, True, False]])  # True == invalid

        out = masked_softmax(x, mask, dim=-1)

        assert out[0, 1].item() == 0.0
        assert torch.allclose(out.sum(dim=-1), torch.ones(1), atol=1e-6)
        expected = torch.softmax(torch.tensor([[1.0, 3.0]]), dim=-1)
        assert torch.allclose(out[:, [0, 2]], expected, atol=1e-6)

    def test_returns_all_zeros_when_everything_is_masked(self):
        x = torch.randn(2, 3)
        mask = torch.tensor([[True, True, True], [False, False, False]])

        out = masked_softmax(x, mask, dim=-1)

        assert torch.equal(out[0], torch.zeros(3))
        assert torch.allclose(out[1].sum(), torch.tensor(1.0), atol=1e-6)

    def test_is_numerically_stable(self):
        x = torch.tensor([[1e4, 1e4, -1e4]])
        mask = torch.tensor([[False, False, True]])

        out = masked_softmax(x, mask, dim=-1)

        assert torch.isfinite(out).all()
        assert torch.allclose(out, torch.tensor([[0.5, 0.5, 0.0]]), atol=1e-6)


class TestMakeCausalMask:
    def test_is_strictly_upper_triangular_and_boolean(self):
        mask = make_causal_mask(4)

        assert mask.dtype == torch.bool
        assert mask.shape == (4, 4)
        assert torch.equal(mask, torch.ones(4, 4, dtype=torch.bool).triu(diagonal=1))

    def test_diagonal_and_past_are_allowed(self):
        T_ = 5
        mask = make_causal_mask(T_)

        for i in range(T_):
            assert not mask[i, i]  # a position may attend to itself
            assert not mask[i, :i].any()  # and to the past
            assert mask[i, i + 1 :].all()  # but never to the future

    def test_respects_the_device_argument(self):
        mask = make_causal_mask(3, device="cpu")

        assert mask.device.type == "cpu"


class TestApplyCausalMask:
    def test_masks_future_positions_with_the_given_value(self):
        logits = torch.zeros(T, T)

        out = apply_causal_mask(logits, value=-1e9)

        assert out.shape == (T, T)
        assert torch.equal(out[0], torch.tensor([0.0, -1e9, -1e9, -1e9]))
        assert torch.equal(out[-1], torch.zeros(T))

    def test_broadcasts_across_leading_dims_and_leaves_input_unchanged(self):
        logits = torch.randn(B, H, T, T)
        original = logits.clone()

        out = apply_causal_mask(logits)

        assert out.shape == (B, H, T, T)
        causal = torch.ones(T, T, dtype=torch.bool).triu(diagonal=1)
        assert (out[..., causal] == -1e9).all()
        assert torch.equal(out[..., ~causal], logits[..., ~causal])
        assert torch.equal(logits, original)

    def test_softmax_of_masked_logits_is_causal(self):
        logits = torch.randn(B, H, T, T)

        probs = torch.softmax(apply_causal_mask(logits), dim=-1)

        causal = torch.ones(T, T, dtype=torch.bool).triu(diagonal=1)
        assert torch.allclose(probs[..., causal], torch.zeros_like(probs[..., causal]), atol=1e-6)
        assert torch.allclose(probs.sum(dim=-1), torch.ones(B, H, T), atol=1e-6)
