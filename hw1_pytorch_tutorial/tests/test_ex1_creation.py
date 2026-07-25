"""Tests for the tensor-creation helpers of exercise 1."""

import pytest
import torch

from ex1 import (
    cast_dtype_and_move,
    make_arange,
    make_linspace,
    make_ones_like,
    make_randn,
    make_tensor,
    make_zeros,
)


class TestMakeTensor:
    def test_builds_nested_list_with_requested_dtype(self):
        x = make_tensor([[1, 2], [3, 4]], dtype=torch.float32)

        assert x.shape == (2, 2)
        assert x.dtype == torch.float32
        assert torch.equal(x, torch.tensor([[1.0, 2.0], [3.0, 4.0]]))

    def test_infers_dtype_from_python_values_when_none_given(self):
        ints = make_tensor([1, 2, 3])
        floats = make_tensor([1.0, 2.0, 3.0])

        assert ints.dtype == torch.int64
        assert floats.dtype == torch.float32


class TestMakeZeros:
    def test_shape_dtype_and_values(self):
        z = make_zeros((2, 3), dtype=torch.float64)

        assert z.shape == (2, 3)
        assert z.dtype == torch.float64
        assert torch.equal(z, torch.zeros(2, 3, dtype=torch.float64))

    def test_respects_device(self):
        z = make_zeros((4,), device="cpu")

        assert z.device.type == "cpu"
        assert not z.any()


class TestMakeOnesLike:
    def test_matches_shape_dtype_and_device_of_input(self):
        base = torch.randn(2, 3, dtype=torch.float64)

        ones = make_ones_like(base)

        assert ones.shape == base.shape
        assert ones.dtype == base.dtype
        assert ones.device == base.device
        assert torch.equal(ones, torch.ones_like(base))

    def test_does_not_alias_or_modify_the_input(self):
        base = torch.zeros(3, 2)

        ones = make_ones_like(base)
        ones[0, 0] = 99.0

        assert torch.equal(base, torch.zeros(3, 2))


class TestMakeArange:
    def test_is_end_exclusive_and_honours_step(self):
        ar = make_arange(0, 5, 2, dtype=torch.int64)

        assert torch.equal(ar, torch.tensor([0, 2, 4]))

    def test_end_value_is_never_included(self):
        ar = make_arange(0, 4)

        assert ar.numel() == 4
        assert ar[-1].item() == 3


class TestMakeLinspace:
    def test_is_end_inclusive(self):
        ls = make_linspace(0.0, 1.0, steps=5, dtype=torch.float32)

        assert ls.shape == (5,)
        assert ls[0].item() == pytest.approx(0.0)
        assert ls[-1].item() == pytest.approx(1.0)

    def test_spacing_is_uniform(self):
        ls = make_linspace(-1.0, 1.0, steps=9)

        diffs = ls[1:] - ls[:-1]
        assert torch.allclose(diffs, torch.full_like(diffs, 0.25))


class TestMakeRandn:
    def test_same_seed_is_reproducible(self):
        a = make_randn((2, 3), seed=123, dtype=torch.float32)
        b = make_randn((2, 3), seed=123, dtype=torch.float32)

        assert a.shape == (2, 3)
        assert a.dtype == torch.float32
        assert torch.equal(a, b)

    def test_different_seeds_give_different_draws(self):
        a = make_randn((4, 4), seed=0)
        b = make_randn((4, 4), seed=1)

        assert not torch.equal(a, b)


class TestCastDtypeAndMove:
    def test_casts_dtype_and_keeps_values(self):
        casted = cast_dtype_and_move(
            torch.tensor([1, 2, 3]), torch.device("cpu"), torch.float32
        )

        assert casted.dtype == torch.float32
        assert casted.device.type == "cpu"
        assert torch.equal(casted, torch.tensor([1.0, 2.0, 3.0]))

    def test_float_to_int_cast_truncates_towards_zero(self):
        casted = cast_dtype_and_move(
            torch.tensor([1.7, -1.7]), torch.device("cpu"), torch.int64
        )

        assert casted.dtype == torch.int64
        assert torch.equal(casted, torch.tensor([1, -1]))
