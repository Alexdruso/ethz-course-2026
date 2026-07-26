"""Tests for the parameter-initialization helpers of exercise 2.

The uniform initializers are checked with two complementary assertions:
a *hard* one (nothing may exceed the analytic bound) and a *soft* one (the
largest draw must come close to that bound), which together pin the scale
down without depending on a particular RNG stream.
"""

import math

import torch
from torch import nn

from ex2 import fan_in_fan_out, init_linear_, kaiming_uniform_, xavier_uniform_


def _xavier_bound(fan_in: int, fan_out: int, gain: float = 1.0) -> float:
    return gain * math.sqrt(6.0 / (fan_in + fan_out))


def _kaiming_bound(fan_in: int) -> float:
    # bound = sqrt(3) * gain / sqrt(fan_in) with gain = sqrt(2)
    return math.sqrt(3.0) * math.sqrt(2.0) / math.sqrt(fan_in)


def _assert_uniform_within(weight: torch.Tensor, bound: float) -> None:
    largest = weight.abs().max().item()
    assert largest <= bound + 1e-6, f"draws exceed the bound {bound}"
    assert largest > 0.9 * bound, f"draws never approach the bound {bound}"
    assert abs(weight.mean().item()) < 0.05 * bound, "not centred on zero"


class TestFanInFanOut:
    def test_linear_weight_is_out_features_by_in_features(self):
        weight = torch.empty(5, 3)  # nn.Linear(in_features=3, out_features=5)

        assert fan_in_fan_out(weight) == (3, 5)

    def test_does_not_swap_the_two_values(self):
        weight = torch.empty(2, 7)

        fan_in, fan_out = fan_in_fan_out(weight)

        assert fan_in == 7
        assert fan_out == 2

    def test_matches_the_shape_of_a_real_linear_layer(self):
        layer = nn.Linear(11, 4)

        fan_in, fan_out = fan_in_fan_out(layer.weight)

        assert (fan_in, fan_out) == (11, 4)


class TestXavierUniform:
    def test_fills_in_place_within_the_xavier_bound(self):
        weight = torch.zeros(400, 250)
        bound = _xavier_bound(fan_in=250, fan_out=400)

        out = xavier_uniform_(weight)

        assert out.shape == weight.shape
        assert out.data_ptr() == weight.data_ptr()  # in-place, not a copy
        _assert_uniform_within(weight, bound)

    def test_gain_scales_the_bound_linearly(self):
        plain = torch.zeros(300, 300)
        gained = torch.zeros(300, 300)
        bound = _xavier_bound(fan_in=300, fan_out=300)

        xavier_uniform_(plain)
        xavier_uniform_(gained, gain=4.0)

        _assert_uniform_within(plain, bound)
        _assert_uniform_within(gained, 4.0 * bound)
        assert gained.abs().max().item() > plain.abs().max().item()

    def test_bound_depends_on_both_fan_in_and_fan_out(self):
        wide = torch.zeros(100, 900)  # fan_in + fan_out = 1000
        tall = torch.zeros(900, 100)  # same sum -> same bound

        xavier_uniform_(wide)
        xavier_uniform_(tall)

        bound = _xavier_bound(fan_in=900, fan_out=100)
        _assert_uniform_within(wide, bound)
        _assert_uniform_within(tall, bound)


class TestKaimingUniform:
    def test_fills_in_place_within_the_kaiming_bound(self):
        weight = torch.zeros(400, 256)
        bound = _kaiming_bound(fan_in=256)

        out = kaiming_uniform_(weight)

        assert out.data_ptr() == weight.data_ptr()
        _assert_uniform_within(weight, bound)

    def test_bound_depends_only_on_fan_in(self):
        narrow_out = torch.zeros(50, 128)
        wide_out = torch.zeros(500, 128)
        bound = _kaiming_bound(fan_in=128)

        kaiming_uniform_(narrow_out)
        kaiming_uniform_(wide_out)

        _assert_uniform_within(narrow_out, bound)
        _assert_uniform_within(wide_out, bound)

    def test_is_wider_than_xavier_for_the_same_weight(self):
        shape = (512, 512)
        kaiming = torch.zeros(*shape)
        xavier = torch.zeros(*shape)

        kaiming_uniform_(kaiming)
        xavier_uniform_(xavier)

        # sqrt(6/fan_in) vs sqrt(6/(fan_in+fan_out)): a factor of sqrt(2)
        assert kaiming.std().item() > 1.2 * xavier.std().item()


class TestInitLinear:
    def test_xavier_scheme_sets_weights_and_zeroes_the_bias(self):
        layer = nn.Linear(256, 400)

        out = init_linear_(layer, scheme="xavier")

        assert out is layer
        _assert_uniform_within(layer.weight.detach(), _xavier_bound(256, 400))
        assert torch.equal(layer.bias.detach(), torch.zeros(400))

    def test_kaiming_relu_scheme_uses_the_kaiming_bound(self):
        layer = nn.Linear(256, 400)

        init_linear_(layer, scheme="kaiming_relu")

        _assert_uniform_within(layer.weight.detach(), _kaiming_bound(256))
        assert torch.equal(layer.bias.detach(), torch.zeros(400))

    def test_zero_scheme_clears_weights_and_bias(self):
        layer = nn.Linear(6, 4)

        init_linear_(layer, scheme="zero")

        assert torch.equal(layer.weight.detach(), torch.zeros(4, 6))
        assert torch.equal(layer.bias.detach(), torch.zeros(4))

    def test_leaves_the_parameters_trainable(self):
        layer = nn.Linear(64, 32)

        init_linear_(layer, scheme="xavier")

        assert layer.weight.requires_grad
        assert layer.bias.requires_grad
        assert layer.weight.shape == (32, 64)
