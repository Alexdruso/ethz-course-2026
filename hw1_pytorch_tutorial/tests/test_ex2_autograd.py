"""Tests for the autograd fundamentals of exercise 2.

All three functions differentiate a closed-form objective, so every expected
value below is written out analytically instead of being read back from
autograd.
"""

import torch

from ex2 import grad_with_autograd_grad, grad_with_backward, grad_wrt_multiple_inputs


class TestGradWithAutogradGrad:
    def test_returns_two_x_for_a_plain_input(self):
        # f(x) = sum(x^2)  ->  df/dx = 2x, even though x arrives without grad.
        x = torch.tensor([1.0, -2.0, 3.0])

        g = grad_with_autograd_grad(x)

        assert g.shape == x.shape
        assert torch.allclose(g, torch.tensor([2.0, -4.0, 6.0]))

    def test_works_on_a_multi_dim_input_that_already_requires_grad(self):
        x = torch.randn(3, 4, dtype=torch.float64, requires_grad=True)

        g = grad_with_autograd_grad(x)

        assert g.shape == x.shape
        assert g.dtype == x.dtype
        assert torch.allclose(g, 2.0 * x.detach())

    def test_does_not_write_into_dot_grad(self):
        # torch.autograd.grad returns the gradient; it must not populate .grad.
        x = torch.tensor([1.0, 2.0])

        grad_with_autograd_grad(x)

        assert x.grad is None


class TestGradWithBackward:
    def test_returns_two_x(self):
        x = torch.tensor([1.0, -2.0, 3.0], requires_grad=True)

        g = grad_with_backward(x)

        assert g.shape == x.shape
        assert torch.allclose(g, torch.tensor([2.0, -4.0, 6.0]))

    def test_repeated_calls_do_not_accumulate(self):
        x = torch.randn(2, 3, requires_grad=True)

        first = grad_with_backward(x).clone()
        second = grad_with_backward(x).clone()

        assert torch.allclose(first, 2.0 * x.detach())
        assert torch.allclose(second, first)  # not 4x

    def test_ignores_gradients_left_over_from_earlier_work(self):
        x = torch.tensor([1.0, 2.0], requires_grad=True)
        x.grad = torch.tensor([100.0, 100.0])

        g = grad_with_backward(x)

        assert torch.allclose(g, torch.tensor([2.0, 4.0]))


class TestGradWrtMultipleInputs:
    def test_gradients_of_a_hand_computed_example(self):
        # f(a, b) = sum(a^2 + a*b)  ->  df/da = 2a + b, df/db = a
        a = torch.tensor([1.0, 2.0])
        b = torch.tensor([3.0, 4.0])

        ga, gb = grad_wrt_multiple_inputs(a, b)

        assert torch.allclose(ga, torch.tensor([5.0, 8.0]))
        assert torch.allclose(gb, torch.tensor([1.0, 2.0]))

    def test_returns_the_pair_in_order_for_a_multi_dim_input(self):
        a = torch.randn(2, 3, dtype=torch.float64)
        b = torch.randn(2, 3, dtype=torch.float64)

        out = grad_wrt_multiple_inputs(a, b)

        assert isinstance(out, tuple)
        assert len(out) == 2
        ga, gb = out
        assert ga.shape == a.shape
        assert gb.shape == b.shape
        assert torch.allclose(ga, 2.0 * a + b)
        assert torch.allclose(gb, a)

    def test_accepts_inputs_that_do_not_require_grad(self):
        a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        b = torch.tensor([[0.5, 0.5], [1.0, 1.0]])
        assert not a.requires_grad and not b.requires_grad

        ga, gb = grad_wrt_multiple_inputs(a, b)

        assert torch.allclose(ga, 2.0 * a + b)
        assert torch.allclose(gb, a)
