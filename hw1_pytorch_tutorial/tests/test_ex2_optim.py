"""Tests for the from-scratch AdamW of exercise 2.

The expected values come from two independent oracles:

* `_reference_step`, a literal transcription of the update rule in the
  docstring, and
* `torch.optim.AdamW`, which implements the very same decoupled update
  (`p.mul_(1 - lr*wd)` followed by `addcdiv_` composes to
  `p - lr*(m_hat/(sqrt(v_hat) + eps) + wd*p)`).
"""

import torch

from ex2 import AdamWState, adamw_step_, adamw_step_many_, init_adamw_state


def _reference_step(p, grad, m, v, t, lr, betas, eps, weight_decay):
    """One AdamW step, written out exactly as the docstring specifies."""
    beta1, beta2 = betas
    t = t + 1
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * grad * grad
    m_hat = m / (1 - beta1**t)
    v_hat = v / (1 - beta2**t)
    p = p - lr * (m_hat / (v_hat.sqrt() + eps) + weight_decay * p)
    return p, m, v, t


class TestInitAdamwState:
    def test_moments_start_at_zero_with_the_parameter_shape(self):
        p = torch.randn(3, 4)

        state = init_adamw_state(p)

        assert isinstance(state, AdamWState)
        assert state.t == 0
        assert state.m.shape == p.shape
        assert state.v.shape == p.shape
        assert torch.equal(state.m, torch.zeros_like(p))
        assert torch.equal(state.v, torch.zeros_like(p))

    def test_matches_dtype_and_device_and_carries_no_grad(self):
        p = torch.randn(2, 3, 5, dtype=torch.float64, requires_grad=True)

        state = init_adamw_state(p)

        assert state.m.dtype == torch.float64
        assert state.v.dtype == torch.float64
        assert state.m.device == p.device
        assert state.v.device == p.device
        assert not state.m.requires_grad
        assert not state.v.requires_grad

    def test_state_tensors_are_independent_of_the_parameter(self):
        p = torch.randn(4)

        state = init_adamw_state(p)
        state.m.add_(1.0)

        assert state.m.data_ptr() != p.data_ptr()
        assert state.v.data_ptr() != state.m.data_ptr()
        assert torch.equal(state.v, torch.zeros_like(p))


class TestAdamwStep:
    def test_first_step_matches_the_written_out_update(self):
        torch.manual_seed(0)
        p = torch.randn(3, 4, dtype=torch.float64)
        grad = torch.randn(3, 4, dtype=torch.float64)
        lr, betas, eps, wd = 0.05, (0.9, 0.999), 1e-8, 0.1

        expected_p, expected_m, expected_v, expected_t = _reference_step(
            p.clone(),
            grad.clone(),
            torch.zeros_like(p),
            torch.zeros_like(p),
            0,
            lr,
            betas,
            eps,
            wd,
        )

        state = AdamWState(torch.zeros_like(p), torch.zeros_like(p), 0)
        new_state = adamw_step_(p, grad, state, lr, betas, eps, wd)

        assert new_state.t == expected_t == 1
        assert torch.allclose(p, expected_p)
        assert torch.allclose(new_state.m, expected_m)
        assert torch.allclose(new_state.v, expected_v)

    def test_several_steps_match_torch_optim_adamw(self):
        torch.manual_seed(0)
        lr, betas, eps, wd = 1e-2, (0.9, 0.999), 1e-8, 0.05
        init = torch.randn(5, 3, dtype=torch.float64)
        grads = [torch.randn(5, 3, dtype=torch.float64) for _ in range(6)]

        reference = init.clone().requires_grad_(True)
        opt = torch.optim.AdamW(
            [reference], lr=lr, betas=betas, eps=eps, weight_decay=wd
        )

        p = init.clone()
        state = AdamWState(torch.zeros_like(p), torch.zeros_like(p), 0)
        for step, g in enumerate(grads, start=1):
            reference.grad = g.clone()
            opt.step()

            state = adamw_step_(p, g, state, lr, betas, eps, wd)
            assert state.t == step
            assert torch.allclose(p, reference.detach(), rtol=1e-9, atol=1e-12)

    def test_updates_the_parameter_in_place_and_leaves_the_grad_alone(self):
        p = torch.full((2, 3), 1.0)
        original_ptr = p.data_ptr()
        grad = torch.full((2, 3), 0.5)
        grad_copy = grad.clone()

        state = adamw_step_(p, grad, init_adamw_state(p), lr=0.1)

        assert p.data_ptr() == original_ptr  # mutated, not rebound
        assert not torch.allclose(p, torch.ones(2, 3))
        assert torch.equal(grad, grad_copy)
        assert state.t == 1

    def test_works_for_a_3d_parameter_with_custom_hyperparameters(self):
        torch.manual_seed(1)
        p = torch.randn(2, 3, 4, dtype=torch.float64)
        grad = torch.randn(2, 3, 4, dtype=torch.float64)
        lr, betas, eps, wd = 0.2, (0.5, 0.9), 1e-6, 0.3

        expected_p, expected_m, expected_v, _ = _reference_step(
            p.clone(),
            grad.clone(),
            torch.zeros_like(p),
            torch.zeros_like(p),
            0,
            lr,
            betas,
            eps,
            wd,
        )

        state = adamw_step_(p, grad, init_adamw_state(p), lr, betas, eps, wd)

        assert torch.allclose(p, expected_p)
        assert torch.allclose(state.m, expected_m)
        assert torch.allclose(state.v, expected_v)

    def test_weight_decay_shrinks_a_parameter_with_zero_gradient(self):
        p = torch.tensor([2.0, -4.0])
        grad = torch.zeros(2)

        adamw_step_(p, grad, init_adamw_state(p), lr=0.1, weight_decay=0.5)

        # p <- p - lr*(0 + wd*p) = p * (1 - 0.05)
        assert torch.allclose(p, torch.tensor([2.0, -4.0]) * 0.95)


class TestAdamwStepMany:
    def test_matches_stepping_each_parameter_individually(self):
        torch.manual_seed(0)
        shapes = [(3,), (2, 4), (2, 2, 3)]
        params = [torch.randn(s, dtype=torch.float64) for s in shapes]
        grads = [torch.randn(s, dtype=torch.float64) for s in shapes]
        lr, betas, eps, wd = 0.03, (0.9, 0.99), 1e-8, 0.02

        expected = []
        for p, g in zip(params, grads):
            ep, em, ev, et = _reference_step(
                p.clone(),
                g.clone(),
                torch.zeros_like(p),
                torch.zeros_like(p),
                0,
                lr,
                betas,
                eps,
                wd,
            )
            expected.append((ep, em, ev, et))

        states = [init_adamw_state(p) for p in params]
        new_states = adamw_step_many_(params, grads, states, lr, betas, eps, wd)

        assert len(new_states) == len(params)
        for p, state, (ep, em, ev, et) in zip(params, new_states, expected):
            assert torch.allclose(p, ep)
            assert torch.allclose(state.m, em)
            assert torch.allclose(state.v, ev)
            assert state.t == et

    def test_two_steps_stay_in_sync_with_torch_optim_adamw(self):
        torch.manual_seed(2)
        lr, betas, eps, wd = 5e-2, (0.9, 0.999), 1e-8, 0.1
        inits = [
            torch.randn(4, dtype=torch.float64),
            torch.randn(2, 3, dtype=torch.float64),
        ]

        references = [t.clone().requires_grad_(True) for t in inits]
        opt = torch.optim.AdamW(
            references, lr=lr, betas=betas, eps=eps, weight_decay=wd
        )

        params = [t.clone() for t in inits]
        states = [init_adamw_state(p) for p in params]
        for step in (1, 2):
            grads = [torch.randn_like(p) for p in params]
            for ref, g in zip(references, grads):
                ref.grad = g.clone()
            opt.step()

            states = adamw_step_many_(params, grads, states, lr, betas, eps, wd)
            for p, ref, state in zip(params, references, states):
                assert state.t == step
                assert torch.allclose(p, ref.detach(), rtol=1e-9, atol=1e-12)

    def test_updates_every_parameter_in_place_without_touching_the_grads(self):
        params = [torch.ones(3), torch.full((2, 2), -1.0)]
        ptrs = [p.data_ptr() for p in params]
        grads = [torch.full((3,), 0.5), torch.full((2, 2), -0.25)]
        grad_copies = [g.clone() for g in grads]

        states = adamw_step_many_(
            params, grads, [init_adamw_state(p) for p in params], lr=0.1
        )

        for p, ptr in zip(params, ptrs):
            assert p.data_ptr() == ptr
        assert not torch.allclose(params[0], torch.ones(3))
        assert not torch.allclose(params[1], torch.full((2, 2), -1.0))
        for g, copy in zip(grads, grad_copies):
            assert torch.equal(g, copy)
        assert [s.t for s in states] == [1, 1]
