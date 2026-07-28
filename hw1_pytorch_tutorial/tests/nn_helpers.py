"""Small helpers shared by the exercise 3 and 4 tests.

The exercises fix only the *interface*: the constructor signature, what
`forward` maps to what, and the shapes involved. Everything else - how many
submodules you use, what you call them, in which order you define them - is
yours to choose. So the tests here reach a module through its public surface
only: constructing it, calling it, and the `nn.Module` API (`parameters()`,
`modules()`, `train()`/`eval()`). They never read an attribute by name.

Two tricks do most of the work:

- `make_all_projections_identity` turns every matrix in a module into the
  identity, which collapses a whole block into a closed-form expression and
  exposes its non-linear structure regardless of how it is wired internally.
- `recover_affine` reconstructs a normalization layer's learnable scale and
  shift from its outputs, so tests can compare against a PyTorch reference
  without knowing which parameter is which.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def make_all_projections_identity(module: nn.Module) -> None:
    """Set every 2-D parameter to the identity and every 1-D parameter to zero.

    Only meaningful when all of the module's matrices are square (i.e. the
    hidden width equals the model width). Then every projection becomes an
    identity map and whatever is left is exactly the module's non-linear
    structure: `FeedForward` collapses to `gelu(x)`, a GEGLU collapses to
    `gelu(x) * x`, and so on. Because every matrix gets the same value, the
    result does not depend on the order the parameters are defined in.
    """
    with torch.no_grad():
        for p in module.parameters():
            if p.dim() == 2:
                assert p.shape[0] == p.shape[1], (
                    f"expected square projections, got {tuple(p.shape)}"
                )
                p.copy_(torch.eye(p.shape[0], dtype=p.dtype))
            elif p.dim() == 1:
                p.zero_()


def projection_shapes(module: nn.Module) -> list[tuple[int, ...]]:
    """Shapes of all 2-D parameters, sorted so definition order does not matter.

    A dense layer is the only thing in these exercises that owns a matrix, so
    this doubles as a way to see the width chain of a network without knowing
    how its layers are named or nested.
    """
    return sorted(tuple(p.shape) for p in module.parameters() if p.dim() == 2)


def count_projections(module: nn.Module) -> int:
    """How many dense layers a module contains, counted by their matrices."""
    return sum(1 for p in module.parameters() if p.dim() == 2)


def vector_shapes(module: nn.Module) -> list[tuple[int, ...]]:
    """Shapes of all 1-D parameters (biases, norm scales), sorted."""
    return sorted(tuple(p.shape) for p in module.parameters() if p.dim() == 1)


def randomize_parameters(
    module: nn.Module, low: float = 0.5, high: float = 1.5
) -> None:
    """Fill every parameter with distinct non-zero values.

    Normalization layers initialize their scale to ones and their shift to
    zeros, which makes an *unused* parameter indistinguishable from a used one.
    Moving both off their defaults is what lets `recover_affine` prove they are
    actually applied.
    """
    with torch.no_grad():
        for p in module.parameters():
            p.uniform_(low, high)


def recover_affine(
    norm: nn.Module, d: int, eps: float, subtract_mean: bool = True
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reconstruct a normalization layer's scale and shift from its outputs.

    A normalizer computes `scale * normalize(x) + shift`, so:

    - `normalize` maps an all-zero input to zero, hence `norm(0)` *is* `shift`;
    - for any other input, `(norm(u) - shift) / normalize(u)` *is* `scale`.

    Recovering them this way means a test can compare against `F.layer_norm`
    without ever touching `.weight` or `.bias` - the layer only has to behave
    like a normalizer, not to name its parameters a particular way.
    """
    with torch.no_grad():
        shift = norm(torch.zeros(d))
        # A deterministic probe with no component sitting near the mean, so the
        # division below never lands on a near-zero denominator.
        u = torch.linspace(-2.0, 2.0, d) + 0.7
        if subtract_mean:
            u_hat = F.layer_norm(u, (d,), eps=eps)
        else:
            u_hat = u * torch.rsqrt(u.pow(2).mean(dim=-1, keepdim=True) + eps)
        assert u_hat.abs().min() > 0.1, "probe vector is too close to degenerate"
        scale = (norm(u) - shift) / u_hat
    return scale, shift


def assert_is_affine(fn, x: torch.Tensor, atol: float = 1e-5) -> None:
    """Check `fn` is an affine map of its input, i.e. contains no activation.

    An affine `f` satisfies `f(a + b) - f(0) == (f(a) - f(0)) + (f(b) - f(0))`.
    Any element-wise non-linearity anywhere in the chain breaks this.
    """
    a, b = x, torch.randn_like(x)
    zero = fn(torch.zeros_like(x))
    lhs = fn(a + b) - zero
    rhs = (fn(a) - zero) + (fn(b) - zero)
    assert torch.allclose(lhs, rhs, atol=atol), (
        "expected an affine (activation-free) map"
    )


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


class ModeSpy(nn.Module):
    """Wraps a module and records the train/eval mode of every forward call."""

    def __init__(self, inner: nn.Module):
        super().__init__()
        self.inner = inner
        self.modes: list[bool] = []

    def forward(self, *args, **kwargs):
        self.modes.append(self.training)
        return self.inner(*args, **kwargs)


class ZeroModule(nn.Module):
    """Returns zeros, but keeps a parameter so gradients still have somewhere to go."""

    def __init__(self, d_model: int):
        super().__init__()
        self.scale = nn.Parameter(torch.zeros(d_model))

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return x * self.scale
