# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: rl2
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Exercise 3: Neural networks in PyTorch
#
# In this exercise you’ll implement small neural-network building blocks from scratch and use them to train a simple classifier.
#
# You’ll cover:
# - **Basic layers**: Linear, Embedding, Dropout
# - **Normalization**: LayerNorm and RMSNorm
# - **MLPs + residual**: composing layers into deeper networks
# - **Classification**: generating a learnable dataset, implementing cross-entropy from logits, and writing a minimal training loop
#
# As before: fill in all `TODO`s without changing function names or signatures.
# Use small sanity checks and compare to PyTorch reference implementations when useful.

# %%
from __future__ import annotations

import torch
from torch import nn

# %% [markdown]
# ## Basic layers
#
# In this section you’ll implement a few core layers that appear everywhere:
#
# ### `Linear`
# A fully-connected layer that follows nn.Linear conventions:
# `y = x @ Wᵀ + b`
#
# Important details:
# - Parameters should be registered as `nn.Parameter`
# - Store weight as (out_features, in_features) like nn.Linear.
# - The forward pass should support leading batch dimensions: `x` can be shape `(..., in_features)`
#
# ### `Embedding`
# An embedding table maps integer ids to vectors:
# - input: token ids `idx` of shape `(...,)`
# - output: vectors of shape `(..., embedding_dim)`
#
# This is essentially a learnable lookup table.
#
# ### `Dropout`
# Dropout randomly zeroes activations during training to reduce overfitting.
# Implementation details:
# - Only active in `model.train()` mode
# - In training: drop with probability `p` and scale the kept values by `1/(1-p)` so the expected value stays the same
# - In eval: return the input unchanged
#
# ## Instructions
# - Do not use PyTorch reference modules for the parts you implement (e.g. don’t call nn.Linear inside your Linear).
# - You may use standard tensor ops that you learned before (matmul, sum, mean, rsqrt, indexing, etc.).
# - Use a parameter initialization method of your choice. We recommend something like Xavier-uniform.
#


# %%
class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()

        weights = torch.nn.init.xavier_uniform(
            torch.zeros(
                size=(out_features, in_features),
                requires_grad=True,
            )
        )

        self._weights: nn.Parameter = nn.Parameter(data=weights, requires_grad=True)

        self._bias: nn.Parameter | None = (
            nn.Parameter(
                data=torch.zeros(
                    size=(out_features,),
                    requires_grad=True,
                ),
                requires_grad=True,
            )
            if bias
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (..., in_features)
        return: (..., out_features)
        """
        bias = self._bias if self._bias is not None else 0.0

        return x @ self._weights.T + bias


# %%
class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self._embeddings = nn.Parameter(
            data=torch.nn.init.xavier_uniform(
                tensor=torch.zeros(
                    size=(num_embeddings, embedding_dim), requires_grad=True
                )
            ),
            requires_grad=True,
        )

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """
        idx: (...,) int64
        return: (..., embedding_dim)
        """
        return self._embeddings[idx]


# %%
class Dropout(nn.Module):
    def __init__(self, p: float):
        super().__init__()
        self._p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        In train mode: drop with prob p and scale by 1/(1-p).
        In eval mode: return x unchanged.
        """
        if self.training:
            drop = torch.rand(size=x.shape, requires_grad=False) >= self._p

            scale = 1 / (1 - self._p)

            x = x * drop * scale

        return x


# %% [markdown]
# ## Normalization
#
# Normalization layers help stabilize training by controlling activation statistics.
#
# ### LayerNorm
# LayerNorm normalizes each example across its **feature dimension** (the last dimension):
#
# - compute mean and variance over the last dimension
# - normalize: `(x - mean) / sqrt(var + eps)`
# - apply learnable per-feature scale and shift (`weight`, `bias`)
#
# **In this exercise, assume `elementwise_affine=True` (always include `weight` and `bias`).**
# `weight` and `bias` each have shape `(D,)`.
#
# LayerNorm is widely used in transformers because it does not depend on batch statistics.
#
# ### RMSNorm
# RMSNorm is similar to LayerNorm but normalizes using only the root-mean-square:
# - `x / sqrt(mean(x^2) + eps)` over the last dimension
# - usually includes a learnable scale (`weight`)
# - no mean subtraction
#
# RMSNorm is popular in modern LLMs because it's faster.
#


# %%
class LayerNorm(nn.Module):
    def __init__(
        self, normalized_shape: int, eps: float = 1e-5, elementwise_affine: bool = True
    ):
        super().__init__()

        self._normalized_shape = normalized_shape
        self._eps = eps

        self._weight: torch.nn.Parameter | None = None
        self._bias: torch.nn.Parameter | None = None

        if elementwise_affine:
            self._weight = torch.nn.Parameter(
                data=torch.ones(size=(normalized_shape,), requires_grad=True),
                requires_grad=True,
            )
            self._bias = torch.nn.Parameter(
                data=torch.zeros(size=(normalized_shape,), requires_grad=True),
                requires_grad=True,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize over the last dimension.
        x: (..., D)
        """

        x = (x - x.mean(dim=-1, keepdim=True)) / torch.sqrt(
            x.var(dim=-1, keepdim=True, unbiased=False) + self._eps
        )

        if self._weight is not None:
            assert self._bias is not None
            x = x * self._weight + self._bias

        return x


# %%
class RMSNorm(nn.Module):
    def __init__(self, normalized_shape: int, eps: float = 1e-8):
        super().__init__()

        self._normalized_shape = normalized_shape
        self._eps = eps

        self._weight: torch.nn.Parameter

        self._weight = torch.nn.Parameter(
            data=torch.ones(size=(normalized_shape,), requires_grad=True),
            requires_grad=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        RMSNorm: x / sqrt(mean(x^2) + eps) * weight
        over the last dimension.
        """
        x = x / torch.sqrt((x**2).mean(dim=-1, keepdim=True) + self._eps)

        return x * self._weight


# %% [markdown]
# ## MLPs and residual networks
#
# Now you’ll build larger networks by composing layers.
#
# ### MLP
# An MLP is a stack of `depth` Linear layers with non-linear activations (use GELU) in between.
# In this exercise you’ll support:
# - configurable depth
# - a hidden dimension
# - optional LayerNorm between layers (a common stabilization trick)
#
# A key skill is building networks using `nn.ModuleList` / `nn.Sequential` while keeping shapes consistent.
#
# ### Transformer-style FeedForward (FFN)
# A transformer block contains a position-wise feedforward network:
# - `D -> 4D -> D` (by default)
# - activation is typically **GELU**
#
# This is essentially an MLP applied independently at each token position.
#
# ### Residual wrapper
# Residual connections are the simplest form of “skip connection”:
# - output is `x + fn(x)`
#
# They improve gradient flow and allow training deeper networks more reliably.


# %%
class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        depth: int,
        use_layernorm: bool = False,
    ):
        super().__init__()
        # TODO: build modules (list of Linear + activation)
        # Optionally insert LayerNorm between layers.
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: implement
        raise NotImplementedError


# %%
class FeedForward(nn.Module):
    """
    Transformer-style FFN: D -> 4D -> D (default)
    """

    def __init__(self, d_model: int, d_ff: int | None = None):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        # TODO: create two Linear layers and choose an activation (GELU)
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: implement
        raise NotImplementedError


# %%
class Residual(nn.Module):
    def __init__(self, fn: nn.Module):
        super().__init__()
        # TODO: implement
        raise NotImplementedError

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        # TODO: return x + fn(x, ...)
        raise NotImplementedError


# %% [markdown]
# ## Classification problem
#
# In this section you’ll put everything together in a minimal MNIST classification experiment.
#
# You will:
# 1) download and load the MNIST dataset
# 2) implement cross-entropy from logits (stable, using log-softmax)
# 3) build a simple MLP-based classifier (flatten MNIST images first)
# 4) write a minimal training loop
# 5) report train loss curve and final accuracy
#
# The goal here is not to reach state-of-the-art accuracy, but to understand the full pipeline:
# data → model → logits → loss → gradients → parameter update.
#
# ### Model notes
# - We want you to combine the MLP we implemented above with the classification head we define below into one model
#
# ### MNIST notes
# - MNIST images are `28×28` grayscale.
# - After `ToTensor()`, each image has shape `(1, 28, 28)` and values in `[0, 1]`.
# - For an MLP classifier, we flatten to a vector of length `784`.
#
# ## Deliverables
# - Include a plot of your train loss curve in the video submission as well as a final accuracy.
# - **NOTE** Here we don't grade on model performance but we expect you to achieve at least 70% accuracy to confirm a correct model implementation.

# %%
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# %%
transform = transforms.ToTensor()  # -> float32 in [0,1], shape (1, 28, 28)

train_ds = datasets.MNIST(root="data", train=True, download=True, transform=transform)
test_ds = datasets.MNIST(root="data", train=False, download=True, transform=transform)

# TODO: define the dataloaders


# %%
def cross_entropy_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """
    Compute mean cross-entropy loss from logits.

    logits: (B, C)
    targets: (B,) int64

    Requirements:
    - Use log-softmax for stability (do not use torch.nn.CrossEntropyLoss, we check this in the autograder).
    """
    # TODO: implement
    raise NotImplementedError


# %%
class ClassificationHead(nn.Module):
    def __init__(self, d_in: int, num_classes: int):
        super().__init__()
        # TODO: implement
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (..., d_in)
        return: (..., num_classes) logits
        """
        # TODO: implement
        raise NotImplementedError


# %%
def accuracy(loader):
    # TODO: You can use this function to evaluate your model accuracy.
    raise NotImplementedError


# %%
def train_classifier(
    model: nn.Module,
    train_data_loader: DataLoader,
    test_data_loader: DataLoader,
    lr: float,
    epochs: int,
    seed: int = 0,
) -> list[float]:
    """
    Minimal training loop for MNIST classification.

    Steps:
    - define optimizer
    - for each epoch:
        - sample minibatches
        - forward -> cross-entropy -> backward -> optimizer step
      - compute test accuracy at the end of each epoch
    - return list of training losses (one per update step)

    Requirements:
    - call model.train() during training and model.eval() during evaluation
    - do not use torch.nn.CrossEntropyLoss (use your cross_entropy_from_logits)
    """
    # TODO: implement
    raise NotImplementedError


# %%
