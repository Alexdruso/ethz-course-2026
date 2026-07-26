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
# # Exercise 2: PyTorch core
#
# In this exercise you’ll build core PyTorch “muscle memory” that you’ll reuse in basically every model you write:
#
# - **Autograd**: how gradients are created, how they accumulate, and how to compute gradients for one or multiple inputs.
# - **Dataloading**: writing small `Dataset`s, using `DataLoader`, and custom `collate_fn`.
# - **Optimizers**: implementing **AdamW** updates from scratch (state, bias correction, weight decay).
# - **Training basics**: a clean single training step.
# - **Initialization**: fan-in/out and common initializers (Xavier / Kaiming), plus a helper to init `nn.Linear`.
#
# As before: fill in all `TODO`s without changing function names or signatures.
# When debugging, print shapes/dtypes/devices, and write tiny sanity checks (e.g. compare to PyTorch’s built-ins).
#

# %%
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

# %% [markdown]
# ## Autograd fundamentals
#
# PyTorch builds a computation graph when you apply operations to tensors with `requires_grad=True`.
# Calling `backward()` (or `torch.autograd.grad`) computes gradients by traversing that graph.
#
# ### Key concepts
# - **Leaf tensor**: a tensor created by you (not the result of an operation) with `requires_grad=True`. Leaf tensors can store gradients in `.grad`.
# - **Gradient accumulation**: calling `backward()` adds into `.grad` (it does not overwrite). You must reset gradients between steps/calls.
# - **`torch.autograd.grad` vs `.backward()`**
#   - `torch.autograd.grad(f, x)` returns `df/dx` directly and does not write into `x.grad` unless you explicitly do so.
#   - `f.backward()` writes gradients into `.grad` of leaf tensors.
#
# In the next functions you’ll compute gradients for a simple scalar function such as `f(x) = sum(x^2)` using both APIs.
#
# ### `torch.no_grad()`
# Wrap inference-only code to avoid tracking gradients and building graphs:
# - saves memory
# - speeds up evaluation
#
# ### `detach()`
# `y = x.detach()` returns a tensor that shares data with `x` but is **not connected** to the autograd graph.
# This is useful when you want to treat something as a constant target.
#
# ### `model.train()` vs `model.eval()`
# - `train()` enables training behavior (e.g. dropout active, batchnorm updates running stats).
# - `eval()` enables inference behavior (e.g. dropout off, batchnorm uses running stats).


# %%
def grad_with_autograd_grad(x: torch.Tensor) -> torch.Tensor:
    """
    Compute gradient of f(x) = sum(x^2) using torch.autograd.grad

    Requirements:
    - Do not call .backward().
    - x should require grad inside the function (don't assume it does).
    - Must return df/dx
    """
    x = x.clone().detach()  # avoid modifying the original tensor

    x.requires_grad_(True)

    result = (x**2).sum()

    return torch.autograd.grad(result, x)[0]


# %%


def grad_with_backward(x: torch.Tensor) -> torch.Tensor:
    """
    Compute gradient of f(x) = sum(x^2) using .backward().

    Requirements:
    - Must return df/dx
    - Must not leak gradients across calls (watch x.grad accumulation)
    """

    x = x.clone().detach()  # avoid modifying the original tensor

    x.requires_grad_(True)

    result = (x**2).sum()

    result.backward()

    gradient = x.grad

    assert gradient is not None, "Gradient should not be None after backward()"

    return gradient


# %%
def grad_wrt_multiple_inputs(
    a: torch.Tensor,
    b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute gradients w.r.t. multiple inputs. The function is f(a, b) = sum(a^2 + ab).

    Return:
        (df/da, df/db)

    Requirements:
    - Use torch.autograd.grad
    - Ensure both a and b require grad in this function.
    """
    a = a.clone().detach()
    b = b.clone().detach()

    a.requires_grad_(True)
    b.requires_grad_(True)

    fn = (a**2 + a * b).sum()

    return torch.autograd.grad(fn, (a, b))


# %% [markdown]
# ## Dataloading
#
# In PyTorch, a `Dataset` defines how to fetch a *single* training example, and a `DataLoader` handles:
# - batching
# - shuffling
# - parallel workers
# - optional custom batching logic via `collate_fn`
#
# ### `Dataset` in one sentence
# A `Dataset` only needs:
# - `__len__`: number of items
# - `__getitem__`: return one item (e.g. `(x, y)`)
#
# ### Why `collate_fn` matters
# The default DataLoader collation stacks items along a new batch dimension.
# That works for fixed-size tensors, but it breaks for **variable-length sequences**.
#
# So we’ll implement padding ourselves:
# - Convert a list of 1D token sequences into a padded tensor `(B, T_max)`
# - Track `lengths` and a `padding_mask`
#
# ### Mask convention for padding
# For padding masks in this exercise:
# - `padding_mask[b, t] == True` means **this is padding / invalid**
# - `padding_mask[b, t] == False` means **this is a real token**

# %%
from torch.utils.data import DataLoader, Dataset


# %%
class TensorPairDataset(Dataset):
    """
    Minimal dataset wrapping (x, y).

    x: (N, ...)
    y: (N, ...)

    N is the number of samples. The dataset should return tuples of (x[i], y[i]).
    """

    def __init__(self, x: torch.Tensor, y: torch.Tensor):
        assert len(x) == len(y)

        self.x = x.clone()
        self.y = y.clone()

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return (self.x[idx], self.y[idx])


# %%
class NextTokenDataset(Dataset):
    """
    Next-token prediction dataset.

    Given tokens of shape (N, T), produce:
      input_ids  = tokens[:, :-1]
      target_ids = tokens[:, 1:]

    Return per item:
      (input_ids, target_ids)

    Notes:
    - Returned tensors should be 1D of length (T-1).
    - dtype should remain integer.
    """

    def __init__(self, tokens: torch.Tensor):
        self.tokens = tokens.clone()

    def __len__(self) -> int:
        return len(self.tokens)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return (self.tokens[idx, :-1], self.tokens[idx, 1:])


# %%


class RandomCropSequenceDataset(Dataset):
    """
    Sequence dataset that returns random crops of fixed length.

    tokens: (N, T_total)
    crop_len: L

    For each __getitem__:
      - sample a start index s so that s+L <= T_total
      - return tokens[idx, s:s+L]

    Requirements:
    - Use a torch.Generator for deterministic behavior if seed is provided.
    - Do NOT use Python's random module.
    """

    def __init__(self, tokens: torch.Tensor, crop_len: int, seed: int | None = None):
        self.t_total = tokens.shape[1]

        assert self.t_total >= crop_len

        self.tokens = tokens
        self.crop_len = crop_len

        self.generator = torch.Generator()

        if seed is not None:
            self.generator = self.generator.manual_seed(seed)

    def __len__(self) -> int:
        return len(self.tokens)

    def __getitem__(self, idx: int) -> torch.Tensor:
        sample = self.tokens[idx]

        start_index = torch.randint(
            low=0,
            high=self.t_total - self.crop_len + 1,
            size=(1,),
            generator=self.generator,
        )

        return sample[start_index : start_index + self.crop_len]


# %%


@dataclass(frozen=True)
class PaddedBatch:
    """
    A padded batch for variable-length sequences.

    tokens: LongTensor (B, T_max)
    lengths: LongTensor (B,)
    padding_mask: BoolTensor (B, T_max) where True means "this is padding"
    """

    tokens: torch.Tensor
    lengths: torch.Tensor
    padding_mask: torch.Tensor


def pad_1d_sequences(seqs: list[torch.Tensor], pad_value: int = 0) -> PaddedBatch:
    """
    Pad a list of 1D integer tensors to the same length.

    Requirements:
    - Return PaddedBatch(tokens, lengths, padding_mask)
    - padding_mask[b, t] == True iff t >= lengths[b]
    - tokens should be dtype long, if not cast them
    """
    max_length: int = max(len(sequence) for sequence in seqs)

    tokens = torch.tensor(data=pad_value, dtype=torch.long).repeat(
        (len(seqs), max_length)
    )

    padding_mask = torch.ones_like(tokens, dtype=torch.bool)

    for idx, sequence in enumerate(seqs):
        tokens[idx, : len(sequence)] = sequence
        padding_mask[idx, : len(sequence)] = False

    return PaddedBatch(
        tokens=tokens,
        lengths=torch.tensor(
            data=tuple(len(sequence) for sequence in seqs), dtype=torch.long
        ),
        padding_mask=padding_mask,
    )


# %%
def collate_next_token_batch(
    batch: list[tuple[torch.Tensor, torch.Tensor]], pad_value: int = 0
) -> dict[str, torch.Tensor]:
    """
    Collate for NextTokenDataset samples that may have variable lengths.

    batch: list of (input_ids, target_ids), each 1D

    Return dict with:
      - input_ids: (B, T_max)
      - target_ids: (B, T_max)
      - attention_mask: (B, T_max) where True means "keep" (NOT padding)
      - padding_mask: (B, T_max) where True means "padding"

    Requirements:
    - pad input_ids and target_ids consistently
    - attention_mask is the logical NOT of padding_mask
    """
    input_padding = pad_1d_sequences(
        seqs=[input_sequence for input_sequence, output_sequence in batch],
        pad_value=pad_value,
    )
    output_padding = pad_1d_sequences(
        seqs=[output_sequence for input_sequence, output_sequence in batch],
        pad_value=pad_value,
    )

    return {
        "input_ids": input_padding.tokens,
        "target_ids": output_padding.tokens,
        "attention_mask": ~input_padding.padding_mask,
        "padding_mask": input_padding.padding_mask,
    }


# %%
def make_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = True,
    drop_last: bool = False,
    collate_fn=None,
    num_workers: int = 0,
) -> DataLoader:
    """
    Create a DataLoader with optional collate_fn.
    """
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )


# %% [markdown]
# ## Optimizers (AdamW from scratch)
#
# PyTorch optimizers keep **state** for each parameter (e.g. moment estimates in Adam).
# In this section you’ll implement **AdamW**, which is Adam + *decoupled* weight decay.
#
# ### AdamW state
# For each parameter tensor `p` we store:
# - `m`: first moment (EMA of gradients)
# - `v`: second moment (EMA of squared gradients)
# - `t`: step counter
#
# ### Update overview (high level)
# 1) Update moments `m, v`
# 2) Bias-correct them (`m_hat, v_hat`)
# 3) Apply parameter update:
#    `p -= lr * ( m_hat / (sqrt(v_hat) + eps) + weight_decay * p )`
#
# Notes:
# - This update is **in-place** (mutates `p`).
# - Gradients should not be modified.
# - State tensors must match parameter shape/device/dtype.


# %%
@dataclass
class AdamWState:
    """
    Per-parameter AdamW state.

    m: first moment
    v: second moment
    t: step count
    """

    m: torch.Tensor
    v: torch.Tensor
    t: int


def init_adamw_state(p: torch.Tensor) -> AdamWState:
    """
    Initialize AdamW state tensors for a parameter tensor p.

    What to create:
    - m: zeros like p, same shape/device/dtype
    - v: zeros like p, same shape/device/dtype
    - t: step counter starting at 0

    Notes / requirements:
    - Use torch.zeros_like(p) for m and v.
    - Do NOT attach gradients to the state (initialize under torch.no_grad()).
    - t starts at 0. In adamw_step_, increment t to 1 on the first update *before*
      computing bias correction terms (1 - beta1^t) and (1 - beta2^t).
    - State tensors must live on the same device as p (CPU vs GPU) and have the
      same dtype as p.
    """
    with torch.no_grad():
        return AdamWState(m=torch.zeros_like(p), v=torch.zeros_like(p), t=0)


# %%
def adamw_step_(
    p: torch.Tensor,
    grad: torch.Tensor,
    state: AdamWState,
    lr: float,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    weight_decay: float = 0.01,
) -> AdamWState:
    """
    In-place AdamW parameter update (updates p).

    Algorithm (AdamW):
      m = beta1*m + (1-beta1)*grad
      v = beta2*v + (1-beta2)*grad^2
      m_hat = m / (1 - beta1^t)
      v_hat = v / (1 - beta2^t)
      p = p - lr * (m_hat / (sqrt(v_hat) + eps) + weight_decay * p)

    Requirements:
    - Update p in-place.
    - Return updated state (with incremented t).
    - Do not modify grad.
    - Should work for any tensor shape.
    """
    with torch.no_grad():
        t = state.t
        t += 1

        beta1, beta2 = betas

        m = beta1 * state.m + (1 - beta1) * grad
        v = beta2 * state.v + (1 - beta2) * grad**2

        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)

        p = p - lr * (m_hat / (v_hat**2 + eps) + weight_decay * p)

        return AdamWState(m=m, v=v, t=t)


# %%
def adamw_step_many_(
    params: list[torch.Tensor],
    grads: list[torch.Tensor],
    states: list[AdamWState],
    lr: float,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    weight_decay: float = 0.01,
) -> list[AdamWState]:
    """
    Apply AdamW to many parameters.

    Requirements:
    - len(params) == len(grads) == len(states)
    - Update each param in-place.
    - Return the list of updated states.
    """
    # TODO: implement
    raise NotImplementedError


# %% [markdown]
# ## Training basics
#
# A minimal training step follows the same pattern almost everywhere:
#
# 1) set model to train mode
# 2) reset gradients
# 3) forward pass
# 4) compute loss
# 5) backward pass
# 6) step optimizer
#
# In this exercise you’ll implement a single MSE training step using a standard PyTorch optimizer.
# Return a Python float loss value.


# %%
def train_step_mse(
    model: nn.Module,
    batch: tuple[torch.Tensor, torch.Tensor],
    optimizer: torch.optim.Optimizer,
) -> float:
    """
    One MSE train step using standard torch optimizer.
    """
    # TODO: implement
    raise NotImplementedError


# %% [markdown]
# ## Parameter initialization
#
# Initialization matters because it controls signal and gradient scales at the start of training.
#
# ### Fan-in / fan-out
# - `fan_in`: number of input connections to a unit
# - `fan_out`: number of output connections from a unit
#
# For a Linear layer weight of shape `(out_features, in_features)`:
# - `fan_in = in_features`
# - `fan_out = out_features`
#
# ### Common schemes
# - **Xavier / Glorot** (often good for tanh / linear-ish nets):
#   keeps variance stable across layers when activations are roughly symmetric.
# - **Kaiming / He** (often good for ReLU-like nets):
#   accounts for the fact that ReLU zeroes out about half the inputs.
#
# In this section you’ll implement Xavier uniform and Kaiming uniform and use them to initialize `nn.Linear`.
# We also always zero the bias unless explicitly told otherwise.


# %%
def fan_in_fan_out(weight: torch.Tensor) -> tuple[int, int]:
    """Compute (fan_in, fan_out) for a weight tensor."""
    # TODO: implement
    raise NotImplementedError


# %%
def xavier_uniform_(weight: torch.Tensor, gain: float = 1.0) -> torch.Tensor:
    """
    In-place Xavier/Glorot uniform init:
      bound = gain * sqrt(6 / (fan_in + fan_out))
      U(-bound, bound)
    """
    # TODO: implement
    raise NotImplementedError


# %%
def kaiming_uniform_(weight: torch.Tensor, nonlinearity: str = "relu") -> torch.Tensor:
    """
    In-place Kaiming/He uniform init.

    Follow this common choice:
      gain = sqrt(2) for ReLU
      std = gain / sqrt(fan_in)
      bound = sqrt(3) * std
      U(-bound, bound)
    """
    # TODO: implement
    raise NotImplementedError


# %%
def init_linear_(layer: nn.Linear, scheme: str = "xavier") -> nn.Linear:
    """
    Initialize an nn.Linear in-place.

    scheme:
      - "xavier"
      - "kaiming_relu"
      - "zero" (weights and bias = 0)
    """
    # TODO: implement
    raise NotImplementedError
