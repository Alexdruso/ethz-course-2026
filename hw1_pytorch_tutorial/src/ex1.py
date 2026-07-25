# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: hw1_pytorch_tutorial (3.12.13.final.0)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Exercise 1: Tensor basics 
# In this exercise you will learn the basics of tensor creation, manipulation, indexing, broadcasting, vectorization, einsum, and attention masking fundamentals. These basics are important for understanding any complex implementation later on so make sure you understand them well.
#
# **To complete this exercise fill in all TODOs in the functions below.** 
#
# Make sure to check the output of your function and whether or not it fulfills the requirements outlined in the function definition. Do NOT change the function signature or name since we will be running checks on your functions during grading.
#
# ### Shape legend used in this notebook
# - `B`: batch size
# - `T`: sequence length / time
# - `D`: feature dimension
# - `H`: number of attention heads
# - `Dh`: per-head feature dimension
#
# ### Debugging tip: what to print
# When you get a shape error, print:
# - `x.shape`, `x.dtype`, `x.device`
# - `x.is_contiguous()` (important for `view`)
# For masks also print:
# - `mask.shape`, `mask.dtype`, `mask.sum()` and a small slice like `mask[0, :10]`
#
# ### Reproducibility tip: seeding in PyTorch
# Many operations in deep learning involve randomness (e.g., initializing model weights, shuffling data, dropout, random augmentations).
# **Seeding** sets the starting state of PyTorch’s random number generator so that these random choices become **repeatable**.
#
# - If you set the same seed and run the same code again, you should get the same *random* tensors / initial weights.
# - If you don’t set a seed, results can vary between runs.
#
# Common usage: `torch.manual_seed(seed)`
#
# Note: even with fixed seeds, some GPU operations can still be non-deterministic due to performance optimizations. For this assignment, seeding is mainly to make debugging easier and to ensure everyone can reproduce the same intermediate results. If you are given a seed, make sure to use it when creating tensors or performing other operations.

# %% [markdown]
# ## Tensor creation
# This warmup exercise teaches you how to create tensors with different shapes and values. A few details about tensor creation that are good to know:
# - `torch.tensor([...])` infers dtype from Python values (ints → integer tensor, floats → float tensor).
# - `torch.arange(start, end)` is **end-exclusive**.
# - `torch.linspace(start, end, steps)` is **end-inclusive**.

# %%
from collections.abc import Sequence
import torch


# %%
def make_tensor(data, dtype: torch.dtype | None = None, device: torch.device | str | None = None) -> torch.Tensor:
    """ Create a tensor from Python data (list/tuple/nested lists). """
    return torch.tensor(data, dtype=dtype, device=device)

x = make_tensor([[1, 2], [3, 4]], dtype=torch.float32)

x


# %%
def make_zeros(shape: Sequence[int], dtype: torch.dtype | None = None, device: torch.device | str | None = None) -> torch.Tensor:
    """Create a tensor filled with zeros."""
    return torch.zeros(shape, dtype=dtype, device=device)

z = make_zeros((2, 3), dtype=torch.float64)

z


# %%
def make_ones_like(x: torch.Tensor) -> torch.Tensor:
    """Create a tensor of ones with the same shape, dtype, and device as x. """
    return torch.ones_like(x)

base = torch.randn(2, 3, dtype=torch.float32)
ones = make_ones_like(base)

ones


# %%
def make_arange(start: int, end: int, step: int = 1, dtype: torch.dtype | None = None, device: torch.device | str | None = None) -> torch.Tensor:
    """Create a 1D tensor containing values [start, start+step, ..., < end]."""
    return torch.arange(start, end, step, dtype=dtype, device=device)

ar = make_arange(0, 5, 2, dtype=torch.int64)

ar


# %%
def make_linspace(start: float, end: float, steps: int, dtype: torch.dtype | None = None, device: torch.device | str | None = None) -> torch.Tensor:
    """Create a 1D tensor with evenly spaced values from start to end (inclusive)."""
    return torch.linspace(start, end, steps, dtype=dtype, device=device)

ls = make_linspace(0.0, 1.0, steps=5, dtype=torch.float32)

# ls

# %%
def make_randn(shape: Sequence[int], seed: int | None = None, dtype: torch.dtype | None = None, device: torch.device | str | None = None) -> torch.Tensor:
    """Create a tensor filled with values from a standard normal distribution."""
    if seed is not None:
        torch.manual_seed(seed)
    return torch.randn(shape, dtype=dtype, device=device)

a = make_randn((2, 3), seed=123, dtype=torch.float32)

a


# %%
def cast_dtype_and_move(x: torch.Tensor, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Convert tensor dtype and move to device."""
    return x.to(device=device, dtype=dtype)

casted = cast_dtype_and_move(torch.tensor([1, 2, 3]), torch.device("cpu"), torch.float32)

casted.device


# %% [markdown]
# ## Shape manipulation
# Now that we covered the basic tensor creation schemes, we want to focus on shape manipulation. Understanding the difference between these mechanisms is key for building larger systems and many people still get it wrong. 
# The core ideas to understand are:
# - **Contiguous tensors** store data in a single, row-major memory layout.
# - Many ops (especially slicing like `x[:, ::2]`, `transpose`, `permute`) often create **non-contiguous** tensors (no copy but different strides).
# - `view(...)` is **zero-copy** but typically requires **contiguous** memory → may throw an error.
# - `reshape(...)` tries to return a view, but if the tensor is non-contiguous it will **allocate/copy**.
# - `contiguous()` forces a contiguous copy when the tensor isn’t contiguous.
#
# If you *need* a view after reordering dims: call `x = x.contiguous()` first (this makes a contiguous copy).

# %%
def reshape_tensor(x: torch.Tensor, new_shape: Sequence[int]) -> torch.Tensor:
    """Reshape tensor to new_shape (may return a view or a copy)."""
    return x.reshape(new_shape)

x = torch.arange(6)
y = reshape_tensor(x, (2, 3))


# %%
def view_tensor(x: torch.Tensor, new_shape: Sequence[int]) -> torch.Tensor:
    """View tensor as new_shape (requires contiguous memory and doesn't allocate new memory for the tensor data)."""
    return x.contiguous().view(new_shape)

y_view = view_tensor(x, (2, 3))


# %%
def flatten_from_dim(x: torch.Tensor, start_dim: int = 0) -> torch.Tensor:
    """Flatten a tensor starting from start_dim into a single dimension."""
    return x.flatten(start_dim=start_dim)

x2 = torch.randn(2, 3, 4)
flat = flatten_from_dim(x2, start_dim=1)

print(x2)
print(flat)


# %%
def add_singleton_dim(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Insert a size-1 dimension at position dim."""
    return torch.unsqueeze(x, dim=dim)

x3 = torch.randn(5, 7)
x3s = add_singleton_dim(x3, dim=1)


# %%
def remove_singleton_dims(x: torch.Tensor, dim: int | None = None) -> torch.Tensor:
    """Remove size-1 dimensions."""
    return x.squeeze(dim=dim) if dim else x.squeeze()

x4 = torch.randn(2, 1, 3)
x4s = remove_singleton_dims(x4)


# %%
def transpose_last_two(x: torch.Tensor) -> torch.Tensor:
    """Swap the last two dimensions of x."""
    return x.transpose(-1, -2)

x6 = torch.randn(2, 3, 4)
x6t = transpose_last_two(x6)
print(x6.shape)
print(x6t.shape)


# %%
def permute_bhwc_to_bchw(x: torch.Tensor) -> torch.Tensor:
    """Convert (B, H, W, C) tensor into (B, C, H, W)."""
    return x.permute(0, 3, 1, 2)

x7 = torch.randn(8, 32, 32, 3)
x7p = permute_bhwc_to_bchw(x7)

print(x7.shape)
print(x7p.shape)


# %%
def make_contiguous(x: torch.Tensor) -> torch.Tensor:
    """Check if tensor is contiguous and if not make contiguous."""
    return x.contiguous() if not x.is_contiguous() else x

x8 = torch.randn(4, 6)[:, ::2]
x8c = make_contiguous(x8)


# %% [markdown]
# ## Indexing
# Now that we know how to create tensors and manipulate them we need to understand how we can extract certain components from them using indexing. 
# - Basic slicing (`x[a:b]`) returns a view when possible.
# - “Fancy” indexing (lists/tensors of indices) usually allocates a new tensor.
# - In-place vs out-of-place matters: if a function says “return a copy, leave the input unchanged”, you need `clone()`.

# %%
def slice_rows(x: torch.Tensor, start: int, end: int) -> torch.Tensor:
    """Slice rows in a 2D tensor: x[start:end, :]."""
    return x[start:end, :]

x = torch.arange(12).reshape(4, 3)
rows = slice_rows(x, 1, 3)

rows


# %%
def select_columns(x: torch.Tensor, cols: Sequence[int]) -> torch.Tensor:
    """Select specific columns from a 2D tensor."""
    return x[:, cols]

cols = select_columns(x, [0, 2])

cols


# %%
def get_diagonal(x: torch.Tensor) -> torch.Tensor:
    """Get the diagonal of a 2D tensor."""
    return x.diag()

d = get_diagonal(torch.tensor([[1, 2], [3, 4]]))

d


# %%
def set_subtensor(x: torch.Tensor, row_idx: int, col_idx: int, value: float) -> torch.Tensor:
    """Return a copy of x where x[row_idx, col_idx] is set to value."""
    result = x.clone()
    result[row_idx, col_idx] = value
    return result
    

base = torch.zeros(2, 2)
out = set_subtensor(base, 0, 1, 5.0)

out


# %%
def gather_rows(x: torch.Tensor, row_indices: torch.Tensor) -> torch.Tensor:
    """Gather (concat) rows from x using row_indices."""
    return x.gather(0, row_indices.unsqueeze(1).expand(-1, x.size(1)))

x2 = torch.tensor([[10, 11], [20, 21], [30, 31]])
idx = torch.tensor([2, 0])
gathered = gather_rows(x2, idx)

print(x2)
print(idx)
print(gathered)


# %% [markdown]
# ## Broadcasting and reducing
# Now we're covering a pytorch mechanism that lets you apply elementwise ops without using python loops. It's important to understand how it works to trace your shapes in complicated systems. The broadcasting rules to know are:
# - Dimensions align from the **right**.
# - A dimension can broadcast if it’s equal or one of them is **1**.
#
# ### Reduction ops and `keepdim`
#
# When you reduce over a dimension (e.g. `sum`, `mean`, `max`), PyTorch can either:
#
# - **remove** the reduced dimension (`keepdim=False`, default), or
# - **keep** it as size 1 (`keepdim=True`)
#
# Keeping the dimension is often helpful because it makes broadcasting back “just work”.
#
# #### Shape diagram examples
#
# Assume `x` has shape `(B, T, D)`:
#
# **Sum over time**
# - `x.sum(dim=1)` → shape `(B, D)`
# - `x.sum(dim=1, keepdim=True)` → shape `(B, 1, D)`
#
# **Mean over features**
# - `x.mean(dim=2)` → shape `(B, T)`
# - `x.mean(dim=2, keepdim=True)` → shape `(B, T, 1)`
#
# #### Why `keepdim=True` helps with broadcasting
#
# Example: center `x` by subtracting the mean over `T`
#
# - If `m = x.mean(dim=1)` has shape `(B, D)`, then `x - m` **fails** (shapes `(B,T,D)` and `(B,D)` don't align).
# - If `m = x.mean(dim=1, keepdim=True)` has shape `(B,1,D)`, then `x - m` **works** via broadcasting.

# %%
def sum_over_dim(x: torch.Tensor, dim: int, keepdim: bool = False) -> torch.Tensor:
    """Sum tensor values along dimension dim."""
    return x.sum(dim=dim, keepdim=keepdim)

x = torch.ones(2, 3)
y = sum_over_dim(x, dim=1)

print(x)
print(y)


# %%
def mean_over_dim(x: torch.Tensor, dim: int, keepdim: bool = False) -> torch.Tensor:
    """Mean along dimension dim."""
    return x.mean(dim=dim, keepdim=keepdim)

x2 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
y2 = mean_over_dim(x2, dim=0)


# %%
def max_over_dim(x: torch.Tensor, dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Max values and argmax indices along dimension dim."""
    return torch.max(x, dim=dim)

x3 = torch.tensor([[1.0, 5.0], [3.0, 2.0]])
values, idx = max_over_dim(x3, dim=1)

print(x3)
print(values)
print(idx)


# %%
def argmax_over_dim(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Argmax indices along dimension dim."""
    return x.argmax(dim=dim)

idx2 = argmax_over_dim(x3, dim=1)


# %%
def broadcast_add_vector(x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Add a vector v to each row of a 2D tensor x using broadcasting."""
    return x + v

x4 = torch.zeros(3, 2)
v = torch.tensor([10.0, 20.0])
y4 = broadcast_add_vector(x4, v)

print(x4)
print(v)
print(y4)


# %% [markdown]
# ## Vectorization
# We want to avoid slow (due to per-iteration overhead) python loops as much as possible and pytorch gives us many tools to avoid it. We cover these basics:
# - `cat` vs `stack` (concatenate existing dims vs create a new dim)
# - `repeat` vs `expand`
# - `scatter_add` / `index_add` for accumulation
# - `where` for conditional selection
#
# ### `expand` vs `repeat`
#
# - `repeat(...)` **copies** data → larger tensor with independent storage.
# - `expand(...)` **does not copy** data → it creates a *view* with clever strides.
#
# This has two important implications:
#
# 1) `expand` only works when expanding a **size-1 dimension** (broadcasting a singleton).
# 2) The expanded tensor may have **many positions pointing to the same memory**.  
#    Modifying the expanded tensor can therefore produce surprising results (multiple rows change).
#
# Rule of thumb:
# - Use `expand` for read-only broadcasting.
# - Use `repeat` if you truly need independent copies.
#
#
# NOTE: We implore you to write your own quick checks from now on for calling the functions and checking their output. As before you are still required to fill in the TODOs in each function.

# %%
def concat_tensors(tensors: Sequence[torch.Tensor], dim: int = 0) -> torch.Tensor:
    """Concatenate tensors along dim. NOTE: This will always allocate new memory"""
    return torch.cat(tensors, dim=dim)


# %%
def stack_tensors(tensors: Sequence[torch.Tensor], dim: int = 0) -> torch.Tensor:
    """Stack tensors along a new dimension dim."""
    return torch.stack(tensors, dim=dim)


# %%
def repeat_tensor(x: torch.Tensor, repeats: Sequence[int]) -> torch.Tensor:
    """Repeat tensor along each dimension."""
    return x.repeat(*repeats)


# %%
def expand_tensor(x: torch.Tensor, *sizes: int) -> torch.Tensor:
    """Expand tensor to a larger size without copying data.(Sizes can be -1 to keep original dimension.)"""
    return x.expand(*sizes)


# %%
def cumsum_over_dim(x: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """Cumulative sum along dim."""
    return x.cumsum(dim=dim)


# %%
def where_select(mask: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Elementwise select: return a where mask is True else b. mask must be broadcastable to a and b."""
    return torch.where(mask, a, b)


# %%
def one_hot(indices: torch.Tensor, num_classes: int, dtype: torch.dtype | None = None) -> torch.Tensor:
    """
    Create one-hot encodings.
    Output is a tensor of the same shape as indices with an added dimension of size num_classes at the end, 
    where the value along that dimension is 1 if it matches the index and 0 otherwise.

    Shapes:
    - indices: (...,) integer tensor
    Return:
    - out: (..., num_classes)

    Requirements:
    - Must work for arbitrary leading shape.
    - No Python loops.
    """
    encoding = torch.zeros(*indices.shape, num_classes, dtype=dtype)
    return encoding.scatter(-1, indices.unsqueeze(-1), 1)

example_indices = torch.tensor([0, 2, 1])
num_classes = 3
one_hot_encoded = one_hot(example_indices, num_classes)

one_hot_encoded


# %%
def scatter_add_1d(
    values: torch.Tensor, indices: torch.Tensor, size: int
) -> torch.Tensor:
    """
    Sum `values` into an output vector at positions `indices`.

    Shapes:
    - values: (N,)
    - indices: (N,) integer indices in [0, size)
    Return:
    - out: (size,) with same dtype and device as values

    Requirement:
    - no Python loops
    """
    out = torch.zeros(size, dtype=values.dtype, device=values.device)
    return out.scatter_add(0, indices, values)
    


# %%
def batched_token_histogram(tokens: torch.Tensor, vocab_size: int) -> torch.Tensor:
    """
    Count token occurrences per batch item.

    Shapes:
    - tokens: (B, T) int64
    Return:
    - counts: (B, vocab_size) where counts[b, v] = number of times token v appears in tokens[b] 

    Requirements:
    - No Python loops over B or T.
    """
    return one_hot(tokens, vocab_size).sum(dim=1)


# %%
def masked_mean(x: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Mean over `dim` considering only mask==True entries.

    Convention:
    - mask: bool tensor broadcastable to x
    - mask==True means "keep this entry"

    Return: same shape as x.mean(dim=dim)

    Requirements:
    - Avoid division by zero: if all mask are False along `dim`, define mean as 0.
    """
    masked = x * mask
    return masked.sum(dim=dim) / mask.sum(dim=dim).clamp(min=1)  # Avoid division by zero


example_x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
example_mask = torch.tensor([[True, False], [False, True]])
mean_result = masked_mean(example_x, example_mask, dim=1)

mean_result


# %% [markdown]
# ## Einsum warmup
# Now that you’re comfortable with shapes and broadcasting, we’ll introduce `torch.einsum`, a concise way to express tensor operations by explicitly naming axes and summing over repeated indices.
#
# ### The idea
# You describe each input tensor by labeling its dimensions with letters, e.g.
# - `x: (B, T, D)` → `"btd"`
# - `W: (D, H)`    → `"dh"`
#
# Then you tell einsum what output labels you want:
# - `"btd,dh->bth"`
#
# ### Rules of einsum
# 1) **Same letter = same axis** (must match in size, except broadcastable size-1).
# 2) **Repeated letters are summed over** (a “contraction”).
# 3) **Letters that appear in the output are kept** (in that order).
# 4) You can **reorder axes** just by changing the output label order.
#
# ### Tiny cheat sheet
# - Sum over an axis: `"btd->bt"` (sums over `d`)
# - Transpose: `"ij->ji"`
# - Dot product: `"d,d->"` or batched `"btd,btd->bt"`
# - Matrix multiply: `"ik,kj->ij"`
# - Batched matmul: `"bij,bjk->bik"`
# - Outer product: `"i,j->ij"`
#
# ### How to derive an einsum (recommended workflow)
# 1) Write down shapes with named axes (e.g. `q: b h t d`, `k: b h s d`).
# 2) Decide which axes you want to **sum over** (give them the same letter in both inputs).
# 3) Decide which axes you want to **keep** in the output (write them after `->`).
#
# In this section, you’ll use einsum to implement building blocks that show up in attention:
# - linear projections (`x @ W`)
# - dot products
# - attention score matrices (`QKᵀ`)
# - applying attention weights (`softmax(scores) @ V`)
#
# NOTE: For these exercises you are required to use `torch.einsum` not `matmul` (we check). You are also not required to understand the attention mechanism at this point and the exercises are sovable without. It is good however, to remember the implementations in this exercise for future implementations.

# %%
def einsum_linear_btd_dh_to_bth(x: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    """
    Linear projection using einsum.

    Shapes:
    - x: (B, T, D)
    - W: (D, H)
    Return:
    - y: (B, T, H)
    """
    return torch.einsum("btd, dh -> bth", x, W)


# %%
def einsum_pairwise_dot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Pairwise dot product between x and y.

    Shapes:
    - x: (B, T, D)
    - y: (B, T, D)
    Return:
    - dots: (B, T) where dots[b,t] = dot(x[b,t], y[b,t])
    """
    return torch.einsum("btd, btd -> bt", x, y)


# %%
def einsum_qk_scores(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """
    Compute attention scores QK^T using einsum.

    Shapes:
    - q: (B, H, T, Dh)
    - k: (B, H, T, Dh)
    Return:
    - scores: (B, H, T, T) where scores[b,h,i,j] = dot(q[b,h,i], k[b,h,j])
    """
    transposed_k = torch.einsum("bhtd -> bhdt", k)

    return torch.einsum("bhti, bhjt -> bhij", q, transposed_k)


# %%
def einsum_apply_attention(weights: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Apply attention weights to values using einsum.

    Shapes:
    - weights: (B, H, T, T)
    - v:       (B, H, T, Dh)
    Return:
    - out:     (B, H, T, Dh) where out[b,h,i] = sum_j weights[b,h,i,j] * v[b,h,j]
    """
    return torch.einsum("bhij, bhjd -> bhid", weights, v)


# %% [markdown]
# ## Attention Fundamentals
# This exercise introduces some building blocks of the attention mechanism which we will encounter extensively throughout the course. It's not yet required for you to fully understand the mechanism to implement the exercises. However, it's good to remember these building blocks for the future. 
#
# To complete the exercises you should familiarize yourself with these topics:
# - Stable softmax read: https://jaykmody.com/blog/stable-softmax/
# - Masking: typically this means setting masked logits to -inf *before* softmax.
# - For attention: causal masks are upper-triangular (no attending to the future).

# %%
def stable_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Numerically stable softmax along `dim`.

    Requirements:
    - Must not overflow for large values in x.
    - Output sums to 1 along `dim`.
    """
    x_max = x.max(dim=dim, keepdim=True).values
    x_exp = torch.exp(x - x_max)
    return x_exp / x_exp.sum(dim=dim, keepdim=True)


# %%
def masked_fill_tensor(x: torch.Tensor, mask: torch.Tensor, value: float) -> torch.Tensor:
    """
    Return a copy of x where positions with mask == True are replaced by `value`.
    
    Requirements:
    - mask must be broadcastable to x.
    - do NOT modify x in-place.
    """
    return torch.where(mask, torch.full_like(x, value), x)


# %%
def masked_softmax(x: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Softmax over x with a boolean mask.

    Convention:
    - mask == True means "invalid and must receive probability 0".
    - Do masking before softmax (i.e., set invalid logits to a large negative).”

    Requirements:
    - Must be numerically stable.
    - Output must be exactly 0 where mask==True.
    - If all entries are masked along `dim`, return all zeros along `dim`.
    - You may reuse functions you implemented above.
    """
    x_hat = masked_fill_tensor(x, mask, -torch.inf)
    return stable_softmax(x_hat, dim=dim)



# %%
def make_causal_mask(T: int, device: torch.device | str | None = None) -> torch.Tensor:
    """
    Create a causal (future-masking) boolean mask of shape (T, T).

    Convention:
    - mask[i, j] == True  => position (i attends to j) is NOT allowed (j is in the future)
    - mask[i, j] == False => allowed

    So this is an upper-triangular mask above the diagonal.

    Return:
    - mask: boolean tensor on the specified device

    Example (T=4):
        [[F, T, T, T],
         [F, F, T, T],
         [F, F, F, T],
         [F, F, F, F]]
    """
    return ~torch.tril(torch.ones(T, T, dtype=torch.bool, device=device))

# %%
def apply_causal_mask(attn_logits: torch.Tensor, value: float = -1e9) -> torch.Tensor:
    """
    Apply a causal mask to attention logits.

    Expected shapes:
    - attn_logits: (..., T, T)

    Returns:
    - masked logits (same shape) where masked positions have been set to `value`.

    Notes:
    - Create a causal mask for the final two dims.
    - Broadcast it across leading dims.
    - You may reuse functions declared above.
    """
    causal_mask = make_causal_mask(attn_logits.shape[-1], device=attn_logits.device)
    return masked_softmax(attn_logits, causal_mask)
