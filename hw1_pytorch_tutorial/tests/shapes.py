"""Shared shape constants for the exercise 1 tests.

Every axis gets a *different* size on purpose: if two axes had the same length,
a permutation or contraction mistake could pass the test by coincidence.
"""

B = 2  # batch
T = 4  # sequence length / time
D = 3  # feature dimension
H = 5  # attention heads (or projection width)
Dh = 7  # per-head feature dimension
