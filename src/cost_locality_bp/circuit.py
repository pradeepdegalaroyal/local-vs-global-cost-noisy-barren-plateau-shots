"""Layered RY + CZ brick-wall ansatz: structure helpers."""

from __future__ import annotations

import math
from functools import lru_cache

import torch


def depth_for(n: int, schedule: str) -> int:
    """Number of layers for ``n`` qubits under a depth schedule.

    ``log``: ceil(log2 n) + 1 (the O(log n) regime in which local costs stay trainable
    without noise); ``linear``: n (the regime in which noise-induced barren plateaus are proven).
    """
    if schedule == "log":
        return math.ceil(math.log2(n)) + 1
    if schedule == "linear":
        return n
    if schedule.startswith("fixed"):
        return int(schedule[5:])
    raise ValueError(f"unknown depth schedule: {schedule}")


def brick_pairs(n: int, parity: int) -> list[tuple[int, int]]:
    """Nearest-neighbour pairs (i, i+1) with i % 2 == parity."""
    return [(i, i + 1) for i in range(parity, n - 1, 2)]


def cz_sign_vectors(n: int, device: str | torch.device = "cpu",
                    dtype: torch.dtype = torch.float64) -> list[torch.Tensor]:
    """Diagonal sign vectors of the two CZ half-layers (even bonds, odd bonds).

    Qubit 0 is the most significant bit of the basis index.
    """
    idx = torch.arange(2 ** n)
    bits = [(idx >> (n - 1 - j)) & 1 for j in range(n)]
    out = []
    for parity in (0, 1):
        parity_sum = torch.zeros(2 ** n, dtype=torch.long)
        for a, b in brick_pairs(n, parity):
            parity_sum += bits[a] * bits[b]
        out.append((1 - 2 * (parity_sum % 2)).to(device=device, dtype=dtype))
    return out


@lru_cache(maxsize=None)
def popcount_table(n: int, device: str = "cpu") -> torch.Tensor:
    """Hamming weight of every basis index (long tensor of length 2**n)."""
    idx = torch.arange(2 ** n)
    w = torch.zeros(2 ** n, dtype=torch.long)
    for j in range(n):
        w += (idx >> j) & 1
    return w.to(device)
