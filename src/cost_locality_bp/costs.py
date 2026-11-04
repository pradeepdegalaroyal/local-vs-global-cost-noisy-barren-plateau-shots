"""Global and local cost functions and their shot statistics.

Both costs follow the state-preparation setting of Cerezo et al. (2021): the ansatz acts on |0...0>
and the cost measures how far the output is from |0...0>.
"""

from __future__ import annotations

import torch

from .circuit import popcount_table


def global_cost(probs: torch.Tensor) -> torch.Tensor:
    """C_G = 1 - P(0...0), the global projector cost."""
    return 1.0 - probs[:, 0]


def local_cost(probs: torch.Tensor, n: int) -> torch.Tensor:
    """C_L = 1 - (1/n) sum_j P(qubit j = 0), the single-qubit-projector cost."""
    w = popcount_table(n, str(probs.device)).to(probs.dtype)
    mean_ones = (probs * w).sum(1) / n
    return mean_ones


def shot_variances(probs: torch.Tensor, n: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-shot variance of the single-shot estimators of C_G and C_L."""
    q = probs[:, 0]
    var_g = q * (1.0 - q)
    w = (popcount_table(n, str(probs.device)).to(probs.dtype)) / n
    mean = (probs * w).sum(1)
    second = (probs * w * w).sum(1)
    return var_g, second - mean ** 2


def sampled_costs(probs: torch.Tensor, shots: int, n: int,
                  generator: torch.Generator | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Finite-shot estimates of (C_G, C_L) from ``shots`` samples per row."""
    p = probs / probs.sum(1, keepdim=True)
    idx = torch.multinomial(p.float(), shots, replacement=True, generator=generator)
    c_g = (idx != 0).to(probs.dtype).mean(1)
    w = popcount_table(n, str(probs.device))[idx].to(probs.dtype) / n
    c_l = w.mean(1)
    return c_g, c_l
