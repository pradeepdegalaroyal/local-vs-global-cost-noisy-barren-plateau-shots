"""Batched exact density-matrix simulator for the RY + CZ ansatz with layer-wise depolarising noise.

All gates are real (RY, CZ) and the channel is a real Pauli channel, so the density matrix stays
real and a float tensor suffices.
"""

from __future__ import annotations

import torch

from .circuit import cz_sign_vectors


def _apply_ry_layer(rho: torch.Tensor, theta: torch.Tensor, n: int) -> torch.Tensor:
    """Apply RY(theta[:, q]) on every qubit q to ``rho`` of shape (B, 2**n, 2**n)."""
    b = rho.shape[0]
    d = 2 ** n
    c = torch.cos(theta / 2)
    s = torch.sin(theta / 2)
    mats = torch.stack([torch.stack([c, -s], -1), torch.stack([s, c], -1)], -2)  # (B, n, 2, 2)
    for q in range(n):
        r = mats[:, q]
        v = rho.reshape(b, 2 ** q, 2, (2 ** (n - q - 1)) * d)
        rho = torch.einsum("bpa,bxay->bxpy", r, v).reshape(b, d, d)
        v = rho.reshape(b, d * 2 ** q, 2, 2 ** (n - q - 1))
        rho = torch.einsum("bpa,bxay->bxpy", r, v).reshape(b, d, d)
    return rho


def _depolarise(rho: torch.Tensor, n: int, q: int, lam: float) -> torch.Tensor:
    """Single-qubit depolarising map rho -> (1 - lam) rho + lam * tr_q(rho) (x) I/2."""
    b = rho.shape[0]
    d = 2 ** n
    r = rho.reshape(b, 2 ** q, 2, 2 ** (n - q - 1), 2 ** q, 2, 2 ** (n - q - 1))
    tr = r[:, :, 0, :, :, 0, :] + r[:, :, 1, :, :, 1, :]
    out = r * (1.0 - lam)
    out[:, :, 0, :, :, 0, :] += 0.5 * lam * tr
    out[:, :, 1, :, :, 1, :] += 0.5 * lam * tr
    return out.reshape(b, d, d)


def simulate_probs(theta: torch.Tensor, p: float, n: int,
                   signs: list[torch.Tensor] | None = None) -> torch.Tensor:
    """Computational-basis probabilities after the noisy ansatz.

    Args:
        theta: angles of shape (B, L, n); layer l applies RY on all qubits, then CZ on even bonds,
            CZ on odd bonds, then single-qubit Pauli noise of total error probability ``p`` on
            every qubit.
        p: single-qubit Pauli error probability per qubit per layer (X, Y, Z each p / 3).
        n: number of qubits.
        signs: optional precomputed CZ sign vectors.

    Returns:
        Tensor (B, 2**n) of outcome probabilities (diagonal of the final density matrix).
    """
    b, n_layers, _ = theta.shape
    d = 2 ** n
    dev, dt = theta.device, theta.dtype
    if signs is None:
        signs = cz_sign_vectors(n, dev, dt)
    sign_mat = (signs[0][:, None] * signs[0][None, :], signs[1][:, None] * signs[1][None, :])
    rho = torch.zeros(b, d, d, device=dev, dtype=dt)
    rho[:, 0, 0] = 1.0
    lam = 4.0 * p / 3.0
    for layer in range(n_layers):
        rho = _apply_ry_layer(rho, theta[:, layer], n)
        rho = rho * sign_mat[0]
        rho = rho * sign_mat[1]
        if lam > 0.0:
            for q in range(n):
                rho = _depolarise(rho, n, q, lam)
    probs = torch.diagonal(rho, dim1=1, dim2=2)
    return probs.clamp_min(0.0)


def pauli_survival(p: float, n_layers: int, weight: int) -> float:
    """Survival factor (1 - 4p/3)**(weight * L) of a Pauli expectation of the given weight."""
    return (1.0 - 4.0 * p / 3.0) ** (weight * n_layers)
