"""Fast correctness tests for the noisy simulator, costs and gradients."""

import math

import numpy as np
import pytest
import torch

from cost_locality_bp.circuit import cz_sign_vectors, depth_for
from cost_locality_bp.costs import global_cost, local_cost, sampled_costs, shot_variances
from cost_locality_bp.gradients import loglog_slope, sample_gradients
from cost_locality_bp.noisy_sim import pauli_survival, simulate_probs


def _statevector_probs(theta: np.ndarray, n: int) -> np.ndarray:
    """Independent reference: dense Kronecker-product statevector simulation."""
    d = 2 ** n
    psi = np.zeros(d)
    psi[0] = 1.0
    idx = np.arange(d)
    bits = [(idx >> (n - 1 - j)) & 1 for j in range(n)]
    for layer in range(theta.shape[0]):
        u = np.array([[1.0]])
        for q in range(n):
            c, s = math.cos(theta[layer, q] / 2), math.sin(theta[layer, q] / 2)
            u = np.kron(u, np.array([[c, -s], [s, c]]))
        psi = u @ psi
        for parity in (0, 1):
            for a in range(parity, n - 1, 2):
                psi = psi * (1 - 2 * (bits[a] * bits[a + 1]))
    return psi ** 2


@pytest.mark.parametrize("n", [2, 3, 4])
def test_noiseless_matches_statevector(n):
    g = torch.Generator().manual_seed(1)
    theta = 2 * math.pi * torch.rand(2, 3, n, generator=g, dtype=torch.float64)
    probs = simulate_probs(theta, 0.0, n).numpy()
    for b in range(2):
        np.testing.assert_allclose(probs[b], _statevector_probs(theta[b].numpy(), n), atol=1e-12)


def test_noise_preserves_trace_and_mixes():
    g = torch.Generator().manual_seed(2)
    theta = 2 * math.pi * torch.rand(3, 4, 3, generator=g, dtype=torch.float64)
    probs = simulate_probs(theta, 0.05, 3)
    np.testing.assert_allclose(probs.sum(1).numpy(), 1.0, atol=1e-12)
    # full depolarisation (p = 3/4 gives lam = 1) leaves the maximally mixed state
    mixed = simulate_probs(theta, 0.75, 3)
    np.testing.assert_allclose(mixed.numpy(), 1 / 8, atol=1e-12)


def test_weight_one_noise_attenuation():
    """One RY layer on |0>: <Z> = cos(theta); one noisy layer scales it by (1 - 4p/3)."""
    p, theta = 0.03, 0.9
    probs = simulate_probs(torch.full((1, 1, 1), theta, dtype=torch.float64), p, 1)[0]
    z = float(probs[0] - probs[1])
    assert z == pytest.approx(math.cos(theta) * pauli_survival(p, 1, 1), abs=1e-12)


def test_costs_and_shot_variance_bounds():
    g = torch.Generator().manual_seed(3)
    theta = 2 * math.pi * torch.rand(5, 3, 4, generator=g, dtype=torch.float64)
    probs = simulate_probs(theta, 0.01, 4)
    cg, cl = global_cost(probs), local_cost(probs, 4)
    assert torch.all((cg >= 0) & (cg <= 1)) and torch.all((cl >= 0) & (cl <= 1))
    assert torch.all(cl <= cg + 1e-12)  # C_L <= C_G for the same state
    vg, vl = shot_variances(probs, 4)
    assert torch.all(vg >= 0) and torch.all(vl >= -1e-15)
    sg, sl = sampled_costs(probs, 20000, 4)
    assert torch.allclose(sg, cg, atol=0.03) and torch.allclose(sl, cl, atol=0.03)


def test_parameter_shift_matches_finite_difference():
    n, n_layers, p = 3, 2, 0.02
    g = torch.Generator().manual_seed(4)
    theta = 2 * math.pi * torch.rand(1, n_layers, n, generator=g, dtype=torch.float64)
    h = 1e-5
    plus, minus = theta.clone(), theta.clone()
    plus[0, 1, 2] += h
    minus[0, 1, 2] -= h
    fd = (global_cost(simulate_probs(plus, p, n)) - global_cost(simulate_probs(minus, p, n))) / (2 * h)
    ps_p, ps_m = theta.clone(), theta.clone()
    ps_p[0, 1, 2] += math.pi / 2
    ps_m[0, 1, 2] -= math.pi / 2
    ps = (global_cost(simulate_probs(ps_p, p, n)) - global_cost(simulate_probs(ps_m, p, n))) / 2
    assert float(ps) == pytest.approx(float(fd), abs=1e-7)


def test_depth_schedules_and_signs():
    assert depth_for(8, "linear") == 8 and depth_for(8, "log") == 4 and depth_for(5, "fixed3") == 3
    s0, s1 = cz_sign_vectors(3)
    assert s0.tolist() == [1, 1, 1, 1, 1, 1, -1, -1]
    assert s1.tolist() == [1, 1, 1, -1, 1, 1, 1, -1]


def test_gradient_sampler_and_slope():
    g = torch.Generator().manual_seed(5)
    s = sample_gradients(3, 2, 0.01, 20, g)
    assert s.g_global.shape == (20,) and np.all(np.abs(s.g_global) <= 0.5 + 1e-9)
    slope, _ = loglog_slope(np.array([2.0, 4.0, 8.0]), np.array([1.0, 0.5, 0.25]))
    assert slope == pytest.approx(-1.0)
