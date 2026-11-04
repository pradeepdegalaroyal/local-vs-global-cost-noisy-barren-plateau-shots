"""Parameter-shift gradient statistics of the global and local costs under noise."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from .circuit import cz_sign_vectors
from .costs import global_cost, local_cost, shot_variances
from .noisy_sim import simulate_probs


@dataclass
class GradientSamples:
    """Exact parameter-shift gradients of a random parameter at random initialisations."""

    g_global: np.ndarray
    g_local: np.ndarray
    shotvar_global: np.ndarray  # (var_plus + var_minus) / 4 per shot
    shotvar_local: np.ndarray
    layer: np.ndarray


def sample_gradients(n: int, n_layers: int, p: float, n_samples: int, gen: torch.Generator,
                     device: str = "cpu", dtype: torch.dtype = torch.float64,
                     chunk: int = 128) -> GradientSamples:
    """Draw ``n_samples`` random initialisations and one random parameter each; return the
    exact parameter-shift gradient (C(+pi/2) - C(-pi/2)) / 2 of both costs, plus the per-shot
    variance of the shot estimator of that gradient.
    """
    signs = cz_sign_vectors(n, device, dtype)
    out: dict[str, list[torch.Tensor]] = {k: [] for k in ("gg", "gl", "vg", "vl", "layer")}
    done = 0
    while done < n_samples:
        m = min(chunk, n_samples - done)
        theta = 2 * math.pi * torch.rand(m, n_layers, n, generator=gen, dtype=dtype)
        layer = torch.randint(0, n_layers, (m,), generator=gen)
        qubit = torch.randint(0, n, (m,), generator=gen)
        ar = torch.arange(m)
        plus, minus = theta.clone(), theta.clone()
        plus[ar, layer, qubit] += math.pi / 2
        minus[ar, layer, qubit] -= math.pi / 2
        probs = simulate_probs(torch.cat([plus, minus]).to(device), p, n, signs)
        pp, pm = probs[:m], probs[m:]
        gg = (global_cost(pp) - global_cost(pm)) / 2
        gl = (local_cost(pp, n) - local_cost(pm, n)) / 2
        vgp, vlp = shot_variances(pp, n)
        vgm, vlm = shot_variances(pm, n)
        out["gg"].append(gg.cpu())
        out["gl"].append(gl.cpu())
        out["vg"].append(((vgp + vgm) / 4).cpu())
        out["vl"].append(((vlp + vlm) / 4).cpu())
        out["layer"].append(layer)
        done += m
    cat = {k: torch.cat(v).numpy() for k, v in out.items()}
    return GradientSamples(cat["gg"], cat["gl"], cat["vg"], cat["vl"], cat["layer"])


def loglog_slope(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Least-squares slope and intercept of log10 y against log10 x."""
    slope, intercept = np.polyfit(np.log10(x), np.log10(y), 1)
    return float(slope), float(intercept)
