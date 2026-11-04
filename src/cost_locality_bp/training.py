"""Shot-limited training of the ansatz: Adam with parameter-shift, SPSA, and random search."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .circuit import cz_sign_vectors
from .costs import global_cost, local_cost, sampled_costs
from .noisy_sim import simulate_probs


@dataclass
class TrainResult:
    """Final parameters and ideal (noise-free) global infidelity."""

    theta: torch.Tensor
    ideal_infidelity: float
    ideal_local_cost: float
    shots_used: int


class Evaluator:
    """Finite-shot cost oracle for one (n, L, p, cost kind) setting that counts shots."""

    def __init__(self, n: int, n_layers: int, p: float, kind: str, shots: int,
                 device: str = "cpu", dtype: torch.dtype = torch.float64):
        if kind not in ("global", "local"):
            raise ValueError(kind)
        self.n, self.n_layers, self.p, self.kind, self.shots = n, n_layers, p, kind, shots
        self.device, self.dtype = device, dtype
        self.signs = cz_sign_vectors(n, device, dtype)
        self.used = 0

    def both(self, theta: torch.Tensor, chunk: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
        """Shot-estimated (C_G, C_L) for each row of ``theta`` (B, L, n); shots are not counted."""
        gs, ls = [], []
        for i in range(0, theta.shape[0], chunk):
            part = theta[i:i + chunk].to(self.device, self.dtype)
            probs = simulate_probs(part, self.p, self.n, self.signs)
            c_g, c_l = sampled_costs(probs, self.shots, self.n)
            gs.append(c_g.cpu().double())
            ls.append(c_l.cpu().double())
        return torch.cat(gs), torch.cat(ls)

    def __call__(self, theta: torch.Tensor, chunk: int = 64) -> torch.Tensor:
        """Shot-estimated cost of the configured kind for each row of ``theta`` (B, L, n)."""
        c_g, c_l = self.both(theta, chunk)
        self.used += theta.shape[0] * self.shots
        return c_g if self.kind == "global" else c_l


def ideal_metrics(theta: torch.Tensor, n: int, device: str = "cpu",
                  dtype: torch.dtype = torch.float64) -> tuple[float, float]:
    """Noise-free (global infidelity, local cost) of the parameters."""
    probs = simulate_probs(theta[None].to(device, dtype), 0.0, n)
    return float(global_cost(probs)[0]), float(local_cost(probs, n)[0])


def init_theta(n: int, n_layers: int, gen: torch.Generator) -> torch.Tensor:
    """Uniform random angles in [0, 2 pi)."""
    return 2 * math.pi * torch.rand(n_layers, n, generator=gen, dtype=torch.float64)


def train_adam(ev: Evaluator, theta0: torch.Tensor, steps: int, lr: float = 0.1) -> TrainResult:
    """Adam with parameter-shift gradients, every shifted circuit costing ``ev.shots`` shots."""
    theta = theta0.clone()
    n_layers, n = theta.shape
    n_par = n_layers * n
    m = torch.zeros(n_par, dtype=torch.float64)
    v = torch.zeros(n_par, dtype=torch.float64)
    b1, b2, eps = 0.9, 0.999, 1e-8
    eye = torch.eye(n_par, dtype=torch.float64) * (math.pi / 2)
    for t in range(1, steps + 1):
        flat = theta.reshape(-1)
        batch = torch.cat([flat[None] + eye, flat[None] - eye]).reshape(2 * n_par, n_layers, n)
        c = ev(batch)
        g = (c[:n_par] - c[n_par:]) / 2
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        step = lr * (m / (1 - b1 ** t)) / ((v / (1 - b2 ** t)).sqrt() + eps)
        theta = (flat - step).reshape(n_layers, n)
    inf, loc = ideal_metrics(theta, ev.n, ev.device, ev.dtype)
    return TrainResult(theta, inf, loc, ev.used)


def train_spsa_batch(ev: Evaluator, theta0: torch.Tensor, is_global: torch.Tensor, evals: int,
                     gen: torch.Generator, a: float = 3.0, c: float = 0.2,
                     big_a_frac: float = 0.1) -> list[TrainResult]:
    """Run R independent SPSA optimisers in one batch (two cost evaluations per step each).

    ``theta0`` has shape (R, L, n); ``is_global[r]`` selects whether run r minimises C_G or C_L.
    Each run is charged ``evals * ev.shots`` shots.
    """
    theta = theta0.clone()
    steps = max(1, evals // 2)
    big_a = big_a_frac * steps
    for k in range(1, steps + 1):
        ak = a / (k + big_a) ** 0.602
        ck = c / k ** 0.101
        delta = 2.0 * torch.randint(0, 2, theta.shape, generator=gen).to(torch.float64) - 1.0
        c_g, c_l = ev.both(torch.cat([theta + ck * delta, theta - ck * delta]))
        r = theta.shape[0]
        cost = torch.where(is_global.repeat(2), c_g, c_l)
        ghat = ((cost[:r] - cost[r:]) / (2 * ck))[:, None, None] * delta
        theta = theta - ak * ghat
    out = []
    for i in range(theta.shape[0]):
        inf, loc = ideal_metrics(theta[i], ev.n, ev.device, ev.dtype)
        out.append(TrainResult(theta[i], inf, loc, steps * 2 * ev.shots))
    return out


def train_spsa(ev: Evaluator, theta0: torch.Tensor, evals: int, gen: torch.Generator,
               **kwargs: float) -> TrainResult:
    """Single-run SPSA of the cost kind configured on ``ev``."""
    flag = torch.tensor([ev.kind == "global"])
    res = train_spsa_batch(ev, theta0[None], flag, evals, gen, **kwargs)[0]
    ev.used += res.shots_used
    return res


def random_search(ev: Evaluator, n_layers: int, n: int, evals: int,
                  gen: torch.Generator) -> TrainResult:
    """Best of ``evals`` uniformly random parameter sets, chosen by measured cost."""
    cand = 2 * math.pi * torch.rand(evals, n_layers, n, generator=gen, dtype=torch.float64)
    c = ev(cand)
    best = cand[int(torch.argmin(c))]
    inf, loc = ideal_metrics(best, ev.n, ev.device, ev.dtype)
    return TrainResult(best, inf, loc, ev.used)
