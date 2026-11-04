"""Experiment drivers: gradient-variance scaling study and shot-limited training study."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch

from .circuit import depth_for
from .gradients import loglog_slope, sample_gradients
from .training import (Evaluator, TrainResult, ideal_metrics, init_theta, random_search,
                       train_adam, train_spsa_batch)


@dataclass
class ScalingConfig:
    """Gradient-statistics sweep."""

    qubits: tuple[int, ...] = (3, 4, 5, 6, 7, 8)
    schedules: tuple[str, ...] = ("log", "linear")
    noise: tuple[float, ...] = (0.0, 0.005, 0.01, 0.02)
    seeds: tuple[int, ...] = (0, 1, 2)
    samples: int = 256


@dataclass
class TrainingConfig:
    """Shot-limited training sweep (log-depth schedule)."""

    qubits: tuple[int, ...] = (4, 6, 8)
    noise: tuple[float, ...] = (0.0, 0.01)
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    shots: int = 200
    adam_steps: int = 25
    methods: tuple[str, ...] = field(default=(
        "adam-global", "adam-local", "spsa-global", "spsa-local", "random-global"))


def run_scaling(cfg: ScalingConfig, device: str) -> pd.DataFrame:
    """Gradient variance and unit-SNR shot cost for both costs on every grid point."""
    rows = []
    for sched in cfg.schedules:
        for p in cfg.noise:
            for n in cfg.qubits:
                n_layers = depth_for(n, sched)
                for seed in cfg.seeds:
                    gen = torch.Generator().manual_seed(1000 * seed + 17 * n + 3)
                    s = sample_gradients(n, n_layers, p, cfg.samples, gen, device=device)
                    for kind, g, sv in (("global", s.g_global, s.shotvar_global),
                                        ("local", s.g_local, s.shotvar_local)):
                        var = float(np.var(g))
                        rows.append({
                            "schedule": sched, "p": p, "n": n, "layers": n_layers, "seed": seed,
                            "cost": kind, "grad_var": var, "grad_rms": float(np.sqrt(np.mean(g ** 2))),
                            # shots per circuit pair for the shot-noise std to equal the rms gradient
                            "shots_unit_snr": float(np.mean(sv) / max(np.mean(g ** 2), 1e-300)),
                        })
    return pd.DataFrame(rows)


def fit_slopes(df: pd.DataFrame, column: str = "grad_var") -> pd.DataFrame:
    """Per-seed log-log slope of ``column`` against Hilbert-space dimension 2**n."""
    out = []
    for (sched, p, kind, seed), g in df.groupby(["schedule", "p", "cost", "seed"]):
        g = g.sort_values("n")
        slope, icpt = loglog_slope(2.0 ** g["n"].to_numpy(), g[column].to_numpy())
        out.append({"schedule": sched, "p": p, "cost": kind, "seed": seed, "column": column,
                    "slope": slope, "intercept": icpt})
    return pd.DataFrame(out)


def run_training(cfg: TrainingConfig, device: str, schedule: str = "log",
                 verbose: bool = True) -> pd.DataFrame:
    """Train with equal total shots per run; record the noise-free global infidelity.

    Methods: Adam and SPSA on either cost, random search (selected by the measured global cost),
    and the untrained random initialisation as a reference.
    """
    rows = []
    for p in cfg.noise:
        for n in cfg.qubits:
            n_layers = depth_for(n, schedule)
            evals = cfg.adam_steps * 2 * n * n_layers
            inits, gens = {}, {}
            for seed in cfg.seeds:
                gens[seed] = torch.Generator().manual_seed(7919 * seed + 31 * n + 5)
                inits[seed] = init_theta(n, n_layers, gens[seed])

            def add(method: str, seed: int, res: TrainResult, secs: float) -> None:
                rows.append({"p": p, "n": n, "layers": n_layers, "seed": seed, "method": method,
                             "shots_used": res.shots_used, "infidelity": res.ideal_infidelity,
                             "local_cost": res.ideal_local_cost, "seconds": secs})

            for seed in cfg.seeds:
                inf, loc = ideal_metrics(inits[seed], n, device)
                add("init", seed, TrainResult(inits[seed], inf, loc, 0), 0.0)
                for kind in ("global", "local"):
                    if f"adam-{kind}" in cfg.methods:
                        t0 = time.time()
                        ev = Evaluator(n, n_layers, p, kind, cfg.shots, device)
                        add(f"adam-{kind}", seed, train_adam(ev, inits[seed], cfg.adam_steps),
                            time.time() - t0)
                if "random-global" in cfg.methods:
                    t0 = time.time()
                    ev = Evaluator(n, n_layers, p, "global", cfg.shots, device)
                    add("random-global", seed, random_search(ev, n_layers, n, evals, gens[seed]),
                        time.time() - t0)
            spsa_kinds = [k for k in ("global", "local") if f"spsa-{k}" in cfg.methods]
            if spsa_kinds:
                t0 = time.time()
                ev = Evaluator(n, n_layers, p, "global", cfg.shots, device)
                runs = [(k, sd) for k in spsa_kinds for sd in cfg.seeds]
                th0 = torch.stack([inits[sd] for _, sd in runs])
                flag = torch.tensor([k == "global" for k, _ in runs])
                gen = torch.Generator().manual_seed(99 + n)
                res = train_spsa_batch(ev, th0, flag, evals, gen)
                for (k, sd), r in zip(runs, res):
                    add(f"spsa-{k}", sd, r, (time.time() - t0) / len(runs))
            if verbose:
                sub = pd.DataFrame([r for r in rows if r["p"] == p and r["n"] == n])
                print(f"  training p={p} n={n}: " + ", ".join(
                    f"{m}={v:.3f}" for m, v in sub.groupby("method")["infidelity"].mean().items()),
                    flush=True)
    return pd.DataFrame(rows)
