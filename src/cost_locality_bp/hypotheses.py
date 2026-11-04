"""Evaluate the pre-stated hypotheses H1-H5 from the saved result tables."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .gradients import loglog_slope


def prefactor_slopes(scaling: pd.DataFrame) -> pd.DataFrame:
    """Post-hoc check: slope of n^2 * Var(local gradient) against d = 2^n, per seed.

    The local cost is a mean over n single-qubit terms, so one parameter's gradient carries a 1/n
    prefactor and its variance a 1/n^2 prefactor that is unrelated to barren plateaus.
    """
    rows = []
    loc = scaling[scaling.cost == "local"]
    for (sched, p, seed), g in loc.groupby(["schedule", "p", "seed"]):
        g = g.sort_values("n")
        slope, _ = loglog_slope(2.0 ** g["n"].to_numpy(), (g["n"] ** 2 * g["grad_var"]).to_numpy())
        rows.append({"schedule": sched, "p": p, "seed": seed, "slope_n2_scaled_local": slope})
    return pd.DataFrame(rows)


def _slope(sl: pd.DataFrame, sched: str, cost: str, p: float) -> pd.Series:
    s = sl[(sl.column == "grad_var") & (sl.schedule == sched) & (sl.cost == cost) & (sl.p == p)]
    return s.sort_values("seed")["slope"].reset_index(drop=True)


def evaluate_hypotheses(slopes: pd.DataFrame, training: pd.DataFrame,
                        noise_mid: float = 0.01, noise_hi: float = 0.02) -> dict:
    """Return a JSON-serialisable verdict for each hypothesis with the numbers behind it."""
    out: dict = {}
    g0, l0 = _slope(slopes, "log", "global", 0.0), _slope(slopes, "log", "local", 0.0)
    out["H1"] = {"global_slope_mean": g0.mean(), "local_slope_mean": l0.mean(),
                 "criterion": "global <= -0.7 and local >= -0.5",
                 "supported": bool(g0.mean() <= -0.7 and l0.mean() >= -0.5)}
    gm, lm = _slope(slopes, "log", "global", noise_mid), _slope(slopes, "log", "local", noise_mid)
    diff = lm - gm
    out["H2"] = {"local_minus_global_mean": diff.mean(), "std": diff.std(ddof=1),
                 "criterion": ">= 0.4", "supported": bool(diff.mean() >= 0.4)}
    dg = _slope(slopes, "linear", "global", noise_hi) - _slope(slopes, "linear", "global", 0.0)
    dl = _slope(slopes, "linear", "local", noise_hi) - _slope(slopes, "linear", "local", 0.0)
    gap = dl - dg
    out["H3"] = {"delta_global_mean": dg.mean(), "delta_local_mean": dl.mean(),
                 "gap_mean": gap.mean(), "gap_std": gap.std(ddof=1), "criterion": "gap >= 0.1",
                 "supported": bool(gap.mean() >= 0.1)}
    nmax = int(training.n.max())
    sub = training[(training.n == nmax) & (training.p == noise_mid)]
    med = sub.groupby("method")["infidelity"].median()
    out["H4"] = {"n": nmax, "p": noise_mid, "median_adam_local": med.get("adam-local", np.nan),
                 "median_adam_global": med.get("adam-global", np.nan),
                 "supported": bool(med.get("adam-local", np.inf) < med.get("adam-global", -np.inf))}
    beats = {}
    for (p, n), g in training.groupby(["p", "n"]):
        m = g.groupby("method")["infidelity"].mean()
        beats[f"p={p},n={n}"] = {k: bool(m[k] < m["random-global"])
                                 for k in ("adam-global", "adam-local", "spsa-global", "spsa-local")}
    out["H5"] = {"beats_random_search_by_mean": beats,
                 "supported": bool(all(all(v.values()) for v in beats.values()))}
    return _clean(out)


def _clean(obj):
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    return obj


def success_counts(training: pd.DataFrame, threshold: float = 0.1) -> pd.DataFrame:
    """Number of runs per (p, n, method) whose final noise-free infidelity is below ``threshold``."""
    flag = training.assign(success=training.infidelity < threshold)
    out = flag.groupby(["p", "n", "method"])["success"].agg(["sum", "count"]).reset_index()
    return out.rename(columns={"sum": "runs_below_0.1", "count": "runs"})
