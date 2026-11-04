"""Tests for the hypothesis evaluation helpers on synthetic tables."""

import numpy as np
import pandas as pd

from cost_locality_bp.hypotheses import evaluate_hypotheses, prefactor_slopes
from cost_locality_bp.study import fit_slopes


def _synthetic_scaling():
    rows = []
    for sched in ("log", "linear"):
        for p in (0.0, 0.01, 0.02):
            for cost, a in (("global", -1.8), ("local", -0.6)):
                for seed in (0, 1):
                    for n in (3, 4, 5, 6):
                        rows.append({"schedule": sched, "p": p, "n": n, "seed": seed, "cost": cost,
                                     "grad_var": (2.0 ** n) ** (a - 4 * p) * (1 / n ** 2 if cost == "local" else 1),
                                     "shots_unit_snr": 1.0})
    return pd.DataFrame(rows)


def test_prefactor_slope_removes_n_squared():
    sc = _synthetic_scaling()
    pre = prefactor_slopes(sc)
    row = pre[(pre.schedule == "log") & (pre.p == 0.0)]
    assert np.allclose(row["slope_n2_scaled_local"], -0.6, atol=1e-9)


def test_hypothesis_verdict_keys():
    sc = _synthetic_scaling()
    sl = fit_slopes(sc, "grad_var")
    tr = pd.DataFrame([{"p": p, "n": n, "method": m, "seed": s, "infidelity": v}
                       for p in (0.0, 0.01) for n in (4, 8) for s in (0, 1)
                       for m, v in (("adam-global", 0.2), ("adam-local", 0.1), ("spsa-global", 0.1),
                                    ("spsa-local", 0.1), ("random-global", 0.5), ("init", 0.9))])
    v = evaluate_hypotheses(sl, tr)
    assert set(v) == {"H1", "H2", "H3", "H4", "H5"}
    assert v["H4"]["supported"] and v["H5"]["supported"]
