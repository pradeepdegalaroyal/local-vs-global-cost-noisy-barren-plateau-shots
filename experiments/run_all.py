"""Reproduce every result and figure: ``python experiments/run_all.py`` (``--quick`` for a smoke run)."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cost_locality_bp import plotting  # noqa: E402
from cost_locality_bp.hypotheses import (evaluate_hypotheses, prefactor_slopes,  # noqa: E402
                                          success_counts)
from cost_locality_bp.study import (ScalingConfig, TrainingConfig, fit_slopes,  # noqa: E402
                                    run_scaling, run_training)


def summarise_training(df: pd.DataFrame) -> pd.DataFrame:
    """Mean, std and median of infidelity per (p, n, method)."""
    g = df.groupby(["p", "n", "method"])["infidelity"]
    out = g.agg(["mean", "std", "median", "count"]).reset_index()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="tiny smoke run (< 1 minute)")
    ap.add_argument("--out", type=str, default=None, help="output root (default: repository root)")
    ap.add_argument("--cpu", action="store_true", help="force CPU")
    args = ap.parse_args()

    torch.set_num_threads(6)
    device = "cuda" if (torch.cuda.is_available() and not args.cpu) else "cpu"
    if args.out:
        out_root = Path(args.out)
    elif args.quick:  # never overwrite the full-run results with smoke-run output
        out_root = Path(tempfile.mkdtemp(prefix="cost_locality_quick_"))
    else:
        out_root = ROOT
    res_dir, fig_dir = out_root / "results", out_root / "figures"
    res_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    if args.quick:
        scfg = ScalingConfig(qubits=(3, 4, 5), noise=(0.0, 0.01), seeds=(0, 1), samples=48)
        tcfg = TrainingConfig(qubits=(3, 4), noise=(0.0, 0.01), seeds=(0, 1), shots=100,
                              adam_steps=8)
    else:
        scfg, tcfg = ScalingConfig(), TrainingConfig()

    t0 = time.time()
    print(f"device={device}  quick={args.quick}", flush=True)
    print("[1/2] gradient-variance scaling", flush=True)
    scaling = run_scaling(scfg, device)
    scaling.to_csv(res_dir / "gradient_scaling.csv", index=False)
    slopes = pd.concat([fit_slopes(scaling, "grad_var"), fit_slopes(scaling, "shots_unit_snr")])
    slopes.to_csv(res_dir / "scaling_slopes_per_seed.csv", index=False)
    slope_summary = (slopes.groupby(["column", "schedule", "cost", "p"])["slope"]
                     .agg(["mean", "std"]).reset_index())
    slope_summary.to_csv(res_dir / "scaling_slopes_summary.csv", index=False)
    t1 = time.time()

    print("[2/2] shot-limited training", flush=True)
    training = run_training(tcfg, device)
    training.to_csv(res_dir / "training_runs.csv", index=False)
    tsum = summarise_training(training)
    tsum.to_csv(res_dir / "training_summary.csv", index=False)
    t2 = time.time()

    pre = prefactor_slopes(scaling)
    pre.to_csv(res_dir / "prefactor_check_per_seed.csv", index=False)
    pre.groupby(["schedule", "p"])["slope_n2_scaled_local"].agg(["mean", "std"]).reset_index()\
        .to_csv(res_dir / "prefactor_check_summary.csv", index=False)
    success_counts(training).to_csv(res_dir / "training_success_counts.csv", index=False)
    if not args.quick:
        (res_dir / "hypotheses.json").write_text(
            json.dumps(evaluate_hypotheses(slopes, training), indent=2))

    var_slopes = slopes[slopes.column == "grad_var"]
    plotting.plot_variance_scaling(scaling, var_slopes, fig_dir / "fig1_variance_scaling.png",
                                   scfg.noise if len(scfg.noise) <= 2 else (0.0, scfg.noise[2]))
    plotting.plot_slope_vs_noise(var_slopes, fig_dir / "fig2_slope_vs_noise.png")
    plotting.plot_shot_cost_bars(scaling, max(scfg.qubits), fig_dir / "fig3_shot_cost.png")
    plotting.plot_training_bars(training, fig_dir / "fig4_training_bars.png")
    plotting.plot_training_ecdf(training, fig_dir / "fig5_training_ecdf.png")

    meta = {
        "quick": args.quick, "device": device, "python": platform.python_version(),
        "torch": torch.__version__, "numpy": np.__version__, "pandas": pd.__version__,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "scaling_seconds": round(t1 - t0, 1), "training_seconds": round(t2 - t1, 1),
        "total_seconds": round(time.time() - t0, 1),
        "scaling_config": scfg.__dict__, "training_config": tcfg.__dict__,
    }
    (res_dir / "run_meta.json").write_text(json.dumps(meta, indent=2, default=list))
    print(json.dumps({"device": device, "total_seconds": meta["total_seconds"],
                      "output": str(out_root)}), flush=True)


if __name__ == "__main__":
    main()
