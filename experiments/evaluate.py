"""Re-derive the hypothesis verdicts and the prefactor check from saved CSVs (no re-simulation)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cost_locality_bp.hypotheses import (evaluate_hypotheses, prefactor_slopes,
                                          success_counts)  # noqa: E402


def main(res_dir: Path = ROOT / "results") -> None:
    scaling = pd.read_csv(res_dir / "gradient_scaling.csv")
    slopes = pd.read_csv(res_dir / "scaling_slopes_per_seed.csv")
    training = pd.read_csv(res_dir / "training_runs.csv")
    pre = prefactor_slopes(scaling)
    pre.to_csv(res_dir / "prefactor_check_per_seed.csv", index=False)
    pre.groupby(["schedule", "p"])["slope_n2_scaled_local"].agg(["mean", "std"]).reset_index() \
        .to_csv(res_dir / "prefactor_check_summary.csv", index=False)
    success_counts(training).to_csv(res_dir / "training_success_counts.csv", index=False)
    verdict = evaluate_hypotheses(slopes, training)
    (res_dir / "hypotheses.json").write_text(json.dumps(verdict, indent=2))
    print(json.dumps(verdict, indent=2))


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results")
