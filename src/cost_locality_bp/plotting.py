"""Greyscale figures: log-log scaling, hatched grouped bars, and empirical CDFs."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

DPI = 200
SCHEDULE_TITLE = {"log": "logarithmic depth, L = ceil(log2 n) + 1", "linear": "linear depth, L = n"}
COST_STYLE = {
    "global": dict(marker="o", ls="-", color="black", mfc="white"),
    "local": dict(marker="s", ls="--", color="0.35", mfc="0.35"),
}
METHOD_ORDER = ["init", "random-global", "spsa-global", "spsa-local", "adam-global", "adam-local"]
METHOD_LABEL = {
    "init": "random initialisation", "random-global": "random search",
    "spsa-global": "SPSA, global cost", "spsa-local": "SPSA, local cost",
    "adam-global": "Adam, global cost", "adam-local": "Adam, local cost",
}
HATCH = {"init": "", "random-global": "...", "spsa-global": "///", "spsa-local": "\\\\\\",
         "adam-global": "xxx", "adam-local": "+++"}
SHADE = {"init": "1.0", "random-global": "0.85", "spsa-global": "0.7", "spsa-local": "0.55",
         "adam-global": "0.4", "adam-local": "0.2"}


def _save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def plot_variance_scaling(df: pd.DataFrame, slopes: pd.DataFrame, path: Path,
                          noise_levels: tuple[float, ...]) -> None:
    """Gradient variance against Hilbert-space dimension with fitted log-log slopes."""
    scheds = [s for s in ("log", "linear") if s in set(df["schedule"])]
    fig, axes = plt.subplots(1, len(scheds), figsize=(5.6 * len(scheds), 4.4), squeeze=False)
    for ax, sched in zip(axes[0], scheds):
        for cost in ("global", "local"):
            for p in noise_levels:
                sub = df[(df.schedule == sched) & (df.cost == cost) & (df.p == p)]
                m = sub.groupby("n")["grad_var"].agg(["mean", "std"]).reset_index()
                style = dict(COST_STYLE[cost])
                alpha = 1.0 if p == 0.0 else 0.55
                filled = p == 0.0
                d = 2.0 ** m["n"].to_numpy()
                ax.errorbar(d, m["mean"], yerr=m["std"].fillna(0), capsize=2, lw=1.2, ms=5,
                            marker=style["marker"], ls=style["ls"] if filled else ":",
                            color=style["color"], mfc=style["mfc"] if filled else "none",
                            alpha=alpha)
                s = slopes[(slopes.schedule == sched) & (slopes.cost == cost) & (slopes.p == p)]
                if p in (0.0, noise_levels[-1]) and len(s):
                    ax.annotate(f"{s['slope'].mean():+.2f}", (d[-1], m["mean"].iloc[-1]),
                                textcoords="offset points", xytext=(6, -2), fontsize=8)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("Hilbert-space dimension d = 2^n (dimensionless)")
        ax.set_ylabel("variance of parameter-shift gradient (cost units squared)")
        ax.set_title(SCHEDULE_TITLE[sched], fontsize=10)
        ax.grid(alpha=0.3, which="both")
    handles = []
    for cost in ("global", "local"):
        for p in (0.0, noise_levels[-1]):
            st = COST_STYLE[cost]
            handles.append(plt.Line2D([], [], color=st["color"], marker=st["marker"],
                                      ls=st["ls"] if p == 0.0 else ":",
                                      mfc=st["mfc"] if p == 0.0 else "none",
                                      label=f"{cost} cost, p = {p:g}"))
    axes[0][0].legend(handles=handles, fontsize=8, loc="lower left")
    fig.suptitle("Gradient variance versus dimension (numbers: fitted log-log slope, mean of seeds; "
                 "dotted = noisy)", fontsize=9)
    _save(fig, path)


def plot_slope_vs_noise(slopes: pd.DataFrame, path: Path) -> None:
    """Fitted log-log slope against the single-qubit Pauli error probability."""
    scheds = [s for s in ("log", "linear") if s in set(slopes["schedule"])]
    fig, axes = plt.subplots(1, len(scheds), figsize=(5.2 * len(scheds), 4.0), squeeze=False)
    for ax, sched in zip(axes[0], scheds):
        for cost in ("global", "local"):
            sub = slopes[(slopes.schedule == sched) & (slopes.cost == cost)]
            m = sub.groupby("p")["slope"].agg(["mean", "std"]).reset_index()
            st = COST_STYLE[cost]
            ax.errorbar(m["p"], m["mean"], yerr=m["std"].fillna(0), capsize=3, lw=1.3, ms=6,
                        marker=st["marker"], ls=st["ls"], color=st["color"], mfc=st["mfc"],
                        label=f"{cost} cost")
        ax.set_xlabel("single-qubit Pauli error probability p per layer")
        ax.set_ylabel("slope of log10 variance vs log10 d")
        ax.set_title(SCHEDULE_TITLE[sched], fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Noise steepens the decay; more strongly for the global cost", fontsize=10)
    _save(fig, path)


def plot_shot_cost_bars(df: pd.DataFrame, n: int, path: Path) -> None:
    """Grouped hatched bars of the shots needed for unit gradient signal-to-noise ratio."""
    scheds = [s for s in ("log", "linear") if s in set(df["schedule"])]
    noise = sorted(df["p"].unique())
    fig, axes = plt.subplots(1, len(scheds), figsize=(5.4 * len(scheds), 4.0), squeeze=False)
    width = 0.38
    for ax, sched in zip(axes[0], scheds):
        for k, cost in enumerate(("global", "local")):
            means, stds = [], []
            for p in noise:
                v = df[(df.schedule == sched) & (df.cost == cost) & (df.p == p) & (df.n == n)]
                means.append(v["shots_unit_snr"].mean())
                stds.append(v["shots_unit_snr"].std() if len(v) > 1 else 0.0)
            x = np.arange(len(noise)) + (k - 0.5) * width
            ax.bar(x, means, width, yerr=stds, capsize=3, color="0.85" if k == 0 else "0.45",
                   edgecolor="black", hatch="///" if k == 0 else "xxx", label=f"{cost} cost")
        ax.set_xticks(np.arange(len(noise)))
        ax.set_xticklabels([f"{p:g}" for p in noise])
        ax.set_yscale("log")
        ax.set_xlabel("single-qubit Pauli error probability p per layer")
        ax.set_ylabel("shots per circuit for unit gradient SNR")
        ax.set_title(SCHEDULE_TITLE[sched], fontsize=10)
        ax.grid(alpha=0.3, axis="y", which="both")
        ax.legend(fontsize=8)
    fig.suptitle(f"Shot cost of resolving one gradient component, n = {n} qubits (mean and std over seeds)",
                 fontsize=10)
    _save(fig, path)


def plot_training_bars(df: pd.DataFrame, path: Path) -> None:
    """Hatched grouped bars: final noise-free infidelity per method, qubit number and noise."""
    noise = sorted(df["p"].unique())
    qubits = sorted(df["n"].unique())
    methods = [m for m in METHOD_ORDER if m in set(df["method"])]
    fig, axes = plt.subplots(1, len(noise), figsize=(5.8 * len(noise), 4.2), squeeze=False)
    width = 0.8 / len(methods)
    for ax, p in zip(axes[0], noise):
        for k, m in enumerate(methods):
            means, stds = [], []
            for n in qubits:
                v = df[(df.p == p) & (df.n == n) & (df.method == m)]["infidelity"]
                means.append(v.mean())
                stds.append(v.std() if len(v) > 1 else 0.0)
            x = np.arange(len(qubits)) + (k - (len(methods) - 1) / 2) * width
            ax.bar(x, means, width, yerr=stds, capsize=2, color=SHADE[m], edgecolor="black",
                   hatch=HATCH[m], label=METHOD_LABEL[m], lw=0.8)
        ax.set_xticks(np.arange(len(qubits)))
        ax.set_xticklabels([str(n) for n in qubits])
        ax.set_xlabel("number of qubits n")
        ax.set_ylabel("final noise-free infidelity 1 - |<0|psi>|^2")
        ax.set_title(f"p = {p:g}, equal shot budget per run", fontsize=10)
        ax.grid(alpha=0.3, axis="y")
    for ax in axes[0]:
        ax.set_ylim(0, 1.45)
    axes[0][0].legend(fontsize=7, loc="upper center", ncol=3)
    fig.suptitle("Trained state quality (mean and std over seeds)", fontsize=10)
    _save(fig, path)


def plot_training_ecdf(df: pd.DataFrame, path: Path) -> None:
    """Empirical CDF of final infidelity per method, pooled over qubit numbers and seeds."""
    noise = sorted(df["p"].unique())
    methods = [m for m in METHOD_ORDER if m in set(df["method"])]
    styles = {"init": (":", "x"), "random-global": ((0, (1, 1)), "+"), "spsa-global": ("-.", "^"),
              "spsa-local": ("-.", "v"), "adam-global": ("-", "o"), "adam-local": ("--", "s")}
    fig, axes = plt.subplots(1, len(noise), figsize=(5.4 * len(noise), 4.0), squeeze=False)
    for ax, p in zip(axes[0], noise):
        for m in methods:
            v = np.sort(df[(df.p == p) & (df.method == m)]["infidelity"].to_numpy())
            y = np.arange(1, len(v) + 1) / len(v)
            ls, mk = styles[m]
            ax.step(v, y, where="post", color="black", ls=ls, lw=1.3, label=METHOD_LABEL[m])
            ax.plot(v, y, mk, color="black", ms=4, mfc="none" if m != "adam-local" else "0.4")
        ax.set_xlabel("final noise-free infidelity (dimensionless)")
        ax.set_ylabel("empirical cumulative fraction of runs")
        ax.set_title(f"p = {p:g}", fontsize=10)
        ax.grid(alpha=0.3)
        ax.set_xlim(-0.02, 1.02)
    axes[0][-1].legend(fontsize=7, loc="center", bbox_to_anchor=(0.5, 0.78), framealpha=0.95, ncol=2)
    fig.suptitle("Empirical CDF of final infidelity, pooled over n and seeds (left-shifted is better)",
                 fontsize=10)
    _save(fig, path)
