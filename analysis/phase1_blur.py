"""Phase 1 — Gaussian blur sweep analysis.

Reads results/blur_sweep_B/summary.csv and experiments/blur_sweep_B.yaml,
plots ATE RMSE vs sigma with clean stereo + mono baselines.

Usage:
    python analysis/phase1_blur.py
"""

import yaml
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SWEEP_CSV   = ROOT / "results" / "blur_sweep_B" / "summary.csv"
SWEEP_YAML  = ROOT / "experiments" / "blur_sweep_B.yaml"
CLEAN_STEREO_STATS = ROOT / "results" / "clean" / "traj1" / "clean" / "stereo" / "stats.yaml"
CLEAN_MONO_STATS   = ROOT / "results" / "clean" / "traj1" / "clean" / "mono"   / "stats.yaml"
OUT_PNG = ROOT / "analysis" / "phase1_blur.png"


def load_sigma_map(yaml_path: Path) -> dict[str, float]:
    """Return {condition_name: sigma} from experiment YAML."""
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)
    return {k: float(v["sigma"]) for k, v in cfg["conditions"].items()}


def load_baseline(stats_path: Path) -> float:
    with open(stats_path) as f:
        return yaml.safe_load(f)["rmse"]


def main():
    sigma_map = load_sigma_map(SWEEP_YAML)
    clean_stereo_rmse = load_baseline(CLEAN_STEREO_STATS)
    clean_mono_rmse   = load_baseline(CLEAN_MONO_STATS)

    df = pd.read_csv(SWEEP_CSV)
    df = df[df["mode"] == "stereo"].copy()
    df["sigma"] = df["condition"].map(sigma_map)
    df["failed"] = df["ate_rmse"] == "FAIL"
    df["rmse"] = pd.to_numeric(df["ate_rmse"], errors="coerce")
    df = df.sort_values("sigma")

    success = df[~df["failed"]]
    failed  = df[df["failed"]]

    fig, ax = plt.subplots(figsize=(10, 5))

    # --- failure region ---
    if not failed.empty:
        fail_start = failed["sigma"].min()
        ax.axvspan(fail_start, df["sigma"].max() + 0.5, color="red", alpha=0.08, label="SLAM failure region")
        ax.axvline(fail_start, color="red", linestyle="--", linewidth=1.2, alpha=0.7)
        ax.text(fail_start + 0.05, clean_mono_rmse * 1.02, f"σ = {fail_start}", color="red", fontsize=9)

    # --- baselines ---
    ax.axhline(clean_stereo_rmse, color="steelblue", linestyle="--", linewidth=1.2,
               label=f"Clean stereo baseline ({clean_stereo_rmse:.4f} m)")
    ax.axhline(clean_mono_rmse, color="darkorange", linestyle="--", linewidth=1.2,
               label=f"Clean mono baseline ({clean_mono_rmse:.4f} m)")

    # --- crossover annotation ---
    # point where stereo RMSE crosses mono baseline
    crossover = success[success["rmse"] >= clean_mono_rmse]
    if not crossover.empty:
        cx = crossover.iloc[0]["sigma"]
        cy = crossover.iloc[0]["rmse"]
        ax.annotate("Crossover\n(stereo ≥ mono)",
                    xy=(cx, cy), xytext=(cx - 1.5, cy + 0.0012),
                    arrowprops=dict(arrowstyle="->", color="black"),
                    fontsize=8.5)

    # --- sweep curve ---
    ax.plot(success["sigma"], success["rmse"], "o-", color="steelblue",
            linewidth=2, markersize=6, label="Stereo ATE (blur sweep B)")

    ax.set_xlabel("Gaussian blur sigma (σ)", fontsize=12)
    ax.set_ylabel("ATE RMSE (m)", fontsize=12)
    ax.set_title("Phase 1 — Stereo ATE vs Gaussian Blur Severity\n(kernel_size = 6σ, Option B)", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    print(f"Saved → {OUT_PNG}")

    # --- text summary ---
    print(f"\n{'─'*55}")
    print(f"  {'sigma':>6}  {'RMSE (m)':>10}  {'vs clean':>10}  {'vs mono':>10}")
    print(f"{'─'*55}")
    print(f"  {'0 (clean)':>6}  {clean_stereo_rmse:>10.4f}  {'—':>10}  {clean_stereo_rmse/clean_mono_rmse:>9.1%}")
    for _, row in success.iterrows():
        pct_clean = (row["rmse"] - clean_stereo_rmse) / clean_stereo_rmse
        pct_mono  = row["rmse"] / clean_mono_rmse
        print(f"  {row['sigma']:>6.1f}  {row['rmse']:>10.4f}  {pct_clean:>+10.1%}  {pct_mono:>9.1%}")
    for _, row in failed.iterrows():
        print(f"  {row['sigma']:>6.1f}  {'FAIL':>10}")
    print(f"{'─'*55}")
    print(f"\nMono baseline: {clean_mono_rmse:.4f} m")
    print(f"Cliff edge:    σ = {fail_start:.1f}")


if __name__ == "__main__":
    main()
