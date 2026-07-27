"""Phase 1 — Brightness/exposure sweep analysis.

Reads results/brightness_sweep/summary.csv and experiments/brightness_sweep.yaml,
plots ATE RMSE vs alpha with clean stereo + mono baselines.

Usage:
    python analysis/phase1_brightness.py
"""

import yaml
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SWEEP_CSV  = ROOT / "results" / "brightness_sweep" / "summary.csv"
SWEEP_YAML = ROOT / "experiments" / "brightness_sweep.yaml"
CLEAN_STEREO_STATS = ROOT / "results" / "clean" / "traj1" / "clean" / "stereo" / "stats.yaml"
CLEAN_MONO_STATS   = ROOT / "results" / "clean" / "traj1" / "clean" / "mono"   / "stats.yaml"
OUT_PNG = ROOT / "analysis" / "phase1_brightness.png"


def load_alpha_map(yaml_path: Path) -> dict[str, float]:
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)
    return {k: float(v["alpha"]) for k, v in cfg["conditions"].items()}


def load_baseline(stats_path: Path) -> float:
    with open(stats_path) as f:
        return yaml.safe_load(f)["rmse"]


def main():
    alpha_map = load_alpha_map(SWEEP_YAML)
    clean_stereo_rmse = load_baseline(CLEAN_STEREO_STATS)
    clean_mono_rmse   = load_baseline(CLEAN_MONO_STATS)

    df = pd.read_csv(SWEEP_CSV)
    df = df[df["mode"] == "stereo"].copy()
    df["alpha"]  = df["condition"].map(alpha_map)
    df["failed"] = df["ate_rmse"] == "FAIL"
    df["rmse"]   = pd.to_numeric(df["ate_rmse"], errors="coerce")
    df = df.sort_values("alpha")

    success = df[~df["failed"]]
    failed  = df[df["failed"]]

    fig, ax = plt.subplots(figsize=(11, 5))

    # --- underexposure failure region ---
    under_fail = failed[failed["alpha"] < 1.0]
    if not under_fail.empty:
        fail_max = under_fail["alpha"].max()
        ax.axvspan(0, fail_max, color="red", alpha=0.08, label="SLAM failure (underexposure)")
        ax.axvline(fail_max, color="red", linestyle="--", linewidth=1.2, alpha=0.7)
        ax.text(fail_max + 0.01, clean_mono_rmse * 1.01,
                f"α = {fail_max:.2f}", color="red", fontsize=9)

    # --- baselines ---
    ax.axhline(clean_stereo_rmse, color="steelblue", linestyle="--", linewidth=1.2,
               label=f"Clean stereo baseline ({clean_stereo_rmse:.4f} m)")
    ax.axhline(clean_mono_rmse, color="darkorange", linestyle="--", linewidth=1.2,
               label=f"Clean mono baseline ({clean_mono_rmse:.4f} m)")

    # --- clean marker ---
    ax.axvline(1.0, color="green", linestyle=":", linewidth=1.2, alpha=0.6, label="Clean (α = 1.0)")

    # --- sweep curve ---
    ax.plot(success["alpha"], success["rmse"], "o-", color="steelblue",
            linewidth=2, markersize=6, label="Stereo ATE (brightness sweep)")

    # --- overexposure / underexposure annotations ---
    ax.text(0.15, clean_stereo_rmse + 0.0005, "← underexposure", fontsize=9, color="gray", ha="center")
    ax.text(5.0, clean_stereo_rmse + 0.0005, "overexposure →", fontsize=9, color="gray", ha="center")

    ax.set_xscale("log")
    ax.set_xlabel("Brightness scale factor α  (log scale)", fontsize=12)
    ax.set_ylabel("ATE RMSE (m)", fontsize=12)
    ax.set_title("Phase 1 — Stereo ATE vs Brightness/Exposure\n(asymmetric: underexposure cliff at α≈0.03, overexposure graceful)", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which="both")

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    print(f"Saved → {OUT_PNG}")

    # --- text summary ---
    print(f"\n{'─'*65}")
    print(f"  {'alpha':>8}  {'RMSE (m)':>10}  {'vs clean':>10}  {'vs mono':>10}  {'direction':>12}")
    print(f"{'─'*65}")
    print(f"  {'1.0 (clean)':>8}  {clean_stereo_rmse:>10.4f}  {'—':>10}  {clean_stereo_rmse/clean_mono_rmse:>9.1%}")
    for _, row in df.sort_values("alpha").iterrows():
        direction = "under" if row["alpha"] < 1.0 else "over"
        if row["failed"]:
            print(f"  {row['alpha']:>8.2f}  {'FAIL':>10}  {'':>10}  {'':>10}  {direction:>12}")
        else:
            pct_clean = (row["rmse"] - clean_stereo_rmse) / clean_stereo_rmse
            pct_mono  = row["rmse"] / clean_mono_rmse
            print(f"  {row['alpha']:>8.2f}  {row['rmse']:>10.4f}  {pct_clean:>+10.1%}  {pct_mono:>9.1%}  {direction:>12}")
    print(f"{'─'*65}")
    print(f"\nMono baseline:       {clean_mono_rmse:.4f} m")
    print(f"Underexposure cliff: α ≈ 0.03–0.05")
    print(f"Overexposure:        graceful, no failure up to α=10")


if __name__ == "__main__":
    main()
