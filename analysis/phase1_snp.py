"""Phase 1 — Salt & pepper noise sweep analysis.

Reads results/snp_sweep/summary.csv and experiments/snp_sweep.yaml,
plots ATE RMSE vs noise density with clean stereo + mono baselines.

Usage:
    python analysis/phase1_snp.py
"""

import yaml
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SWEEP_CSV  = ROOT / "results" / "snp_sweep" / "summary.csv"
SWEEP_YAML = ROOT / "experiments" / "snp_sweep.yaml"
CLEAN_STEREO_STATS = ROOT / "results" / "clean" / "traj1" / "clean" / "stereo" / "stats.yaml"
CLEAN_MONO_STATS   = ROOT / "results" / "clean" / "traj1" / "clean" / "mono"   / "stats.yaml"
OUT_PNG = ROOT / "analysis" / "phase1_snp.png"


def load_density_map(yaml_path: Path) -> dict[str, float]:
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)
    return {k: float(v["density"]) for k, v in cfg["conditions"].items()}


def load_baseline(stats_path: Path) -> float:
    with open(stats_path) as f:
        return yaml.safe_load(f)["rmse"]


def main():
    density_map = load_density_map(SWEEP_YAML)
    clean_stereo_rmse = load_baseline(CLEAN_STEREO_STATS)
    clean_mono_rmse   = load_baseline(CLEAN_MONO_STATS)

    df = pd.read_csv(SWEEP_CSV)
    df = df[df["mode"] == "stereo"].copy()
    df["density"] = df["condition"].map(density_map)
    df["failed"]  = df["ate_rmse"] == "FAIL"
    df["rmse"]    = pd.to_numeric(df["ate_rmse"], errors="coerce")
    df = df.sort_values("density")

    success = df[~df["failed"]]
    failed  = df[df["failed"]]

    fig, ax = plt.subplots(figsize=(10, 5))

    # --- failure region ---
    if not failed.empty:
        fail_start = failed["density"].min()
        ax.axvspan(fail_start, df["density"].max() + 0.02, color="red", alpha=0.08,
                   label="SLAM failure region")
        ax.axvline(fail_start, color="red", linestyle="--", linewidth=1.2, alpha=0.7)
        ax.text(fail_start + 0.003, clean_mono_rmse * 1.01,
                f"density = {fail_start:.2f}", color="red", fontsize=9)

    # --- baselines ---
    ax.axhline(clean_stereo_rmse, color="steelblue", linestyle="--", linewidth=1.2,
               label=f"Clean stereo baseline ({clean_stereo_rmse:.4f} m)")
    ax.axhline(clean_mono_rmse, color="darkorange", linestyle="--", linewidth=1.2,
               label=f"Clean mono baseline ({clean_mono_rmse:.4f} m)")

    # --- sweep curve ---
    ax.plot(success["density"], success["rmse"], "o-", color="steelblue",
            linewidth=2, markersize=6, label="Stereo ATE (S&P sweep)")

    # --- flat region annotation ---
    flat = success[success["density"] <= 0.30]
    if not flat.empty:
        ax.annotate("Flat — SLAM unaffected\n(RANSAC absorbs corrupted keypoints)",
                    xy=(flat["density"].mean(), flat["rmse"].mean()),
                    xytext=(0.05, clean_stereo_rmse + 0.0025),
                    arrowprops=dict(arrowstyle="->", color="gray"),
                    fontsize=8.5, color="gray")

    ax.set_xlabel("Salt & pepper noise density (fraction of pixels corrupted)", fontsize=12)
    ax.set_ylabel("ATE RMSE (m)", fontsize=12)
    ax.set_title("Phase 1 — Stereo ATE vs Salt & Pepper Noise\n(binary cliff: flat until ~33%, then failure)", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    print(f"Saved → {OUT_PNG}")

    # --- text summary ---
    print(f"\n{'─'*60}")
    print(f"  {'density':>8}  {'RMSE (m)':>10}  {'vs clean':>10}  {'vs mono':>10}")
    print(f"{'─'*60}")
    print(f"  {'0 (clean)':>8}  {clean_stereo_rmse:>10.4f}  {'—':>10}  {clean_stereo_rmse/clean_mono_rmse:>9.1%}")
    for _, row in success.iterrows():
        pct_clean = (row["rmse"] - clean_stereo_rmse) / clean_stereo_rmse
        pct_mono  = row["rmse"] / clean_mono_rmse
        print(f"  {row['density']:>8.2f}  {row['rmse']:>10.4f}  {pct_clean:>+10.1%}  {pct_mono:>9.1%}")
    for _, row in failed.iterrows():
        print(f"  {row['density']:>8.2f}  {'FAIL':>10}")
    print(f"{'─'*60}")
    print(f"\nMono baseline: {clean_mono_rmse:.4f} m")
    print(f"Cliff edge:    density ≈ {fail_start:.2f}")


if __name__ == "__main__":
    main()
