"""
Phase 1 — Combined analysis for all 5 degradation types across both trajectories.

Reads results/<sweep>/summary.csv for each experiment and plots ATE RMSE vs
degradation parameter, with traj1 and traj2 as separate lines and clean
stereo + mono baselines for each trajectory.

Also generates individual PNGs per degradation type for inclusion in key_findings.tex.

Usage:
    python analysis/phase1_analysis.py
"""

import yaml
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

ROOT    = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
EXPS    = ROOT / "experiments"
OUT_DIR = ROOT / "analysis"

TRAJS = ["traj1", "traj2"]
TRAJ_COLORS = {
    "traj1": {"stereo": "steelblue",  "mono": "darkorange"},
    "traj2": {"stereo": "seagreen",   "mono": "firebrick"},
}
TRAJ_LABELS = {"traj1": "Room 1", "traj2": "Room 2"}

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_baseline(traj, mode="stereo"):
    p = RESULTS / "clean" / traj / "clean" / mode / "stats.yaml"
    with open(p) as f:
        return yaml.safe_load(f)["rmse"]


def load_sweep(exp_name, param_fn):
    """
    Load summary.csv for an experiment, map condition → param value,
    split by traj, return dict {traj: (success_df, failed_df)}.
    """
    csv  = RESULTS / exp_name / "summary.csv"
    cyml = EXPS / f"{exp_name}.yaml"
    with open(cyml) as f:
        cfg = yaml.safe_load(f)

    param_map = param_fn(cfg["conditions"])
    df = pd.read_csv(csv)
    df = df[df["mode"] == "stereo"].copy()
    df["param"]  = df["condition"].map(param_map)
    df["failed"] = df["ate_rmse"] == "FAIL"
    df["rmse"]   = pd.to_numeric(df["ate_rmse"], errors="coerce")
    df = df.dropna(subset=["param"]).sort_values("param")

    result = {}
    for traj in TRAJS:
        sub = df[df["sequence"] == traj]
        if sub.empty:
            continue
        result[traj] = (sub[~sub["failed"]], sub[sub["failed"]])
    return result


def draw_baselines(ax, traj, stereo_rmse, mono_rmse, draw_label=True):
    lbl_s = f"{TRAJ_LABELS[traj]} stereo baseline ({stereo_rmse:.4f} m)" if draw_label else None
    lbl_m = f"{TRAJ_LABELS[traj]} mono baseline ({mono_rmse:.4f} m)"     if draw_label else None
    ax.axhline(stereo_rmse, color=TRAJ_COLORS[traj]["stereo"],
               linestyle="--", linewidth=1.1, alpha=0.6, label=lbl_s)
    ax.axhline(mono_rmse, color=TRAJ_COLORS[traj]["mono"],
               linestyle=":",  linewidth=1.1, alpha=0.8, label=lbl_m)


def draw_cliff(ax, failed_df, param_col, y_ref, color, label=None):
    if failed_df.empty:
        return
    cliff = failed_df[param_col].min()
    ax.axvline(cliff, color=color, linestyle="--", linewidth=1.0, alpha=0.6)
    ax.text(cliff, y_ref * 1.02, f"{cliff:.2g}", color=color, fontsize=7.5,
            ha="left", va="bottom")


def finalize(ax, fig, out_path, title, xlabel):
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel("ATE RMSE (m)", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


# ── Per-degradation plots ─────────────────────────────────────────────────────

def plot_blur():
    data = load_sweep("blur_sweep_B",
                      lambda c: {k: float(v["sigma"]) for k, v in c.items()})
    fig, ax = plt.subplots(figsize=(10, 5))

    for traj, (success, failed) in data.items():
        s = load_baseline(traj, "stereo")
        m = load_baseline(traj, "mono")
        draw_baselines(ax, traj, s, m)
        ax.plot(success["param"], success["rmse"], "o-",
                color=TRAJ_COLORS[traj]["stereo"], linewidth=2, markersize=5,
                label=f"{TRAJ_LABELS[traj]} stereo ATE")
        draw_cliff(ax, failed, "param", m, TRAJ_COLORS[traj]["stereo"])

        crossover = success[success["rmse"] >= m]
        if not crossover.empty:
            cx, cy = crossover.iloc[0]["param"], crossover.iloc[0]["rmse"]
            ax.annotate(f"crossover\n({TRAJ_LABELS[traj]})",
                        xy=(cx, cy), xytext=(cx - 1.2, cy + 0.001),
                        arrowprops=dict(arrowstyle="->", color="gray"), fontsize=7.5)

    ax.set_xlim(left=0)
    finalize(ax, fig, OUT_DIR / "phase1_blur.png",
             "Phase 1 — Stereo ATE vs Gaussian Blur (σ)\nOption B: kernel = 6σ",
             "Gaussian blur sigma (σ)")


def plot_motion_blur():
    data = load_sweep("motion_blur_sweep",
                      lambda c: {k: int(v["length"]) for k, v in c.items()})
    fig, ax = plt.subplots(figsize=(10, 5))

    for traj, (success, failed) in data.items():
        s = load_baseline(traj, "stereo")
        m = load_baseline(traj, "mono")
        draw_baselines(ax, traj, s, m)
        ax.plot(success["param"], success["rmse"], "s-",
                color=TRAJ_COLORS[traj]["stereo"], linewidth=2, markersize=5,
                label=f"{TRAJ_LABELS[traj]} stereo ATE")
        draw_cliff(ax, failed, "param", m, TRAJ_COLORS[traj]["stereo"])

        crossover = success[success["rmse"] >= m]
        if not crossover.empty:
            cx, cy = crossover.iloc[0]["param"], crossover.iloc[0]["rmse"]
            ax.annotate(f"crossover\n({TRAJ_LABELS[traj]})",
                        xy=(cx, cy), xytext=(cx - 4, cy + 0.001),
                        arrowprops=dict(arrowstyle="->", color="gray"), fontsize=7.5)

    ax.set_xlim(left=0)
    finalize(ax, fig, OUT_DIR / "phase1_motion_blur.png",
             "Phase 1 — Stereo ATE vs Motion Blur (kernel length px)",
             "Motion blur kernel length (pixels)")


def plot_snp():
    data = load_sweep("snp_sweep",
                      lambda c: {k: float(v["density"]) for k, v in c.items()})
    fig, ax = plt.subplots(figsize=(10, 5))

    for traj, (success, failed) in data.items():
        s = load_baseline(traj, "stereo")
        m = load_baseline(traj, "mono")
        draw_baselines(ax, traj, s, m)
        ax.plot(success["param"], success["rmse"], "o-",
                color=TRAJ_COLORS[traj]["stereo"], linewidth=2, markersize=5,
                label=f"{TRAJ_LABELS[traj]} stereo ATE")
        draw_cliff(ax, failed, "param", m, TRAJ_COLORS[traj]["stereo"])

    ax.annotate("Flat — RANSAC absorbs\ncorrupted keypoints",
                xy=(0.15, 0.0075), xytext=(0.05, 0.010),
                arrowprops=dict(arrowstyle="->", color="gray"), fontsize=8, color="gray")
    ax.set_xlim(left=0)
    finalize(ax, fig, OUT_DIR / "phase1_snp.png",
             "Phase 1 — Stereo ATE vs Salt & Pepper Noise Density\n(flat then cliff — RANSAC absorbs random corruption)",
             "S&P noise density (fraction of pixels corrupted)")


def plot_brightness():
    data = load_sweep("brightness_sweep",
                      lambda c: {k: float(v["alpha"]) for k, v in c.items()})
    fig, ax = plt.subplots(figsize=(11, 5))

    for traj, (success, failed) in data.items():
        s = load_baseline(traj, "stereo")
        m = load_baseline(traj, "mono")
        draw_baselines(ax, traj, s, m)
        ax.plot(success["param"], success["rmse"], "o-",
                color=TRAJ_COLORS[traj]["stereo"], linewidth=2, markersize=5,
                label=f"{TRAJ_LABELS[traj]} stereo ATE")
        under_fail = failed[failed["param"] < 1.0]
        draw_cliff(ax, under_fail, "param", m, TRAJ_COLORS[traj]["stereo"])

    ax.axvline(1.0, color="green", linestyle=":", linewidth=1.2, alpha=0.6, label="Clean (α=1.0)")
    ax.set_xscale("log")
    ax.grid(True, alpha=0.25, which="both")
    finalize(ax, fig, OUT_DIR / "phase1_brightness.png",
             "Phase 1 — Stereo ATE vs Brightness Scale Factor α\n(asymmetric: underexposure cliff, overexposure graceful)",
             "Brightness scale factor α (log scale)")


def plot_occlusion():
    data = load_sweep("occlusion_sweep",
                      lambda c: {k: float(v["frac"]) for k, v in c.items()})
    fig, ax = plt.subplots(figsize=(10, 5))

    for traj, (success, failed) in data.items():
        s = load_baseline(traj, "stereo")
        m = load_baseline(traj, "mono")
        draw_baselines(ax, traj, s, m)
        ax.plot(success["param"], success["rmse"], "o-",
                color=TRAJ_COLORS[traj]["stereo"], linewidth=2, markersize=5,
                label=f"{TRAJ_LABELS[traj]} stereo ATE")
        draw_cliff(ax, failed, "param", m, TRAJ_COLORS[traj]["stereo"])

    ax.annotate("Flat — RANSAC absorbs\nocclusion patch",
                xy=(0.30, 0.0068), xytext=(0.10, 0.010),
                arrowprops=dict(arrowstyle="->", color="gray"), fontsize=8, color="gray")
    ax.set_xlim(left=0)
    finalize(ax, fig, OUT_DIR / "phase1_occlusion.png",
             "Phase 1 — Stereo ATE vs Occlusion Fraction\n(flat then cliff — RANSAC absorbs occluded region)",
             "Fraction of image area blacked out")


def plot_combined():
    """5-panel summary figure."""
    fig, axes = plt.subplots(1, 5, figsize=(26, 5))
    fig.suptitle("Phase 1 — Stereo ATE vs Degradation Severity  (Room 1 & Room 2)",
                 fontsize=13)

    sweeps = [
        ("blur_sweep_B",      lambda c: {k: float(v["sigma"])   for k, v in c.items()}, "σ",          "Gaussian Blur (σ)"),
        ("motion_blur_sweep", lambda c: {k: int(v["length"])    for k, v in c.items()}, "length (px)", "Motion Blur (px)"),
        ("snp_sweep",         lambda c: {k: float(v["density"]) for k, v in c.items()}, "density",     "Salt & Pepper"),
        ("brightness_sweep",  lambda c: {k: float(v["alpha"])   for k, v in c.items()}, "α",           "Brightness (α)"),
        ("occlusion_sweep",   lambda c: {k: float(v["frac"])    for k, v in c.items()}, "frac",        "Occlusion (frac)"),
    ]

    for ax, (exp, pfn, xlabel, title) in zip(axes, sweeps):
        data = load_sweep(exp, pfn)
        for traj, (success, failed) in data.items():
            s = load_baseline(traj, "stereo")
            m = load_baseline(traj, "mono")
            ax.axhline(s, color=TRAJ_COLORS[traj]["stereo"], linestyle="--",
                       linewidth=0.9, alpha=0.5)
            ax.axhline(m, color=TRAJ_COLORS[traj]["mono"],   linestyle=":",
                       linewidth=0.9, alpha=0.7)
            ax.plot(success["param"], success["rmse"], "o-",
                    color=TRAJ_COLORS[traj]["stereo"], linewidth=1.8, markersize=4,
                    label=TRAJ_LABELS[traj])
            if not failed.empty:
                ax.axvline(failed["param"].min(), color=TRAJ_COLORS[traj]["stereo"],
                           linestyle="--", linewidth=0.9, alpha=0.5)

        if exp == "brightness_sweep":
            ax.set_xscale("log")
            ax.grid(True, alpha=0.2, which="both")
        else:
            ax.grid(True, alpha=0.2)
            ax.set_xlim(left=0)

        ax.set_title(title, fontsize=10)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel("ATE RMSE (m)" if ax == axes[0] else "", fontsize=9)
        ax.legend(fontsize=7)

    # shared legend for baselines
    patches = [
        mpatches.Patch(color="steelblue", linestyle="--", label="Room 1 stereo baseline"),
        mpatches.Patch(color="darkorange", linestyle=":",  label="Room 1 mono baseline"),
        mpatches.Patch(color="seagreen",   linestyle="--", label="Room 2 stereo baseline"),
        mpatches.Patch(color="firebrick",  linestyle=":",  label="Room 2 mono baseline"),
    ]
    fig.legend(handles=patches, loc="lower center", ncol=4, fontsize=8,
               bbox_to_anchor=(0.5, -0.04))

    fig.tight_layout()
    fig.savefig(OUT_DIR / "phase1_combined.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {OUT_DIR / 'phase1_combined.png'}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    OUT_DIR.mkdir(exist_ok=True)
    print("Generating Phase 1 analysis plots...")
    plot_blur()
    plot_motion_blur()
    plot_snp()
    plot_brightness()
    plot_occlusion()
    plot_combined()
    print("\nAll plots saved to analysis/")
