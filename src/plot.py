import csv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def plot_results(results: dict, results_dir: Path, experiment_name: str = ""):
    """Grouped bar chart: conditions on x-axis, bars per (sequence, mode).

    Args:
        results: {(sequence, condition, mode): stats_dict_or_None}
        results_dir: where to save the plot and CSV
    """
    # Extract unique axes
    sequences = sorted({s for s, c, m in results})
    conditions = list(dict.fromkeys(c for s, c, m in results))  # preserve YAML order
    modes = sorted({m for s, c, m in results})

    # Build groups: one bar per (sequence, mode) combo
    group_labels = []
    for seq in sequences:
        for mode in modes:
            label = f"{seq}/{mode}" if len(sequences) > 1 else mode
            group_labels.append((seq, mode, label))

    n_groups = len(group_labels)
    n_conditions = len(conditions)
    x = np.arange(n_conditions)
    width = 0.8 / max(n_groups, 1)

    fig, ax = plt.subplots(figsize=(max(10, n_conditions * 1.5), 6))
    colors = plt.cm.Set2(np.linspace(0, 1, n_groups))

    for i, (seq, mode, label) in enumerate(group_labels):
        rmses = []
        for cond in conditions:
            stats = results.get((seq, cond, mode))
            rmse = stats.get("rmse") if stats and isinstance(stats, dict) else None
            rmses.append(rmse)

        offsets = x + (i - n_groups / 2 + 0.5) * width
        bar_vals = [r if r is not None else 0 for r in rmses]
        bars = ax.bar(offsets, bar_vals, width, label=label, color=colors[i])

        # Mark failures
        for j, r in enumerate(rmses):
            if r is None:
                ax.text(offsets[j], 0.001, "FAIL", ha="center", va="bottom",
                        fontsize=7, color="red", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("ATE RMSE (m)", fontsize=12)
    title = "ATE RMSE by Degradation Condition"
    if experiment_name:
        title += f"\n{experiment_name}"
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    plot_path = results_dir / "ate_by_condition.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved: {plot_path}")
    plt.close(fig)


def print_summary(results: dict, results_dir: Path):
    """Print summary table and save CSV.

    Args:
        results: {(sequence, condition, mode): stats_dict_or_None}
        results_dir: where to save summary.csv
    """
    csv_path = results_dir / "summary.csv"

    # Determine clean baselines for pct change
    baselines = {}
    for (seq, cond, mode), stats in results.items():
        if cond == "clean" and stats and isinstance(stats, dict) and "rmse" in stats:
            baselines[(seq, mode)] = stats["rmse"]

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sequence", "condition", "mode", "ate_rmse", "ate_mean",
                         "ate_median", "ate_std", "pct_vs_clean"])

        header = f"{'Seq':<10} {'Condition':<28} {'Mode':<8} {'RMSE':>10} {'Mean':>10} {'Median':>10} {'Std':>10} {'vs clean':>10}"
        print(f"\n{'='*len(header)}")
        print(header)
        print(f"{'─'*len(header)}")

        for (seq, cond, mode), stats in sorted(results.items()):
            if stats is None or not isinstance(stats, dict) or "rmse" not in stats:
                writer.writerow([seq, cond, mode, "FAIL", "", "", "", ""])
                print(f"{seq:<10} {cond:<28} {mode:<8} {'FAIL':>10}")
                continue

            rmse = stats["rmse"]
            mean = stats.get("mean", 0)
            median = stats.get("median", 0)
            std = stats.get("std", 0)

            baseline = baselines.get((seq, mode))
            if baseline and baseline > 0 and cond != "clean":
                pct = ((rmse - baseline) / baseline) * 100
                pct_str = f"{pct:+.1f}%"
            else:
                pct = ""
                pct_str = ""

            writer.writerow([seq, cond, mode, f"{rmse:.6f}", f"{mean:.6f}",
                             f"{median:.6f}", f"{std:.6f}", pct_str])
            print(f"{seq:<10} {cond:<28} {mode:<8} {rmse:>10.4f} {mean:>10.4f} "
                  f"{median:>10.4f} {std:>10.4f} {pct_str:>10}")

        print(f"{'='*len(header)}")

    print(f"Summary saved: {csv_path}")