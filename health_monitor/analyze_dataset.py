"""
Dataset analyzer — summarizes severity distribution per condition.

Run after build_dataset_index.py:
    python health_monitor/analyze_dataset.py
    python health_monitor/analyze_dataset.py --dataset data/health_monitor_dataset
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path

from config import DATASET_ROOT

SEVERITY_BINS = [
    (0.0,  0.25, "healthy     "),
    (0.25, 0.50, "mild        "),
    (0.50, 0.75, "degraded    "),
    (0.75, 1.00, "severe      "),
    (1.00, 1.01, "failed(=1.0)"),
]


def analyze(dataset_root: Path):
    seq_root = dataset_root / "sequence"
    if not seq_root.exists():
        raise FileNotFoundError(f"No sequence directory found at {seq_root}")

    # collect all (traj, condition, csv_path) tuples across all trajs
    all_data   = []   # list of {"traj": ..., "condition": ..., "severity": Series}
    csv_by_key = {}   # (traj, cond) -> csv_path (for n_inf computation in summary)

    print(f"\n{'Traj/Condition':<36} {'Windows':>7} {'Failed':>7} {'Mean sev':>9} "
          f"{'<0.25':>7} {'0.25-0.5':>9} {'0.5-0.75':>9} {'>0.75':>7} {'=1.0':>6}")
    print("─" * 108)

    for traj_dir in sorted(seq_root.iterdir()):
        if not traj_dir.is_dir():
            continue
        degrad_root = traj_dir / "degradations"
        if not degrad_root.exists():
            continue

        for cond_dir in sorted(degrad_root.iterdir()):
            if not cond_dir.is_dir():
                continue
            csv_path = cond_dir / "dataset.csv"
            if not csv_path.exists():
                print(f"  {traj_dir.name}/{cond_dir.name:<24}  no dataset.csv — run build_dataset_index.py")
                continue

            df  = pd.read_csv(csv_path)
            sev = df["severity"]
            n   = len(sev)
            n_inf = np.isinf(df["rte_stereo"]).sum()

            counts = []
            for lo, hi, _ in SEVERITY_BINS:
                if hi > 1.0:
                    counts.append((sev == 1.0).sum())
                else:
                    counts.append(((sev >= lo) & (sev < hi)).sum())

            label = f"{traj_dir.name}/{cond_dir.name}"
            print(f"  {label:<34}  {n:>6}  {n_inf:>6}  {sev.mean():>8.3f}  "
                  f"{counts[0]:>6}  {counts[1]:>8}  {counts[2]:>8}  {counts[3]:>6}  {counts[4]:>5}")

            all_data.append({"traj": traj_dir.name, "condition": cond_dir.name, "severity": sev})
            csv_by_key[(traj_dir.name, cond_dir.name)] = csv_path

    # overall summary
    if all_data:
        combined = pd.concat([d["severity"] for d in all_data], ignore_index=True)
        print("─" * 108)
        n = len(combined)
        n_inf = sum(np.isinf(pd.read_csv(p)["rte_stereo"]).sum() for p in csv_by_key.values())
        counts = []
        for lo, hi, _ in SEVERITY_BINS:
            if hi > 1.0:
                counts.append((combined == 1.0).sum())
            else:
                counts.append(((combined >= lo) & (combined < hi)).sum())
        print(f"  {'ALL':<34}  {n:>6}  {n_inf:>6}  {combined.mean():>8.3f}  "
              f"{counts[0]:>6}  {counts[1]:>8}  {counts[2]:>8}  {counts[3]:>6}  {counts[4]:>5}")

        print(f"\nSeverity breakdown (all conditions, n={n}):")
        for (lo, hi, label), count in zip(SEVERITY_BINS, counts):
            bar = "█" * int(40 * count / n)
            print(f"  {label}  {count:>5} ({100*count/n:5.1f}%)  {bar}")

    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=None,
                        help="Path to health_monitor_dataset root "
                             "(default: data/health_monitor_dataset)")
    args = parser.parse_args()

    dataset_root = Path(args.dataset) if args.dataset else DATASET_ROOT

    analyze(dataset_root)
